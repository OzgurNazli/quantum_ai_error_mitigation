"""
diffusion_compiler.py

Production-ready Conditional Diffusion Model for Quantum Circuit Compilation
and Error Mitigation (QEM) — scaled to 40-qubit HPC workloads.

Architecture overview
─────────────────────
Conditioning path (★ refactored for scalability):
    • TargetObservableEncoder  — (B, num_observables) → (B, context_dim)
      Replaces the exponentially-large UnitaryEncoder.
      Conditions on Pauli expectation values / physical observables instead
      of a dense 2^n × 2^n unitary matrix.

    • SyndromeEncoder          — (B, num_syndromes)   → (B, context_dim)
      Replaces the dense NoiseProfileEncoder.
      Conditions on QEC stabiliser syndrome bits or noisy local observables.

Generation path (unchanged — scales linearly):
    • CircuitEmbedding   — token matrix (B, Q, D) ↔ continuous tensor (B, Q, D, E)
    • TimestepEmbedding  — scalar t → dense vector
    • UNetDenoiser       — 2-D U-Net over (num_qubits × circuit_depth) "image"
    • DDPMScheduler      — linear/cosine β-schedule, 40-step inference

Mathematical grounding:
    Forward process  : q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1−ᾱ_t)·I)
    Reverse process  : p(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²·I)
    CFG guidance     : ε_guided = ε_uncond + g·(ε_cond − ε_uncond)  [g=10]
    Infidelity metric: F(U,V) = 1 − |Tr(U†V)/2ⁿ|²
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── GATE VOCABULARY  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitVocab:
    """
    Gate vocabulary for circuit tokenization.

    Token convention (from notebook):
        0   → padding / end-of-circuit
        +k  → gate vocab[k-1] applied on this qubit at this time step
        −k  → this qubit is a *control* qubit for gate vocab[k-1] at this time step

    Attributes:
        gates          : Ordered list of gate name strings, e.g. ['h', 'cx', 'ccx'].
        token_to_gate  : {int → str}  mapping for decoding.
        gate_to_token  : {str → int}  mapping for encoding.
        vocab_size     : Total number of tokens including padding (len(gates) + 1).
    """
    gates: List[str]
    token_to_gate: Dict[int, str] = field(init=False, repr=False)
    gate_to_token: Dict[str, int] = field(init=False, repr=False)
    vocab_size: int = field(init=False, repr=False)

    def __post_init__(self):
        self.token_to_gate = {i + 1: g for i, g in enumerate(self.gates)}
        self.gate_to_token = {g: i + 1 for i, g in enumerate(self.gates)}
        self.vocab_size = len(self.gates) + 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── CIRCUIT EMBEDDING & DECODING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class CircuitEmbedding(nn.Module):
    """
    Bidirectional encoder/decoder between integer token circuits and
    continuous embedding tensors.

    Shape contract
    ──────────────
    encode  input : (B, num_qubits, circuit_depth)             [int64]
    encode  output: (B, num_qubits, circuit_depth, embed_dim)  [float32]
    decode  input : (B, num_qubits, circuit_depth, embed_dim)  [float32]
    decode  output: (B, num_qubits, circuit_depth)             [int64]
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        if embed_dim < vocab_size:
            raise ValueError(
                f"embed_dim={embed_dim} must be ≥ vocab_size={vocab_size} "
                "for orthonormal initialisation."
            )
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        basis = self._orthonormal_init(vocab_size, embed_dim)
        self.embedding = nn.Parameter(basis, requires_grad=True)

    @staticmethod
    def _orthonormal_init(n: int, d: int) -> Tensor:
        A = torch.randn(d, n)
        Q, _ = torch.linalg.qr(A)
        return Q[:, :n].T

    def forward(self, tokens: Tensor) -> Tensor:
        signs      = tokens.sign().float()
        abs_tokens = tokens.abs().clamp(max=self.vocab_size - 1)
        embedded   = self.embedding[abs_tokens]
        return embedded * signs.unsqueeze(-1)

    def decode(self, tensor: Tensor) -> Tensor:
        norm_t   = F.normalize(tensor,         dim=-1)
        norm_b   = F.normalize(self.embedding, dim=-1)
        cos_sim  = norm_t @ norm_b.T
        best_idx = cos_sim.abs().argmax(dim=-1)
        sign     = cos_sim.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1).sign()
        return (sign * best_idx.float()).long()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── CONDITIONING ENCODERS  ★ REFACTORED
#
# Both encoders now accept compact 1-D vectors instead of exponentially-large
# dense matrices.  At 40 qubits:
#
#   OLD UnitaryEncoder      input: (B, 2, 2^40, 2^40) ≈ 2.4 × 10^24 floats — IMPOSSIBLE
#   NEW TargetObservableEncoder input: (B, num_observables)  e.g. (B, 160)  — trivial
#
#   OLD NoiseProfileEncoder input: (B, 2 · 2^40) ≈ 2.2 × 10^12 floats — IMPOSSIBLE
#   NEW SyndromeEncoder     input: (B, num_syndromes)  e.g. (B, 80)   — trivial
# ══════════════════════════════════════════════════════════════════════════════

class TargetObservableEncoder(nn.Module):  # ★ replaces UnitaryEncoder
    """
    Projects a vector of target physical observables into a context vector.

    Motivation
    ──────────
    Conditioning on a dense 2^n × 2^n unitary is infeasible beyond ~25 qubits.
    Instead we condition on a compact set of Pauli expectation values or other
    physically meaningful scalars that *characterise* the target operation:

        ⟨P_i⟩_target  for a chosen set of Pauli strings P_i

    These can be computed cheaply from the target circuit description or from
    classical simulation of small patches, and they capture the action of the
    target unitary on the relevant observable subspace.

    Shape contract
    ──────────────
    Input : (B, num_observables)   — 1-D vector of real-valued observables
    Output: (B, context_dim)
    """

    def __init__(self, num_observables: int, context_dim: int) -> None:
        """
        Args:
            num_observables: Number of target observable values in the input vector.
                             Typical choices:
                               • 4^1 = 4   for single-qubit characterisation
                               • 4·n = 160 for n=40 single-body Pauli expectations
                               • Custom: any problem-specific observable count.
            context_dim    : Width of the output context vector.
        """
        super().__init__()
        self.num_observables = num_observables

        self.encoder = nn.Sequential(
            nn.Linear(num_observables, context_dim * 4),
            nn.SiLU(),
            nn.LayerNorm(context_dim * 4),
            nn.Linear(context_dim * 4, context_dim * 2),
            nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(self, observables: Tensor) -> Tensor:
        """
        Args:
            observables: Shape (B, num_observables).
                         Each element is a real-valued Pauli expectation value
                         or other physical observable in any numeric range
                         (no normalisation enforced here — caller's responsibility).

        Returns:
            Context vector of shape (B, context_dim).
        """
        return self.encoder(observables)


class SyndromeEncoder(nn.Module):  # ★ replaces NoiseProfileEncoder
    """
    Projects a vector of QEC syndrome measurements into a context vector.

    Motivation
    ──────────
    Conditioning on a dense 2^n-dimensional noisy state vector is infeasible
    at scale.  Instead we condition on a compact QEC syndrome vector:

        s ∈ {−1, 0, +1}^m  or  ℝ^m

    where each element is either a binary stabiliser syndrome bit
    (from a surface code / flag-qubit readout) or a noisy local
    observable (⟨Z_i⟩_noisy for each physical qubit i).

    A *null* (learned zero-signal) embedding is maintained for the
    unconditional branch required by Classifier-Free Guidance.

    Shape contract
    ──────────────
    Input : (B, num_syndromes)  — 1-D real-valued syndrome / observable vector
    Output: (B, context_dim)
    """

    def __init__(
        self,
        num_syndromes: int,
        context_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            num_syndromes: Number of syndrome / local-observable values.
                           Typical choices:
                             • Surface-d code: (d−1)² X-type + (d−1)² Z-type ancillas
                             • Flag-qubit readout: 1 bit per physical qubit → n bits
                             • Noisy local observables: ⟨Z_i⟩ for i=1..n → n values
            context_dim  : Width of the output context vector.
            dropout      : Dropout rate inside the encoder MLP.
        """
        super().__init__()
        self.num_syndromes = num_syndromes

        self.encoder = nn.Sequential(
            nn.Linear(num_syndromes, context_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )
        # Learnable null embedding for CFG unconditional branch — kept intact
        self.null_embedding = nn.Parameter(torch.zeros(context_dim))

    def forward(self, syndromes: Tensor, use_null: bool = False) -> Tensor:
        """
        Args:
            syndromes: Shape (B, num_syndromes).
                       Real-valued syndrome bits or local observables.
            use_null : If True, returns the learned null embedding broadcast
                       over the batch — used for the unconditional CFG branch.

        Returns:
            Context vector of shape (B, context_dim).
        """
        if use_null:
            return self.null_embedding.unsqueeze(0).expand(syndromes.shape[0], -1)
        return self.encoder(syndromes)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── TIMESTEP EMBEDDING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    Maps a scalar diffusion timestep t ∈ {0, …, T−1} to a dense vector.

    Shape contract
    ──────────────
    Input : (B,)
    Output: (B, time_embed_dim)
    """

    def __init__(self, time_embed_dim: int, max_timesteps: int = 1000) -> None:
        super().__init__()
        assert time_embed_dim % 2 == 0, "time_embed_dim must be even."
        half = time_embed_dim // 2

        freqs = torch.exp(
            -math.log(max_timesteps)
            * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        args = t.float()[:, None] * self.freqs[None, :]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(emb)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── U-NET DENOISER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Conv2D residual block with AdaGN-style timestep + context injection.

    Shape contract
    ──────────────
    x     : (B, in_channels,  num_qubits, circuit_depth)
    t_emb : (B, time_embed_dim)
    ctx   : (B, context_dim)
    output: (B, out_channels, num_qubits, circuit_depth)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        groups = lambda c: min(8, c)

        self.norm1 = nn.GroupNorm(groups(in_channels),  in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_proj    = nn.Linear(time_embed_dim, out_channels)
        self.context_proj = nn.Linear(context_dim,    out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor, ctx: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = h + self.context_proj(self.act(ctx))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between circuit features (query) and context vector (key/value).

    Shape contract
    ──────────────
    x  : (B, channels, num_qubits, circuit_depth)
    ctx: (B, context_dim)
    out: (B, channels, num_qubits, circuit_depth)
    """

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm     = nn.GroupNorm(min(8, channels), channels)
        self.ctx_proj = nn.Linear(context_dim, channels)
        self.attn     = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h  = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        kv = self.ctx_proj(ctx).unsqueeze(1)
        attended, _ = self.attn(h, kv, kv)
        attended    = attended.permute(0, 2, 1).reshape(B, C, H, W)
        return x + attended


class UNetDenoiser(nn.Module):
    """
    U-Net noise predictor.  Treats the circuit tensor as a 2-D feature map:
        height   = num_qubits
        width    = circuit_depth
        channels = embed_dim

    Shape contract
    ──────────────
    x    : (B, embed_dim,     num_qubits, circuit_depth)  ← noisy circuit tensor
    t_emb: (B, time_embed_dim)
    ctx  : (B, context_dim)
    out  : (B, embed_dim,     num_qubits, circuit_depth)  ← predicted noise ε
    """

    def __init__(
        self,
        embed_dim: int,
        context_dim: int,
        time_embed_dim: int,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        channels = [base_channels * m for m in channel_mults]

        self.input_proj = nn.Conv2d(embed_dim, base_channels, 1)

        self.enc_blocks  : nn.ModuleList = nn.ModuleList()
        self.downsamples : nn.ModuleList = nn.ModuleList()
        in_ch = base_channels

        for out_ch in channels:
            self.enc_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch,  out_ch, time_embed_dim, context_dim),
                ResidualBlock(out_ch, out_ch, time_embed_dim, context_dim),
            ]))
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 2, stride=2))
            in_ch = out_ch

        self.btn_res1 = ResidualBlock(in_ch, in_ch, time_embed_dim, context_dim)
        self.btn_attn = CrossAttentionBlock(in_ch, context_dim, num_heads)
        self.btn_res2 = ResidualBlock(in_ch, in_ch, time_embed_dim, context_dim)

        self.upsamples  : nn.ModuleList = nn.ModuleList()
        self.dec_blocks : nn.ModuleList = nn.ModuleList()

        for out_ch in reversed(channels):
            self.upsamples.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.dec_blocks.append(nn.ModuleList([
                ResidualBlock(out_ch * 2, out_ch, time_embed_dim, context_dim),
                ResidualBlock(out_ch,     out_ch, time_embed_dim, context_dim),
            ]))
            in_ch = out_ch

        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, embed_dim, 1),
        )

    def forward(self, x: Tensor, t_emb: Tensor, ctx: Tensor) -> Tensor:
        h = self.input_proj(x)

        skips: List[Tensor] = []
        for (res1, res2), down in zip(self.enc_blocks, self.downsamples):
            h = res1(h, t_emb, ctx)
            h = res2(h, t_emb, ctx)
            skips.append(h)
            h = down(h)

        h = self.btn_res1(h, t_emb, ctx)
        h = self.btn_attn(h, ctx)
        h = self.btn_res2(h, t_emb, ctx)

        for up, (res1, res2), skip in zip(
            self.upsamples, self.dec_blocks, reversed(skips)
        ):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.pad(h, [
                    0, skip.shape[-1] - h.shape[-1],
                    0, skip.shape[-2] - h.shape[-2],
                ])
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb, ctx)
            h = res2(h, t_emb, ctx)

        return self.output_proj(h)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── DDPM NOISE SCHEDULER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """
    DDPM noise scheduler for the forward and reverse diffusion processes.

    Forward  : x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε,    ε ~ N(0,I)
    Reverse  : x_{t-1} = (1/√α_t)·(x_t − β_t/√(1−ᾱ_t)·ε_θ) + σ_t·z
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 40,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        self.num_train_timesteps = num_train_timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'.")

        alphas      = 1.0 - betas
        alphas_cump = torch.cumprod(alphas, dim=0)

        self.betas         = betas
        self.alphas        = alphas
        self.alphas_cump   = alphas_cump
        self.sqrt_acp      = alphas_cump.sqrt()
        self.sqrt_one_macp = (1.0 - alphas_cump).sqrt()

        self.set_inference_timesteps(num_inference_steps)

    @staticmethod
    def _cosine_schedule(T: int, s: float = 0.008) -> Tensor:
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        f     = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
        acp   = f / f[0]
        betas = 1.0 - acp[1:] / acp[:-1]
        return betas.clamp(0.0, 0.999)

    def set_inference_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1).long() * ratio

    def to(self, device: torch.device) -> "DDPMScheduler":
        for attr in ("betas", "alphas", "alphas_cump",
                     "sqrt_acp", "sqrt_one_macp", "timesteps"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        extra   = [1] * (x0.dim() - 1)
        s_acp   = self.sqrt_acp[t].reshape(-1, *extra)
        s_1macp = self.sqrt_one_macp[t].reshape(-1, *extra)
        return s_acp * x0 + s_1macp * noise

    def step(self, eps_pred: Tensor, t: int, x_t: Tensor) -> Tensor:
        beta_t      = self.betas[t]
        alpha_t     = self.alphas[t]
        acp_t       = self.alphas_cump[t]
        sqrt_1m_acp = self.sqrt_one_macp[t]

        mean = (x_t - (beta_t / sqrt_1m_acp) * eps_pred) / alpha_t.sqrt()

        if t > 0:
            acp_prev = self.alphas_cump[t - 1]
            variance = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
            mean     = mean + variance.clamp(min=1e-20).sqrt() * torch.randn_like(x_t)

        return mean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── MAIN MODULE: DiffusionCompiler  ★ REFACTORED
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionCompiler(nn.Module):
    """
    Conditional Diffusion Model for Quantum Circuit Compilation & Error Mitigation.
    Designed for HPC-scale workloads up to 40+ qubits.

    Scalability design
    ──────────────────
    The conditioning path no longer references 2^num_qubits anywhere.
    All exponential blow-up has been eliminated:

        Condition A — target_observables  : (B, num_observables)
            A fixed-length vector of Pauli expectation values ⟨P_i⟩ computed
            from the target circuit.  `num_observables` is a hyperparameter
            chosen by the user — typically O(n) or O(n²), never O(2^n).

        Condition B — syndromes           : (B, num_syndromes)
            A fixed-length vector of QEC stabiliser syndrome bits or noisy
            local observables ⟨Z_i⟩_noisy.  Again O(n), never O(2^n).

    The generation path (U-Net + circuit token matrix) was already linear in
    num_qubits and circuit_depth — it remains unchanged.

    Dynamic tensor shapes
    ─────────────────────
        target_observables : (B, num_observables)        ← NO 2^n dependence
        syndromes          : (B, num_syndromes)          ← NO 2^n dependence
        circuit tensor     : (B, embed_dim, num_qubits, circuit_depth)  ← O(n)
        output token matrix: (B, num_qubits, circuit_depth)             ← O(n)
    """

    def __init__(
        self,
        num_qubits: int,
        circuit_depth: int,
        vocab: CircuitVocab,
        num_observables: int,                      # ★ replaces unitary condition
        num_syndromes: int,                        # ★ replaces noisy state condition
        embed_dim: int = 16,
        context_dim: int = 256,
        time_embed_dim: int = 128,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_heads: int = 4,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 40,
        cfg_scale: float = 10.0,
        noise_schedule: str = "linear",
        syndrome_encoder_dropout: float = 0.1,
    ) -> None:
        """
        Args:
            num_qubits        : Number of physical qubits.
                                Used only to size the circuit token matrix
                                (num_qubits × circuit_depth).  NOT used to
                                compute 2^num_qubits anywhere.
            circuit_depth     : Maximum number of gate time-steps per circuit.
            vocab             : Gate vocabulary object (see CircuitVocab).
            num_observables   : Length of the target observable vector.
                                Recommended: 4·num_qubits (one-body Pauli
                                expectations on each qubit) or any custom value.
                                Typical range: 10 – 1000.  Must be ≥ 1.
            num_syndromes     : Length of the QEC syndrome / noisy-observable vector.
                                For a distance-d surface code: 2·(d−1)².
                                For simple local readout: num_qubits.
                                Must be ≥ 1.
            embed_dim         : Embedding width per token. Must be ≥ vocab.vocab_size.
            context_dim       : Width of the unified conditioning context vector.
            time_embed_dim    : Width of the timestep embedding (must be even).
            base_channels     : U-Net base channel count at the shallowest level.
            channel_mults     : Per-level channel multipliers for the U-Net.
            num_heads         : Cross-attention heads in the U-Net bottleneck.
            num_train_timesteps: Total DDPM training timesteps T.
            num_inference_steps: Reverse diffusion steps at inference (default 40,
                                 matching pipeline.scheduler.set_timesteps(40)).
            cfg_scale         : Classifier-Free Guidance scale g
                                (default 10, matching notebook's g=10).
            noise_schedule    : Beta schedule: "linear" or "cosine".
            syndrome_encoder_dropout: Dropout rate inside SyndromeEncoder MLP.
        """
        super().__init__()

        self.num_qubits    = num_qubits
        self.circuit_depth = circuit_depth
        self.vocab         = vocab
        self.embed_dim     = embed_dim
        self.cfg_scale     = cfg_scale

        # ── Sub-modules ───────────────────────────────────────────────────────

        self.circuit_embedding = CircuitEmbedding(
            vocab_size=vocab.vocab_size,
            embed_dim=embed_dim,
        )

        self.timestep_embedding = TimestepEmbedding(
            time_embed_dim=time_embed_dim,
            max_timesteps=num_train_timesteps,
        )

        # ★ Replaces UnitaryEncoder — no 2^n dependence
        self.observable_encoder = TargetObservableEncoder(
            num_observables=num_observables,
            context_dim=context_dim,
        )

        # ★ Replaces NoiseProfileEncoder — no 2^n dependence
        self.syndrome_encoder = SyndromeEncoder(
            num_syndromes=num_syndromes,
            context_dim=context_dim,
            dropout=syndrome_encoder_dropout,
        )

        # Fuses observable context + syndrome context → single context vector
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim * 2),
            nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )

        self.unet = UNetDenoiser(
            embed_dim=embed_dim,
            context_dim=context_dim,
            time_embed_dim=time_embed_dim,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_heads=num_heads,
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            schedule=noise_schedule,
        )

    # ── Context encoding helper ───────────────────────────────────────────────

    def _encode_context(  # ★ signature updated
        self,
        target_observables: Tensor,
        syndromes: Tensor,
        use_null_condition: bool = False,
    ) -> Tensor:
        """
        Encodes target observables and QEC syndromes into a unified context vector.

        Args:
            target_observables: Pauli expectation values of the target operation.
                                Shape (B, num_observables).
                                Values are real-valued scalars (any range).
            syndromes         : QEC syndrome bits or noisy local observables.
                                Shape (B, num_syndromes).
            use_null_condition: If True, the syndrome encoder returns its learned
                                null embedding — used for the unconditional CFG
                                branch without zeroing out observable context.

        Returns:
            Fused context vector of shape (B, context_dim).
        """
        obs_ctx  = self.observable_encoder(target_observables)
        syn_ctx  = self.syndrome_encoder(syndromes, use_null=use_null_condition)
        combined = torch.cat([obs_ctx, syn_ctx], dim=-1)
        return self.context_fusion(combined)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(  # ★ signature updated
        self,
        circuit_tokens: Tensor,
        target_observables: Tensor,
        syndromes: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Training forward pass.  Predicts the noise ε added at step t.

        Training objective:  L = ‖ ε_θ(x_t, t, ctx) − ε ‖²

        Args:
            circuit_tokens    : Clean gate-token matrix.
                                Shape (B, num_qubits, circuit_depth) [int64].
            target_observables: Target Pauli expectation values.
                                Shape (B, num_observables) [float32].
            syndromes         : QEC syndromes or noisy local observables.
                                Shape (B, num_syndromes) [float32].
            t                 : Diffusion timestep per sample.
                                Shape (B,) [int64 ∈ {0, …, T−1}].

        Returns:
            Tuple of:
                eps_pred : Predicted noise, shape (B, embed_dim, num_qubits, circuit_depth).
                eps_true : Ground-truth noise ε (same shape) — used to compute loss.
        """
        # Embed clean tokens → continuous circuit tensor
        x0 = self.circuit_embedding(circuit_tokens)   # (B, Q, D, E)
        x0 = x0.permute(0, 3, 1, 2)                  # (B, E, Q, D)

        # Sample noise and corrupt x0 → x_t
        eps_true = torch.randn_like(x0)
        x_t      = self.scheduler.add_noise(x0, eps_true, t)

        # Encode conditioning
        ctx   = self._encode_context(target_observables, syndromes)
        t_emb = self.timestep_embedding(t)

        # Predict noise
        eps_pred = self.unet(x_t, t_emb, ctx)
        return eps_pred, eps_true

    # ── Inference: reverse-diffusion sampling ─────────────────────────────────

    @torch.no_grad()
    def generate_mitigating_circuit(  # ★ signature updated
        self,
        target_observables: Tensor,
        syndromes: Tensor,
        num_samples: int = 1,
        cfg_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generates mitigating quantum circuits via guided reverse diffusion.

        Uses Classifier-Free Guidance (CFG):
            ε_guided = ε_uncond + g · (ε_cond − ε_uncond)
        where g is the CFG scale (default g=10, matching notebook).

        Args:
            target_observables: Target Pauli expectation values.
                                Shape (B, num_observables) OR (num_observables,).
                                Neither dimension involves 2^num_qubits.
            syndromes         : QEC syndromes or noisy local observables.
                                Shape (B, num_syndromes) OR (num_syndromes,).
                                Neither dimension involves 2^num_qubits.
            num_samples       : Number of circuit candidates generated in parallel.
            cfg_scale         : Override CFG scale g.  None → uses self.cfg_scale.
            device            : Target device.  None → inferred from model params.
            seed              : Optional RNG seed for reproducible sampling.

        Returns:
            Dict with three keys:

            "token_matrix"  : Tensor (num_samples, num_qubits, circuit_depth) [int64]
                Decoded gate-token matrix.
                    +k  → gate vocab.token_to_gate[k] on this qubit
                    −k  → control qubit for gate k
                     0  → padding (circuit ends here)

            "raw_tensor"    : Tensor (num_samples, embed_dim, num_qubits, circuit_depth)
                Raw continuous output before discrete decoding.

            "cudaq_params"  : List[Dict] of length num_samples.
                Each dict contains:
                    "gates"    : List[(gate_name: str,
                                       qubits   : List[int],
                                       time_step: int)]
                    "n_qubits" : int
                    "depth"    : int  — last non-padding column + 1
        """
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        g = cfg_scale if cfg_scale is not None else self.cfg_scale

        # ── Normalise batch dimensions ────────────────────────────────────────
        if target_observables.dim() == 1:
            target_observables = target_observables.unsqueeze(0)
        if syndromes.dim() == 1:
            syndromes = syndromes.unsqueeze(0)

        if target_observables.shape[0] == 1 and num_samples > 1:
            target_observables = target_observables.expand(num_samples, -1)
            syndromes          = syndromes.expand(num_samples, -1)

        target_observables = target_observables.to(device)
        syndromes          = syndromes.to(device)

        # ── Encode conditional + unconditional contexts for CFG ───────────────
        ctx_cond   = self._encode_context(
            target_observables, syndromes, use_null_condition=False
        )
        ctx_uncond = self._encode_context(
            target_observables, syndromes, use_null_condition=True
        )

        # ── Start from pure noise in circuit-tensor space ─────────────────────
        # Shape: (num_samples, embed_dim, num_qubits, circuit_depth)
        x_t = torch.randn(
            num_samples, self.embed_dim, self.num_qubits, self.circuit_depth,
            device=device,
        )

        # ── Reverse diffusion loop (T → 0, 40 steps) ─────────────────────────
        self.scheduler.to(device)

        for t_val in self.scheduler.timesteps:
            t_batch = t_val.expand(num_samples)
            t_emb   = self.timestep_embedding(t_batch.float())

            eps_cond   = self.unet(x_t, t_emb, ctx_cond)
            eps_uncond = self.unet(x_t, t_emb, ctx_uncond)

            # Classifier-Free Guidance (g=10 from notebook)
            eps_guided = eps_uncond + g * (eps_cond - eps_uncond)

            x_t = self.scheduler.step(eps_guided, int(t_val.item()), x_t)

        # ── Decode continuous tensor → integer token matrix ───────────────────
        raw_tensor   = x_t.clone()
        token_matrix = self.circuit_embedding.decode(x_t.permute(0, 2, 3, 1))

        cudaq_params = self._tokens_to_cudaq_params(token_matrix)

        return {
            "token_matrix" : token_matrix,
            "raw_tensor"   : raw_tensor,
            "cudaq_params" : cudaq_params,
        }

    # ── Token matrix → CUDA-Q parameters  (unchanged) ────────────────────────

    def _tokens_to_cudaq_params(self, token_matrix: Tensor) -> List[Dict]:
        """
        Parses a batch of token matrices into CUDA-Q compatible gate-parameter dicts.

        Args:
            token_matrix: (B, num_qubits, circuit_depth) [int64].

        Returns:
            List of dicts, one per sample:
                "gates"    : List of (gate_name, [qubit_indices], time_step).
                "n_qubits" : int
                "depth"    : int  — last non-padding column + 1.
        """
        results: List[Dict] = []
        np_matrix = token_matrix.cpu().numpy()

        for b in range(np_matrix.shape[0]):
            mat, gates, depth = np_matrix[b], [], 0

            for t in range(self.circuit_depth):
                col = mat[:, t]
                if not np.any(col != 0):
                    break
                depth = t + 1
                target_qubits  = np.where(col > 0)[0].tolist()
                control_qubits = np.where(col < 0)[0].tolist()
                for tq in target_qubits:
                    gate_name = self.vocab.token_to_gate.get(int(col[tq]))
                    if gate_name is None:
                        continue
                    gates.append((gate_name, sorted(control_qubits) + [tq], t))

            results.append({"gates": gates, "n_qubits": self.num_qubits, "depth": depth})

        return results

    # ── Utility: infidelity  (unchanged) ─────────────────────────────────────

    @staticmethod
    def compute_infidelity(
        want_unitary: np.ndarray,
        got_unitary: np.ndarray,
    ) -> float:
        """
        Infidelity(U, V) = 1 − |Tr(U†V) / 2^n|²

        Used to score generated circuits when a reference unitary is available
        (e.g. classical simulation of small patches up to ~25 qubits).
        """
        n       = want_unitary.shape[0]
        overlap = np.trace(want_unitary.conj().T @ got_unitary) / n
        return float(1.0 - abs(overlap) ** 2)

    # ── Factory method ★ updated ──────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> "DiffusionCompiler":
        """
        Constructs a DiffusionCompiler from a flat configuration dict.

        Required keys:
            "gates"            : List[str]  — gate vocabulary
            "num_observables"  : int        — length of observable condition vector
            "num_syndromes"    : int        — length of syndrome condition vector

        All other keys map 1-to-1 to __init__ parameters.

        Example:
            model = DiffusionCompiler.from_config({
                "num_qubits"         : 40,
                "circuit_depth"      : 12,
                "gates"              : ["h", "cx", "z", "x", "ccx", "swap"],
                "num_observables"    : 160,   # 4 * 40  single-body Paulis
                "num_syndromes"      : 40,    # one ⟨Z_i⟩ per qubit
                "embed_dim"          : 16,
                "context_dim"        : 256,
                "cfg_scale"          : 10.0,
                "num_inference_steps": 40,
            })

        Args:
            config: Configuration dict (shallow-copied — caller's dict is not mutated).
            device: If provided, moves the model to this device after construction.

        Returns:
            Initialised DiffusionCompiler.
        """
        cfg   = dict(config)
        vocab = CircuitVocab(gates=cfg.pop("gates"))
        model = cls(vocab=vocab, **cfg)
        if device is not None:
            model = model.to(device)
        return model


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST  —  python diffusion_compiler.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    NUM_QUBITS = 40   # ← the actual target scale

    model = DiffusionCompiler.from_config({
        "num_qubits"          : NUM_QUBITS,
        "circuit_depth"       : 12,
        "gates"               : ["h", "cx", "z", "x", "ccx", "swap"],
        # 4 * n single-body Pauli expectations (X,Y,Z per qubit + identity)
        "num_observables"     : 4 * NUM_QUBITS,
        # one noisy ⟨Z_i⟩ readout per physical qubit
        "num_syndromes"       : NUM_QUBITS,
        "embed_dim"           : 16,
        "context_dim"         : 128,
        "time_embed_dim"      : 64,
        "base_channels"       : 32,
        "channel_mults"       : (1, 2),
        "num_train_timesteps" : 1000,
        "num_inference_steps" : 40,
        "cfg_scale"           : 10.0,
    })

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DiffusionCompiler (40-qubit scale) | trainable params: {n_params:,}")
    print(f"  TargetObservableEncoder input dim : {4 * NUM_QUBITS}   (was 2·2^40 ≈ 2.2T)")
    print(f"  SyndromeEncoder input dim         : {NUM_QUBITS}   (was 2·2^40 ≈ 2.2T)")

    # Dummy conditions — all O(n), no 2^n anywhere
    obs      = torch.randn(1, 4 * NUM_QUBITS)   # (1, 160)
    syndromes = torch.randn(1, NUM_QUBITS)       # (1, 40)

    result = model.generate_mitigating_circuit(
        target_observables=obs,
        syndromes=syndromes,
        num_samples=4,
        seed=42,
    )

    print(f"\ntoken_matrix shape : {result['token_matrix'].shape}")
    print(f"raw_tensor shape   : {result['raw_tensor'].shape}")
    print(f"\nFirst circuit (CUDA-Q params):")
    for gate_name, qubits, t_step in result["cudaq_params"][0]["gates"]:
        print(f"  t={t_step:02d}  {gate_name:6s}  qubits={qubits}")
