"""
diffusion_compiler.py

Production-ready Conditional Diffusion Model for Quantum Circuit Compilation
and Error Mitigation (QEM).

Extracted and refactored from NVIDIA's academic notebook:
    "01_compiling_unitaries_using_diffusion_models.ipynb"

Mathematical grounding:
    Forward process  : q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1−ᾱ_t)·I)
    Reverse process  : p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²·I)
    CFG guidance     : ε_guided = ε_uncond + g·(ε_cond − ε_uncond)   [g=10 in notebook]
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
# SECTION 1 ── GATE VOCABULARY
# Maps integer token IDs ↔ gate name strings.
# Notebook reference: vocab_list = ['h', 'cx', 'z', 'x', 'ccx', 'swap']
# Convention: 0 = padding, +k = gate k on target qubit, −k = control qubit for gate k
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitVocab:
    """
    Gate vocabulary for circuit tokenization.

    Token convention (from notebook):
        0   → padding / end-of-circuit
        +k  → gate vocab[k-1] applied on this qubit at this time step
        −k  → this qubit is a *control* qubit for gate vocab[k-1] at this time step

    Attributes:
        gates          : Ordered list of gate name strings, e.g. ['h', 'cx', 'ccx'].
        token_to_gate  : {int → str}  mapping for decoding.
        gate_to_token  : {str → int}  mapping for encoding.
        vocab_size     : Total number of tokens including padding (len(gates) + 1).
    """
    gates: List[str]
    token_to_gate: Dict[int, str] = field(init=False, repr=False)
    gate_to_token: Dict[str, int] = field(init=False, repr=False)
    vocab_size: int = field(init=False, repr=False)

    def __post_init__(self):
        # token 0 is reserved for padding
        self.token_to_gate = {i + 1: g for i, g in enumerate(self.gates)}
        self.gate_to_token = {g: i + 1 for i, g in enumerate(self.gates)}
        self.vocab_size = len(self.gates) + 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── CIRCUIT EMBEDDING & DECODING
#
# Notebook section: "Preparing Quantum Circuit Data for the Model"
#
# A circuit is a (num_qubits × circuit_depth) integer token matrix.
# Each integer is replaced by an orthonormal basis vector of dim `embed_dim`
# (sign is preserved for control-qubit tokens).
#
# Decoding (notebook section "Decoding the Generated Tensors"):
#   1. Compute cosine similarity of each position vector against all basis vectors.
#   2. Token = argmax(|cos_sim|), sign = sign(cos_sim at that index).
# ══════════════════════════════════════════════════════════════════════════════

class CircuitEmbedding(nn.Module):
    """
    Bidirectional encoder/decoder between integer token circuits and
    continuous embedding tensors.

    Shape contract
    ──────────────
    encode  input : (B, num_qubits, circuit_depth)             [int64]
    encode  output: (B, num_qubits, circuit_depth, embed_dim)  [float32]
    decode  input : (B, num_qubits, circuit_depth, embed_dim)  [float32]
    decode  output: (B, num_qubits, circuit_depth)             [int64]
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Args:
            vocab_size : Number of distinct tokens *excluding* padding (= len(gates)).
                         The actual lookup index range is [0, vocab_size].
            embed_dim  : Embedding vector width.  Must satisfy embed_dim >= vocab_size
                         so QR decomposition can yield vocab_size orthonormal rows.
        """
        super().__init__()
        if embed_dim < vocab_size:
            raise ValueError(
                f"embed_dim={embed_dim} must be ≥ vocab_size={vocab_size} "
                "for orthonormal initialisation."
            )
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

        # Learnable basis matrix, initialised to orthonormal columns via QR.
        # Shape: (vocab_size, embed_dim)
        basis = self._orthonormal_init(vocab_size, embed_dim)
        self.embedding = nn.Parameter(basis, requires_grad=True)

    # ── initialisation ────────────────────────────────────────────────────────

    @staticmethod
    def _orthonormal_init(n: int, d: int) -> Tensor:
        """
        Returns n orthonormal row-vectors of dimension d via QR decomposition.

        Notebook ref: generate_orthonormal_vectors_qr(vocab_length, d)

        Args:
            n: Number of vectors to generate (vocab_size).
            d: Dimension of each vector (embed_dim).

        Returns:
            Tensor of shape (n, d).
        """
        A = torch.randn(d, n)
        Q, _ = torch.linalg.qr(A)   # Q: (d, n) with orthonormal columns
        return Q[:, :n].T            # (n, d) — each row is a unit vector

    # ── forward (tokenised → embedded) ───────────────────────────────────────

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Maps an integer token matrix to a continuous embedding tensor.

        Args:
            tokens: Integer tensor of shape (B, num_qubits, circuit_depth).
                    Positive = gate applied on qubit, negative = control qubit,
                    zero = padding.

        Returns:
            Embedded tensor of shape (B, num_qubits, circuit_depth, embed_dim).
        """
        signs      = tokens.sign().float()                        # (B, Q, D)
        abs_tokens = tokens.abs().clamp(max=self.vocab_size - 1)  # guard OOV

        embedded = self.embedding[abs_tokens]                     # (B, Q, D, E)
        embedded = embedded * signs.unsqueeze(-1)                 # apply sign
        return embedded

    # ── decode (embedded → tokenised) ────────────────────────────────────────

    def decode(self, tensor: Tensor) -> Tensor:
        """
        Recovers an integer token matrix from a continuous embedding tensor.

        Algorithm (notebook section "Decoding the Generated Tensors"):
            cos_sim  = normalised(tensor) @ normalised(basis).T
            token    = argmax(|cos_sim|)       # closest basis vector
            sign     = sign(cos_sim[token])    # determines control vs target

        Args:
            tensor: Float tensor of shape (B, num_qubits, circuit_depth, embed_dim).

        Returns:
            Token tensor of shape (B, num_qubits, circuit_depth) [int64].
        """
        # Normalise along embedding axis
        norm_t = F.normalize(tensor,         dim=-1)   # (..., E)
        norm_b = F.normalize(self.embedding, dim=-1)   # (V, E)

        cos_sim  = norm_t @ norm_b.T                   # (..., V)
        abs_cos  = cos_sim.abs()
        best_idx = abs_cos.argmax(dim=-1)              # (...) in [0, V)

        # Recover sign from cosine similarity at the winning index
        sign = cos_sim.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1).sign()

        return (sign * best_idx.float()).long()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── CONDITIONING ENCODERS
#
# The notebook conditions the model on a target unitary U, supplied as two
# real-valued (2^n × 2^n) matrices (real part, imaginary part):
#   U_tensor = torch.stack([U_r, U_i], dim=0)  →  shape (2, 2^n, 2^n)
#
# For QEM we add a second condition: the *noisy* quantum state or noise
# profile that guides the model towards the *correcting* unitary.
# ══════════════════════════════════════════════════════════════════════════════

class UnitaryEncoder(nn.Module):
    """
    Projects a complex unitary matrix (split into real + imaginary channels)
    into a fixed-width context vector.

    Notebook ref:
        U_r, U_i = torch.Tensor(np.real(U)), torch.Tensor(np.imag(U))
        U_tensor = torch.stack([U_r, U_i], dim=0)   # shape (2, 2^n, 2^n)

    Shape contract
    ──────────────
    Input : (B, 2, 2**num_qubits, 2**num_qubits)
    Output: (B, context_dim)

    The flat input size scales as  2 · (2^num_qubits)².
    """

    def __init__(self, num_qubits: int, context_dim: int) -> None:
        """
        Args:
            num_qubits : Number of qubits — sets unitary dimensions dynamically.
            context_dim: Width of the output context vector.
        """
        super().__init__()
        hilbert_dim    = 2 ** num_qubits
        flat_input_dim = 2 * hilbert_dim * hilbert_dim  # real + imag, flattened

        self.encoder = nn.Sequential(
            nn.Linear(flat_input_dim, context_dim * 4),
            nn.SiLU(),
            nn.LayerNorm(context_dim * 4),
            nn.Linear(context_dim * 4, context_dim * 2),
            nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim),
        )

    def forward(self, unitary: Tensor) -> Tensor:
        """
        Args:
            unitary: Shape (B, 2, 2**num_qubits, 2**num_qubits).
                     unitary[:, 0] = real part,  unitary[:, 1] = imaginary part.

        Returns:
            Context vector of shape (B, context_dim).
        """
        flat = unitary.reshape(unitary.shape[0], -1)  # (B, 2·H²)
        return self.encoder(flat)


class NoiseProfileEncoder(nn.Module):
    """
    Encodes a noise profile or noisy quantum state into a context vector.

    This is the QEM-specific conditioning signal.  The model learns to map
    a noise descriptor onto the space of correcting gate sequences.

    Supported condition shapes (after external flattening):
        • Noisy state vector  : (B, 2 · 2**num_qubits)  — real+imag parts
        • Arbitrary descriptor: (B, condition_input_dim)

    A *null* (learned zero-signal) embedding is maintained for the
    unconditional branch required by Classifier-Free Guidance.

    Shape contract
    ──────────────
    Input : (B, condition_input_dim)
    Output: (B, context_dim)
    """

    def __init__(
        self,
        condition_input_dim: int,
        context_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            condition_input_dim: Flat dimension of the noise condition vector.
            context_dim         : Width of the output context vector.
            dropout             : Dropout applied inside encoder (regularisation).
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(condition_input_dim, context_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )
        # Learnable null embedding for CFG unconditional branch
        self.null_embedding = nn.Parameter(torch.zeros(context_dim))

    def forward(self, condition: Tensor, use_null: bool = False) -> Tensor:
        """
        Args:
            condition: Shape (B, condition_input_dim).
            use_null : If True, returns the learned null embedding (no condition).
                       Used for the unconditional branch of CFG.

        Returns:
            Context vector of shape (B, context_dim).
        """
        if use_null:
            return self.null_embedding.unsqueeze(0).expand(condition.shape[0], -1)
        return self.encoder(condition)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── TIMESTEP EMBEDDING
#
# Standard sinusoidal positional encoding used in DDPM for timestep t,
# followed by a learned two-layer MLP projection.
# Implicitly used by the genQC scheduler referenced in the notebook.
# ══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """
    Maps a scalar diffusion timestep t ∈ {0, …, T−1} to a dense vector.

    Algorithm:
        sinusoidal(t)_i = sin(t / 10000^(2i/d))   for i < d/2
        sinusoidal(t)_i = cos(t / 10000^(2i/d))   for i ≥ d/2
    then projected through a two-layer MLP.

    Shape contract
    ──────────────
    Input : (B,)    — integer or float timesteps
    Output: (B, time_embed_dim)
    """

    def __init__(self, time_embed_dim: int, max_timesteps: int = 1000) -> None:
        """
        Args:
            time_embed_dim: Output embedding width (must be even).
            max_timesteps : Maximum T used during training.
        """
        super().__init__()
        assert time_embed_dim % 2 == 0, "time_embed_dim must be even."
        half = time_embed_dim // 2

        freqs = torch.exp(
            -math.log(max_timesteps)
            * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)  # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Timestep tensor of shape (B,).

        Returns:
            Timestep embedding of shape (B, time_embed_dim).
        """
        args  = t.float()[:, None] * self.freqs[None, :]          # (B, half)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)        # (B, time_embed_dim)
        return self.mlp(emb)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── U-NET DENOISER
#
# Predicts the noise ε added to a circuit tensor at timestep t.
# Treats the circuit tensor as a 2D "image" where:
#   height  = num_qubits
#   width   = circuit_depth
#   channels= embed_dim
#
# Architecture:
#   Encoder  : [ResBlock × 2  →  Downsample] × len(channel_mults)
#   Bottleneck: ResBlock → CrossAttention(context) → ResBlock
#   Decoder  : [Upsample → ResBlock × 2 (+ skip)] × len(channel_mults)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Conv2D residual block with AdaGN-style timestep + context injection.

    Injects both the timestep embedding and the conditioning context as
    channel-wise biases after the first normalisation layer.

    Shape contract
    ──────────────
    x     : (B, in_channels,  num_qubits, circuit_depth)
    t_emb : (B, time_embed_dim)
    ctx   : (B, context_dim)
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
        groups = lambda c: min(8, c)  # GroupNorm groups — safe for small channels

        self.norm1 = nn.GroupNorm(groups(in_channels),  in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_proj    = nn.Linear(time_embed_dim, out_channels)
        self.context_proj = nn.Linear(context_dim,    out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor, ctx: Tensor) -> Tensor:
        """
        Args:
            x    : Feature map (B, in_channels,  H, W).
            t_emb: Timestep embedding (B, time_embed_dim).
            ctx  : Context vector (B, context_dim).

        Returns:
            Feature map (B, out_channels, H, W).
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Broadcast timestep and context as per-channel biases
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = h + self.context_proj(self.act(ctx))[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between circuit features (query) and context vector (key/value).

    Used in the U-Net bottleneck to fuse conditioning information with the
    circuit latent representation.

    The context vector is treated as a single key/value token.

    Shape contract
    ──────────────
    x  : (B, channels, num_qubits, circuit_depth)   — spatial feature map
    ctx: (B, context_dim)                            — conditioning vector
    out: (B, channels, num_qubits, circuit_depth)
    """

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm     = nn.GroupNorm(min(8, channels), channels)
        self.ctx_proj = nn.Linear(context_dim, channels)
        self.attn     = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        """
        Args:
            x  : (B, C, H, W)
            ctx: (B, context_dim)

        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        h  = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)   # (B, H*W, C)
        kv = self.ctx_proj(ctx).unsqueeze(1)                        # (B, 1,   C)

        attended, _ = self.attn(h, kv, kv)                         # (B, H*W, C)
        attended    = attended.permute(0, 2, 1).reshape(B, C, H, W)
        return x + attended


class UNetDenoiser(nn.Module):
    """
    U-Net that predicts the added noise ε given a noisy circuit tensor,
    timestep t, and conditioning context.

    All spatial dimensions (num_qubits × circuit_depth) scale dynamically
    at inference time — no fixed sizes are baked in.

    Shape contract
    ──────────────
    x    : (B, embed_dim,     num_qubits, circuit_depth)  ← noisy circuit tensor
    t_emb: (B, time_embed_dim)
    ctx  : (B, context_dim)
    out  : (B, embed_dim,     num_qubits, circuit_depth)  ← predicted noise ε
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
        """
        Args:
            embed_dim      : Input/output channel count (circuit embedding width).
            context_dim    : Width of the conditioning context vector.
            time_embed_dim : Width of the timestep embedding.
            base_channels  : Channel count at the first (shallowest) U-Net level.
            channel_mults  : Per-level channel width multipliers.
            num_heads      : Number of attention heads in the bottleneck.
        """
        super().__init__()
        channels = [base_channels * m for m in channel_mults]

        # Input projection: embed_dim → base_channels
        self.input_proj = nn.Conv2d(embed_dim, base_channels, 1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks  : nn.ModuleList = nn.ModuleList()
        self.downsamples : nn.ModuleList = nn.ModuleList()
        in_ch = base_channels

        for out_ch in channels:
            self.enc_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch,   out_ch, time_embed_dim, context_dim),
                ResidualBlock(out_ch,  out_ch, time_embed_dim, context_dim),
            ]))
            # Halve spatial dims; stride-2 conv preserves gradients better than pooling
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 2, stride=2))
            in_ch = out_ch

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.btn_res1 = ResidualBlock(in_ch, in_ch, time_embed_dim, context_dim)
        self.btn_attn = CrossAttentionBlock(in_ch, context_dim, num_heads)
        self.btn_res2 = ResidualBlock(in_ch, in_ch, time_embed_dim, context_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.upsamples  : nn.ModuleList = nn.ModuleList()
        self.dec_blocks : nn.ModuleList = nn.ModuleList()

        for out_ch in reversed(channels):
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            )
            # *2 on in_channels because of skip-connection concatenation
            self.dec_blocks.append(nn.ModuleList([
                ResidualBlock(out_ch * 2, out_ch, time_embed_dim, context_dim),
                ResidualBlock(out_ch,     out_ch, time_embed_dim, context_dim),
            ]))
            in_ch = out_ch

        # Output projection: base_channels → embed_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, embed_dim, 1),
        )

    def forward(self, x: Tensor, t_emb: Tensor, ctx: Tensor) -> Tensor:
        """
        Args:
            x    : Noisy circuit tensor (B, embed_dim, num_qubits, circuit_depth).
            t_emb: Timestep embedding (B, time_embed_dim).
            ctx  : Context vector (B, context_dim).

        Returns:
            Predicted noise ε of shape (B, embed_dim, num_qubits, circuit_depth).
        """
        h = self.input_proj(x)

        # Encoder pass — collect skip connections
        skips: List[Tensor] = []
        for (res1, res2), down in zip(self.enc_blocks, self.downsamples):
            h = res1(h, t_emb, ctx)
            h = res2(h, t_emb, ctx)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.btn_res1(h, t_emb, ctx)
        h = self.btn_attn(h, ctx)
        h = self.btn_res2(h, t_emb, ctx)

        # Decoder pass — restore spatial dims with skip connections
        for up, (res1, res2), skip in zip(
            self.upsamples, self.dec_blocks, reversed(skips)
        ):
            h = up(h)
            # Pad if odd spatial sizes caused a 1-pixel mismatch
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.pad(h, [
                    0, skip.shape[-1] - h.shape[-1],
                    0, skip.shape[-2] - h.shape[-2],
                ])
            h = torch.cat([h, skip], dim=1)  # channel-wise skip concat
            h = res1(h, t_emb, ctx)
            h = res2(h, t_emb, ctx)

        return self.output_proj(h)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── DDPM NOISE SCHEDULER
#
# Implements the linear β-schedule and the DDPM forward / reverse processes.
#
# Notebook ref: pipeline.scheduler.set_timesteps(40)
#   → 40 denoising steps at inference time.
#
# Key equations
# ─────────────
# Forward process (add noise):
#   x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε,    ε ~ N(0, I)
#
# Reverse step (denoise):
#   μ_θ = (1/√α_t) · (x_t − β_t / √(1−ᾱ_t) · ε_θ(x_t, t))
#   x_{t−1} = μ_θ + σ_t · z,               z ~ N(0, I)  [z=0 at t=0]
# ══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """
    DDPM noise scheduler for the forward and reverse diffusion processes.

    Attributes (all plain tensors, not nn.Parameters — this is not an nn.Module):
        betas          : β_t schedule,          shape (T,)
        alphas         : α_t = 1 − β_t,         shape (T,)
        alphas_cump    : ᾱ_t = ∏_{s≤t} α_s,    shape (T,)
        sqrt_acp       : √ᾱ_t,                  shape (T,)
        sqrt_one_macp  : √(1−ᾱ_t),              shape (T,)
        timesteps      : Inference-time sub-sampled timestep sequence (descending).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 40,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        """
        Args:
            num_train_timesteps: Total training timesteps T.
            num_inference_steps: Inference denoising steps  (notebook uses 40).
            beta_start          : β_1 — starting noise variance.
            beta_end            : β_T — ending noise variance.
            schedule            : "linear" or "cosine" β-schedule.
        """
        self.num_train_timesteps = num_train_timesteps

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif schedule == "cosine":
            betas = self._cosine_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'. Use 'linear' or 'cosine'.")

        alphas      = 1.0 - betas
        alphas_cump = torch.cumprod(alphas, dim=0)

        self.betas         = betas
        self.alphas        = alphas
        self.alphas_cump   = alphas_cump
        self.sqrt_acp      = alphas_cump.sqrt()
        self.sqrt_one_macp = (1.0 - alphas_cump).sqrt()

        self.set_inference_timesteps(num_inference_steps)

    @staticmethod
    def _cosine_schedule(T: int, s: float = 0.008) -> Tensor:
        """
        Cosine β-schedule (Nichol & Dhariwal, "Improved DDPM", 2021).
        Produces a smoother noise profile than the linear schedule.
        """
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        f     = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
        acp   = f / f[0]
        betas = 1.0 - acp[1:] / acp[:-1]
        return betas.clamp(0.0, 0.999)

    def set_inference_timesteps(self, num_inference_steps: int) -> None:
        """
        Computes the sub-sampled timestep sequence for inference.

        Mirrors diffusers' scheduler.set_timesteps():  evenly spaced steps
        from T−1 down to 0, capped at num_inference_steps steps.

        Args:
            num_inference_steps: Number of denoising steps (notebook: 40).
        """
        self.num_inference_steps = num_inference_steps
        ratio = self.num_train_timesteps // num_inference_steps
        # Descending order so we go T → 0 in the sampling loop
        steps = torch.arange(num_inference_steps - 1, -1, -1) * ratio
        self.timesteps = steps.long()

    def to(self, device: torch.device) -> "DDPMScheduler":
        """Moves all buffers to `device` (mirrors nn.Module.to)."""
        self.betas         = self.betas.to(device)
        self.alphas        = self.alphas.to(device)
        self.alphas_cump   = self.alphas_cump.to(device)
        self.sqrt_acp      = self.sqrt_acp.to(device)
        self.sqrt_one_macp = self.sqrt_one_macp.to(device)
        self.timesteps     = self.timesteps.to(device)
        return self

    # ── Forward process (used during training) ────────────────────────────────

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Applies the forward diffusion process:  q(x_t | x_0).

        x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε

        Args:
            x0   : Clean circuit tensor (B, ...).
            noise: Gaussian noise ε of same shape as x0.
            t    : Integer timesteps (B,).

        Returns:
            Noisy tensor x_t with the same shape as x0.
        """
        extra = [1] * (x0.dim() - 1)                             # broadcast shape
        s_acp   = self.sqrt_acp[t].reshape(-1, *extra)
        s_1macp = self.sqrt_one_macp[t].reshape(-1, *extra)
        return s_acp * x0 + s_1macp * noise

    # ── Reverse process (used during inference) ───────────────────────────────

    def step(self, eps_pred: Tensor, t: int, x_t: Tensor) -> Tensor:
        """
        Performs one reverse diffusion step:  p(x_{t-1} | x_t).

        μ_θ(x_t, t) = (1/√α_t) · (x_t − β_t / √(1−ᾱ_t) · ε_θ)
        x_{t−1} = μ_θ + σ_t · z,   z ~ N(0, I)   (z = 0 when t = 0)

        Args:
            eps_pred: Predicted noise ε_θ(x_t, t), same shape as x_t.
            t       : Current integer timestep (scalar Python int).
            x_t     : Noisy tensor at step t.

        Returns:
            Denoised estimate x_{t−1} of the same shape as x_t.
        """
        beta_t       = self.betas[t]
        alpha_t      = self.alphas[t]
        acp_t        = self.alphas_cump[t]
        sqrt_1m_acp  = self.sqrt_one_macp[t]

        # Predicted mean
        coeff = beta_t / sqrt_1m_acp
        mean  = (x_t - coeff * eps_pred) / alpha_t.sqrt()

        # Posterior variance (0 at the final step t=0)
        if t > 0:
            acp_prev = self.alphas_cump[t - 1]
            variance = beta_t * (1.0 - acp_prev) / (1.0 - acp_t)
            std      = variance.clamp(min=1e-20).sqrt()
            mean     = mean + std * torch.randn_like(x_t)

        return mean


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── MAIN MODULE: DiffusionCompiler
#
# Ties every sub-module together.
# Exposes two public entry-points:
#   forward(...)                  — training loss computation
#   generate_mitigating_circuit() — full reverse-diffusion inference with CFG
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionCompiler(nn.Module):
    """
    Conditional Diffusion Model for Quantum Circuit Compilation & Error Mitigation.

    Given a target unitary U and (optionally) a noise profile, generates a
    quantum circuit gate sequence that implements U — or counteracts the noise.

    Dynamic shape scaling
    ─────────────────────
    Every tensor dimension that depends on the number of qubits is computed
    at construction time from `num_qubits` and `circuit_depth`:

        Unitary tensor  : (B, 2,          2**num_qubits, 2**num_qubits)
        Circuit tensor  : (B, embed_dim,  num_qubits,    circuit_depth)
        State condition : (B, 2 · 2**num_qubits)  — real+imag state vector

    Key hyper-parameters (from notebook)
    ─────────────────────────────────────
        num_inference_steps = 40   (pipeline.scheduler.set_timesteps(40))
        cfg_scale           = 10   (g=10 in infer_comp.generate_comp_tensors)
    """

    def __init__(
        self,
        num_qubits: int,
        circuit_depth: int,
        vocab: CircuitVocab,
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
        condition_input_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            num_qubits          : Number of qubits. Drives unitary size (2^n × 2^n)
                                  and circuit height (num_qubits rows).
            circuit_depth       : Maximum number of gate time-steps per circuit
                                  (= circuit width in the token matrix).
            vocab               : Gate vocabulary object (see CircuitVocab).
            embed_dim           : Embedding width per token.  embed_dim ≥ vocab_size.
            context_dim         : Width of the unified conditioning vector.
            time_embed_dim      : Width of the timestep embedding (must be even).
            base_channels       : U-Net base channel count.
            channel_mults       : Per-level channel width multipliers for the U-Net.
            num_heads           : Cross-attention heads in the U-Net bottleneck.
            num_train_timesteps : Total DDPM timesteps T (used during training).
            num_inference_steps : Reverse diffusion steps at inference (notebook: 40).
            cfg_scale           : Classifier-Free Guidance scale g (notebook: 10).
            noise_schedule      : β-schedule type: "linear" or "cosine".
            condition_input_dim : Flat dimension of the noise-profile condition.
                                  Defaults to 2 · 2**num_qubits (state vector).
        """
        super().__init__()

        self.num_qubits    = num_qubits
        self.circuit_depth = circuit_depth
        self.vocab         = vocab
        self.embed_dim     = embed_dim
        self.cfg_scale     = cfg_scale

        # Default noise condition size: real + imag parts of a state vector
        if condition_input_dim is None:
            condition_input_dim = 2 * (2 ** num_qubits)
        self.condition_input_dim = condition_input_dim

        # ── Sub-modules ───────────────────────────────────────────────────────

        self.circuit_embedding = CircuitEmbedding(
            vocab_size=vocab.vocab_size,
            embed_dim=embed_dim,
        )

        self.timestep_embedding = TimestepEmbedding(
            time_embed_dim=time_embed_dim,
            max_timesteps=num_train_timesteps,
        )

        self.unitary_encoder = UnitaryEncoder(
            num_qubits=num_qubits,
            context_dim=context_dim,
        )

        self.noise_encoder = NoiseProfileEncoder(
            condition_input_dim=condition_input_dim,
            context_dim=context_dim,
        )

        # Fuses unitary context + noise context → single context vector
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

    # ── Conditioning helper ───────────────────────────────────────────────────

    def _encode_context(
        self,
        unitary: Tensor,
        noisy_condition: Tensor,
        use_null_condition: bool = False,
    ) -> Tensor:
        """
        Encodes the target unitary and noise profile into a single context vector.

        Args:
            unitary           : (B, 2, 2**num_qubits, 2**num_qubits)
                                Channel 0 = Re(U), Channel 1 = Im(U).
            noisy_condition   : (B, condition_input_dim)
            use_null_condition: If True, the noise encoder returns its null embedding.
                                Used for the unconditional branch of CFG.

        Returns:
            Fused context vector of shape (B, context_dim).
        """
        u_ctx    = self.unitary_encoder(unitary)
        n_ctx    = self.noise_encoder(noisy_condition, use_null=use_null_condition)
        combined = torch.cat([u_ctx, n_ctx], dim=-1)
        return self.context_fusion(combined)

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        circuit_tokens: Tensor,
        unitary: Tensor,
        noisy_condition: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Training forward pass.  Predicts the noise ε that was added at step t.

        Training objective:  L = ‖ ε_θ(x_t, t, ctx) − ε ‖²

        Args:
            circuit_tokens : Clean token matrix (B, num_qubits, circuit_depth) [int64].
            unitary        : Target unitary (B, 2, 2**num_qubits, 2**num_qubits).
            noisy_condition: Noise profile (B, condition_input_dim).
            t              : Timestep per sample (B,) [int64 ∈ {0,…,T−1}].

        Returns:
            Tuple of:
                eps_pred : Predicted noise (B, embed_dim, num_qubits, circuit_depth).
                eps_true : Ground-truth noise ε  (same shape) — needed for loss.
        """
        # 1. Embed clean circuit tokens to continuous tensor
        #    (B, Q, D) → embed → (B, Q, D, E) → permute → (B, E, Q, D)
        x0 = self.circuit_embedding(circuit_tokens)  # (B, Q, D, E)
        x0 = x0.permute(0, 3, 1, 2)                 # (B, E, Q, D)

        # 2. Add noise according to the forward diffusion process
        eps_true = torch.randn_like(x0)
        x_t      = self.scheduler.add_noise(x0, eps_true, t)

        # 3. Encode conditioning
        ctx   = self._encode_context(unitary, noisy_condition)
        t_emb = self.timestep_embedding(t)

        # 4. Predict noise
        eps_pred = self.unet(x_t, t_emb, ctx)
        return eps_pred, eps_true

    # ── Inference: reverse-diffusion sampling ─────────────────────────────────

    @torch.no_grad()
    def generate_mitigating_circuit(
        self,
        unitary: Tensor,
        noisy_condition: Tensor,
        num_samples: int = 1,
        cfg_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generates mitigating quantum circuits via guided reverse diffusion.

        Reverse diffusion with Classifier-Free Guidance (CFG):
            ε_guided = ε_uncond + g · (ε_cond − ε_uncond)

        where g is the CFG scale (notebook uses g = 10).

        Args:
            unitary         : Target unitary.
                              Shape (B, 2, 2**num_qubits, 2**num_qubits)
                              OR   (2, 2**num_qubits, 2**num_qubits)  — batch added.
            noisy_condition : Noise profile guiding error mitigation.
                              Shape (B, condition_input_dim) OR (condition_input_dim,).
            num_samples     : Number of circuit candidates generated in parallel.
            cfg_scale       : Override CFG scale g.  None → uses self.cfg_scale.
            device          : Target device.  None → inferred from model params.
            seed            : Optional RNG seed for reproducible sampling.

        Returns:
            Dict with three keys:

            "token_matrix"  : Tensor (num_samples, num_qubits, circuit_depth) [int64]
                Decoded gate-token matrix.
                    +k  → gate vocab.token_to_gate[k] on this qubit
                    −k  → control qubit for gate k
                     0  → padding (circuit ends here)

            "raw_tensor"    : Tensor (num_samples, embed_dim, num_qubits, circuit_depth)
                The raw continuous output of the reverse diffusion before decoding.

            "cudaq_params"  : List[Dict] of length num_samples.
                Each dict is CUDA-Q compatible and contains:
                    "gates"    : List[(gate_name: str,
                                       qubits   : List[int],
                                       time_step: int)]
                    "n_qubits" : int
                    "depth"    : int  (last non-zero column + 1)
        """
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        g = cfg_scale if cfg_scale is not None else self.cfg_scale

        # ── Normalise input batch dimensions ──────────────────────────────────
        if unitary.dim() == 3:
            unitary = unitary.unsqueeze(0)           # (1, 2, H, H)
        if noisy_condition.dim() == 1:
            noisy_condition = noisy_condition.unsqueeze(0)   # (1, C)

        # Broadcast single condition to num_samples
        if unitary.shape[0] == 1 and num_samples > 1:
            unitary         = unitary.expand(num_samples, -1, -1, -1)
            noisy_condition = noisy_condition.expand(num_samples, -1)

        unitary         = unitary.to(device)
        noisy_condition = noisy_condition.to(device)

        # ── Encode conditional + unconditional contexts for CFG ───────────────
        ctx_cond   = self._encode_context(unitary, noisy_condition, use_null_condition=False)
        ctx_uncond = self._encode_context(unitary, noisy_condition, use_null_condition=True)

        # ── Initialise with pure Gaussian noise in circuit-tensor space ───────
        # Shape: (num_samples, embed_dim, num_qubits, circuit_depth)
        x_t = torch.randn(
            num_samples, self.embed_dim, self.num_qubits, self.circuit_depth,
            device=device,
        )

        # ── Reverse diffusion loop  (T → 0, 40 steps) ────────────────────────
        # Notebook ref: pipeline.scheduler.set_timesteps(40)
        self.scheduler.to(device)

        for t_val in self.scheduler.timesteps:   # descending: T-1 … 0
            t_batch = t_val.expand(num_samples)  # (B,)
            t_emb   = self.timestep_embedding(t_batch.float())

            # Conditional and unconditional noise predictions
            eps_cond   = self.unet(x_t, t_emb, ctx_cond)
            eps_uncond = self.unet(x_t, t_emb, ctx_uncond)

            # Classifier-Free Guidance
            # Notebook ref: g=10 in infer_comp.generate_comp_tensors(g=10, ...)
            eps_guided = eps_uncond + g * (eps_cond - eps_uncond)

            # DDPM reverse step
            x_t = self.scheduler.step(eps_guided, int(t_val.item()), x_t)

        # ── Decode final continuous tensor → integer token matrix ─────────────
        raw_tensor = x_t.clone()

        # (B, E, Q, D) → (B, Q, D, E) for the embedding decoder
        token_matrix = self.circuit_embedding.decode(x_t.permute(0, 2, 3, 1))
        # token_matrix: (B, num_qubits, circuit_depth) [int64]

        # ── Convert to CUDA-Q compatible gate parameter dicts ─────────────────
        cudaq_params = self._tokens_to_cudaq_params(token_matrix)

        return {
            "token_matrix" : token_matrix,
            "raw_tensor"   : raw_tensor,
            "cudaq_params" : cudaq_params,
        }

    # ── Token matrix → CUDA-Q parameters ─────────────────────────────────────

    def _tokens_to_cudaq_params(self, token_matrix: Tensor) -> List[Dict]:
        """
        Parses a batch of token matrices into structured CUDA-Q gate parameter dicts.

        Notebook ref: genqc_to_cudaq(decoded, vocab_dict) — replicates the core
        parsing logic without requiring CUDA-Q as an import dependency here.

        Token conventions:
            +k  → vocab.token_to_gate[k] applied, this qubit is the *target*.
            −k  → this qubit is a *control* qubit for gate k.
             0  → padding; marks end-of-circuit when the entire column is zero.

        Args:
            token_matrix: (B, num_qubits, circuit_depth) [int64].

        Returns:
            List of dicts, one per sample:
                "gates"    : List of (gate_name, [qubit_indices], time_step).
                             qubit_indices = control qubits (sorted) + [target].
                "n_qubits" : int
                "depth"    : int  — index of the last non-padding column + 1.
        """
        results: List[Dict] = []
        np_matrix = token_matrix.cpu().numpy()  # (B, Q, D)

        for b in range(np_matrix.shape[0]):
            mat    = np_matrix[b]    # (num_qubits, circuit_depth)
            gates  = []
            depth  = 0

            for t in range(self.circuit_depth):
                col = mat[:, t]

                # All zeros = padding column → circuit has ended
                if not np.any(col != 0):
                    break

                depth = t + 1
                target_qubits  = np.where(col > 0)[0].tolist()
                control_qubits = np.where(col < 0)[0].tolist()

                for tq in target_qubits:
                    token     = int(col[tq])
                    gate_name = self.vocab.token_to_gate.get(token)
                    if gate_name is None:
                        continue  # unknown/OOV token — skip
                    # control qubits listed first (sorted), then target
                    all_qubits = sorted(control_qubits) + [tq]
                    gates.append((gate_name, all_qubits, t))

            results.append({
                "gates"    : gates,
                "n_qubits" : self.num_qubits,
                "depth"    : depth,
            })

        return results

    # ── Utility: infidelity ───────────────────────────────────────────────────

    @staticmethod
    def compute_infidelity(
        want_unitary: np.ndarray,
        got_unitary: np.ndarray,
    ) -> float:
        """
        Computes the unitary infidelity between two matrices.

        Notebook section: "Computing Infidelity"

        Infidelity(U, V) = 1 − |Tr(U† V) / 2^n|²

        Values:
            0.0 → circuits are identical up to global phase.
            1.0 → circuits are completely orthogonal.

        Args:
            want_unitary: Target unitary (2^n, 2^n) [complex].
            got_unitary : Generated unitary (2^n, 2^n) [complex].

        Returns:
            Scalar infidelity ∈ [0, 1].
        """
        n       = want_unitary.shape[0]
        overlap = np.trace(want_unitary.conj().T @ got_unitary) / n
        return float(1.0 - abs(overlap) ** 2)

    # ── Factory: build from config dict ──────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: Dict,
        device: Optional[torch.device] = None,
    ) -> "DiffusionCompiler":
        """
        Constructs a DiffusionCompiler from a flat configuration dictionary.

        The dict must contain a "gates" key (List[str]).
        All other keys map 1-to-1 to __init__ parameters.

        Example:
            model = DiffusionCompiler.from_config({
                "num_qubits"          : 3,
                "circuit_depth"       : 12,
                "gates"               : ["h", "cx", "z", "x", "ccx", "swap"],
                "embed_dim"           : 16,
                "context_dim"         : 256,
                "cfg_scale"           : 10.0,
                "num_inference_steps" : 40,
            })

        Args:
            config: Configuration dict.  Modified in-place (gates key popped).
            device: If provided, moves the model to this device after construction.

        Returns:
            Initialised DiffusionCompiler.
        """
        cfg   = dict(config)                   # shallow copy — don't mutate caller's dict
        vocab = CircuitVocab(gates=cfg.pop("gates"))
        model = cls(vocab=vocab, **cfg)
        if device is not None:
            model = model.to(device)
        return model


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE-TEST  (python diffusion_compiler.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Build model ───────────────────────────────────────────────────────────
    model = DiffusionCompiler.from_config({
        "num_qubits"          : 3,
        "circuit_depth"       : 12,
        "gates"               : ["h", "cx", "z", "x", "ccx", "swap"],
        "embed_dim"           : 16,
        "context_dim"         : 128,
        "time_embed_dim"      : 64,
        "base_channels"       : 32,
        "channel_mults"       : (1, 2),
        "num_train_timesteps" : 1000,
        "num_inference_steps" : 40,
        "cfg_scale"           : 10.0,
    })

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DiffusionCompiler ready  |  trainable params: {n_params:,}")

    hilbert = 2 ** 3   # 3 qubits → 8-dimensional Hilbert space

    # ── Dummy 3-qubit unitary (real + imag channels) ──────────────────────────
    U_real = torch.eye(hilbert).unsqueeze(0).unsqueeze(0)          # (1, 1, 8, 8)
    U_imag = torch.zeros_like(U_real)
    unitary = torch.cat([U_real, U_imag], dim=1)                   # (1, 2, 8, 8)

    # ── Dummy noise condition (noisy state vector, real+imag) ────────────────
    noisy_state = torch.randn(1, 2 * hilbert)                      # (1, 16)

    # ── Generate 4 candidate circuits ────────────────────────────────────────
    result = model.generate_mitigating_circuit(
        unitary=unitary,
        noisy_condition=noisy_state,
        num_samples=4,
        seed=42,
    )

    print(f"\nGenerated token matrix shape : {result['token_matrix'].shape}")
    print(f"Raw tensor shape             : {result['raw_tensor'].shape}")
    print(f"\nFirst circuit (CUDA-Q params):")
    for gate_name, qubits, t_step in result["cudaq_params"][0]["gates"]:
        print(f"  t={t_step:02d}  {gate_name:6s}  qubits={qubits}")
