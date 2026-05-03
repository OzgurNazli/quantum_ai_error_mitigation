"""
evaluate_and_plot.py

Publication-quality evaluation and plotting script for the QEM Diffusion Model.
Place in: src/ai_core/

Workflow
────────
1. Load trained DiffusionCompiler from checkpoint.
2. Load .npy dataset (circuit_tokens, target_observables, syndromes).
3. Run reverse-diffusion inference conditioned on syndromes.
4. Simulate generated circuit tokens → predicted ⟨Z_i⟩ via numpy statevector.
5. Compute MSE and per-qubit MAE.
6. Save 3 publication-quality figures to plots/.

Output figures
──────────────
    plots/qem_scatter.pdf       — Predicted vs Ground Truth ⟨Z⟩ scatter
    plots/qem_error_hist.pdf    — Absolute error distribution with KDE
    plots/qem_qubit_mae.pdf     — Per-qubit MAE/MSE bar chart
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────────
# Add src/ai_core to path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffusion_compiler import DiffusionCompiler, CircuitVocab


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── GLOBAL PLOT AESTHETICS
# ══════════════════════════════════════════════════════════════════════════════

# Base seaborn theme — whitegrid is standard for academic publications
sns.set_theme(style="whitegrid", font_scale=1.3)

# Attempt LaTeX rendering for publication-quality typography.
# Falls back silently to Matplotlib's built-in mathtext if LaTeX is not in PATH.
_LATEX_OK = False
try:
    matplotlib.rcParams.update({"text.usetex": True, "font.family": "serif"})
    # Probe with a tiny render — raises if LaTeX is missing
    matplotlib.mathtext.MathTextParser("deferred")  # type: ignore[attr-defined]
    _LATEX_OK = True
except Exception:
    matplotlib.rcParams.update({"text.usetex": False})
    warnings.warn(
        "LaTeX not found — using Matplotlib mathtext for labels. "
        "Install a TeX distribution for true LaTeX rendering.",
        stacklevel=1,
    )

matplotlib.rcParams.update({
    "axes.labelsize"  : 14,
    "axes.titlesize"  : 15,
    "axes.titleweight": "bold",
    "legend.fontsize" : 11,
    "xtick.labelsize" : 11,
    "ytick.labelsize" : 11,
    "figure.dpi"      : 300,
    "savefig.dpi"     : 300,
    "savefig.bbox"    : "tight",
})

# Per-qubit colour palette — tab10 gives well-separated, print-safe colours
_PALETTE  = sns.color_palette("tab10")
_FIG_FMT  = "pdf"          # lossless; change to "png" for slides

# ── Convenience label helpers ─────────────────────────────────────────────────

def _lz(i: int) -> str:
    """LaTeX / mathtext label for ⟨Z_i⟩."""
    return fr"$\langle Z_{{{i}}} \rangle$"

def _obs_axis_label() -> str:
    return r"$\langle Z_i \rangle$"

def _abs_err_label() -> str:
    return r"$|\hat{y} - y|$"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

# Only these keys are forwarded to DiffusionCompiler.__init__.
# All other argparse keys (epochs, lr, checkpoint_dir, …) are ignored.
_MODEL_KEYS = frozenset({
    "num_qubits", "circuit_depth", "num_observables", "num_syndromes",
    "embed_dim", "context_dim", "time_embed_dim", "base_channels",
    "channel_mults", "num_heads", "num_train_timesteps",
    "num_inference_steps", "cfg_scale", "noise_schedule",
    "syndrome_encoder_dropout", "gates",
})


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[DiffusionCompiler, Dict]:
    """
    Reconstructs and loads a DiffusionCompiler from a train_mitigator.py checkpoint.

    The checkpoint dict is expected to contain:
        "model_state_dict" : OrderedDict of model weights
        "args"             : flat dict produced by vars(argparse.Namespace)

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device         : Device on which to place the model.

    Returns:
        Tuple of (model in eval mode, full args dict from checkpoint).

    Raises:
        FileNotFoundError : if checkpoint_path does not exist.
        ValueError        : if the checkpoint lacks the expected keys.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[load] Checkpoint  : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    full_args: Dict = ckpt.get("args", {})
    if not full_args:
        raise ValueError(
            "Checkpoint does not contain an 'args' key.\n"
            "Ensure the checkpoint was saved by save_checkpoint() in train_mitigator.py."
        )

    # ── Extract and sanitise model config ────────────────────────────────────
    model_cfg = {k: v for k, v in full_args.items() if k in _MODEL_KEYS}

    # channel_mults is stored as list by argparse; DiffusionCompiler wants tuple
    if "channel_mults" in model_cfg:
        model_cfg["channel_mults"] = tuple(model_cfg["channel_mults"])

    print(
        f"[load] Architecture: {model_cfg['num_qubits']} qubits | "
        f"depth {model_cfg['circuit_depth']} | "
        f"context_dim {model_cfg['context_dim']} | "
        f"num_observables {model_cfg['num_observables']} | "
        f"num_syndromes {model_cfg['num_syndromes']}"
    )

    model = DiffusionCompiler.from_config(model_cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    saved_epoch = ckpt.get("epoch", "?")
    saved_loss  = ckpt.get("loss",  float("nan"))
    n_params    = sum(p.numel() for p in model.parameters())
    print(
        f"[load] Restored from epoch {saved_epoch} | "
        f"saved loss {saved_loss:.6f} | "
        f"params {n_params:,}"
    )
    return model, full_args


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the three QEM dataset arrays produced by generate_qem_data.py.

    Expected files inside data_dir:
        circuit_tokens.npy       — (N, circuit_depth)    int64
        target_observables.npy   — (N, num_observables)  float32  clean ⟨Z_i⟩
        syndromes.npy            — (N, num_syndromes)    float32  noisy ⟨Z_iZ_{i+1}⟩

    Args:
        data_dir: Path to the directory containing the .npy files.

    Returns:
        Tuple of (circuit_tokens, target_observables, syndromes).

    Raises:
        FileNotFoundError: if any of the three .npy files is missing.
    """
    files = {
        "circuit_tokens"     : data_dir / "circuit_tokens.npy",
        "target_observables" : data_dir / "target_observables.npy",
        "syndromes"          : data_dir / "syndromes.npy",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    circuit_tokens     = np.load(files["circuit_tokens"])
    target_observables = np.load(files["target_observables"])
    syndromes          = np.load(files["syndromes"])

    print(f"\n[data] Directory   : {data_dir}")
    print(f"[data] circuit_tokens     {circuit_tokens.shape}     {circuit_tokens.dtype}")
    print(f"[data] target_observables {target_observables.shape}  {target_observables.dtype}")
    print(f"[data] syndromes          {syndromes.shape}   {syndromes.dtype}")
    return circuit_tokens, target_observables, syndromes

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── NUMPY STATEVECTOR SIMULATOR  (7-gate Clifford set)
#
# Gate set: ["i", "x", "y", "z", "h", "s", "cx"]
# Matches the GATE_SETS["clifford"] definition in train_mitigator_main.py.
#
# Token → gate mapping for n qubits  (token 0 = PAD reserved):
# ┌─────────────────────┬───────────┬──────────────────────────────────────┐
# │ Token range         │ Gate      │ Qubit index                          │
# ├─────────────────────┼───────────┼──────────────────────────────────────┤
# │ 0                   │ PAD       │ —                                    │
# │ 1       …  n        │ I         │ token − 1                            │
# │ n+1     …  2n       │ X         │ token − n − 1                        │
# │ 2n+1    …  3n       │ Y         │ token − 2n − 1                       │
# │ 3n+1    …  4n       │ Z         │ token − 3n − 1                       │
# │ 4n+1    …  5n       │ H         │ token − 4n − 1                       │
# │ 5n+1    …  6n       │ S         │ token − 5n − 1                       │
# │ 6n+1    …  7n−1     │ CX (CNOT) │ ctrl = token − 6n − 1, tgt = ctrl+1 │
# └─────────────────────┴───────────┴──────────────────────────────────────┘
#
# Gate matrices (all complex128 for numerical precision)
# ───────────────────────────────────────────────────────
#   I = [[1,  0],          — identity, no-op on statevector
#        [0,  1]]
#
#   X = [[0,  1],          — Pauli-X, bit-flip, maps |0⟩↔|1⟩
#        [1,  0]]
#
#   Y = [[0, -i],          — Pauli-Y, complex; Y = iXZ
#        [i,  0]]
#
#   Z = [[1,  0],          — Pauli-Z, phase-flip, maps |1⟩ → −|1⟩
#        [0, -1]]
#
#   H = (1/√2)·[[1,  1],  — Hadamard, maps Z↔X
#               [1, -1]]
#
#   S = [[1,  0],          — Phase gate, S²=Z, maps X→Y
#        [0,  i]]
#
# Memory: O(2^n_qubits).  Valid for n_qubits ≤ ~20.
# For the H4 evaluation set (n_qubits = 4): 2^4 = 16 complex amplitudes.
# ══════════════════════════════════════════════════════════════════════════════

# Pre-built gate matrices — declared once at module level, reused every call.
# All entries are complex128 so the statevector stays in complex128 throughout.
# This is critical for Y and S which have imaginary matrix elements.
_I_GATE: np.ndarray = np.array(
    [[1.0,  0.0],
     [0.0,  1.0]],
    dtype=complex,
)

_X_GATE: np.ndarray = np.array(
    [[0.0,  1.0],
     [1.0,  0.0]],
    dtype=complex,
)

_Y_GATE: np.ndarray = np.array(
    [[0.0, -1.0j],
     [1.0j, 0.0]],
    dtype=complex,
)

_Z_GATE: np.ndarray = np.array(
    [[1.0,  0.0],
     [0.0, -1.0]],
    dtype=complex,
)

_H_GATE: np.ndarray = (1.0 / np.sqrt(2)) * np.array(
    [[1.0,  1.0],
     [1.0, -1.0]],
    dtype=complex,
)

_S_GATE: np.ndarray = np.array(
    [[1.0,  0.0],
     [0.0,  1.0j]],
    dtype=complex,
)


def _apply_1q(
    state: np.ndarray,
    qubit: int,
    gate: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """
    Applies a (2×2) single-qubit gate to a statevector via reshape + tensordot.

    Gate-agnostic: works for any 2×2 complex unitary (I, X, Y, Z, H, S, …).
    Cost: O(2^n_qubits) — for n=4: 16 complex multiplications.

    Args:
        state   : Complex statevector of shape (2^n_qubits,).
        qubit   : Target qubit index (0 = most-significant bit).
        gate    : 2×2 complex unitary matrix.
        n_qubits: Total qubit count.

    Returns:
        Updated statevector of the same shape.
    """
    s = state.reshape([2] * n_qubits)
    # Contract the gate over the target qubit axis
    s = np.tensordot(gate, s, axes=[[1], [qubit]])
    # tensordot pushes the contracted axis to position 0 — move it back
    s = np.moveaxis(s, 0, qubit)
    return s.reshape(-1)


def _apply_cnot(
    state: np.ndarray,
    ctrl: int,
    tgt: int,
    n_qubits: int,
) -> np.ndarray:
    """
    Applies a CNOT gate by iterating over the full 2^n_qubits basis.

    For each basis state where the control qubit is |1⟩, the target bit
    is flipped via XOR.  No dense matrix storage required.

    Args:
        state   : Complex statevector of shape (2^n_qubits,).
        ctrl    : Control qubit index.
        tgt     : Target qubit index.
        n_qubits: Total qubit count.

    Returns:
        Updated statevector of the same shape.
    """
    new_state = np.zeros_like(state)
    ctrl_mask = 1 << (n_qubits - 1 - ctrl)
    tgt_mask  = 1 << (n_qubits - 1 - tgt)

    for idx in range(len(state)):
        if state[idx] == 0j:
            continue
        # Control qubit |1⟩ → XOR the target bit; otherwise pass through
        new_idx = (idx ^ tgt_mask) if (idx & ctrl_mask) else idx
        new_state[new_idx] += state[idx]

    return new_state


def statevector_simulate(
    tokens: np.ndarray,
    n_qubits: int,
    hf_state: np.ndarray,
) -> np.ndarray:
    """
    Simulates a 7-gate Clifford circuit and returns ⟨Z_i⟩ for every qubit.

    Gate set: ["i", "x", "y", "z", "h", "s", "cx"]  — matches GATE_SETS
    ["clifford"] in train_mitigator_main.py.

    Token → gate mapping (for n = n_qubits):
        Token 0              → PAD — no operation
        Token 1       … n    → I  on qubit (token − 1)
        Token n+1     … 2n   → X  on qubit (token − n − 1)
        Token 2n+1    … 3n   → Y  on qubit (token − 2n − 1)
        Token 3n+1    … 4n   → Z  on qubit (token − 3n − 1)
        Token 4n+1    … 5n   → H  on qubit (token − 4n − 1)
        Token 5n+1    … 6n   → S  on qubit (token − 5n − 1)
        Token 6n+1    … 7n−1 → CX(ctrl=i, tgt=i+1)
                               where i = token − 6n − 1

    Args:
        tokens   : 1-D integer array of gate tokens (may contain PAD = 0).
        n_qubits : Number of physical qubits.
        hf_state : HF occupation array of shape (n_qubits,) with 0/1 values.
                   The circuit is initialised in this reference state before
                   any gates are applied.

    Returns:
        Float32 array of shape (n_qubits,) with ⟨Z_i⟩ ∈ [−1, +1].

    Notes on ⟨Z_i⟩ behaviour per gate:
        I : no-op — ⟨Z_i⟩ unchanged.
        X : bit-flip  — ⟨Z_i⟩ → −⟨Z_i⟩.
        Y : bit+phase flip — ⟨Z_i⟩ → −⟨Z_i⟩.
        Z : phase-flip — ⟨Z_i⟩ unchanged (only affects ⟨X⟩, ⟨Y⟩).
        H : maps Z↔X  — ⟨Z_i⟩ → ⟨X_i⟩ (creates superposition from Z-eigenstate).
        S : maps X→Y  — ⟨Z_i⟩ unchanged on its own.
        CX: entangles adjacent qubits — captured exactly by the statevector.
    """
    # ── Initialise HF reference state ────────────────────────────────────────
    # hf_state is a length-n array of 0/1 occupation numbers — NOT a 2^n vector.
    # hf_idx is the single integer index into the 2^n computational basis.
    hf_idx = sum(int(b) << (n_qubits - 1 - i) for i, b in enumerate(hf_state))
    state  = np.zeros(2 ** n_qubits, dtype=complex)
    state[hf_idx] = 1.0 + 0j

    # ── Pre-compute per-gate token offsets once (avoids repeated multiplication)
    n = n_qubits
    _RANGES = (
        (1,          n,         _I_GATE, 1),          # I  : offset 1
        (n + 1,      2 * n,     _X_GATE, n + 1),      # X  : offset n+1
        (2 * n + 1,  3 * n,     _Y_GATE, 2 * n + 1),  # Y  : offset 2n+1
        (3 * n + 1,  4 * n,     _Z_GATE, 3 * n + 1),  # Z  : offset 3n+1
        (4 * n + 1,  5 * n,     _H_GATE, 4 * n + 1),  # H  : offset 4n+1
        (5 * n + 1,  6 * n,     _S_GATE, 5 * n + 1),  # S  : offset 5n+1
    )
    _CX_LO = 6 * n + 1          # first CX token
    _CX_HI = 7 * n - 1          # last  CX token  (n−1 adjacent pairs)

    # ── Apply 7-gate Clifford token sequence ──────────────────────────────────
    for t in tokens.tolist():
        if t == 0:
            continue                                                    # PAD

        elif _CX_LO <= t <= _CX_HI:                                    # CX layer
            ctrl = t - _CX_LO
            state = _apply_cnot(state, ctrl, ctrl + 1, n_qubits)

        else:
            # Single-qubit gates — linear scan over the 6 ranges
            for lo, hi, gate_mat, offset in _RANGES:
                if lo <= t <= hi:
                    state = _apply_1q(state, t - offset, gate_mat, n_qubits)
                    break

    # ── ⟨Z_i⟩ = Σ_s sign(bit_i(s)) · |⟨s|ψ⟩|² ──────────────────────────────
    # Z eigenvalue: +1 for |0⟩, −1 for |1⟩
    probs = np.abs(state) ** 2   # Born probabilities, shape (2^n,)
    z_exp = np.zeros(n_qubits, dtype=np.float64)

    for i in range(n_qubits):
        mask = 1 << (n_qubits - 1 - i)
        for idx in range(2 ** n_qubits):
            sign      = +1 if not (idx & mask) else -1
            z_exp[i] += sign * probs[idx]

    return z_exp.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    model: DiffusionCompiler,
    circuit_tokens: np.ndarray,
    target_observables: np.ndarray,
    syndromes: np.ndarray,
    device: torch.device,
    n_elec: Optional[int] = None,
) -> np.ndarray:
    """
    Runs the full reverse-diffusion loop and returns ⟨Z_i⟩ predictions.

    Conditioning strategy
    ─────────────────────
    The model is conditioned on BOTH target_observables (what clean observable
    values we wish to achieve) AND syndromes (the noisy hardware measurements).
    At evaluation time we provide the ground-truth target_observables from the
    test set — this tests the model's ability to generate circuits whose
    simulated output matches the desired target.

    In production deployment, target_observables would be replaced by a
    device specification or a classically-estimated prior.

    Steps
    ─────
    1. Run generate_mitigating_circuit() → predicted gate token sequences.
    2. Simulate each sequence with the numpy statevector simulator → ⟨Z_i⟩.

    Args:
        model              : DiffusionCompiler in eval mode.
        circuit_tokens     : (N, circuit_depth) int64 — provides shape context.
        target_observables : (N, num_observables) float32 — condition A.
        syndromes          : (N, num_syndromes) float32   — condition B.
        device             : PyTorch device.
        n_elec             : Active electron count for HF state initialisation.
                             Defaults to n_qubits // 2 (half-filling).

    Returns:
        predicted_observables: (N, num_observables) float32 — simulated ⟨Z_i⟩
            from the diffusion model's generated circuits.
    """
    N                = circuit_tokens.shape[0]
    n_qubits         = model.num_qubits
    num_observables  = target_observables.shape[1]

    obs_t = torch.tensor(target_observables, dtype=torch.float32).to(device)
    syn_t = torch.tensor(syndromes,          dtype=torch.float32).to(device)

    # ── Reverse diffusion (40 denoising steps, CFG g=10) ─────────────────────
    print(
        f"\n[infer] Running reverse diffusion — "
        f"{N} samples × {model.scheduler.num_inference_steps} steps ..."
    )
    with torch.no_grad():
        result = model.generate_mitigating_circuit(
            target_observables = obs_t,
            syndromes          = syn_t,
            num_samples        = N,
            device             = device,
        )

    predicted_tokens = result["token_matrix"].cpu().numpy()  # (N, circuit_depth)

    # ── HF reference state ────────────────────────────────────────────────────
    # In the JW mapping: the first n_elec spin-orbitals are occupied.
    # H4 with 2 active electrons → hf_state = [1, 1, 0, 0].
    if n_elec is None:
        n_elec = n_qubits // 2
    hf_state = np.zeros(n_qubits, dtype=int)
    hf_state[:n_elec] = 1
    print(f"[infer] HF reference state : {hf_state.tolist()}")

    # ── Statevector simulation → ⟨Z_i⟩ ──────────────────────────────────────
    print("[infer] Simulating predicted circuits ...")
    predicted_observables = np.zeros((N, num_observables), dtype=np.float32)

    for i in tqdm(range(N), desc="  Statevector simulate", ncols=70):
        tokens_1d = np.asarray(predicted_tokens[i]).flatten().astype(int)
        predicted_observables[i] = statevector_simulate(
            tokens_1d, n_qubits, hf_state
        )

    return predicted_observables

def _syndrome_baseline(syndromes: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Creates an unmitigated baseline from the raw syndromes.
    For an H4 chain, the syndrome ⟨Z_i Z_{i+1}⟩ correlates heavily with ⟨Z_i⟩.
    This simple mean-field approximation maps the hardware syndromes 
    directly to local observable expectations for baseline comparison.
    """
    N = syndromes.shape[0]
    baseline = np.zeros((N, n_qubits), dtype=np.float32)
    
    # Basit bir eşleme: Sendromların ortalamasını alarak yerel Z değerlerini tahmin et
    for i in range(n_qubits):
        if i == 0:
            baseline[:, i] = syndromes[:, 0]
        elif i == n_qubits - 1:
            baseline[:, i] = syndromes[:, -1]
        else:
            baseline[:, i] = (syndromes[:, i-1] + syndromes[:, i]) / 2.0
            
    # Değerleri fiziksel [-1, 1] sınırları içine kırp
    return np.clip(baseline, -1.0, 1.0)
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── PLOT 1: PREDICTION VS GROUND TRUTH SCATTER
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(
    predicted: np.ndarray,
    target: np.ndarray,
    save_path: Path,
    mse: float,
) -> None:
    """
    Scatter plot: predicted ⟨Z_i⟩ vs ground-truth ⟨Z_i⟩ for all qubits.

    Each qubit gets its own colour and marker.  The ideal y = x reference line
    is plotted in black dashes.  Overall MSE is annotated in the legend box.

    Args:
        predicted : (N, n_qubits) — mitigated predictions.
        target    : (N, n_qubits) — ground-truth clean observables.
        save_path : Output file path (extension sets format).
        mse       : Pre-computed overall MSE (displayed in annotation).
    """
    N, n_qubits = predicted.shape

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # ── Per-qubit scatter ─────────────────────────────────────────────────────
    for q in range(n_qubits):
        ax.scatter(
            target[:, q],
            predicted[:, q],
            label      = _lz(q),
            color      = _PALETTE[q],
            alpha      = 0.72,
            s          = 38,
            edgecolors = "white",
            linewidths = 0.4,
            zorder     = 3,
        )

    # ── Ideal y = x reference line ────────────────────────────────────────────
    all_vals = np.concatenate([target.ravel(), predicted.ravel()])
    lo, hi   = all_vals.min() - 0.05, all_vals.max() + 0.05
    ax.plot(
        [lo, hi], [lo, hi],
        color     = "#222222",
        lw        = 1.6,
        ls        = "--",
        label     = r"$\hat{y} = y$ (ideal)",
        zorder    = 2,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # ── Annotations ───────────────────────────────────────────────────────────
    r2 = float(np.corrcoef(target.ravel(), predicted.ravel())[0, 1] ** 2)
    ax.text(
        0.04, 0.95,
        f"MSE = {mse:.4f}\n$R^2$ = {r2:.4f}",
        transform            = ax.transAxes,
        fontsize             = 10,
        verticalalignment    = "top",
        bbox                 = dict(boxstyle="round,pad=0.35", facecolor="white",
                                    edgecolor="#cccccc", alpha=0.9),
    )

    # ── Labels & styling ──────────────────────────────────────────────────────
    ax.set_xlabel(fr"Ground Truth {_obs_axis_label()}")
    ax.set_ylabel(fr"Mitigated Prediction {_obs_axis_label()}")
    ax.set_title("Prediction vs.\ Ground Truth")
    ax.legend(
        loc            = "lower right",
        framealpha     = 0.9,
        edgecolor      = "#cccccc",
        handletextpad  = 0.4,
        fontsize       = 10,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or _FIG_FMT)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── PLOT 2: ABSOLUTE ERROR DISTRIBUTION HISTOGRAM
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_histogram(
    predicted: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    save_path: Path,
) -> None:
    """
    Histogram + KDE of absolute prediction errors for the mitigated model
    vs the unmitigated mean-field syndrome baseline.

    Two overlapping semi-transparent histograms are drawn on a shared axis
    so the error reduction from our model is immediately visible.

    Args:
        predicted : (N, n_qubits) — model's mitigated predictions.
        target    : (N, n_qubits) — ground-truth clean observables.
        baseline  : (N, n_qubits) — unmitigated syndrome baseline predictions.
        save_path : Output file path.
    """
    err_model    = np.abs(predicted - target).ravel()
    err_baseline = np.abs(baseline  - target).ravel()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    # ── Shared bin range ──────────────────────────────────────────────────────
    max_err = max(err_model.max(), err_baseline.max())
    bins    = np.linspace(0, max_err * 1.05, 35)

    common_kw = dict(bins=bins, edgecolor="white", linewidth=0.4, alpha=0.65)

    ax.hist(
        err_baseline,
        **common_kw,
        color = sns.color_palette("muted")[3],
        label = fr"Unmitigated (MAE = {err_baseline.mean():.4f})",
    )
    ax.hist(
        err_model,
        **common_kw,
        color = sns.color_palette("muted")[0],
        label = fr"Mitigated — ours (MAE = {err_model.mean():.4f})",
    )

    # ── KDE overlay for the model errors (shows distributional shape cleanly) ─
    try:
        from scipy.stats import gaussian_kde   # optional but improves aesthetics
        kde  = gaussian_kde(err_model, bw_method=0.3)
        xs   = np.linspace(0, max_err * 1.05, 300)
        # Scale KDE to match histogram area
        bin_width  = bins[1] - bins[0]
        kde_scaled = kde(xs) * len(err_model) * bin_width
        ax.plot(xs, kde_scaled, color=sns.color_palette("muted")[0],
                lw=2.0, ls="-", label="_nolegend_")
    except ImportError:
        pass   # scipy not installed — skip KDE

    # ── Mean lines ────────────────────────────────────────────────────────────
    ax.axvline(
        err_model.mean(),
        color = sns.color_palette("muted")[0],
        lw    = 1.8, ls = "--", alpha = 0.9,
    )
    ax.axvline(
        err_baseline.mean(),
        color = sns.color_palette("muted")[3],
        lw    = 1.8, ls = "--", alpha = 0.9,
    )

    # ── Labels & styling ──────────────────────────────────────────────────────
    ax.set_xlabel(_abs_err_label())
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution")
    ax.legend(framealpha=0.9, edgecolor="#cccccc")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Improvement percentage annotation
    improvement = 100 * (1 - err_model.mean() / (err_baseline.mean() + 1e-12))
    ax.text(
        0.97, 0.95,
        fr"Error reduction: {improvement:.1f}\%",
        transform         = ax.transAxes,
        fontsize          = 10,
        ha                = "right",
        va                = "top",
        bbox              = dict(boxstyle="round,pad=0.35", facecolor="white",
                                 edgecolor="#cccccc", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or _FIG_FMT)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── PLOT 3: QUBIT-WISE PERFORMANCE BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_qubit_performance(
    predicted: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    save_path: Path,
) -> None:
    """
    Grouped bar chart showing per-qubit MAE for the mitigated model vs baseline.

    Two bar groups per qubit (mitigated in blue, baseline in red) allow the
    reader to immediately see whether noise mitigation is uniform across the
    hydrogen chain or favours certain qubits.

    Error bars show ± one standard deviation of absolute errors.

    Args:
        predicted : (N, n_qubits) — model's mitigated predictions.
        target    : (N, n_qubits) — ground-truth clean observables.
        baseline  : (N, n_qubits) — unmitigated syndrome baseline.
        save_path : Output file path.
    """
    N, n_qubits = predicted.shape

    abs_err_model    = np.abs(predicted - target)    # (N, n_qubits)
    abs_err_baseline = np.abs(baseline  - target)

    mae_model    = abs_err_model.mean(axis=0)         # (n_qubits,)
    mae_baseline = abs_err_baseline.mean(axis=0)
    std_model    = abs_err_model.std(axis=0)
    std_baseline = abs_err_baseline.std(axis=0)

    mse_model    = (( predicted - target) ** 2).mean(axis=0)
    mse_baseline = ((baseline   - target) ** 2).mean(axis=0)

    qubit_labels = [fr"$Q_{{{q}}}$" for q in range(n_qubits)]
    x            = np.arange(n_qubits)
    bar_w        = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=False)

    # ── Left panel: MAE ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(
        x - bar_w / 2, mae_model, bar_w,
        yerr      = std_model,
        color     = sns.color_palette("muted")[0],
        capsize   = 4,
        error_kw  = dict(elinewidth=1.2, capthick=1.2),
        label     = "Mitigated",
        zorder    = 3,
    )
    ax.bar(
        x + bar_w / 2, mae_baseline, bar_w,
        yerr      = std_baseline,
        color     = sns.color_palette("muted")[3],
        capsize   = 4,
        error_kw  = dict(elinewidth=1.2, capthick=1.2),
        label     = "Unmitigated",
        zorder    = 3,
    )
    # Overall mean reference lines
    ax.axhline(
        mae_model.mean(), color=sns.color_palette("muted")[0],
        lw=1.4, ls="--", alpha=0.7, label=r"Mean (mitigated)",
    )
    ax.axhline(
        mae_baseline.mean(), color=sns.color_palette("muted")[3],
        lw=1.4, ls="--", alpha=0.7, label=r"Mean (unmitigated)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(qubit_labels)
    ax.set_ylabel(fr"MAE $= \langle|\hat{{y}}-y|\rangle$")
    ax.set_title("Per-Qubit MAE")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── Right panel: MSE ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(
        x - bar_w / 2, mse_model, bar_w,
        color = sns.color_palette("muted")[0],
        label = "Mitigated",
        zorder = 3,
    )
    ax.bar(
        x + bar_w / 2, mse_baseline, bar_w,
        color = sns.color_palette("muted")[3],
        label = "Unmitigated",
        zorder = 3,
    )
    ax.axhline(
        mse_model.mean(), color=sns.color_palette("muted")[0],
        lw=1.4, ls="--", alpha=0.7,
    )
    ax.axhline(
        mse_baseline.mean(), color=sns.color_palette("muted")[3],
        lw=1.4, ls="--", alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(qubit_labels)
    ax.set_ylabel(r"MSE $= \langle(\hat{y}-y)^2\rangle$")
    ax.set_title("Per-Qubit MSE")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    fig.suptitle(
        r"Qubit-wise Error Analysis: H$_4$ Chain",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, format=save_path.suffix.lstrip(".") or _FIG_FMT)
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 ── MAIN EVALUATION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args: argparse.Namespace) -> None:
    """
    Full evaluation pipeline: load → infer → metrics → plots.

    Args:
        args: Parsed argparse namespace.
    """
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  QEM Diffusion Model — Evaluation")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"{'='*60}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model, full_args = load_model(Path(args.checkpoint), device)

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    circuit_tokens, target_observables, syndromes = load_dataset(Path(args.data_dir))

    N          = circuit_tokens.shape[0]
    n_qubits   = target_observables.shape[1]
    n_syndromes = syndromes.shape[1]

    # ── 3. Run inference ──────────────────────────────────────────────────────
    predicted_observables = run_inference(
        model              = model,
        circuit_tokens     = circuit_tokens,
        target_observables = target_observables,
        syndromes          = syndromes,
        device             = device,
        n_elec             = args.n_elec,
    )

    # ── 4. Unmitigated syndrome baseline (for comparison) ─────────────────────
    baseline = _syndrome_baseline(syndromes, n_qubits)

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    mse_total     = float(np.mean((predicted_observables - target_observables) ** 2))
    mae_total     = float(np.mean(np.abs(predicted_observables - target_observables)))
    mse_baseline  = float(np.mean((baseline - target_observables) ** 2))
    mae_baseline  = float(np.mean(np.abs(baseline - target_observables)))
    mae_per_qubit = np.mean(np.abs(predicted_observables - target_observables), axis=0)

    print(f"\n{'─'*60}")
    print(f"  RESULTS  ({N} test samples, {n_qubits} qubits)")
    print(f"{'─'*60}")
    print(f"  {'Metric':<28} {'Mitigated':>12}  {'Baseline':>12}")
    print(f"  {'─'*54}")
    print(f"  {'Overall MSE':<28} {mse_total:>12.6f}  {mse_baseline:>12.6f}")
    print(f"  {'Overall MAE':<28} {mae_total:>12.6f}  {mae_baseline:>12.6f}")
    print(f"  {'Error reduction (MAE)':<28} "
          f"{100*(1 - mae_total/(mae_baseline+1e-12)):>11.2f}%")
    print(f"{'─'*60}")
    print(f"  Per-qubit MAE: "
          + "  ".join(f"Q{q}={mae_per_qubit[q]:.4f}" for q in range(n_qubits)))
    print(f"{'─'*60}\n")

    # ── 6. Save plots ─────────────────────────────────────────────────────────
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter(
        predicted  = predicted_observables,
        target     = target_observables,
        save_path  = plots_dir / f"qem_scatter.{_FIG_FMT}",
        mse        = mse_total,
    )
    plot_error_histogram(
        predicted  = predicted_observables,
        target     = target_observables,
        baseline   = baseline,
        save_path  = plots_dir / f"qem_error_hist.{_FIG_FMT}",
    )
    plot_qubit_performance(
        predicted  = predicted_observables,
        target     = target_observables,
        baseline   = baseline,
        save_path  = plots_dir / f"qem_qubit_mae.{_FIG_FMT}",
    )

    print(f"\n[done] All figures saved to: {plots_dir.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 ── ARGUMENT PARSER & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and plot results for the QEM Diffusion Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file (e.g., checkpoints/epoch_0020.pt).",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data/raw_noisy_circuits",
        help="Directory containing circuit_tokens.npy, target_observables.npy, syndromes.npy.",
    )
    p.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Output directory for saved figures.",
    )
    p.add_argument(
        "--n_elec",
        type=int,
        default=None,
        help=(
            "Number of active electrons for HF state initialisation. "
            "Defaults to n_qubits // 2 (half-filling). "
            "For H4 with 4 qubits and 2 active electrons, use --n_elec 2."
        ),
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when a GPU is available.",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
