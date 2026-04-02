"""
generate_qem_data.py

Data factory for the QEM Diffusion Model training pipeline.
Produces three .npy arrays consumed by QEMDataset in train_mitigator.py:

    data/raw_noisy_circuits/
        circuit_tokens.npy       (N, circuit_depth)   int64
        target_observables.npy   (N, n_qubits)        float32  — clean ⟨Z_i⟩
        syndromes.npy            (N, n_qubits - 1)    float32  — noisy ⟨Z_i Z_{i+1}⟩

Memory architecture
───────────────────
All O(2^n) objects have been eliminated:
    ✗  REMOVED  expand_operator_pool_by_lie_closure  — stored O(2^n × 2^n) matrices
    ✗  REMOVED  create_shadow_unitary_from_commutator — scipy.linalg.expm on 2^n × 2^n
    ✗  REMOVED  qml.matrix()  calls — always materialises full 2^n × 2^n tensor
    ✗  REMOVED  qml.QubitUnitary(np.eye(2**nq))      — 2^n × 2^n identity matrix
    ✗  REMOVED  PennyLane simulation backend

    ✓  REPLACED with hardware-efficient ansatz (HEA) gate pool — O(n) tokens
    ✓  REPLACED with cudaq.observe() — C++ statevector, never stores 2^n × 2^n
    ✓  KEPT     select_active_space() — operates on (n_AO × n_AO) RDM1
    ✓  KEPT     GPTQE / GPT Transformer — finds optimal gate sequences
    ✓  KEPT     PennyLane qchem — Hamiltonian as sparse Pauli strings (no qml.matrix)
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import cudaq
from cudaq import spin as cudaq_spin

from pyscf import gto, scf
import pennylane as qml          # used ONLY for qchem Hamiltonian (no qml.matrix calls)

# GQE Transformer — kept from original; model.py must be in the same directory
from model import GPT, GPTConfig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── HARDWARE-EFFICIENT ANSATZ (HEA) GATE POOL
#
# Replaces the dense Lie-closure operator pool entirely.
#
# Memory cost of the OLD pool at 40 qubits:
#   Each operator in the pool was stored as a (2^40 × 2^40) complex matrix.
#   Even a pool of 10 operators = 10 × (2^40)² × 16 bytes ≈ impossible.
#
# Memory cost of the NEW pool:
#   Each token is a single integer. A pool of 3n tokens = 3 × 40 = 120 integers.
#
# Token vocabulary
# ────────────────
#   0          : [PAD] / BOS — no operation
#   1  …  n    : Ry(RY_ANGLE) on qubit (token − 1)
#   n+1 … 2n   : Rz(RZ_ANGLE) on qubit (token − n − 1)
#   2n+1… 3n−1 : CNOT with control=i, target=i+1  for i = 0 … n−2
#
# Total vocabulary size = 3n  (plus 0 = PAD gives 3n+1 tokens in the GPT)
# ══════════════════════════════════════════════════════════════════════════════

RY_ANGLE: float = math.pi / 4   # fixed rotation angle for Ry gates
RZ_ANGLE: float = math.pi / 4   # fixed rotation angle for Rz gates


def hea_vocab_size(n_qubits: int) -> int:
    """
    Returns the number of non-padding gate tokens for a given qubit count.

    Tokens 1 … vocab_size(n) are valid gate tokens.
    Token 0 is reserved for PAD / BOS.

    Memory: O(n) — a constant per qubit, never O(2^n).

    Args:
        n_qubits: Number of qubits.

    Returns:
        Integer vocabulary size (excluding padding token 0).
    """
    # n Ry tokens + n Rz tokens + (n-1) CNOT tokens = 3n - 1
    return 3 * n_qubits - 1


def describe_token(token: int, n_qubits: int) -> str:
    """Human-readable description of a gate token (for logging)."""
    if token == 0:
        return "[PAD]"
    elif 1 <= token <= n_qubits:
        return f"Ry({RY_ANGLE:.3f}, q{token - 1})"
    elif n_qubits < token <= 2 * n_qubits:
        return f"Rz({RZ_ANGLE:.3f}, q{token - n_qubits - 1})"
    elif 2 * n_qubits < token <= 3 * n_qubits - 1:
        i = token - 2 * n_qubits - 1
        return f"CNOT(q{i}→q{i + 1})"
    return f"[UNKNOWN token {token}]"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── ACTIVE SPACE SELECTION  (kept from original, unchanged)
#
# This function is memory-safe at any qubit count because:
#   - mf.make_rdm1() returns a (n_AO × n_AO) 1-electron RDM, not 2^n × 2^n
#   - np.linalg.eigh(dm) operates on that small matrix
#   - No 2^n objects are created anywhere in this function
# ══════════════════════════════════════════════════════════════════════════════

def select_active_space(
    atom_str: str,
    basis: str = "cc-pvdz",
    symmetry: bool = True,
    spin: int = 0,
    manual_orbital_indices: List[int] | None = None,
    max_active_orbs: int | None = None,
    lower_occ_threshold: float | None = None,
    upper_occ_threshold: float | None = None,
    verbose: bool = False,
) -> Tuple[List[int], int, int]:
    """
    Selects the active orbital space from a PySCF RHF/UHF calculation.

    Memory profile: O(n_AO²) for the 1-RDM — safe at all scales.

    Args:
        atom_str            : PySCF atom specification string.
        basis               : Basis set name.
        symmetry            : Use molecular symmetry.
        spin                : Spin multiplicity − 1 (0 = singlet).
        manual_orbital_indices: Force-include these orbital indices.
        max_active_orbs     : Hard cap on number of active orbitals.
        lower_occ_threshold : Lower natural occupation cutoff.
        upper_occ_threshold : Upper natural occupation cutoff.
        verbose             : Print selection details.

    Returns:
        Tuple of (active_orbital_indices, n_orbs, n_elec).
    """
    mol = gto.M(atom=atom_str, basis=basis, symmetry=symmetry, spin=spin)
    mf  = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf.verbose = 0
    mf.kernel()

    # 1-electron reduced density matrix — shape (n_AO, n_AO), NOT (2^n, 2^n)
    dm              = mf.make_rdm1() if spin == 0 else sum(mf.make_rdm1())
    eigvals, _      = np.linalg.eigh(dm)
    idx_sorted      = np.argsort(eigvals)[::-1]
    natural_occ     = eigvals[idx_sorted]

    energies_raw    = mf.mo_energy if spin == 0 else 0.5 * (mf.mo_energy[0] + mf.mo_energy[1])
    mo_energy       = np.array(energies_raw)[idx_sorted]

    low_thr = np.percentile(natural_occ, 5)  if lower_occ_threshold is None else lower_occ_threshold
    up_thr  = np.percentile(natural_occ, 95) if upper_occ_threshold is None else upper_occ_threshold
    occ_inds = [i for i, o in enumerate(natural_occ) if low_thr < o < up_thr]

    span    = mo_energy.max() - mo_energy.min()
    window  = 0.1 * span
    homo    = mol.nelectron // 2 - 1
    lumo    = homo + 1
    emin    = mo_energy[homo] - window
    emax    = mo_energy[lumo] + window
    energy_inds = [i for i, e in enumerate(mo_energy) if emin <= e <= emax]

    manual = manual_orbital_indices or []
    if spin == 0:
        final = sorted(set(occ_inds) & set(energy_inds) | set(manual))
    else:
        final = sorted(set(occ_inds) | set(manual))

    if not final:
        final = [homo, lumo]
    if spin == 0 and len(final) < 2:
        final = [homo, lumo]
    if max_active_orbs and len(final) > max_active_orbs:
        final = sorted(final, key=lambda i: natural_occ[i], reverse=True)[:max_active_orbs]

    n_orbs = len(final)
    n_elec = (int(round(natural_occ[final].sum()))
              if spin == 0
              else int(round(natural_occ[final].sum() / 2)))
    if spin == 0 and (n_elec % 2) != 0:
        n_elec = 2

    if verbose:
        print(f"  Active orbitals: {final}  ({n_orbs} orbs, {n_elec} elec)")
    return final, n_orbs, n_elec


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── HAMILTONIAN: PennyLane qchem → CUDA-Q SpinOperator
#
# PennyLane qchem.molecular_hamiltonian() returns a Hamiltonian as a sum of
# weighted Pauli strings.  We extract those Pauli strings and convert them to
# a cudaq.SpinOperator — a sparse representation that never materialises the
# full 2^n × 2^n matrix.
#
# Memory cost: O(num_Pauli_terms × n_qubits) — polynomial, not exponential.
#
# CRITICAL: qml.matrix() is NEVER called anywhere in this section.
# ══════════════════════════════════════════════════════════════════════════════

def build_hamiltonian(
    symbols: List[str],
    coords: np.ndarray,
    basis: str,
    n_elec: int,
    n_orbs: int,
) -> Tuple[cudaq.SpinOperator, int, np.ndarray]:
    """
    Builds the molecular Hamiltonian as a CUDA-Q SpinOperator.

    Uses PennyLane qchem for the Jordan-Wigner transformation but strips the
    result down to sparse Pauli strings before passing to CUDA-Q.
    qml.matrix() is never called — no dense 2^n × 2^n matrix is ever created.

    Args:
        symbols : Atom symbols, e.g. ['H', 'H', 'H', 'H'].
        coords  : Atomic coordinates, shape (n_atoms, 3).
        basis   : Basis set string.
        n_elec  : Number of active electrons.
        n_orbs  : Number of active orbitals.

    Returns:
        Tuple of:
            spin_op  : cudaq.SpinOperator — sparse Hamiltonian, O(n) memory.
            n_qubits : int — number of qubits (= 2 × n_orbs in Jordan-Wigner).
            hf_state : np.ndarray of shape (n_qubits,) — HF occupation vector
                       with values 0 or 1. NOT a 2^n state vector.
    """
    # PennyLane qchem builds the JW Hamiltonian as sparse Pauli strings.
    # This call is safe — it does NOT materialise any 2^n × 2^n matrix.
    H_pl, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coords,
        basis=basis, charge=0,
        active_electrons=n_elec,
        active_orbitals=n_orbs,
    )

    # HF reference state: length-n_qubits array of 0/1 occupation numbers.
    # This is a length-n vector, NOT a 2^n state vector — memory safe.
    hf_state = qml.qchem.hf_state(n_elec, n_qubits)

    spin_op = _pennylane_ham_to_cudaq(H_pl, n_qubits)
    return spin_op, n_qubits, hf_state


def _pennylane_ham_to_cudaq(H_pl, n_qubits) -> cudaq.SpinOperator:
    """
    Converts a PennyLane Hamiltonian to a CUDA-Q SpinOperator.

    Iterates over the sparse Pauli-string representation of H_pl.
    Memory cost per term: O(n_qubits).  Total: O(num_terms × n_qubits).
    qml.matrix() is NEVER called — no 2^n × 2^n matrix is ever created.

    Args:
        H_pl     : PennyLane Hamiltonian object (sum of weighted Pauli strings).
        n_qubits : Number of qubits.

    Returns:
        cudaq.SpinOperator representing the same Hamiltonian.
    """
    total_op: cudaq.SpinOperator | None = None
    
    coeffs, ops = H_pl.terms()

    for coeff, op in zip(coeffs, ops):
        pauli_word = _extract_pauli_word(op, n_qubits)   # e.g. "IXZYI"
        term_op    = _pauli_word_to_spin_op(pauli_word)
        weighted   = float(np.real(coeff)) * term_op

        total_op = weighted if total_op is None else total_op + weighted

    if total_op is None:
        raise ValueError("Empty Hamiltonian — no Pauli terms found.")
    return total_op


def _extract_pauli_word(op: qml.operation.Operator, n_qubits: int) -> str:
    """
    Extracts a length-n_qubits Pauli word string from a PennyLane operator.

    Handles both single-qubit Paulis and tensor products (old and new API).

    Args:
        op       : A PennyLane Pauli operator or tensor product thereof.
        n_qubits : Total qubit count (sets the length of the output string).

    Returns:
        String of length n_qubits over alphabet {I, X, Y, Z}.
    """
    pauli_word = ['I'] * n_qubits
    _PAULI_MAP = {'PauliX': 'X', 'PauliY': 'Y', 'PauliZ': 'Z'}

    # Support both qml.operation.Tensor (legacy) and qml.ops.Prod (new API)
    if hasattr(op, 'obs'):          # Tensor product — legacy PennyLane
        sub_ops = op.obs
    elif hasattr(op, 'operands'):   # Prod — PennyLane >= 0.36
        sub_ops = list(op.operands)
    else:                            # Single operator
        sub_ops = [op]

    for sub in sub_ops:
        name = type(sub).__name__
        if name in _PAULI_MAP:
            for wire in sub.wires:
                pauli_word[int(wire)] = _PAULI_MAP[name]
        # Identity ('Identity' or 'I') → leave as 'I'

    return ''.join(pauli_word)


def _pauli_word_to_spin_op(pauli_word: str) -> cudaq.SpinOperator:
    """
    Converts a Pauli word string to a CUDA-Q SpinOperator.

    Builds the operator by multiplying single-qubit spin operators together.
    Pure identity terms are represented via cudaq.spin.i(0).

    Args:
        pauli_word: String of length n over {I, X, Y, Z}.

    Returns:
        cudaq.SpinOperator for this Pauli word.
    """
    term: cudaq.SpinOperator | None = None

    for i, p in enumerate(pauli_word):
        if   p == 'X': factor = cudaq_spin.x(i)
        elif p == 'Y': factor = cudaq_spin.y(i)
        elif p == 'Z': factor = cudaq_spin.z(i)
        else:          continue   # 'I' — identity on this qubit, skip

        term = factor if term is None else term * factor

    # Pure identity Pauli word (all 'I') → represent as spin.i(0)
    # cudaq.spin.i(0) is the identity operator; ⟨I⟩ = 1 always
    if term is None:
        term = cudaq_spin.i(0)

    return term


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── CUDA-Q CIRCUIT EXECUTOR
#
# Replaces the PennyLane qnode entirely.
# cudaq.observe() uses a C++ statevector/density-matrix backend and never
# materialises the full operator matrix — it evaluates each Pauli term
# independently via the Hadamard test or statevector contraction.
#
# Memory: cudaq.observe with qpp-cpu uses O(2^n) for the statevector, but
# this is unavoidable for exact simulation.  The key difference from the
# original is that we NEVER store 2^n × 2^n matrices.
# ══════════════════════════════════════════════════════════════════════════════

def build_cudaq_kernel(
    token_seq: np.ndarray,
    n_qubits: int,
    hf_state: np.ndarray,
) -> cudaq.Kernel:
    """
    Builds a CUDA-Q kernel from a gate token sequence.

    Tokens are decoded into gate operations using the HEA vocabulary.
    No matrix multiplication or dense linear algebra is performed.

    Token → gate mapping:
        0              → skip (PAD)
        1 … n          → Ry(RY_ANGLE) on qubit (token − 1)
        n+1 … 2n       → Rz(RZ_ANGLE) on qubit (token − n − 1)
        2n+1 … 3n−1    → CNOT(control=i, target=i+1) where i = token − 2n − 1

    Args:
        token_seq : 1-D array of gate tokens (int), may include PAD (0).
        n_qubits  : Number of physical qubits.
        hf_state  : Array of shape (n_qubits,) with 0/1 HF occupation numbers.
                    Used to initialise the circuit in the HF reference state.
                    This is a length-n array, NOT a 2^n state vector.

    Returns:
        cudaq.Kernel ready for cudaq.observe().
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n_qubits)

    # Initialise HF reference state by flipping occupied qubits
    # hf_state is a length-n array — memory safe
    for i, occ in enumerate(hf_state):
        if int(occ) == 1:
            kernel.x(qubits[i])

    # Decode tokens → gates
    for token in token_seq:
        t = int(token)
        if t == 0:
            continue                                          # PAD — no-op

        elif 1 <= t <= n_qubits:
            qubit_idx = t - 1
            kernel.ry(RY_ANGLE, qubits[qubit_idx])           # Ry layer

        elif n_qubits < t <= 2 * n_qubits:
            qubit_idx = t - n_qubits - 1
            kernel.rz(RZ_ANGLE, qubits[qubit_idx])           # Rz layer

        elif 2 * n_qubits < t <= 3 * n_qubits - 1:
            i = t - 2 * n_qubits - 1
            kernel.cx(qubits[i], qubits[i + 1])              # entanglement layer

    return kernel


def evaluate_energy(
    token_seq: np.ndarray,
    n_qubits: int,
    hf_state: np.ndarray,
    spin_op: cudaq.SpinOperator,
) -> float:
    """
    Evaluates the energy ⟨H⟩ of a circuit using cudaq.observe().

    cudaq.observe() evaluates each Pauli term in spin_op independently
    using the C++ backend — it never materialises the 2^n × 2^n operator.

    Args:
        token_seq : Gate token sequence (1-D int array).
        n_qubits  : Number of qubits.
        hf_state  : HF reference occupation vector of length n_qubits.
        spin_op   : CUDA-Q SpinOperator (the molecular Hamiltonian).

    Returns:
        Scalar energy ⟨ψ|H|ψ⟩ as a Python float.
    """
    kernel = build_cudaq_kernel(token_seq, n_qubits, hf_state)
    return cudaq.observe(kernel, spin_op).expectation()


def get_subsequence_energies(
    token_sequences: np.ndarray,
    n_qubits: int,
    hf_state: np.ndarray,
    spin_op: cudaq.SpinOperator,
) -> np.ndarray:
    """
    Computes the energy at each prefix length for every token sequence.

    This is the energy oracle called during GQE training data preparation.
    Replaces the original PennyLane-based get_subsequence_energies().

    Args:
        token_sequences : Shape (N, seq_len) — 1-indexed gate token sequences.
        n_qubits        : Number of qubits.
        hf_state        : HF occupation vector of length n_qubits.
        spin_op         : CUDA-Q SpinOperator (molecular Hamiltonian).

    Returns:
        Float32 array of shape (N, seq_len) where [i, k] = ⟨H⟩ after k gates
        of sequence i have been applied.
    """
    N, seq_len = token_sequences.shape
    energies = np.zeros((N, seq_len), dtype=np.float32)

    for i in tqdm(range(N), desc="  Computing subsequence energies", leave=False):
        for k in range(1, seq_len + 1):
            energies[i, k - 1] = evaluate_energy(
                token_sequences[i, :k], n_qubits, hf_state, spin_op
            )
    return energies


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── GPTQE MODEL  (kept from original, no architectural changes)
#
# The Transformer's role is unchanged: given a partial token sequence,
# predict the next gate token and its cumulative energy contribution.
# The only change is that the energy oracle is now cudaq.observe() instead
# of a PennyLane qnode.
# ══════════════════════════════════════════════════════════════════════════════

class GPTQE(GPT):
    """
    Generative Pre-trained Transformer for Quantum Eigensolving (GQE).

    Extends the base GPT with an auxiliary time-prediction head.
    Generates gate token sequences that minimise the molecular energy.

    Architecture (unchanged from original hydrogen_chain.py):
        - Token embedding + positional embedding
        - Stack of Transformer blocks (self-attention + MLP)
        - lm_head  : predicts energy contribution per next token
        - time_head: predicts cumulative gate time per step
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.time_head   = torch.nn.Linear(config.n_embd, 1, bias=False)
        self.lambda_time = 1.0

    def forward(
        self,
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Token sequence (B, T) [int64].

        Returns:
            Tuple of (energy_logits (B, T, V), time_logits (B, T)).
        """
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(
            torch.arange(idx.size(1), device=idx.device)
        )
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x), self.time_head(x).squeeze(-1)

    def calculate_loss(
        self,
        tokens: torch.Tensor,
        energies: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the combined energy + time prediction loss.

        Args:
            tokens      : Shape (B, seq_len+1) — token sequences with BOS prefix.
            energies    : Shape (B, seq_len)   — subsequence energies from CUDA-Q.
            target_times: Shape (B, seq_len+1) — cumulative gate step counts.

        Returns:
            Scalar loss tensor.
        """
        curr, nxt  = tokens[:, :-1], tokens[:, 1:]
        e_logits, t_logits = self(curr)
        mask    = F.one_hot(nxt, num_classes=self.config.vocab_size)
        tok_e   = (e_logits * mask).sum(dim=2)
        cum_e   = torch.cumsum(tok_e, dim=1)
        cum_t   = torch.cumsum(t_logits, dim=1)
        return (
            F.mse_loss(cum_e, energies)
            + self.lambda_time * F.mse_loss(cum_t, target_times[:, 1:])
        )

    @torch.no_grad()
    def generate(
        self,
        n_seq: int,
        max_new: int,
        temp: float = 1.0,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively samples gate token sequences.

        Starts from BOS (token 0) and samples greedily-softmax from the
        negative energy logits (lower predicted energy = higher probability).

        Args:
            n_seq  : Number of sequences to generate in parallel.
            max_new: Number of gate tokens to generate per sequence.
            temp   : Softmax temperature (lower = greedier).
            device : PyTorch device string.

        Returns:
            Tuple of:
                idx   : (n_seq, max_new+1) — generated token sequences (BOS + gates).
                total : (n_seq, 1)         — cumulative predicted energy scores.
        """
        idx   = torch.zeros((n_seq, 1), dtype=torch.long, device=device)
        total = torch.zeros((n_seq, 1), device=device)

        for _ in range(max_new):
            cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            e_logits, _ = self(cond)
            last        = e_logits[:, -1, :]
            last[:, 0]  = float("inf")   # never sample PAD token again
            probs       = F.softmax(-last / temp, dim=-1)
            nxt         = torch.multinomial(probs, num_samples=1)
            total      += last.gather(1, nxt)
            idx         = torch.cat([idx, nxt], dim=1)

        return idx, total


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── GQE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_gqe(
    n_qubits: int,
    hf_state: np.ndarray,
    spin_op: cudaq.SpinOperator,
    train_cfg: dict,
    torch_device: torch.device,
) -> GPTQE:
    """
    Trains the GPTQE Transformer to generate low-energy gate sequences.

    Workflow:
        1. Sample random gate-token sequences from the HEA vocabulary.
        2. Evaluate their prefix energies using cudaq.observe() (no dense matrices).
        3. Train the GPT to predict these cumulative energies autoregressively.
        4. After training, generate() samples circuits that minimise the energy.

    Args:
        n_qubits    : Number of qubits.
        hf_state    : HF occupation vector of length n_qubits.
        spin_op     : CUDA-Q SpinOperator (molecular Hamiltonian).
        train_cfg   : Dict with keys: train_size, seq_len, n_epochs, n_batches,
                      eval_freq, lr, weight_decay.
        torch_device: PyTorch device for model and tensors.

    Returns:
        Trained GPTQE model (in eval mode).
    """
    train_size = train_cfg["train_size"]
    seq_len    = train_cfg["seq_len"]
    n_epochs   = train_cfg["n_epochs"]
    n_batches  = train_cfg["n_batches"]
    eval_freq  = train_cfg["eval_freq"]
    vsize      = hea_vocab_size(n_qubits)   # 3n-1

    # ── Build training token sequences ────────────────────────────────────────
    # Tokens are sampled from [1, vsize] — 0 is reserved for PAD/BOS
    # Shape: (train_size, seq_len) — integer gate indices, memory O(n)
    print("  Sampling random HEA sequences for training data...")
    raw_tokens = np.random.randint(
        low=1, high=vsize + 1,
        size=(train_size, seq_len),
    ).astype(np.int64)

    # Prepend BOS token (0) → shape (train_size, seq_len+1)
    bos        = np.zeros((train_size, 1), dtype=np.int64)
    tokens_arr = np.concatenate([bos, raw_tokens], axis=1)

    # ── Compute subsequence energies via CUDA-Q ───────────────────────────────
    print("  Evaluating subsequence energies via cudaq.observe()...")
    cudaq.set_target("qpp-cpu")
    subseq_energies = get_subsequence_energies(
        raw_tokens, n_qubits, hf_state, spin_op
    )  # shape (train_size, seq_len)

    # ── Cumulative time array (one unit per gate step) ────────────────────────
    # time_lookup: token 0 → 0.0, tokens 1..vsize → 1.0
    time_lookup        = np.ones(vsize + 1, dtype=np.float32)
    time_lookup[0]     = 0.0
    times_arr          = np.cumsum(time_lookup[tokens_arr], axis=1).astype(np.float32)

    # ── Move to PyTorch tensors ───────────────────────────────────────────────
    tk = torch.tensor(tokens_arr,    dtype=torch.long,    device=torch_device)
    en = torch.tensor(subseq_energies, dtype=torch.float32, device=torch_device)
    tm = torch.tensor(times_arr,     dtype=torch.float32, device=torch_device)

    # ── Build GPTQE ───────────────────────────────────────────────────────────
    gptqe = GPTQE(GPTConfig(
        vocab_size  = vsize + 1,       # tokens 0..vsize
        block_size  = seq_len,
        dropout     = 0.1,
        bias        = False,
    )).to(torch_device)

    optimizer = gptqe.configure_optimizers(
        weight_decay  = train_cfg.get("weight_decay", 0.01),
        learning_rate = train_cfg.get("lr", 5e-5),
        betas         = (0.9, 0.999),
        device_type   = torch_device.type,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_avg_energy = float("inf")

    for epoch in tqdm(range(n_epochs), desc="  Training GPTQE"):
        gptqe.train()
        perm = np.random.permutation(train_size)

        for b in range(n_batches):
            idxs = perm[b::n_batches]
            optimizer.zero_grad()
            loss = gptqe.calculate_loss(tk[idxs], en[idxs], tm[idxs])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_freq == 0:
            gptqe.eval()
            gen_tk, gen_sc = gptqe.generate(
                n_seq=64, max_new=seq_len, temp=5e-4, device=torch_device
            )
            gen_tokens_np = gen_tk[:, 1:].cpu().numpy()   # remove BOS

            # Evaluate true energies of fully-generated sequences
            true_E = np.array([
                evaluate_energy(gen_tokens_np[i], n_qubits, hf_state, spin_op)
                for i in range(len(gen_tokens_np))
            ])
            avg_e = float(np.mean(true_E))

            if avg_e < best_avg_energy:
                best_avg_energy = avg_e

            tqdm.write(
                f"  epoch {epoch + 1:>4d} | loss {loss.item():.5f} | "
                f"avg_E {avg_e:.6f} | best_E {best_avg_energy:.6f}"
            )
            gptqe.train()

    print(f"  GQE training complete. Best average energy: {best_avg_energy:.6f} Ha")
    gptqe.eval()
    return gptqe


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 ── QEM DATASET GENERATION
#
# Once the GQE is trained, we generate the three .npy arrays by running
# the optimal circuits through two measurement modes:
#
#   Pass 1 — Clean (qpp-cpu, no noise):
#     target_observables[i, j] = ⟨Z_j⟩ under the ideal circuit
#
#   Pass 2 — Noisy (density-matrix-cpu + DepolarizationChannel):
#     syndromes[i, j] = ⟨Z_j Z_{j+1}⟩ under the noisy circuit
#
# These are the inputs and outputs of the DiffusionCompiler at inference time.
# ══════════════════════════════════════════════════════════════════════════════

def generate_qem_dataset(
    gptqe: GPTQE,
    n_qubits: int,
    hf_state: np.ndarray,
    spin_op: cudaq.SpinOperator,
    num_samples: int,
    circuit_depth: int,
    depol_p: float,
    torch_device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the three QEM dataset arrays using the trained GPTQE.

    Measurement operators (all O(n) memory — no 2^n objects):
        Z_i     = cudaq_spin.z(i)           for i in 0..n-1  → target_observables
        Z_iZ_{i+1} = z(i)*z(i+1)           for i in 0..n-2  → syndromes

    Args:
        gptqe        : Trained GPTQE model (in eval mode).
        n_qubits     : Number of qubits.
        hf_state     : HF occupation vector of length n_qubits.
        spin_op      : CUDA-Q molecular Hamiltonian (for energy logging only).
        num_samples  : Number of (circuit, observables, syndromes) triplets.
        circuit_depth: Number of gate tokens per circuit.
        depol_p      : Depolarisation probability applied after CX gates.
        torch_device : PyTorch device for GQE inference.

    Returns:
        Tuple of three arrays:
            circuit_tokens     : (num_samples, circuit_depth)   int64
            target_observables : (num_samples, n_qubits)        float32
            syndromes          : (num_samples, n_qubits - 1)    float32
    """
    # ── Generate gate sequences from trained GQE ──────────────────────────────
    print(f"  Generating {num_samples} circuits from trained GPTQE...")
    gen_tokens, _ = gptqe.generate(
        n_seq    = num_samples,
        max_new  = circuit_depth,
        temp     = 1e-3,
        device   = torch_device,
    )
    # gen_tokens shape: (num_samples, circuit_depth+1) with BOS at col 0
    # Remove BOS; tokens are now in range [0, vsize]
    token_array = gen_tokens[:, 1:].cpu().numpy().astype(np.int64)
    # token_array shape: (num_samples, circuit_depth)

    # ── Pre-build all observable operators (O(n) objects, never O(2^n)) ──────
    # Single-qubit Z observables: one per qubit → target_observables
    z_ops  = [cudaq_spin.z(i) for i in range(n_qubits)]
    # ZZ parity checks: one per adjacent pair → syndromes (QEC stabilisers)
    zz_ops = [cudaq_spin.z(i) * cudaq_spin.z(i + 1) for i in range(n_qubits - 1)]

    # ── Noise model: depolarising channel on entanglement gates ───────────────
    # Two-qubit CX gates receive full depol_p; single-qubit gates receive 10×
    noise_model = cudaq.NoiseModel()
    noise_model.add_all_qubit_channel(
        "cx", cudaq.DepolarizationChannel(depol_p)
    )
    noise_model.add_all_qubit_channel(
        "ry", cudaq.DepolarizationChannel(depol_p / 10.0)
    )
    noise_model.add_all_qubit_channel(
        "rz", cudaq.DepolarizationChannel(depol_p / 10.0)
    )

    # ── Pre-allocate output arrays ────────────────────────────────────────────
    circuit_tokens_out     = token_array.copy()
    target_observables_out = np.zeros((num_samples, n_qubits),      dtype=np.float32)
    syndromes_out          = np.zeros((num_samples, n_qubits - 1),  dtype=np.float32)

    # ── PASS 1: Clean measurements (qpp-cpu, no noise) ────────────────────────
    # target_observables[i, j] = ⟨Z_j⟩_ideal
    print("  Pass 1/2 — Clean measurements (qpp-cpu)...")
    cudaq.set_target("qpp-cpu")

    for i in tqdm(range(num_samples), desc="  Clean Z expectations"):
        kernel = build_cudaq_kernel(token_array[i], n_qubits, hf_state)
        for j, z_op in enumerate(z_ops):
            target_observables_out[i, j] = cudaq.observe(kernel, z_op).expectation()

    # ── PASS 2: Noisy measurements (density-matrix-cpu + depolarising noise) ──
    # syndromes[i, j] = ⟨Z_j Z_{j+1}⟩_noisy
    print("  Pass 2/2 — Noisy ZZ syndrome measurements (density-matrix-cpu)...")
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_noise(noise_model)

    for i in tqdm(range(num_samples), desc="  Noisy ZZ syndromes"):
        kernel = build_cudaq_kernel(token_array[i], n_qubits, hf_state)
        for j, zz_op in enumerate(zz_ops):
            syndromes_out[i, j] = cudaq.observe(kernel, zz_op).expectation()

    # Restore clean simulation settings
    cudaq.unset_noise()
    cudaq.set_target("qpp-cpu")

    return circuit_tokens_out, target_observables_out, syndromes_out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 ── MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QEM Data Factory: GQE → CUDA-Q → .npy dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Molecule & chemistry
    p.add_argument("--basis",          type=str,   default="sto-3g",
                   choices=["sto-3g", "6-31g", "cc-pvdz"],
                   help="Basis set for PySCF/PennyLane qchem.")
    p.add_argument("--bond_length",    type=float, default=0.74,
                   help="H–H bond length in Angstroms.")
    # Circuit
    p.add_argument("--circuit_depth",  type=int,   default=8,
                   help="Number of gate tokens per circuit (GQE sequence length).")
    # GQE training
    p.add_argument("--train_size",     type=int,   default=1024,
                   help="Number of random sequences for GQE training data.")
    p.add_argument("--n_epochs",       type=int,   default=100)
    p.add_argument("--n_batches",      type=int,   default=8)
    p.add_argument("--eval_freq",      type=int,   default=10)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    # Dataset generation
    p.add_argument("--num_samples",    type=int,   default=1000,
                   help="Number of (circuit, observables, syndromes) samples to save.")
    p.add_argument("--depol_p",        type=float, default=0.01,
                   help="Depolarisation probability for noise injection.")
    # I/O
    p.add_argument("--output_dir",     type=str,   default="data/raw_noisy_circuits",
                   help="Directory for output .npy files.")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("  QEM Data Factory — generate_qem_data.py")
    print(f"  Molecule : H4 linear chain  ({args.basis})")
    print(f"  Device   : {torch_device}")
    print(f"  Seed     : {args.seed}")
    print("=" * 64)

    # ── H4 linear chain geometry ──────────────────────────────────────────────
    symbols = ["H", "H", "H", "H"]
    coords  = np.array([
        [0.0, 0.0, i * args.bond_length] for i in range(4)
    ])
    atom_str = "".join(
        f"{s} {x:.4f} {y:.4f} {z:.4f}\n"
        for s, (x, y, z) in zip(symbols, coords)
    )

    # ── Active space selection (PySCF) ────────────────────────────────────────
    print("\n[1/4] Active space selection (PySCF)...")
    thresholds = {"sto-3g": (0.05, 1.75), "6-31g": (0.02, 1.98), "cc-pvdz": (0.01, 1.99)}
    low_thr, up_thr = thresholds.get(args.basis, (0.01, 1.99))
    active_idx, n_orbs, n_elec = select_active_space(
        atom_str,
        basis=args.basis,
        lower_occ_threshold=low_thr,
        upper_occ_threshold=up_thr,
        verbose=True,
    )
    print(f"  → {n_orbs} active orbitals, {n_elec} active electrons")

    # ── Hamiltonian (PennyLane qchem → CUDA-Q SpinOperator) ──────────────────
    print("\n[2/4] Building Hamiltonian as sparse CUDA-Q SpinOperator...")
    spin_op, n_qubits, hf_state = build_hamiltonian(
        symbols, coords, args.basis, n_elec, n_orbs
    )
    vsize = hea_vocab_size(n_qubits)
    print(f"  → {n_qubits} qubits | HEA vocab size: {vsize} tokens")
    print(f"  → HF state: {hf_state}")

    # ── GQE training ──────────────────────────────────────────────────────────
    print("\n[3/4] Training GPTQE...")
    train_cfg = {
        "train_size"  : args.train_size,
        "seq_len"     : args.circuit_depth,
        "n_epochs"    : args.n_epochs,
        "n_batches"   : args.n_batches,
        "eval_freq"   : args.eval_freq,
        "lr"          : args.lr,
        "weight_decay": args.weight_decay,
    }
    gptqe = train_gqe(n_qubits, hf_state, spin_op, train_cfg, torch_device)

    # ── QEM dataset generation ────────────────────────────────────────────────
    print("\n[4/4] Generating QEM dataset...")
    circuit_tokens, target_observables, syndromes = generate_qem_dataset(
        gptqe         = gptqe,
        n_qubits      = n_qubits,
        hf_state      = hf_state,
        spin_op       = spin_op,
        num_samples   = args.num_samples,
        circuit_depth = args.circuit_depth,
        depol_p       = args.depol_p,
        torch_device  = torch_device,
    )

    # ── Save .npy files ───────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "circuit_tokens.npy",      circuit_tokens)
    np.save(out_dir / "target_observables.npy",  target_observables)
    np.save(out_dir / "syndromes.npy",            syndromes)

    print("\n" + "=" * 64)
    print("  Dataset saved:")
    print(f"    {out_dir}/circuit_tokens.npy      {circuit_tokens.shape}  int64")
    print(f"    {out_dir}/target_observables.npy  {target_observables.shape}  float32")
    print(f"    {out_dir}/syndromes.npy            {syndromes.shape}  float32")
    print(f"\n  DiffusionCompiler config for train_mitigator.py:")
    print(f"    num_qubits      = {n_qubits}")
    print(f"    circuit_depth   = {args.circuit_depth}")
    print(f"    num_observables = {target_observables.shape[1]}  (= n_qubits)")
    print(f"    num_syndromes   = {syndromes.shape[1]}  (= n_qubits - 1)")
    print(f"    gates           = {[describe_token(t, n_qubits) for t in range(1, min(6, vsize+1))]}...")
    print("=" * 64)
