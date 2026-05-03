# =============================================================================
# Pauli Tokenizer Pipeline for N-Qubit Quantum Noise Channels
# =============================================================================

import numpy as np
from itertools import product
from typing import List, Dict, Optional
import json
import os
from transformers import PreTrainedTokenizer

# =============================================================================
# CONFIGURATION — change this to run for a different qubit count# =============================================================================
N_QUBITS = 4
COEFF_STEP = 0.05
COEFF_MIN = -1.0
COEFF_MAX = 1.0
SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[K_SEP]", "[COEFF_R]", "[COEFF_I]"]


# =============================================================================
# Part 1: Kraus-to-Pauli Converter & Noise Channels
# =============================================================================

class NQubitKrausToPauliConverter:
    """
    Convert n-qubit Kraus operators to Pauli representations.
    K_k = sum_j c_kj P_j   where   c_kj = (1/d) Tr(P_j @ K_k)
    """

    PAULIS = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    def __init__(self, n_qubits: int, sparse: bool = True, tol: float = 1e-10):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.n_paulis = 4 ** n_qubits
        self.sparse = sparse
        self.tol = tol

        self._pauli_cache: Dict[str, np.ndarray] = {}
        self.pauli_labels = self._generate_pauli_labels()

        if n_qubits <= 3:
            self._precompute_paulis()

    def _generate_pauli_labels(self) -> List[str]:
        return [''.join(p) for p in product('IXYZ', repeat=self.n_qubits)]

    def _precompute_paulis(self):
        for label in self.pauli_labels:
            self._pauli_cache[label] = self._compute_pauli(label)

    def _compute_pauli(self, label: str) -> np.ndarray:
        if label in self._pauli_cache:
            return self._pauli_cache[label]
        P = self.PAULIS[label[0]]
        for char in label[1:]:
            P = np.kron(P, self.PAULIS[char])
        if len(self._pauli_cache) < 1000:
            self._pauli_cache[label] = P
        return P

    def get_pauli(self, label: str) -> np.ndarray:
        return self._compute_pauli(label)

    def decompose_operator(self, K: np.ndarray) -> Dict[str, complex]:
        assert K.shape == (self.dim, self.dim), \
            f"Expected {self.dim}x{self.dim} matrix, got {K.shape}"
        result = {}
        for label in self.pauli_labels:
            P = self.get_pauli(label)
            coeff = np.trace(P @ K) / self.dim
            if not self.sparse or np.abs(coeff) > self.tol:
                result[label] = coeff
        return result

    def decompose_kraus_operators(self, kraus_ops: List[np.ndarray]) -> List[Dict[str, complex]]:
        return [self.decompose_operator(K) for K in kraus_ops]

    def kraus_to_chi_matrix(self, kraus_ops: List[np.ndarray]) -> np.ndarray:
        chi = np.zeros((self.n_paulis, self.n_paulis), dtype=complex)
        for K in kraus_ops:
            coeffs = np.array([
                np.trace(self.get_pauli(label) @ K) / self.dim
                for label in self.pauli_labels
            ])
            chi += np.outer(coeffs, coeffs.conj())
        return chi


class NoiseChannels:
    """Pre-defined n-qubit noise channels as Kraus operators."""

    @staticmethod
    def amplitude_damping(n_qubits: int, gamma: float) -> List[np.ndarray]:
        """
        n-qubit local amplitude damping.
        Returns 2^n Kraus operators, each of size (2^n x 2^n).
        """
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        single = [K0, K1]

        kraus_ops = []
        for indices in product(range(2), repeat=n_qubits):
            K = single[indices[0]]
            for idx in indices[1:]:
                K = np.kron(K, single[idx])
            kraus_ops.append(K)
        return kraus_ops


# =============================================================================
# Part 2: Vocabulary Builder
# =============================================================================

def generate_pauli_labels(n_qubits: int = N_QUBITS) -> List[str]:
    return sorted([''.join(p) for p in product('IXYZ', repeat=n_qubits)])


def generate_coeff_tokens(step: float = COEFF_STEP) -> List[str]:
    values = np.arange(COEFF_MIN, COEFF_MAX + step / 2, step)
    values = np.round(values, decimals=2)
    return [f"C:{v:+.2f}" for v in values]


def build_vocab(n_qubits: int = N_QUBITS, coeff_step: float = COEFF_STEP) -> Dict[str, int]:
    vocab = {}
    idx = 0
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1
    for label in generate_pauli_labels(n_qubits):
        vocab[label] = idx
        idx += 1
    for token in generate_coeff_tokens(coeff_step):
        vocab[token] = idx
        idx += 1
    return vocab


def save_vocab(vocab: Dict[str, int], path: str):
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# Part 3: Core Pauli Noise Tokenizer
# =============================================================================

class PauliNoiseTokenizer:
    """
    Tokenizes n-qubit quantum noise channels (Kraus operators)
    into discrete token sequences based on Pauli decomposition.
    """

    def __init__(self, vocab: Dict[str, int], n_qubits: int = N_QUBITS,
                 coeff_step: float = 0.05, skip_zero_imag: bool = True, tol: float = 1e-6):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.coeff_step = coeff_step
        self.skip_zero_imag = skip_zero_imag
        self.tol = tol
        self._converter = NQubitKrausToPauliConverter(n_qubits=n_qubits, sparse=True, tol=tol)

    def _discretize(self, value: float) -> str:
        rounded = round(value / self.coeff_step) * self.coeff_step
        clamped = max(-1.0, min(1.0, rounded))
        clamped = round(clamped, 2)
        return f"C:{clamped:+.2f}"

    def encode_to_strings(self, kraus_ops: List[np.ndarray]) -> List[str]:
        decompositions = self._converter.decompose_kraus_operators(kraus_ops)
        tokens = ["[BOS]"]

        for k_idx, decomp in enumerate(decompositions):
            if k_idx > 0:
                tokens.append("[K_SEP]")

            sorted_terms = sorted(decomp.items(), key=lambda x: -abs(x[1]))

            for pauli_label, coeff in sorted_terms:
                real_token = self._discretize(coeff.real)
                imag_token = self._discretize(coeff.imag)

                if real_token == "C:+0.00" and imag_token == "C:+0.00":
                    continue

                tokens.append(pauli_label)
                tokens.append("[COEFF_R]")
                tokens.append(real_token)

                if not (self.skip_zero_imag and imag_token == "C:+0.00"):
                    tokens.append("[COEFF_I]")
                    tokens.append(imag_token)

        tokens.append("[EOS]")
        return tokens

    def encode(self, kraus_ops: List[np.ndarray]) -> List[int]:
        token_strings = self.encode_to_strings(kraus_ops)
        return [self.vocab.get(t, self.vocab["[UNK]"]) for t in token_strings]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.id_to_token.get(i, "[UNK]") for i in token_ids])

    def decode_to_list(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "[UNK]") for i in token_ids]

    @property
    def vocab_size(self): return len(self.vocab)

    @property
    def pad_token_id(self): return self.vocab["[PAD]"]

    @property
    def bos_token_id(self): return self.vocab["[BOS]"]

    @property
    def eos_token_id(self): return self.vocab["[EOS]"]


# =============================================================================
# Part 4: HuggingFace-Compatible Tokenizer Wrapper
# =============================================================================

class PauliNoiseHFTokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible tokenizer for Pauli noise channel sequences."""

    vocab_files_names = {"vocab_file": "pauli_vocab.json"}

    def __init__(self, vocab_file: Optional[str] = None,
                 vocab_dict: Optional[Dict] = None, **kwargs):
        if vocab_dict is not None:
            self._vocab = vocab_dict
        elif vocab_file is not None:
            with open(vocab_file, "r") as f:
                self._vocab = json.load(f)
        else:
            raise ValueError("Provide either vocab_file or vocab_dict")

        self._id_to_token = {v: k for k, v in self._vocab.items()}

        super().__init__(
            pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]", **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get("[UNK]", 3))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "[UNK]")

    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        fname = f"{filename_prefix}-pauli_vocab.json" if filename_prefix else "pauli_vocab.json"
        filepath = os.path.join(save_directory, fname)
        with open(filepath, "w") as f:
            json.dump(self._vocab, f, indent=2)
        return (filepath,)


# =============================================================================
# Part 5: End-to-End Pipeline & Dataset Generation
# =============================================================================

class PauliTokenizationPipeline:
    """End-to-end pipeline: noise channel parameters -> training-ready token data."""

    def __init__(self, vocab: Dict[str, int], n_qubits: int = N_QUBITS,
                 coeff_step: float = 0.05, skip_zero_imag: bool = True):
        self.n_qubits = n_qubits
        self.tokenizer = PauliNoiseTokenizer(
            vocab, n_qubits=n_qubits,
            coeff_step=coeff_step, skip_zero_imag=skip_zero_imag
        )
        self.vocab = vocab

    def tokenize_amplitude_damping(self, gamma: float) -> Dict:
        kraus_ops = NoiseChannels.amplitude_damping(self.n_qubits, gamma)
        token_strings = self.tokenizer.encode_to_strings(kraus_ops)
        token_ids = self.tokenizer.encode(kraus_ops)
        return {
            "gamma": gamma,
            "token_strings": token_strings,
            "token_ids": token_ids,
            "sequence_length": len(token_ids),
        }

    def tokenize_kraus_operators(self, kraus_ops: List[np.ndarray]) -> Dict:
        token_strings = self.tokenizer.encode_to_strings(kraus_ops)
        token_ids = self.tokenizer.encode(kraus_ops)
        return {
            "token_strings": token_strings,
            "token_ids": token_ids,
            "sequence_length": len(token_ids),
        }

    def generate_dataset(self, gamma_values, output_path: str) -> int:
        count = 0
        with open(output_path, "w") as f:
            for gamma in gamma_values:
                result = self.tokenize_amplitude_damping(gamma)
                entry = {
                    "gamma": float(round(gamma, 4)),
                    "token_ids": result["token_ids"],
                    "text": " ".join(result["token_strings"]),
                    "sequence_length": result["sequence_length"],
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
        print(f"Dataset generated: {count} samples -> {output_path}")
        return count

    def verify_reconstruction(self, kraus_ops: List[np.ndarray]) -> Dict:
        decompositions = self.tokenizer._converter.decompose_kraus_operators(kraus_ops)
        all_errors = []
        for decomp in decompositions:
            for _, coeff in decomp.items():
                for val in (coeff.real, coeff.imag):
                    disc = round(val / self.tokenizer.coeff_step) * self.tokenizer.coeff_step
                    disc = max(-1.0, min(1.0, disc))
                    all_errors.append(abs(val - disc))
        all_errors = np.array(all_errors)
        return {
            "max_error": float(all_errors.max()),
            "mean_error": float(all_errors.mean()),
            "n_coefficients": len(all_errors),
            "max_possible_error": self.tokenizer.coeff_step / 2,
        }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Running Pauli tokenizer pipeline for N_QUBITS = {N_QUBITS}")

    # Build vocab
    vocab = build_vocab(n_qubits=N_QUBITS)
    vocab_path = os.path.join(os.getcwd(), f"{N_QUBITS}q_pauli_vocab.json")
    save_vocab(vocab, vocab_path)
    print(f"Vocab size: {len(vocab)} | Saved to: {vocab_path}")

    # Tokenizer smoke test
    tokenizer = PauliNoiseTokenizer(vocab, n_qubits=N_QUBITS)
    kraus_ops = NoiseChannels.amplitude_damping(N_QUBITS, gamma=0.1)
    token_strings = tokenizer.encode_to_strings(kraus_ops)
    token_ids = tokenizer.encode(kraus_ops)
    print(f"Sequence length (gamma=0.1): {len(token_ids)} tokens")
    print(f"First 20 tokens: {' '.join(token_strings[:20])}")

    # HF tokenizer test
    hf_tokenizer = PauliNoiseHFTokenizer(vocab_dict=vocab)
    hf_ids = hf_tokenizer(" ".join(token_strings))["input_ids"]
    assert hf_ids == token_ids, "HF tokenizer mismatch!"
    print("HF tokenizer: OK")

    # Generate dataset
    pipeline = PauliTokenizationPipeline(vocab, n_qubits=N_QUBITS)
    dataset_path = os.path.join(os.getcwd(), f"{N_QUBITS}q_ad_noise_dataset.jsonl")
    pipeline.generate_dataset(np.arange(0.01, 0.50, 0.01), dataset_path)

    # Verification
    print("\n--- Verification ---")
    kraus_ops = NoiseChannels.amplitude_damping(N_QUBITS, gamma=0.1)
    v = pipeline.verify_reconstruction(kraus_ops)
    print(f"Max error: {v['max_error']:.6f} (limit: {v['max_possible_error']:.6f})")
    assert v['max_error'] <= v['max_possible_error'] + 1e-10
    print("All checks passed.")

