import cudaq
from cudaq import spin
import numpy as np
from itertools import product


@cudaq.kernel
def ghz_circuit():
    qubits = cudaq.qvector(4)
    h(qubits[0])
    cx(qubits[0], qubits[1])
    cx(qubits[0], qubits[2])
    cx(qubits[0], qubits[3])


def build_spin_operator(pauli_combo):
    """Build a CUDA-Q spin operator from a Pauli combo tuple like ('X','Y','Z','Z')."""
    spin_map = {'I': spin.i, 'X': spin.x, 'Y': spin.y, 'Z': spin.z}
    op = None
    for i, p in enumerate(pauli_combo):
        term = spin_map[p](i)
        op = term if op is None else op * term
    return op


def run_tomography(shots=4096):
    """
    Use cudaq.observe() to get expectation values for all 4^4 Pauli strings.
    Returns dict mapping pauli_string -> expectation_value.
    """
    pauli_labels = ['I', 'X', 'Y', 'Z']
    results = {}

    for pauli_combo in product(pauli_labels, repeat=4):
        pauli_str = ''.join(pauli_combo)
        op = build_spin_operator(pauli_combo)
        obs_result = cudaq.observe(ghz_circuit, op, shots_count=shots)
        results[pauli_str] = obs_result.expectation()

    return results


def reconstruct_density_matrix(tomo_results, num_qubits=4):
    """Reconstruct density matrix from Pauli expectation values via linear inversion."""
    dim = 2 ** num_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    paulis = {
        'I': np.array([[1, 0], [0, 1]]),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]])
    }

    for pauli_combo in product('IXYZ', repeat=num_qubits):
        pauli_str = ''.join(pauli_combo)

        pauli_op = paulis[pauli_combo[0]]
        for p in pauli_combo[1:]:
            pauli_op = np.kron(pauli_op, paulis[p])

        rho += tomo_results[pauli_str] * pauli_op

    rho /= dim
    return rho


def compute_fidelity(rho, num_qubits=4):
    """Compute fidelity between reconstructed rho and the ideal GHZ state."""
    dim = 2 ** num_qubits
    ghz_state = np.zeros(dim, dtype=complex)
    ghz_state[0] = 1 / np.sqrt(2)
    ghz_state[-1] = 1 / np.sqrt(2)
    rho_ideal = np.outer(ghz_state, ghz_state.conj())
    return np.real(np.trace(rho_ideal @ rho))


if __name__ == '__main__':
    print("Running quantum state tomography on 4-qubit GHZ state with CUDA-Q...")
    tomo_results = run_tomography(shots=4096)
    rho = reconstruct_density_matrix(tomo_results)
    print(f"Trace of rho:          {np.real(np.trace(rho)):.4f}")
    print(f"Fidelity with ideal GHZ: {compute_fidelity(rho):.4f}")
