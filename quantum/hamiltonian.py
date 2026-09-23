"""
Ising Hamiltonian (Cost & Mixer) Formulation for QAOA
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from quantum.qubo import QuboProblem


@dataclass
class IsingHamiltonian:
    """Encapsulates the Cost Hamiltonian, Mixer Hamiltonian, and energy offset."""
    cost_operator: SparsePauliOp
    mixer_operator: SparsePauliOp
    offset: float
    num_qubits: int
    linear_coeffs: Dict[int, float]
    quadratic_coeffs: Dict[Tuple[int, int], float]


class HamiltonianConverter:
    """Converts a QUBO matrix into Cost and Mixer Hamiltonians using SparsePauliOp."""
    
    @classmethod
    def qubo_to_ising(cls, qubo: QuboProblem) -> IsingHamiltonian:
        """
        Transforms binary quadratic program:
            min x^T Q x + c
        into Ising spin glass model:
            H_C = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + offset * I
        using transformation: x_i = (I - Z_i) / 2.
        
        Returns SparsePauliOp cost and transverse mixer operators.
        """
        Q = qubo.qubo_matrix
        N = qubo.num_qubits
        c = qubo.constant_offset
        
        h: Dict[int, float] = {i: 0.0 for i in range(N)}
        J: Dict[Tuple[int, int], float] = {}
        ising_offset = c
        
        # 1. Diagonal terms Q[i, i] * x_i
        for i in range(N):
            q_ii = Q[i, i]
            if abs(q_ii) > 1e-9:
                ising_offset += 0.5 * q_ii
                h[i] -= 0.5 * q_ii
                
        # 2. Off-diagonal terms Q[i, j] * x_i * x_j (i < j)
        for i in range(N):
            for j in range(i + 1, N):
                q_ij = Q[i, j]
                if abs(q_ij) > 1e-9:
                    ising_offset += 0.25 * q_ij
                    h[i] -= 0.25 * q_ij
                    h[j] -= 0.25 * q_ij
                    J[(i, j)] = 0.25 * q_ij
                    
        # 3. Build SparsePauliOp for Cost Hamiltonian
        pauli_list: List[Tuple[str, float]] = []
        
        # Linear Pauli Z terms: Z on qubit i, I everywhere else
        # Note: in Qiskit, index 0 is rightmost in Pauli string (little endian)
        for i in range(N):
            if abs(h[i]) > 1e-9:
                # String of length N with 'Z' at position N - 1 - i
                pauli_str = list("I" * N)
                pauli_str[N - 1 - i] = "Z"
                pauli_list.append(("".join(pauli_str), round(h[i], 6)))
                
        # Quadratic Pauli ZZ terms: Z on qubit i and qubit j
        for (i, j), j_val in J.items():
            if abs(j_val) > 1e-9:
                pauli_str = list("I" * N)
                pauli_str[N - 1 - i] = "Z"
                pauli_str[N - 1 - j] = "Z"
                pauli_list.append(("".join(pauli_str), round(j_val, 6)))
                
        if not pauli_list:
            # Trivial identity
            cost_op = SparsePauliOp.from_list([("I" * N, 0.0)])
        else:
            cost_op = SparsePauliOp.from_list(pauli_list).simplify()
            
        # 4. Build Transverse-Field Mixer Hamiltonian H_M = sum_i X_i
        mixer_list: List[Tuple[str, float]] = []
        for i in range(N):
            pauli_str = list("I" * N)
            pauli_str[N - 1 - i] = "X"
            mixer_list.append(("".join(pauli_str), 1.0))
        mixer_op = SparsePauliOp.from_list(mixer_list).simplify()
        
        return IsingHamiltonian(
            cost_operator=cost_op,
            mixer_operator=mixer_op,
            offset=round(ising_offset, 6),
            num_qubits=N,
            linear_coeffs={k: round(v, 6) for k, v in h.items() if abs(v) > 1e-9},
            quadratic_coeffs={k: round(v, 6) for k, v in J.items() if abs(v) > 1e-9}
        )
