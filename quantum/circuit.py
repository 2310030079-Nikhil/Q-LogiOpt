"""
QAOA Parameterized Quantum Circuit Builder
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from quantum.hamiltonian import IsingHamiltonian


@dataclass
class CircuitMetrics:
    """Metrics quantifying the quantum resource requirements of the circuit."""
    num_qubits: int
    circuit_depth: int
    total_gates: int
    two_qubit_gates: int
    gate_counts: Dict[str, int]
    qaoa_depth_p: int


class QAOACircuitBuilder:
    """Constructs QAOA variational circuits from Cost and Mixer Hamiltonians."""
    
    @classmethod
    def build_qaoa_circuit(
        cls,
        hamiltonian: IsingHamiltonian,
        p: int = 1,
        gamma_params: Optional[Union[List[float], np.ndarray]] = None,
        beta_params: Optional[Union[List[float], np.ndarray]] = None,
        include_measurements: bool = False
    ) -> Tuple[QuantumCircuit, CircuitMetrics]:
        """
        Builds a parameter-bound or symbolic QAOA QuantumCircuit.
        
        If gamma_params and beta_params are None, creates symbolic parameters:
        gamma_0, ..., gamma_{p-1} and beta_0, ..., beta_{p-1}.
        """
        N = hamiltonian.num_qubits
        qr = QuantumRegister(N, name="q")
        cr = ClassicalRegister(N, name="c") if include_measurements else None
        
        qc = QuantumCircuit(qr, cr) if include_measurements else QuantumCircuit(qr)
        
        # 1. Prepare initial uniform superposition |+>^N
        qc.h(qr)
        
        # Determine whether to use bound numbers or symbolic parameters
        is_bound = (gamma_params is not None and beta_params is not None)
        if not is_bound:
            gammas = ParameterVector("gamma", p)
            betas = ParameterVector("beta", p)
        else:
            gammas = list(gamma_params)
            betas = list(beta_params)
            
        # 2. Apply p alternating layers of Cost and Mixer unitaries
        for l in range(p):
            gamma_l = gammas[l]
            beta_l = betas[l]
            
            # --- Cost Layer e^{-i gamma_l H_C} ---
            # Linear terms h_i * Z_i -> RZ(2 * gamma_l * h_i)
            for i, h_val in hamiltonian.linear_coeffs.items():
                if abs(h_val) > 1e-9:
                    angle = 2.0 * gamma_l * h_val
                    qc.rz(angle, qr[i])
                    
            # Quadratic terms J_ij * Z_i * Z_j -> RZZ(2 * gamma_l * J_ij)
            for (i, j), j_val in hamiltonian.quadratic_coeffs.items():
                if abs(j_val) > 1e-9:
                    angle = 2.0 * gamma_l * j_val
                    qc.rzz(angle, qr[i], qr[j])
                    
            # --- Mixer Layer e^{-i beta_l H_M} ---
            # H_M = sum_i X_i -> RX(2 * beta_l) on all qubits
            for i in range(N):
                angle = 2.0 * beta_l
                qc.rx(angle, qr[i])
                
        # 3. Add measurements if requested
        if include_measurements and cr is not None:
            qc.barrier()
            qc.measure(qr, cr)
            
        # 4. Compute circuit metrics
        gate_dict = dict(qc.count_ops())
        # Two-qubit gates are typically rzz, cx, cz
        two_q_count = gate_dict.get("rzz", 0) + gate_dict.get("cx", 0) + gate_dict.get("cz", 0)
        
        metrics = CircuitMetrics(
            num_qubits=N,
            circuit_depth=qc.depth(),
            total_gates=sum(gate_dict.values()),
            two_qubit_gates=two_q_count,
            gate_counts=gate_dict,
            qaoa_depth_p=p
        )
        
        return qc, metrics

    @staticmethod
    def draw_ascii_circuit(circuit: QuantumCircuit) -> str:
        """Returns clean ASCII text representation of the circuit."""
        try:
            return circuit.draw(output="text", fold=80)
        except Exception as e:
            return f"Circuit text drawing unavailable: {str(e)}"
