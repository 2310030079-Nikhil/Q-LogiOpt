"""
Quantum Simulator Interface (Aer Statevector & Shot-Based Sampler)
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator


@dataclass
class SimulationResult:
    """Encapsulates the output of a quantum simulator run."""
    counts: Dict[str, int]
    probabilities: Dict[str, float]
    shots: int
    simulator_name: str
    execution_time_ms: float


class QuantumSimulator:
    """Manages local statevector and shot-based simulation using Qiskit Aer."""
    
    def __init__(self, seed: Optional[int] = 42, noise_model: Optional[object] = None):
        self.seed = seed
        self.noise_model = noise_model
        if noise_model is not None:
            self.aer_sim = AerSimulator(seed_simulator=seed, noise_model=noise_model)
        else:
            self.aer_sim = AerSimulator(seed_simulator=seed)
        
    def evaluate_expectation_statevector(
        self,
        circuit: QuantumCircuit,
        cost_operator: SparsePauliOp,
        offset: float = 0.0
    ) -> float:
        """
        Computes exact expectation value: <psi|H_C|psi> + offset
        using pure Statevector algebra (zero shot noise, optimal for classical optimizer loops).
        """
        # Remove any measurements before computing statevector
        qc_pure = circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_pure)
        exp_val = sv.expectation_value(cost_operator).real
        return float(exp_val + offset)
        
    def sample_shots(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024
    ) -> SimulationResult:
        """
        Executes shot-based measurement simulation using AerSimulator.
        Automatically ensures measurement operators are present on all qubits.
        """
        import time
        start_time = time.perf_counter()
        
        # Ensure measurements are present
        if circuit.num_clbits == 0:
            qc_measure = circuit.copy()
            qc_measure.measure_all()
        else:
            qc_measure = circuit
            
        compiled_circuit = transpile(qc_measure, self.aer_sim)
        job = self.aer_sim.run(compiled_circuit, shots=shots)
        result = job.result()
        raw_counts = result.get_counts()
        
        # Clean counts dictionary (remove spaces if present)
        counts = {k.replace(" ", ""): v for k, v in raw_counts.items()}
        total_shots = sum(counts.values())
        probabilities = {k: v / total_shots for k, v in counts.items()}
        
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        return SimulationResult(
            counts=counts,
            probabilities=probabilities,
            shots=total_shots,
            simulator_name="Qiskit AerSimulator",
            execution_time_ms=round(runtime_ms, 3)
        )
