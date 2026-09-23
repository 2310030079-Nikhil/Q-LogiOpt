"""
QLDO Noise Robustness Experiment Engine (Experiment 7)
Simulates realistic NISQ noise models (depolarizing gate noise + readout error)
to compare Standard QAOA vs QLDO-QAOA under noisy quantum simulation.
"""
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from data_processing.loader import LocationNode
from quantum.hamiltonian import IsingHamiltonian
from quantum.circuit import QAOACircuitBuilder
from quantum.simulator import QuantumSimulator
from optimization.route_decoder import RouteDecoder
from qldo.probability import ProbabilityDistribution
from qldo.feasibility import FeasibilityEngine
from qldo.concentration import ConcentrationEngine, GoodRouteDefinition


@dataclass
class NoiseComparisonRecord:
    """Record comparing Ideal vs Noisy execution for an algorithm."""
    algorithm: str
    is_noisy: bool
    depolarizing_p1: float
    depolarizing_p2: float
    readout_error_p: float
    feasible_probability: float
    best_route_probability: float
    best_route_cost: float
    expected_cost: float
    cost_variance: float
    shots: int


class NoiseExperimentRunner:
    """Orchestrates noisy quantum simulation benchmarks."""

    @staticmethod
    def build_nisq_noise_model(
        p1_gate: float = 0.001,
        p2_gate: float = 0.01,
        p_readout: float = 0.02
    ) -> NoiseModel:
        """
        Builds a standard NISQ noise model featuring:
        1. 1-qubit depolarizing gate error (rz, rx, h)
        2. 2-qubit depolarizing gate error (cx, rzz)
        3. Measurement readout bit-flip error
        """
        noise_model = NoiseModel()
        
        # 1-qubit gate error
        err_1 = depolarizing_error(p1_gate, 1)
        noise_model.add_all_qubit_quantum_error(err_1, ['rx', 'rz', 'h', 'sx'])
        
        # 2-qubit gate error
        err_2 = depolarizing_error(p2_gate, 2)
        noise_model.add_all_qubit_quantum_error(err_2, ['cx', 'rzz'])
        
        # Readout error
        ro_matrix = [
            [1.0 - p_readout, p_readout],
            [p_readout, 1.0 - p_readout]
        ]
        ro_err = ReadoutError(ro_matrix)
        noise_model.add_all_qubit_readout_error(ro_err)
        
        return noise_model

    @classmethod
    def evaluate_circuit_under_noise(
        cls,
        hamiltonian: IsingHamiltonian,
        opt_gamma: List[float],
        opt_beta: List[float],
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        p1_gate: float = 0.001,
        p2_gate: float = 0.01,
        p_readout: float = 0.02,
        shots: int = 1024,
        seed: int = 42
    ) -> Tuple[Dict[str, any], Dict[str, any]]:
        """
        Compares the final optimal circuit measured under:
        (1) Ideal simulator
        (2) Noisy simulator (with configured noise model)
        """
        p = len(opt_gamma)
        qc, metrics = QAOACircuitBuilder.build_qaoa_circuit(
            hamiltonian=hamiltonian,
            p=p,
            gamma_params=opt_gamma,
            beta_params=opt_beta,
            include_measurements=True
        )

        # 1. Ideal simulation
        sim_ideal = QuantumSimulator(seed=seed)
        res_ideal = sim_ideal.sample_shots(qc, shots=shots)
        
        # 2. Noisy simulation
        noise_model = cls.build_nisq_noise_model(p1_gate, p2_gate, p_readout)
        sim_noisy = QuantumSimulator(noise_model=noise_model, seed=seed)
        res_noisy = sim_noisy.sample_shots(qc, shots=shots)

        # Decode both
        dec_ideal, sum_ideal = RouteDecoder.evaluate_measurement_counts(
            res_ideal.counts, nodes, distance_matrix, cost_matrix
        )
        dec_noisy, sum_noisy = RouteDecoder.evaluate_measurement_counts(
            res_noisy.counts, nodes, distance_matrix, cost_matrix
        )

        valid_ideal = [s for s in dec_ideal if s.is_valid_permutation]
        best_ideal_dist = valid_ideal[0].strict_solution.total_distance if (valid_ideal and valid_ideal[0].strict_solution) else (dec_ideal[0].repaired_solution.total_distance if dec_ideal else 0.0)
        exp_ideal_cost = sum(s.repaired_solution.total_cost * (res_ideal.counts.get(s.bitstring, 0) / shots) for s in dec_ideal) if shots > 0 else 0.0

        valid_noisy = [s for s in dec_noisy if s.is_valid_permutation]
        best_noisy_dist = valid_noisy[0].strict_solution.total_distance if (valid_noisy and valid_noisy[0].strict_solution) else (dec_noisy[0].repaired_solution.total_distance if dec_noisy else 0.0)
        exp_noisy_cost = sum(s.repaired_solution.total_cost * (res_noisy.counts.get(s.bitstring, 0) / shots) for s in dec_noisy) if shots > 0 else 0.0

        ideal_metrics = {
            "mode": "Ideal Simulator",
            "feasible_prob": sum_ideal["constraint_satisfaction_rate"] / 100.0,
            "best_dist": round(best_ideal_dist, 2),
            "expected_cost": round(exp_ideal_cost, 2),
            "unique_valid_tours": len(valid_ideal)
        }

        noisy_metrics = {
            "mode": "Noisy Simulator (NISQ)",
            "feasible_prob": sum_noisy["constraint_satisfaction_rate"] / 100.0,
            "best_dist": round(best_noisy_dist, 2),
            "expected_cost": round(exp_noisy_cost, 2),
            "unique_valid_tours": len(valid_noisy),
            "p1_gate_error": p1_gate,
            "p2_gate_error": p2_gate,
            "readout_error": p_readout
        }

        return ideal_metrics, noisy_metrics
