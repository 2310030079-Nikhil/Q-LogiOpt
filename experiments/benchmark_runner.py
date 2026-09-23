"""
Batch Benchmark and Experiment Execution Engine
"""
from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from classical import NearestNeighborSolver, TwoOptSolver, SimulatedAnnealingSolver, ExactBruteForceSolver
from quantum import QuboBuilder, HamiltonianConverter, QAOAOptimizer
from optimization import RouteDecoder


@dataclass
class BenchmarkRecord:
    """Record of a single benchmark instance comparison."""
    instance_name: str
    num_locations: int
    num_qubits: int
    qaoa_depth_p: int
    shots: int
    classical_nn_distance: float
    classical_2opt_distance: float
    classical_sa_distance: float
    exact_optimal_distance: float
    qaoa_strict_distance: Optional[float]
    qaoa_repaired_distance: float
    qaoa_energy: float
    qaoa_csr_percent: float
    qaoa_approx_ratio: float
    classical_2opt_approx_ratio: float
    classical_runtime_ms: float
    qaoa_runtime_ms: float
    circuit_depth: int
    two_qubit_gates: int


class BenchmarkRunner:
    """Orchestrates multi-parameter experiments and benchmark suites."""
    
    @classmethod
    def run_single_instance_benchmark(
        cls,
        nodes,
        qaoa_p: int = 1,
        optimizer: str = "COBYLA",
        shots: int = 1024,
        max_iterations: int = 40,
        penalty_multiplier: float = 2.0
    ) -> BenchmarkRecord:
        """Runs all classical baselines and QAOA on a given instance."""
        n = len(nodes)
        dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
        
        # 1. Classical Solvers
        nn_sol = NearestNeighborSolver.solve(nodes, dmat)
        two_opt_sol = TwoOptSolver.solve(nodes, dmat)
        sa_sol = SimulatedAnnealingSolver.solve(nodes, dmat)
        
        # Exact solver if n <= 8
        if n <= 8:
            exact_sol = ExactBruteForceSolver.solve(nodes, dmat)
            opt_dist = exact_sol.total_distance
        else:
            opt_dist = min(nn_sol.total_distance, two_opt_sol.total_distance, sa_sol.total_distance)
            
        # 2. Quantum QUBO & QAOA
        qubo = QuboBuilder.build_qubo(nodes, dmat, penalty_multiplier=penalty_multiplier)
        ising = HamiltonianConverter.qubo_to_ising(qubo)
        
        qaoa_res = QAOAOptimizer.optimize(
            hamiltonian=ising,
            p=qaoa_p,
            optimizer_type=optimizer,
            max_iterations=max_iterations,
            shots=shots
        )
        
        decoded, summary = RouteDecoder.evaluate_measurement_counts(
            counts=qaoa_res.simulation_result.counts,
            nodes=nodes,
            distance_matrix=dmat
        )
        
        # Find best strict feasible solution
        valid_states = [s for s in decoded if s.is_valid_permutation]
        strict_dist = valid_states[0].strict_solution.total_distance if valid_states else None
        repaired_dist = decoded[0].repaired_solution.total_distance
        
        # Approximation ratios
        ref_qaoa_dist = strict_dist if strict_dist is not None else repaired_dist
        qaoa_approx = round(ref_qaoa_dist / opt_dist, 3) if opt_dist > 0 else 1.0
        two_opt_approx = round(two_opt_sol.total_distance / opt_dist, 3) if opt_dist > 0 else 1.0
        
        classical_total_ms = nn_sol.runtime_ms + two_opt_sol.runtime_ms + sa_sol.runtime_ms
        
        return BenchmarkRecord(
            instance_name=f"Instance_{n}loc",
            num_locations=n,
            num_qubits=qubo.num_qubits,
            qaoa_depth_p=qaoa_p,
            shots=shots,
            classical_nn_distance=nn_sol.total_distance,
            classical_2opt_distance=two_opt_sol.total_distance,
            classical_sa_distance=sa_sol.total_distance,
            exact_optimal_distance=opt_dist,
            qaoa_strict_distance=strict_dist,
            qaoa_repaired_distance=repaired_dist,
            qaoa_energy=qaoa_res.final_energy,
            qaoa_csr_percent=summary["constraint_satisfaction_rate"],
            qaoa_approx_ratio=qaoa_approx,
            classical_2opt_approx_ratio=two_opt_approx,
            classical_runtime_ms=round(classical_total_ms, 2),
            qaoa_runtime_ms=round(qaoa_res.total_runtime_ms, 2),
            circuit_depth=qaoa_res.circuit_metrics.circuit_depth,
            two_qubit_gates=qaoa_res.circuit_metrics.two_qubit_gates
        )

    @classmethod
    def run_scaling_experiment(
        cls,
        location_counts: List[int] = [3, 4, 5],
        qaoa_p: int = 1,
        shots: int = 1024,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> pd.DataFrame:
        """Runs Experiment 1: Scaling analysis across different node counts."""
        records = []
        total = len(location_counts)
        for idx, count in enumerate(location_counts):
            if progress_callback:
                progress_callback(idx, total, f"Running instance with {count} locations...")
            nodes = SyntheticDataGenerator.generate(num_customers=count - 1, seed=42 + count)
            rec = cls.run_single_instance_benchmark(nodes, qaoa_p=qaoa_p, shots=shots, max_iterations=30)
            records.append(rec.__dict__)
            
        if progress_callback:
            progress_callback(total, total, "Scaling experiment complete.")
        return pd.DataFrame(records)

    @classmethod
    def run_depth_experiment(
        cls,
        nodes,
        depths: List[int] = [1, 2, 3],
        shots: int = 1024,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> pd.DataFrame:
        """Runs Experiment 2: QAOA depth p impact analysis."""
        records = []
        total = len(depths)
        for idx, p_val in enumerate(depths):
            if progress_callback:
                progress_callback(idx, total, f"Evaluating QAOA depth p={p_val}...")
            rec = cls.run_single_instance_benchmark(nodes, qaoa_p=p_val, shots=shots, max_iterations=30)
            records.append(rec.__dict__)
            
        if progress_callback:
            progress_callback(total, total, "Depth experiment complete.")
        return pd.DataFrame(records)
