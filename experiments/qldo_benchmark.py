"""
QLDO Benchmark Runner: Controlled Head-to-Head Comparison
Standard QAOA (Expectation) vs. QLDO-QAOA (Distribution-Aware)
"""
from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from data_processing.loader import LocationNode
from data_processing.distance_matrix import DistanceMatrixCalculator
from classical import NearestNeighborSolver, TwoOptSolver, SimulatedAnnealingSolver, ExactBruteForceSolver
from quantum.qubo import QuboBuilder
from quantum.hamiltonian import HamiltonianConverter
from quantum.qaoa_optimizer import QAOAOptimizer, QAOAExecutionResult
from quantum.qaoa_qldo import QLDOQAOAOptimizer, QLDOExecutionResult
from optimization.route_decoder import RouteDecoder
from qldo.concentration import GoodRouteDefinition


@dataclass
class HeadToHeadResult:
    """Detailed side-by-side comparison of Standard QAOA vs QLDO-QAOA."""
    instance_name: str
    num_locations: int
    qubits: int
    p: int
    shots: int
    ground_truth_cost: Optional[float]
    classical_2opt_cost: float
    
    # Standard QAOA
    std_best_cost: float
    std_expected_cost: float
    std_feasible_prob: float
    std_good_concentration: float
    std_best_prob: float
    std_approx_ratio: Optional[float]
    std_runtime_ms: float
    std_iterations: int
    
    # QLDO-QAOA
    qldo_best_cost: float
    qldo_expected_cost: float
    qldo_feasible_prob: float
    qldo_good_concentration: float
    qldo_best_prob: float
    qldo_approx_ratio: Optional[float]
    qldo_runtime_ms: float
    qldo_iterations: int
    
    # Delta (QLDO - Standard)
    delta_feasible_prob: float
    delta_best_prob: float
    delta_best_cost: float

    def to_dict(self) -> Dict[str, any]:
        return {
            "Instance": self.instance_name,
            "Locations": self.num_locations,
            "Qubits": self.qubits,
            "p": self.p,
            "Ground Truth (C*)": round(self.ground_truth_cost, 2) if self.ground_truth_cost else "N/A",
            "Classical 2-Opt": round(self.classical_2opt_cost, 2),
            "Std QAOA Best": round(self.std_best_cost, 2),
            "QLDO Best": round(self.qldo_best_cost, 2),
            "Std Feasible (P_F)": f"{self.std_feasible_prob * 100:.1f}%",
            "QLDO Feasible (P_F)": f"{self.qldo_feasible_prob * 100:.1f}%",
            "Delta P_F": f"{self.delta_feasible_prob * 100:+.1f}%",
            "Std Best Prob P(r*)": f"{self.std_best_prob * 100:.2f}%",
            "QLDO Best Prob P(r*)": f"{self.qldo_best_prob * 100:.2f}%",
            "Std Approx Ratio": round(self.std_approx_ratio, 3) if self.std_approx_ratio else "N/A",
            "QLDO Approx Ratio": round(self.qldo_approx_ratio, 3) if self.qldo_approx_ratio else "N/A",
            "Std Runtime (ms)": round(self.std_runtime_ms, 1),
            "QLDO Runtime (ms)": round(self.qldo_runtime_ms, 1)
        }


class QLDOBenchmarkRunner:
    """Executes controlled head-to-head experiments."""

    @classmethod
    def run_head_to_head_comparison(
        cls,
        nodes: List[LocationNode],
        cost_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        p: int = 1,
        optimizer_type: str = "COBYLA",
        max_iterations: int = 30,
        shots: int = 1024,
        penalty_lambda: float = 100.0,
        alpha: float = 0.05,
        beta: float = 5.0,
        gamma: float = 5.0,
        epsilon: float = 1e-4,
        good_route_def: GoodRouteDefinition = GoodRouteDefinition.RELATIVE_BEST,
        seed: int = 42,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[HeadToHeadResult, QAOAExecutionResult, QLDOExecutionResult]:
        """
        Runs Experiment 1: Under identical conditions (nodes, QUBO, p, shots, optimizer, iterations, seed),
        compares Standard QAOA minimizing E[C] against QLDO-QAOA.
        """
        n = len(nodes)
        qubits = (n - 1) ** 2

        # 1. Classical ground truth and heuristic
        c_star = None
        if n <= 8:
            exact_res = ExactBruteForceSolver.solve(nodes, distance_matrix, cost_matrix)
            c_star = exact_res.total_cost

        two_opt_res = TwoOptSolver.solve(nodes, distance_matrix, cost_matrix)
        c_2opt = two_opt_res.total_cost

        # 2. Build QUBO and Ising Hamiltonian
        qubo = QuboBuilder.build_qubo(nodes, cost_matrix, penalty_lambda=penalty_lambda)
        ising = HamiltonianConverter.qubo_to_ising(qubo)

        # Standard initial parameters
        gammas_init = [0.1 * (l + 1) / p for l in range(p)]
        betas_init = [0.5 * (1.0 - (l / p)) for l in range(p)]
        init_theta = np.array(gammas_init + betas_init, dtype=float)

        # 3. Run Standard QAOA (Minimizing E[C])
        if progress_callback:
            progress_callback(1, 3, "Running Standard QAOA (Expectation Minimization)...")

        t0_std = time.perf_counter()
        std_res = QAOAOptimizer.optimize(
            hamiltonian=ising,
            p=p,
            optimizer_type=optimizer_type,
            max_iterations=max_iterations,
            shots=shots,
            initial_params=init_theta,
            seed=seed
        )
        t_std_ms = (time.perf_counter() - t0_std) * 1000.0

        std_counts = std_res.simulation_result.counts
        std_decoded, std_summary = RouteDecoder.evaluate_measurement_counts(
            counts=std_counts,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix
        )
        std_valid = [s for s in std_decoded if s.is_valid_permutation]
        std_best_cost = std_valid[0].strict_solution.total_cost if std_valid else std_decoded[0].repaired_solution.total_cost
        std_pf = std_summary["constraint_satisfaction_rate"] / 100.0
        best_bs = std_valid[0].bitstring if std_valid else (std_decoded[0].bitstring if std_decoded else "")
        std_best_prob = (std_counts.get(best_bs, 0) / shots) if shots > 0 else 0.0
        std_exp_c = std_summary.get("expected_cost", std_best_cost)
        std_approx = (c_star / std_best_cost) if c_star and std_best_cost > 0 else 1.0

        # 4. Run QLDO-QAOA (Distribution-Aware)
        if progress_callback:
            progress_callback(2, 3, "Running QLDO-QAOA (Distribution-Aware Minimization)...")

        t0_qldo = time.perf_counter()
        qldo_res = QLDOQAOAOptimizer.optimize(
            nodes=nodes,
            cost_matrix=cost_matrix,
            distance_matrix=distance_matrix,
            hamiltonian=ising,
            p=p,
            optimizer_type=optimizer_type,
            max_iterations=max_iterations,
            shots=shots,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            good_route_def=good_route_def,
            initial_params=init_theta,
            seed=seed,
            optimal_cost=c_star
        )
        t_qldo_ms = (time.perf_counter() - t0_qldo) * 1000.0

        qldo_best_cost = qldo_res.final_metrics.best_route_cost
        qldo_pf = qldo_res.final_metrics.feasible_probability
        qldo_best_prob = qldo_res.final_metrics.best_route_probability
        qldo_exp_c = qldo_res.final_metrics.expected_cost
        qldo_g = qldo_res.final_metrics.good_route_concentration
        qldo_approx = qldo_res.final_metrics.approximation_ratio

        if progress_callback:
            progress_callback(3, 3, "Head-to-head comparison complete!")

        record = HeadToHeadResult(
            instance_name=f"{n}-Node Instance (p={p})",
            num_locations=n,
            qubits=qubits,
            p=p,
            shots=shots,
            ground_truth_cost=c_star,
            classical_2opt_cost=c_2opt,
            std_best_cost=std_best_cost,
            std_expected_cost=std_exp_c,
            std_feasible_prob=std_pf,
            std_good_concentration=0.0,
            std_best_prob=std_best_prob,
            std_approx_ratio=std_approx,
            std_runtime_ms=t_std_ms,
            std_iterations=std_res.iteration_count,
            qldo_best_cost=qldo_best_cost,
            qldo_expected_cost=qldo_exp_c,
            qldo_feasible_prob=qldo_pf,
            qldo_good_concentration=qldo_g,
            qldo_best_prob=qldo_best_prob,
            qldo_approx_ratio=qldo_approx,
            qldo_runtime_ms=t_qldo_ms,
            qldo_iterations=qldo_res.final_metrics.total_iterations,
            delta_feasible_prob=qldo_pf - std_pf,
            delta_best_prob=qldo_best_prob - std_best_prob,
            delta_best_cost=qldo_best_cost - std_best_cost
        )

        return record, std_res, qldo_res
