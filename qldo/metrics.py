"""
QLDO: Comprehensive Research Evaluation Metrics Engine.
Computes standard and distribution-aware benchmarks, approximation ratios,
and statistical telemetry for academic comparison.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import numpy as np


@dataclass
class QLDOResearchMetrics:
    """Consolidated metrics for benchmarking Standard QAOA vs QLDO-QAOA."""
    best_route_cost: float
    expected_cost: float
    cost_variance: float
    cost_std: float
    feasible_probability: float
    constraint_violation_rate: float
    good_route_concentration: float
    best_route_probability: float
    best_bitstring: str
    optimal_ground_truth_cost: Optional[float]
    approximation_ratio: Optional[float]
    total_runtime_ms: float
    total_iterations: int
    total_shots: int
    qubits: int
    circuit_depth: int

    def to_dict(self) -> Dict[str, any]:
        return {
            "Best Route Cost (C_best)": round(self.best_route_cost, 2),
            "Expected Cost (E[C])": round(self.expected_cost, 2),
            "Cost Variance (Var(C))": round(self.cost_variance, 2),
            "Feasible Probability (P_F)": f"{self.feasible_probability * 100:.2f}%",
            "Violation Rate (V_rate)": f"{self.constraint_violation_rate * 100:.2f}%",
            "Good Concentration (G)": round(self.good_route_concentration, 6),
            "Best Route Prob P(r*)": f"{self.best_route_probability * 100:.2f}%",
            "Approximation Ratio": round(self.approximation_ratio, 4) if self.approximation_ratio else "N/A",
            "Runtime (ms)": round(self.total_runtime_ms, 1),
            "Iterations": self.total_iterations,
            "Shots": self.total_shots,
            "Qubits": self.qubits,
            "Circuit Depth": self.circuit_depth
        }


class MetricsEvaluator:
    """Evaluates multi-dimensional research metrics on optimization outputs."""

    @staticmethod
    def evaluate_metrics(
        probabilities: Dict[str, float],
        route_costs: Dict[str, float],
        feasible_subspace: Set[str],
        good_routes: Set[str],
        optimal_cost: Optional[float] = None,
        runtime_ms: float = 0.0,
        iterations: int = 0,
        shots: int = 1024,
        qubits: int = 9,
        circuit_depth: int = 1
    ) -> QLDOResearchMetrics:
        """
        Computes all research evaluation metrics required by Section 19 of the research specification.
        """
        # 1. Expected cost and variance
        exp_c = sum(probabilities.get(r, 0.0) * route_costs.get(r, 0.0) for r in probabilities)
        var_c = sum(probabilities.get(r, 0.0) * ((route_costs.get(r, 0.0) - exp_c) ** 2) for r in probabilities)
        std_c = float(np.sqrt(max(0.0, var_c)))

        # 2. Feasibility probability
        p_f = sum(probabilities.get(r, 0.0) for r in feasible_subspace)
        v_rate = 1.0 - p_f

        # 3. Good-route concentration G
        g_val = sum((probabilities.get(r, 0.0) ** 2) for r in good_routes)

        # 4. Best route and probability
        # Focus on best feasible route if any exist, otherwise best overall
        if feasible_subspace:
            best_r = min(feasible_subspace, key=lambda r: route_costs.get(r, float('inf')))
        else:
            best_r = min(route_costs.keys(), key=lambda r: route_costs.get(r, float('inf')))

        c_best = float(route_costs.get(best_r, 0.0))
        p_best = float(probabilities.get(best_r, 0.0))

        # 5. Approximation ratio
        approx_ratio = None
        if optimal_cost is not None and optimal_cost > 0:
            approx_ratio = float(optimal_cost / c_best) if c_best > 0 else 1.0

        return QLDOResearchMetrics(
            best_route_cost=c_best,
            expected_cost=float(exp_c),
            cost_variance=float(var_c),
            cost_std=std_c,
            feasible_probability=float(p_f),
            constraint_violation_rate=float(v_rate),
            good_route_concentration=float(g_val),
            best_route_probability=p_best,
            best_bitstring=best_r,
            optimal_ground_truth_cost=optimal_cost,
            approximation_ratio=approx_ratio,
            total_runtime_ms=float(runtime_ms),
            total_iterations=iterations,
            total_shots=shots,
            qubits=qubits,
            circuit_depth=circuit_depth
        )
