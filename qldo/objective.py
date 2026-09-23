"""
QLDO: Master Objective Evaluation Engine.
Synthesizes Expected Cost, Cost Variance, Feasibility Barrier, and Good-Route Concentration
into the unified loss function:
J_QLDO(theta) = E[C] + alpha * Var(C) - beta * G - gamma * log(P_F + epsilon)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from qldo.probability import ProbabilityDistribution
from qldo.feasibility import FeasibilityEngine
from qldo.concentration import ConcentrationEngine, GoodRouteDefinition


@dataclass
class QLDOComponents:
    """Detailed breakdown of individual terms comprising the QLDO loss."""
    expected_cost: float
    variance: float
    scaled_variance: float
    good_concentration: float
    scaled_concentration: float
    feasible_probability: float
    feasibility_barrier: float
    total_qldo_objective: float
    alpha: float
    beta: float
    gamma: float
    epsilon: float
    good_routes_count: int
    feasible_routes_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "Expected Cost E[C]": round(self.expected_cost, 4),
            "Variance Var(C)": round(self.variance, 4),
            "Scaled Variance (+alpha*Var)": round(self.scaled_variance, 4),
            "Good-Route Concentration G": round(self.good_concentration, 6),
            "Concentration Reward (-beta*G)": round(self.scaled_concentration, 4),
            "Feasible Probability P_F": round(self.feasible_probability, 4),
            "Feasibility Barrier (-gamma*log)": round(self.feasibility_barrier, 4),
            "Total QLDO Objective J_QLDO": round(self.total_qldo_objective, 4)
        }


class QLDOObjectiveEngine:
    """
    Computes standard expectation and distribution-aware QLDO objectives.
    """

    @classmethod
    def evaluate(
        cls,
        counts_or_probs: Dict[str, float],
        route_costs: Dict[str, float],
        feasible_subspace: Set[str],
        alpha: float = 0.05,
        beta: float = 5.0,
        gamma: float = 5.0,
        epsilon: float = 1e-4,
        good_route_def: GoodRouteDefinition = GoodRouteDefinition.RELATIVE_BEST,
        percentile_q: float = 0.25,
        delta: float = 0.15,
        classical_ref_cost: Optional[float] = None
    ) -> QLDOComponents:
        """
        Evaluates the complete QLDO objective function.
        
        Args:
            counts_or_probs: Measurement counts or normalized probabilities
            route_costs: Dict mapping bitstring -> cost C(r)
            feasible_subspace: Set of strictly feasible bitstrings F
            alpha: Weight for variance regularization term >= 0
            beta: Weight for good-route concentration reward >= 0
            gamma: Weight for logarithmic feasibility barrier >= 0
            epsilon: Stabilization factor > 0
            good_route_def: Strategy for defining high-utility routes G
            percentile_q: Quantile cutoff for Definition A
            delta: Relative cost tolerance for Definition B
            classical_ref_cost: Reference threshold for Definition C
            
        Returns:
            QLDOComponents containing total objective and decomposition
        """
        # Ensure normalized probabilities
        if any(isinstance(v, int) for v in counts_or_probs.values()):
            probs = ProbabilityDistribution.counts_to_probabilities(counts_or_probs)
        else:
            # Check sum
            total_p = sum(counts_or_probs.values())
            if not np.isclose(total_p, 1.0, atol=1e-4):
                probs = {k: v / total_p for k, v in counts_or_probs.items()}
            else:
                probs = counts_or_probs

        # 1. Expected Cost E[C]
        exp_c = ProbabilityDistribution.compute_expected_cost(probs, route_costs)

        # 2. Cost Variance Var(C)
        var_c = ProbabilityDistribution.compute_cost_variance(probs, route_costs, exp_c)
        scaled_var = alpha * var_c

        # 3. Feasible Probability P_F & Logarithmic Barrier
        p_f = FeasibilityEngine.compute_feasible_probability(probs, feasible_subspace)
        barrier = FeasibilityEngine.compute_logarithmic_barrier(p_f, gamma=gamma, epsilon=epsilon)

        # 4. Good Route Set G & Quadratic Concentration G_theta
        good_routes = ConcentrationEngine.identify_good_routes(
            feasible_subspace=feasible_subspace,
            route_costs=route_costs,
            definition=good_route_def,
            percentile_q=percentile_q,
            delta=delta,
            classical_ref_cost=classical_ref_cost
        )
        conc_g = ConcentrationEngine.compute_concentration_metric(probs, good_routes)
        scaled_conc = -beta * conc_g

        # 5. Master QLDO Objective
        # J_QLDO = E[C] + alpha * Var(C) - beta * G - gamma * log(P_F + epsilon)
        j_qldo = exp_c + scaled_var + scaled_conc + barrier

        return QLDOComponents(
            expected_cost=float(exp_c),
            variance=float(var_c),
            scaled_variance=float(scaled_var),
            good_concentration=float(conc_g),
            scaled_concentration=float(scaled_conc),
            feasible_probability=float(p_f),
            feasibility_barrier=float(barrier),
            total_qldo_objective=float(j_qldo),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            epsilon=float(epsilon),
            good_routes_count=len(good_routes),
            feasible_routes_count=len(feasible_subspace)
        )

    @classmethod
    def evaluate_standard_expectation(
        cls,
        counts_or_probs: Dict[str, float],
        route_costs: Dict[str, float]
    ) -> float:
        """
        Evaluates conventional expectation objective J_standard = E[C].
        """
        if any(isinstance(v, int) for v in counts_or_probs.values()):
            probs = ProbabilityDistribution.counts_to_probabilities(counts_or_probs)
        else:
            probs = counts_or_probs
        return ProbabilityDistribution.compute_expected_cost(probs, route_costs)
