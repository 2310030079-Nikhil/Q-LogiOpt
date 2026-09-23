"""
QLDO: Probability Distribution and Statistical Moments Engine.
Computes empirical route probabilities, expected costs, and central variances.
"""
from typing import Dict, List, Tuple
import numpy as np


class ProbabilityDistribution:
    """
    Parses measurement counts and evaluates statistical distribution properties.
    """

    @staticmethod
    def counts_to_probabilities(counts: Dict[str, int]) -> Dict[str, float]:
        """
        Converts raw integer measurement shot counts into a normalized probability distribution.
        
        Args:
            counts: Dict mapping bitstring r -> integer count
            
        Returns:
            Dict mapping bitstring r -> float probability p(r) with sum(p) == 1.0
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            raise ValueError("Total measurement shots cannot be zero.")
            
        probs = {bitstr: count / total_shots for bitstr, count in counts.items()}
        
        # Numerical normalization check
        prob_sum = sum(probs.values())
        if not np.isclose(prob_sum, 1.0, atol=1e-5):
            # Normalize strictly
            probs = {k: v / prob_sum for k, v in probs.items()}
            
        return probs

    @staticmethod
    def compute_expected_cost(
        probabilities: Dict[str, float],
        route_costs: Dict[str, float]
    ) -> float:
        """
        Calculates the first moment E[C] = sum_r p(r) * C(r).
        
        Args:
            probabilities: Dict mapping bitstring -> probability p(r)
            route_costs: Dict mapping bitstring -> cost C(r)
            
        Returns:
            Expected logistics cost E[C]
        """
        expected_cost = 0.0
        for bitstr, p in probabilities.items():
            cost = route_costs.get(bitstr, 0.0)
            expected_cost += p * cost
        return float(expected_cost)

    @staticmethod
    def compute_cost_variance(
        probabilities: Dict[str, float],
        route_costs: Dict[str, float],
        expected_cost: float
    ) -> float:
        """
        Calculates the second central moment Var(C) = sum_r p(r) * (C(r) - E[C])^2.
        
        Args:
            probabilities: Dict mapping bitstring -> probability p(r)
            route_costs: Dict mapping bitstring -> cost C(r)
            expected_cost: Precomputed E[C]
            
        Returns:
            Cost variance Var(C) >= 0.0
        """
        var_cost = 0.0
        for bitstr, p in probabilities.items():
            cost = route_costs.get(bitstr, 0.0)
            var_cost += p * ((cost - expected_cost) ** 2)
        return float(max(0.0, var_cost))

    @staticmethod
    def compute_moments(
        probabilities: Dict[str, float],
        route_costs: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """
        Convenience function returning (E[C], Var(C), Std(C)).
        """
        exp_c = ProbabilityDistribution.compute_expected_cost(probabilities, route_costs)
        var_c = ProbabilityDistribution.compute_cost_variance(probabilities, route_costs, exp_c)
        std_c = float(np.sqrt(var_c))
        return exp_c, var_c, std_c
