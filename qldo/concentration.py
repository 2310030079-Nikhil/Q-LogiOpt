"""
QLDO: Good-Route Set and Probability Concentration Engine.
Implements candidate definitions for the high-quality feasible set G subseteq F,
and computes quadratic and information-theoretic probability concentration metrics.
"""
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class GoodRouteDefinition(str, Enum):
    PERCENTILE = "Definition A: Cost Percentile Q_q"
    RELATIVE_BEST = "Definition B: Relative to Best (1 + delta) * C_best"
    CLASSICAL_BENCHMARK = "Definition C: Classical Reference Threshold"


class ConcentrationEngine:
    """
    Evaluates probability concentration over high-quality feasible logistics routes.
    """

    @staticmethod
    def identify_good_routes(
        feasible_subspace: Set[str],
        route_costs: Dict[str, float],
        definition: GoodRouteDefinition = GoodRouteDefinition.RELATIVE_BEST,
        percentile_q: float = 0.25,
        delta: float = 0.15,
        classical_ref_cost: Optional[float] = None
    ) -> Set[str]:
        """
        Determines the subset G subseteq F of high-quality feasible routes based on
        one of three scientifically investigated criteria.
        
        Args:
            feasible_subspace: Set of strictly feasible bitstrings F
            route_costs: Dict mapping bitstring -> cost C(r)
            definition: Selection strategy for G
            percentile_q: Quantile threshold for Definition A (e.g. 0.25 = top 25%)
            delta: Relative tolerance for Definition B (e.g. 0.15 = within 15% of C_best)
            classical_ref_cost: Cost of classical reference solution for Definition C
            
        Returns:
            Set of bitstrings belonging to G
        """
        if not feasible_subspace:
            return set()
            
        feasible_costs = {r: route_costs[r] for r in feasible_subspace if r in route_costs}
        if not feasible_costs:
            return set()
            
        costs_array = np.array(list(feasible_costs.values()))
        c_best = float(np.min(costs_array))
        
        good_routes = set()
        
        if definition == GoodRouteDefinition.PERCENTILE:
            # Definition A: Routes whose cost is below a selected cost quantile
            cutoff = float(np.quantile(costs_array, percentile_q))
            for r, cost in feasible_costs.items():
                if cost <= cutoff:
                    good_routes.add(r)
                    
        elif definition == GoodRouteDefinition.RELATIVE_BEST:
            # Definition B: Routes within (1 + delta) * C_best
            cutoff = (1.0 + delta) * c_best
            for r, cost in feasible_costs.items():
                if cost <= cutoff:
                    good_routes.add(r)
                    
        elif definition == GoodRouteDefinition.CLASSICAL_BENCHMARK:
            # Definition C: Routes below or equal to a classical heuristic reference
            ref = classical_ref_cost if classical_ref_cost is not None else (1.1 * c_best)
            for r, cost in feasible_costs.items():
                if cost <= ref:
                    good_routes.add(r)
        else:
            good_routes = set(feasible_subspace)
            
        return good_routes

    @staticmethod
    def compute_concentration_metric(
        probabilities: Dict[str, float],
        good_routes: Set[str]
    ) -> float:
        """
        Calculates the primary QLDO quadratic concentration metric:
        G_theta = sum_{r in G} p_theta(r)^2
        
        Properties:
        - G >= 0
        - Diffuse / spread probability across many routes yields low G
        - Sharp concentration onto a single optimal route yields G -> 1.0
        
        Args:
            probabilities: Dict mapping bitstring -> probability p(r)
            good_routes: Set G of high-utility feasible routes
            
        Returns:
            Quadratic concentration value G_theta
        """
        g_val = 0.0
        for r in good_routes:
            p = probabilities.get(r, 0.0)
            g_val += p ** 2
        return float(g_val)

    @staticmethod
    def compute_comparative_concentration_metrics(
        probabilities: Dict[str, float],
        good_routes: Set[str]
    ) -> Dict[str, float]:
        """
        Computes alternative concentration measures for academic comparison:
        1. Quadratic Concentration G = sum_{r in G} p(r)^2
        2. Linear Probability Mass P_G = sum_{r in G} p(r)
        3. Normalized Participation Ratio (NPR) = 1 / (len(G) * sum_{G} (p/P_G)^2)
        4. Shannon Entropy on G: H_G = -sum_{G} (p/P_G) * log(p/P_G)
        """
        quad_g = ConcentrationEngine.compute_concentration_metric(probabilities, good_routes)
        lin_mass = sum(probabilities.get(r, 0.0) for r in good_routes)
        
        # Normalized metrics on G conditional distribution
        if lin_mass > 1e-12:
            cond_p = np.array([probabilities.get(r, 0.0) / lin_mass for r in good_routes])
            sum_cond_sq = float(np.sum(cond_p ** 2))
            inv_pr = 1.0 / (len(good_routes) * sum_cond_sq) if len(good_routes) > 0 and sum_cond_sq > 0 else 0.0
            entropy = float(-np.sum(cond_p * np.log(cond_p + 1e-15)))
        else:
            sum_cond_sq = 0.0
            inv_pr = 0.0
            entropy = 0.0
            
        return {
            "quadratic_concentration_G": quad_g,
            "linear_probability_mass_P_G": float(lin_mass),
            "conditional_hhi": sum_cond_sq,
            "normalized_participation_ratio": inv_pr,
            "conditional_entropy": entropy
        }
