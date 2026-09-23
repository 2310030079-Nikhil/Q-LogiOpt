"""
QLDO: Route Feasibility and Logarithmic Barrier Engine.
Identifies valid logistics routes, computes empirical feasibility probability P_F,
and calculates the interior-point logarithmic constraint barrier -gamma * log(P_F + epsilon).
"""
from typing import Dict, List, Set, Tuple
import numpy as np
from data_processing.loader import LocationNode
from optimization.constraints import ConstraintValidator, ConstraintCheckResult
from optimization.route_decoder import RouteDecoder, DecodedQuantumState


class FeasibilityEngine:
    """
    Evaluates combinatorial feasibility of candidate bitstring solutions and
    computes the logarithmic feasibility penalty barrier.
    """

    @staticmethod
    def identify_feasible_subspace(
        decoded_states: List[DecodedQuantumState]
    ) -> Set[str]:
        """
        Determines the set of bitstrings F that represent strictly feasible logistics routes.
        
        Args:
            decoded_states: Decoded states with matrix and capacity constraint verification
            
        Returns:
            Set of bitstrings r belonging to F
        """
        feasible_bitstrings = set()
        for state in decoded_states:
            # Must satisfy permutation constraints and capacity/distance limits
            if state.is_valid_permutation:
                if state.strict_solution and state.strict_solution.is_feasible:
                    feasible_bitstrings.add(state.bitstring)
        return feasible_bitstrings

    @staticmethod
    def compute_feasible_probability(
        probabilities: Dict[str, float],
        feasible_subspace: Set[str]
    ) -> float:
        """
        Calculates P_F(theta) = sum_{r in F} p_theta(r).
        
        Args:
            probabilities: Dict mapping bitstring -> probability p(r)
            feasible_subspace: Set of valid bitstrings F
            
        Returns:
            Feasibility probability P_F in [0.0, 1.0]
        """
        p_f = sum(probabilities.get(bitstr, 0.0) for bitstr in feasible_subspace)
        return float(np.clip(p_f, 0.0, 1.0))

    @staticmethod
    def compute_logarithmic_barrier(
        p_f: float,
        gamma: float = 5.0,
        epsilon: float = 1e-4
    ) -> float:
        """
        Calculates the interior-point logarithmic feasibility barrier:
        Barrier = -gamma * log(P_F + epsilon)
        
        As P_F -> 0: Barrier -> -gamma * log(epsilon) > 0 (strong penalty)
        As P_F -> 1: Barrier -> -gamma * log(1 + epsilon) approx 0 (vanishing penalty)
        
        Args:
            p_f: Feasibility probability in [0, 1]
            gamma: Barrier scaling coefficient >= 0
            epsilon: Numerical stabilization constant > 0
            
        Returns:
            Barrier penalty value
        """
        if gamma < 0:
            raise ValueError(f"Gamma penalty parameter must be non-negative, got {gamma}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon stability parameter must be strictly positive, got {epsilon}")
            
        barrier = -gamma * float(np.log(p_f + epsilon))
        return float(barrier)
