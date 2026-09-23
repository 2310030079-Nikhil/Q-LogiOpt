"""
Exact Ground Truth Solver: Brute Force Exhaustive Search
"""
import itertools
import time
from typing import List, Optional
import numpy as np
from data_processing.loader import LocationNode
from optimization.objective import ObjectiveEvaluator, RouteSolution


class ExactBruteForceSolver:
    """Computes mathematically provable global optimum for small TSP/VRP instances."""
    
    @classmethod
    def solve(
        cls,
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0
    ) -> RouteSolution:
        """
        Exhaustively checks all (n-1)! customer permutations.
        Computes the true global optimum C* used as denominator for Approximation Ratio.
        """
        start_time = time.perf_counter()
        n = len(nodes)
        
        if n < 3:
            tour = [0, 1, 0] if n == 2 else [0, 0]
            sol = ObjectiveEvaluator.evaluate_tour(tour, nodes, distance_matrix, cost_matrix)
            sol.algorithm = "Exact (Brute Force)"
            sol.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            return sol
            
        customers = list(range(1, n))
        matrix = cost_matrix if cost_matrix is not None else distance_matrix
        
        best_tour = None
        best_cost = float("inf")
        total_perms = 0
        
        for perm in itertools.permutations(customers):
            total_perms += 1
            tour = [0] + list(perm) + [0]
            
            # Fast cost evaluation
            c = sum(matrix[tour[k], tour[k + 1]] for k in range(len(tour) - 1))
            if c < best_cost:
                best_cost = c
                best_tour = tour
                
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        solution = ObjectiveEvaluator.evaluate_tour(
            route_indices=best_tour,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )
        solution.algorithm = "Exact (Brute Force)"
        solution.runtime_ms = round(runtime_ms, 3)
        solution.metadata["total_permutations_evaluated"] = total_perms
        return solution
