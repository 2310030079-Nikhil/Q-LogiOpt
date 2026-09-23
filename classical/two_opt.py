"""
Classical Optimization Baseline 2: 2-Opt Local Search Heuristic
"""
import time
from typing import List, Optional
import numpy as np
from data_processing.loader import LocationNode
from optimization.objective import ObjectiveEvaluator, RouteSolution
from classical.nearest_neighbor import NearestNeighborSolver


class TwoOptSolver:
    """Iterative 2-opt edge-swapping local search heuristic."""
    
    @classmethod
    def solve(
        cls,
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        initial_route: Optional[List[int]] = None,
        max_iterations: int = 500,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0
    ) -> RouteSolution:
        """
        Takes an initial tour (or generates one via Nearest Neighbor)
        and iteratively tests reversing sub-paths [i...j] to reduce tour cost.
        Depot at index 0 and index -1 remain fixed.
        """
        start_time = time.perf_counter()
        n = len(nodes)
        
        if n < 3:
            # For 2 locations (depot + 1 customer), route is trivial [0, 1, 0]
            tour = [0, 1, 0] if n == 2 else [0, 0]
            sol = ObjectiveEvaluator.evaluate_tour(tour, nodes, distance_matrix, cost_matrix)
            sol.algorithm = "2-Opt"
            sol.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            return sol

        if initial_route is None:
            nn_sol = NearestNeighborSolver.solve(nodes, distance_matrix, cost_matrix)
            tour = list(nn_sol.route_indices)
        else:
            tour = list(initial_route)
            
        def tour_cost(r: List[int]) -> float:
            c = 0.0
            matrix = cost_matrix if cost_matrix is not None else distance_matrix
            for k in range(len(r) - 1):
                c += matrix[r[k], r[k + 1]]
            return c

        best_route = list(tour)
        best_cost = tour_cost(best_route)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Sub-segment reversal between indices 1 and len(tour)-2 (keeping depot at 0 and end)
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    # 2-opt swap: reverse segment between i and j
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    new_cost = tour_cost(new_route)
                    if new_cost < best_cost - 1e-6:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
                if improved:
                    break
                    
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        solution = ObjectiveEvaluator.evaluate_tour(
            route_indices=best_route,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )
        solution.algorithm = "2-Opt"
        solution.runtime_ms = round(runtime_ms, 3)
        solution.metadata["iterations"] = iterations
        return solution
