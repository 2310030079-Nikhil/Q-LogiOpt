"""
Classical Optimization Baseline 1: Nearest Neighbor Heuristic
"""
import time
from typing import List, Optional
import numpy as np
from data_processing.loader import LocationNode
from optimization.objective import ObjectiveEvaluator, RouteSolution


class NearestNeighborSolver:
    """Greedy Nearest Neighbor heuristic for TSP/VRP routes."""
    
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
        Constructs a route starting from Depot (0), iteratively picking the
        nearest unvisited customer, and returning to the Depot.
        Complexity: O(n^2).
        """
        start_time = time.perf_counter()
        n = len(nodes)
        
        if n < 2:
            return ObjectiveEvaluator.evaluate_tour([0, 0], nodes, distance_matrix)
            
        unvisited = set(range(1, n))
        current = 0
        tour = [0]
        
        while unvisited:
            # Pick next node that minimizes cost (or distance)
            best_next = None
            best_val = float("inf")
            for candidate in unvisited:
                val = cost_matrix[current, candidate] if cost_matrix is not None else distance_matrix[current, candidate]
                if val < best_val:
                    best_val = val
                    best_next = candidate
                    
            if best_next is not None:
                tour.append(best_next)
                unvisited.remove(best_next)
                current = best_next
            else:
                break
                
        # Return to depot
        tour.append(0)
        
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        solution = ObjectiveEvaluator.evaluate_tour(
            route_indices=tour,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )
        solution.algorithm = "Nearest Neighbor"
        solution.runtime_ms = round(runtime_ms, 3)
        return solution
