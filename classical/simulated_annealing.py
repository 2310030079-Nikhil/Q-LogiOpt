"""
Classical Optimization Baseline 3: Simulated Annealing Metaheuristic
"""
import math
import random
import time
from typing import List, Optional
import numpy as np
from data_processing.loader import LocationNode
from optimization.objective import ObjectiveEvaluator, RouteSolution
from classical.nearest_neighbor import NearestNeighborSolver


class SimulatedAnnealingSolver:
    """Probabilistic Simulated Annealing metaheuristic for logistics routing."""
    
    @classmethod
    def solve(
        cls,
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 1e-3,
        seed: Optional[int] = 42,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0
    ) -> RouteSolution:
        """
        Explores solution space by accepting both cost-reducing moves and
        probabilistic uphill moves (Metropolis criterion: P = exp(-delta / T))
        to escape local minima.
        """
        start_time = time.perf_counter()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        n = len(nodes)
        if n < 3:
            tour = [0, 1, 0] if n == 2 else [0, 0]
            sol = ObjectiveEvaluator.evaluate_tour(tour, nodes, distance_matrix, cost_matrix)
            sol.algorithm = "Simulated Annealing"
            sol.runtime_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            return sol
            
        # Initial tour from Nearest Neighbor
        nn_sol = NearestNeighborSolver.solve(nodes, distance_matrix, cost_matrix)
        current_tour = list(nn_sol.route_indices)
        
        matrix = cost_matrix if cost_matrix is not None else distance_matrix
        
        def calc_cost(r: List[int]) -> float:
            c = 0.0
            for k in range(len(r) - 1):
                c += matrix[r[k], r[k + 1]]
            return c
            
        current_cost = calc_cost(current_tour)
        best_tour = list(current_tour)
        best_cost = current_cost
        
        temp = initial_temp
        iterations = 0
        
        while temp > min_temp:
            iterations += 1
            # Generate neighborhood move: choose between 2-opt sub-path reversal or customer swap
            neighbor_tour = list(current_tour)
            # Customer indices are from 1 to len(tour)-2
            num_cust = len(neighbor_tour) - 2
            if num_cust >= 2:
                i, j = sorted(random.sample(range(1, len(neighbor_tour) - 1), 2))
                if random.random() < 0.5:
                    # 2-opt inversion
                    neighbor_tour = neighbor_tour[:i] + neighbor_tour[i:j + 1][::-1] + neighbor_tour[j + 1:]
                else:
                    # Swap customer i and j
                    neighbor_tour[i], neighbor_tour[j] = neighbor_tour[j], neighbor_tour[i]
                    
            neighbor_cost = calc_cost(neighbor_tour)
            delta = neighbor_cost - current_cost
            
            # Acceptance decision
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                current_tour = neighbor_tour
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = list(current_tour)
                    
            temp *= cooling_rate
            
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        
        solution = ObjectiveEvaluator.evaluate_tour(
            route_indices=best_tour,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )
        solution.algorithm = "Simulated Annealing"
        solution.runtime_ms = round(runtime_ms, 3)
        solution.metadata["iterations"] = iterations
        solution.metadata["final_temp"] = temp
        return solution
