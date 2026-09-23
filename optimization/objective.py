"""
Logistics Objective & Cost Evaluation Engine
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from data_processing.loader import LocationNode


@dataclass
class RouteSolution:
    """Represents the complete result of a routing optimization run."""
    algorithm: str
    route_indices: List[int]
    route_ids: List[str]
    total_distance: float
    total_cost: float
    runtime_ms: float
    is_feasible: bool = True
    constraint_violations: int = 0
    violation_details: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ObjectiveEvaluator:
    """Evaluates logistics objective functions and constraint compliance."""
    
    @staticmethod
    def evaluate_tour(
        route_indices: List[int],
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0,
        capacity_penalty_weight: float = 10.0,
        distance_weight: float = 1.0,
        priority_weight: float = 0.5
    ) -> RouteSolution:
        """
        Calculates exact metrics for a closed tour starting and ending at depot (index 0).
        Route indices format: [0, i1, i2, ..., ik, 0].
        """
        n = len(nodes)
        violations: List[str] = []
        
        # 1. Validation checks on route
        if len(route_indices) < 2:
            violations.append("Tour contains fewer than 2 stops.")
            return RouteSolution(
                algorithm="Evaluator",
                route_indices=route_indices,
                route_ids=[nodes[i].id for i in route_indices if 0 <= i < n],
                total_distance=0.0,
                total_cost=float("inf"),
                runtime_ms=0.0,
                is_feasible=False,
                constraint_violations=len(violations),
                violation_details=violations
            )
            
        if route_indices[0] != 0 or route_indices[-1] != 0:
            violations.append(f"Tour does not start and end at Depot (0). Starts at {route_indices[0]}, ends at {route_indices[-1]}.")
            
        # Check visited customer nodes
        customer_indices = route_indices[1:-1]
        expected_customers = set(range(1, n))
        visited_customers = set(customer_indices)
        
        # Missing customers
        missing = expected_customers - visited_customers
        if missing:
            violations.append(f"Missing customer deliveries: {[nodes[m].id for m in missing]}")
            
        # Duplicate visits
        if len(customer_indices) != len(visited_customers):
            dups = [item for item in customer_indices if customer_indices.count(item) > 1]
            violations.append(f"Duplicate customer visits detected: {set(dups)}")
            
        # Extra out-of-range indices
        out_of_bounds = [idx for idx in route_indices if not (0 <= idx < n)]
        if out_of_bounds:
            violations.append(f"Invalid node indices out of range: {out_of_bounds}")
            return RouteSolution(
                algorithm="Evaluator",
                route_indices=route_indices,
                route_ids=[],
                total_distance=float("inf"),
                total_cost=float("inf"),
                runtime_ms=0.0,
                is_feasible=False,
                constraint_violations=len(violations),
                violation_details=violations
            )

        # 2. Distance and Cost computation
        total_dist = 0.0
        total_cost = 0.0
        total_demand = 0.0
        
        for k in range(len(route_indices) - 1):
            u = route_indices[k]
            v = route_indices[k + 1]
            leg_dist = distance_matrix[u, v]
            total_dist += leg_dist
            
            if cost_matrix is not None:
                leg_cost = cost_matrix[u, v]
            else:
                prio_pen = priority_weight * (5 - nodes[v].priority) if not nodes[v].is_depot else 0.0
                leg_cost = distance_weight * leg_dist + prio_pen
            total_cost += leg_cost
            
            total_demand += nodes[v].demand
            
        # Check Capacity constraint
        if total_demand > vehicle_capacity:
            excess = total_demand - vehicle_capacity
            violations.append(f"Capacity exceeded: {total_demand} > {vehicle_capacity} (excess {excess})")
            total_cost += capacity_penalty_weight * excess
            
        # Check Max Distance constraint
        if total_dist > max_distance:
            excess_dist = total_dist - max_distance
            violations.append(f"Max distance exceeded: {total_dist:.2f} > {max_distance:.2f}")
            total_cost += 5.0 * excess_dist
            
        is_feasible = (len(violations) == 0)
        route_ids = [nodes[i].id for i in route_indices]
        
        return RouteSolution(
            algorithm="Evaluator",
            route_indices=route_indices,
            route_ids=route_ids,
            total_distance=round(total_dist, 3),
            total_cost=round(total_cost, 3),
            runtime_ms=0.0,
            is_feasible=is_feasible,
            constraint_violations=len(violations),
            violation_details=violations,
            metadata={"total_demand": total_demand}
        )
