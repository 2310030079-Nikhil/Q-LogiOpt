"""
Constraint Verification Engine for Logistics Routes
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from data_processing.loader import LocationNode


@dataclass
class ConstraintCheckResult:
    """Detailed result of constraint validation checks."""
    is_valid_permutation: bool
    is_capacity_satisfied: bool
    is_distance_satisfied: bool
    total_violations: int
    messages: List[str]


class ConstraintValidator:
    """Validates structural and physical logistics constraints."""
    
    @classmethod
    def validate_binary_assignment_matrix(cls, X: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validates an (N x N) binary assignment matrix for customer-to-step mapping.
        Requires:
        1. Exactly one 1 per row (every customer visited exactly once)
        2. Exactly one 1 per column (every step occupied by exactly one customer)
        """
        messages: List[str] = []
        n_rows, n_cols = X.shape
        
        row_sums = np.sum(X, axis=1)
        for r_idx, r_sum in enumerate(row_sums):
            if r_sum != 1:
                messages.append(f"Row {r_idx} (Customer {r_idx + 1}): visited {int(r_sum)} times (expected 1).")
                
        col_sums = np.sum(X, axis=0)
        for c_idx, c_sum in enumerate(col_sums):
            if c_sum != 1:
                messages.append(f"Column {c_idx} (Step {c_idx + 1}): assigned {int(c_sum)} customers (expected 1).")
                
        is_valid = (len(messages) == 0)
        return is_valid, messages
        
    @classmethod
    def check_all_constraints(
        cls,
        tour: List[int],
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0
    ) -> ConstraintCheckResult:
        """Runs complete physical checks on a decoded tour [0, c1, ..., ck, 0]."""
        messages: List[str] = []
        n = len(nodes)
        
        # 1. Permutation check
        if len(tour) < 2 or tour[0] != 0 or tour[-1] != 0:
            messages.append("Tour must start and end at the Depot (node 0).")
            
        customers = tour[1:-1]
        expected = set(range(1, n))
        visited = set(customers)
        
        missing = expected - visited
        if missing:
            messages.append(f"Missing deliveries to nodes: {sorted(list(missing))}")
            
        if len(customers) != len(visited):
            messages.append("Duplicate visits present in tour.")
            
        is_valid_perm = (len(messages) == 0)
        
        # 2. Capacity check
        total_demand = sum(nodes[i].demand for i in customers if 0 <= i < n)
        is_cap_ok = (total_demand <= vehicle_capacity)
        if not is_cap_ok:
            messages.append(f"Total demand {total_demand} exceeds capacity {vehicle_capacity}.")
            
        # 3. Distance check
        dist = 0.0
        for k in range(len(tour) - 1):
            dist += distance_matrix[tour[k], tour[k + 1]]
        is_dist_ok = (dist <= max_distance)
        if not is_dist_ok:
            messages.append(f"Total tour distance {dist:.2f} exceeds limit {max_distance:.2f}.")
            
        return ConstraintCheckResult(
            is_valid_permutation=is_valid_perm,
            is_capacity_satisfied=is_cap_ok,
            is_distance_satisfied=is_dist_ok,
            total_violations=len(messages),
            messages=messages
        )
