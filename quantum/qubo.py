"""
QUBO Formulation Engine for Logistics Route Optimization
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from data_processing.loader import LocationNode


@dataclass
class QuboProblem:
    """Encapsulates a formulated QUBO problem."""
    num_locations: int
    num_customers: int
    num_qubits: int
    variable_names: List[str]
    qubo_matrix: np.ndarray          # Upper-triangular N x N matrix
    constant_offset: float
    penalty_lambda: float
    cost_matrix: np.ndarray


class QuboBuilder:
    """
    Constructs the Quadratic Unconstrained Binary Optimization (QUBO)
    matrix for the Anchored Depot TSP/VRP logistics formulation.
    
    Qubit reduction:
    Fixing Depot at step 0 reduces variables from n^2 to (n-1)^2.
    """
    
    @staticmethod
    def get_linear_index(u: int, s: int, m: int) -> int:
        """Customer index u in [0, m-1] and step index s in [0, m-1] -> linear index k."""
        return u * m + s
        
    @staticmethod
    def get_indices_from_linear(k: int, m: int) -> Tuple[int, int]:
        """Linear index k -> (customer u, step s)."""
        return k // m, k % m

    @classmethod
    def build_qubo(
        cls,
        nodes: List[LocationNode],
        cost_matrix: np.ndarray,
        penalty_lambda: Optional[float] = None,
        penalty_multiplier: float = 2.0
    ) -> QuboProblem:
        """
        Builds the upper-triangular QUBO matrix Q such that:
        Objective: min x^T Q x + offset
        
        where x in {0, 1}^N, N = (n-1)^2.
        """
        n = len(nodes)
        m = n - 1  # number of customer nodes (excluding depot)
        num_qubits = m * m
        
        if penalty_lambda is None:
            max_cost = float(np.max(cost_matrix))
            penalty_lambda = max(10.0, max_cost * penalty_multiplier)
            
        P = penalty_lambda
        Q = np.zeros((num_qubits, num_qubits), dtype=float)
        offset = 0.0
        
        # 1. Variable names: x_{customer_id, step_id}
        var_names = []
        for u in range(m):
            for s in range(m):
                var_names.append(f"x[{nodes[u + 1].id},t{s + 1}]")
                
        # 2. Node Visitation Constraint: sum_s x_{u, s} = 1 for each customer u
        # P * (1 - sum_s x_{u, s})^2 = P * (1 - 2 sum_s x_{u, s} + sum_s sum_{s'} x_{u, s} x_{u, s'})
        # Linear: P * (-2 + 1) * x_{u, s} = -P * x_{u, s}
        # Quadratic: +2P * x_{u, s} * x_{u, s'} for s < s'
        for u in range(m):
            offset += P
            for s in range(m):
                k = cls.get_linear_index(u, s, m)
                Q[k, k] += -P
                for s_prime in range(s + 1, m):
                    k_prime = cls.get_linear_index(u, s_prime, m)
                    Q[k, k_prime] += 2.0 * P
                    
        # 3. Step Occupancy Constraint: sum_u x_{u, s} = 1 for each step s
        # P * (1 - sum_u x_{u, s})^2
        # Linear: -P * x_{u, s}
        # Quadratic: +2P * x_{u, s} * x_{u', s} for u < u'
        for s in range(m):
            offset += P
            for u in range(m):
                k = cls.get_linear_index(u, s, m)
                Q[k, k] += -P
                for u_prime in range(u + 1, m):
                    k_prime = cls.get_linear_index(u_prime, s, m)
                    Q[k, k_prime] += 2.0 * P
                    
        # 4. Routing Cost Terms
        # A. Depot to first customer (step 0): C[0, u+1] * x_{u, 0}
        for u in range(m):
            k = cls.get_linear_index(u, 0, m)
            Q[k, k] += cost_matrix[0, u + 1]
            
        # B. Inter-customer transitions (step s to s+1): C[u+1, v+1] * x_{u, s} * x_{v, s+1}
        for s in range(m - 1):
            for u in range(m):
                k_u = cls.get_linear_index(u, s, m)
                for v in range(m):
                    if u == v:
                        continue
                    k_v = cls.get_linear_index(v, s + 1, m)
                    cost_val = cost_matrix[u + 1, v + 1]
                    if k_u < k_v:
                        Q[k_u, k_v] += cost_val
                    else:
                        Q[k_v, k_u] += cost_val
                        
        # C. Last customer (step m-1) to Depot: C[u+1, 0] * x_{u, m-1}
        for u in range(m):
            k = cls.get_linear_index(u, m - 1, m)
            Q[k, k] += cost_matrix[u + 1, 0]
            
        return QuboProblem(
            num_locations=n,
            num_customers=m,
            num_qubits=num_qubits,
            variable_names=var_names,
            qubo_matrix=np.round(Q, 3),
            constant_offset=round(offset, 3),
            penalty_lambda=round(P, 3),
            cost_matrix=cost_matrix
        )

    @staticmethod
    def evaluate_qubo_energy(x: np.ndarray, qubo: QuboProblem) -> float:
        """Evaluates E = x^T Q x + offset for a binary vector x."""
        return float(np.dot(x, np.dot(qubo.qubo_matrix, x)) + qubo.constant_offset)

    @staticmethod
    def to_dataframe(qubo: QuboProblem) -> pd.DataFrame:
        """Converts QUBO matrix to labeled DataFrame for inspection."""
        return pd.DataFrame(
            qubo.qubo_matrix,
            index=qubo.variable_names,
            columns=qubo.variable_names
        )
