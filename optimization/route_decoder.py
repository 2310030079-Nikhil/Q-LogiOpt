"""
Quantum Bitstring to Logistics Route Decoder & Repair Engine
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from data_processing.loader import LocationNode
from optimization.constraints import ConstraintValidator
from optimization.objective import ObjectiveEvaluator, RouteSolution


@dataclass
class DecodedQuantumState:
    """Detailed parsing of a single measured quantum bitstring."""
    bitstring: str
    assignment_matrix: np.ndarray
    is_valid_permutation: bool
    strict_tour: Optional[List[int]]
    repaired_tour: List[int]
    strict_solution: Optional[RouteSolution]
    repaired_solution: RouteSolution
    violation_messages: List[str]


class RouteDecoder:
    """Decodes quantum bitstrings from QAOA/Ising measurements into physical routes."""
    
    @staticmethod
    def bitstring_to_matrix(bitstring: str, num_customers: int, qiskit_endian: bool = True) -> np.ndarray:
        """
        Converts a bitstring of length m^2 into an (m x m) customer assignment matrix.
        In Qiskit, bitstring strings are ordered with qubit N-1 at index 0 and qubit 0 at index -1.
        If qiskit_endian=True, we reverse the string so index k corresponds to qubit k.
        
        k = u * m + s, where:
        u in {0, ..., m-1} is customer index (corresponds to location node u+1)
        s in {0, ..., m-1} is time slot index (corresponds to route step s+1)
        """
        m = num_customers
        if len(bitstring) != m * m:
            raise ValueError(f"Bitstring length {len(bitstring)} does not match m^2 = {m*m} (m={m}).")
            
        ordered_bits = bitstring[::-1] if qiskit_endian else bitstring
        matrix = np.zeros((m, m), dtype=int)
        
        for u in range(m):
            for s in range(m):
                k = u * m + s
                matrix[u, s] = int(ordered_bits[k])
                
        return matrix

    @classmethod
    def decode_single_bitstring(
        cls,
        bitstring: str,
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0,
        qiskit_endian: bool = True
    ) -> DecodedQuantumState:
        """
        Decodes a single bitstring into both strict and repaired tours.
        """
        m = len(nodes) - 1  # Number of customers
        matrix = cls.bitstring_to_matrix(bitstring, m, qiskit_endian=qiskit_endian)
        is_valid_perm, violation_msgs = ConstraintValidator.validate_binary_assignment_matrix(matrix)
        
        strict_tour: Optional[List[int]] = None
        strict_sol: Optional[RouteSolution] = None
        
        if is_valid_perm:
            # Reconstruct exact tour: step s has customer u where matrix[u, s] == 1
            # Customer u corresponds to location index u + 1
            perm = [0] * m
            for s in range(m):
                u = int(np.argmax(matrix[:, s]))
                perm[s] = u + 1
            strict_tour = [0] + perm + [0]
            strict_sol = ObjectiveEvaluator.evaluate_tour(
                route_indices=strict_tour,
                nodes=nodes,
                distance_matrix=distance_matrix,
                cost_matrix=cost_matrix,
                vehicle_capacity=vehicle_capacity,
                max_distance=max_distance
            )
            strict_sol.algorithm = "QAOA (Strict)"
            
        # Repaired tour using Hungarian algorithm on -matrix
        # Maximizes the overlap with the measured bitstring while guaranteeing a valid permutation
        cost_grid = -matrix.astype(float)
        row_ind, col_ind = linear_sum_assignment(cost_grid)
        
        # col_ind[i] is step assigned to customer i (row i)
        # We need customer at each step s
        step_to_cust = [0] * m
        for u, s in zip(row_ind, col_ind):
            step_to_cust[s] = u + 1
            
        repaired_tour = [0] + step_to_cust + [0]
        repaired_sol = ObjectiveEvaluator.evaluate_tour(
            route_indices=repaired_tour,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )
        repaired_sol.algorithm = "QAOA (Repaired)"
        
        return DecodedQuantumState(
            bitstring=bitstring,
            assignment_matrix=matrix,
            is_valid_permutation=is_valid_perm,
            strict_tour=strict_tour,
            repaired_tour=repaired_tour,
            strict_solution=strict_sol,
            repaired_solution=repaired_sol,
            violation_messages=violation_msgs
        )

    @classmethod
    def evaluate_measurement_counts(
        cls,
        counts: Dict[str, int],
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        cost_matrix: Optional[np.ndarray] = None,
        vehicle_capacity: float = 100.0,
        max_distance: float = 1000.0,
        qiskit_endian: bool = True
    ) -> Tuple[List[DecodedQuantumState], Dict[str, float]]:
        """
        Parses full dictionary of measurement counts from Aer simulation.
        Returns:
            decoded_states: List of parsed states sorted by frequency/probability
            summary: Dictionary containing CSR (Constraint Satisfaction Rate), top prob, etc.
        """
        total_shots = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        
        decoded_states: List[DecodedQuantumState] = []
        valid_shots = 0
        
        for bitstring, count in sorted_counts:
            state = cls.decode_single_bitstring(
                bitstring=bitstring,
                nodes=nodes,
                distance_matrix=distance_matrix,
                cost_matrix=cost_matrix,
                vehicle_capacity=vehicle_capacity,
                max_distance=max_distance,
                qiskit_endian=qiskit_endian
            )
            decoded_states.append(state)
            if state.is_valid_permutation:
                valid_shots += count
                
        csr = (valid_shots / total_shots * 100.0) if total_shots > 0 else 0.0
        
        summary = {
            "total_shots": total_shots,
            "unique_bitstrings": len(counts),
            "valid_shots": valid_shots,
            "constraint_satisfaction_rate": round(csr, 2),
            "top_bitstring": sorted_counts[0][0] if sorted_counts else "",
            "top_bitstring_probability": round((sorted_counts[0][1] / total_shots * 100.0), 2) if total_shots > 0 else 0.0
        }
        
        return decoded_states, summary
