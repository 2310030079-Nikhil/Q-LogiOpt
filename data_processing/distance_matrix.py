"""
Distance Matrix Computation (Haversine & Euclidean)
"""
from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
from data_processing.loader import LocationNode


class DistanceMatrixCalculator:
    """Computes pairwise distance and cost matrices for logistics nodes."""
    
    EARTH_RADIUS_KM = 6371.0088
    
    @classmethod
    def haversine_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates great-circle distance between two points on Earth in kilometers
        using the Haversine formula.
        """
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = (np.sin(dphi / 2.0) ** 2 + 
             np.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(cls.EARTH_RADIUS_KM * c)
        
    @classmethod
    def euclidean_distance(cls, x1: float, y1: float, x2: float, y2: float) -> float:
        """Computes simple planar Euclidean distance between two coordinate pairs."""
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        
    @classmethod
    def compute_distance_matrix(
        cls,
        nodes: List[LocationNode],
        metric: Literal["haversine", "euclidean"] = "haversine"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Computes pairwise distance matrix (in km for Haversine, or coordinate units for Euclidean).
        Returns:
            matrix: np.ndarray of shape (N, N)
            labels: List[str] of node IDs corresponding to matrix indices
        """
        n = len(nodes)
        matrix = np.zeros((n, n), dtype=float)
        labels = [node.id for node in nodes]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.0
                elif j > i:
                    if metric == "haversine":
                        dist = cls.haversine_distance(
                            nodes[i].latitude, nodes[i].longitude,
                            nodes[j].latitude, nodes[j].longitude
                        )
                    else:
                        dist = cls.euclidean_distance(
                            nodes[i].latitude, nodes[i].longitude,
                            nodes[j].latitude, nodes[j].longitude
                        )
                    matrix[i, j] = round(dist, 3)
                    matrix[j, i] = round(dist, 3)
                    
        return matrix, labels
        
    @classmethod
    def compute_cost_matrix(
        cls,
        nodes: List[LocationNode],
        distance_matrix: np.ndarray,
        distance_weight: float = 1.0,
        priority_weight: float = 0.5
    ) -> np.ndarray:
        """
        Computes asymmetric/augmented operational cost matrix:
        C[i, j] = w_dist * Dist[i, j] + w_prio * PriorityPenalty[j]
        
        Priority penalty is inversely proportional to customer priority:
        Higher priority (e.g. 5) has lower penalty (0);
        Lower priority (e.g. 1) has higher penalty (max_prio - prio).
        Depot (j=0) has zero priority penalty.
        """
        n = len(nodes)
        max_priority = max([node.priority for node in nodes]) if nodes else 5
        cost_matrix = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cost_matrix[i, j] = 0.0
                else:
                    dist_cost = distance_weight * distance_matrix[i, j]
                    if nodes[j].is_depot:
                        prio_penalty = 0.0
                    else:
                        # Lower priority gets higher penalty
                        prio_penalty = priority_weight * (max_priority - nodes[j].priority)
                    cost_matrix[i, j] = round(dist_cost + prio_penalty, 3)
                    
        return cost_matrix

    @staticmethod
    def to_dataframe(matrix: np.ndarray, labels: List[str]) -> pd.DataFrame:
        """Wraps distance/cost matrix in a labeled DataFrame for display."""
        return pd.DataFrame(matrix, index=labels, columns=labels)
