"""
Synthetic Logistics Dataset Generator
"""
from typing import List, Optional, Tuple
import numpy as np
from data_processing.loader import LocationNode


class SyntheticDataGenerator:
    """Generates realistic synthetic logistics node distributions."""
    
    @staticmethod
    def generate(
        num_customers: int = 3,
        seed: Optional[int] = 42,
        center_lat: float = 17.3850,
        center_lon: float = 78.4867,
        radius_km: float = 8.0,
        demand_range: Tuple[int, int] = (1, 10),
        priority_range: Tuple[int, int] = (1, 5)
    ) -> List[LocationNode]:
        """
        Generates synthetic delivery network with 1 Depot and N customers.
        Coordinates are distributed realistically around the depot center using
        latitude/longitude degree conversions (~111 km/deg lat).
        """
        if seed is not None:
            np.random.seed(seed)
            
        nodes: List[LocationNode] = []
        
        # 1. Depot at the center
        depot = LocationNode(
            id="DEPOT",
            latitude=round(center_lat, 4),
            longitude=round(center_lon, 4),
            demand=0.0,
            priority=0,
            is_depot=True
        )
        nodes.append(depot)
        
        # Convert radius in km to approximate degrees
        deg_lat_per_km = 1.0 / 110.574
        deg_lon_per_km = 1.0 / (111.320 * np.cos(np.radians(center_lat)))
        
        # 2. Customers distributed in polar coordinates around depot
        for i in range(1, num_customers + 1):
            r = np.sqrt(np.random.uniform(0.1, 1.0)) * radius_km
            theta = np.random.uniform(0, 2 * np.pi)
            
            dx_km = r * np.cos(theta)
            dy_km = r * np.sin(theta)
            
            lat = center_lat + dy_km * deg_lat_per_km
            lon = center_lon + dx_km * deg_lon_per_km
            
            demand = float(np.random.randint(demand_range[0], demand_range[1] + 1))
            priority = int(np.random.randint(priority_range[0], priority_range[1] + 1))
            
            nodes.append(
                LocationNode(
                    id=f"C{i}",
                    latitude=round(lat, 4),
                    longitude=round(lon, 4),
                    demand=demand,
                    priority=priority,
                    is_depot=False
                )
            )
            
        return nodes
