"""
Logistics Data Loader & Validator
"""
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np


@dataclass
class LocationNode:
    """Represents a single depot or customer stop in the logistics network."""
    id: str
    latitude: float
    longitude: float
    demand: float
    priority: int
    is_depot: bool = False


class DataLoader:
    """Handles loading and validation of logistics datasets from CSV or raw DataFrames."""
    
    REQUIRED_COLUMNS = ["location_id", "latitude", "longitude", "demand", "priority"]
    
    @classmethod
    def validate_and_parse(cls, df: pd.DataFrame) -> Tuple[List[LocationNode], List[str]]:
        """
        Validates DataFrame and returns list of LocationNode objects and validation errors.
        
        Validation checks:
        1. Required columns presence
        2. Missing values
        3. Duplicate location IDs
        4. Valid latitude [-90, 90] and longitude [-180, 180]
        5. Non-negative demand
        6. Non-negative priority integer
        7. Exactly one DEPOT identified (case-insensitive 'depot' or demand==0 and priority==0)
        """
        errors: List[str] = []
        
        # Check required columns
        missing_cols = [c for c in cls.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return [], errors
            
        # Check null values
        if df[cls.REQUIRED_COLUMNS].isnull().any().any():
            null_cols = df[cls.REQUIRED_COLUMNS].columns[df[cls.REQUIRED_COLUMNS].isnull().any()].tolist()
            errors.append(f"Null values detected in columns: {', '.join(null_cols)}")
            
        # Check duplicates
        ids = df["location_id"].astype(str).str.strip()
        if ids.duplicated().any():
            dups = ids[ids.duplicated()].unique().tolist()
            errors.append(f"Duplicate location IDs found: {', '.join(dups)}")
            
        # Numeric checks
        for idx, row in df.iterrows():
            loc_id = str(row["location_id"]).strip()
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if not (-90.0 <= lat <= 90.0):
                    errors.append(f"Row '{loc_id}': Latitude {lat} out of range [-90, 90].")
                if not (-180.0 <= lon <= 180.0):
                    errors.append(f"Row '{loc_id}': Longitude {lon} out of range [-180, 180].")
            except (ValueError, TypeError):
                errors.append(f"Row '{loc_id}': Non-numeric coordinates detected.")
                
            try:
                dem = float(row["demand"])
                if dem < 0:
                    errors.append(f"Row '{loc_id}': Negative demand ({dem}) is not allowed.")
            except (ValueError, TypeError):
                errors.append(f"Row '{loc_id}': Non-numeric demand detected.")
                
            try:
                prio = int(row["priority"])
                if prio < 0:
                    errors.append(f"Row '{loc_id}': Negative priority ({prio}) is not allowed.")
            except (ValueError, TypeError):
                errors.append(f"Row '{loc_id}': Non-integer priority detected.")
                
        # Depot validation
        is_depot_mask = ids.str.upper().eq("DEPOT")
        if not is_depot_mask.any():
            # Check if any location has demand=0 and priority=0
            zero_mask = (df["demand"] == 0) & (df["priority"] == 0)
            if zero_mask.any():
                depot_idx = df[zero_mask].index[0]
                df.loc[depot_idx, "location_id"] = "DEPOT"
                is_depot_mask = df["location_id"].astype(str).str.upper().eq("DEPOT")
            else:
                errors.append("Missing DEPOT: Dataset must contain a row with location_id='DEPOT' or demand=0, priority=0.")
                
        if is_depot_mask.sum() > 1:
            errors.append(f"Multiple depots detected ({is_depot_mask.sum()}). Exactly one DEPOT is required.")
            
        if errors:
            return [], errors
            
        # Parse into LocationNode list, ensuring DEPOT is at index 0
        nodes: List[LocationNode] = []
        depot_node: Optional[LocationNode] = None
        
        for _, row in df.iterrows():
            loc_id = str(row["location_id"]).strip()
            is_depot = loc_id.upper() == "DEPOT"
            node = LocationNode(
                id=loc_id,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                demand=float(row["demand"]),
                priority=int(row["priority"]),
                is_depot=is_depot,
            )
            if is_depot:
                depot_node = node
            else:
                nodes.append(node)
                
        if depot_node is None:
            return [], ["Internal parsing error: Depot node not found."]
            
        # Place depot at index 0
        ordered_nodes = [depot_node] + nodes
        return ordered_nodes, []

    @classmethod
    def from_csv_file(cls, filepath_or_buffer: Union[str, io.StringIO, io.BytesIO]) -> Tuple[List[LocationNode], List[str]]:
        """Loads and validates dataset from a file path or in-memory buffer."""
        try:
            df = pd.read_csv(filepath_or_buffer)
            return cls.validate_and_parse(df)
        except Exception as e:
            return [], [f"Failed to read CSV: {str(e)}"]

    @classmethod
    def to_dataframe(cls, nodes: List[LocationNode]) -> pd.DataFrame:
        """Converts list of LocationNode objects back to a clean pandas DataFrame."""
        return pd.DataFrame([
            {
                "location_id": n.id,
                "latitude": n.latitude,
                "longitude": n.longitude,
                "demand": n.demand,
                "priority": n.priority,
                "is_depot": n.is_depot
            }
            for n in nodes
        ])
