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
    def detect_column_mapping(cls, df: pd.DataFrame) -> dict:
        """
        Detects potential column mappings for standard logistics fields from any DataFrame.
        Returns a dictionary mapping standard fields to detected DataFrame column names (or None).
        """
        aliases = {
            "location_id": [
                "location_id", "location", "loc_id", "id", "stop_id", "stop", "name", 
                "customer", "customer_id", "station", "node", "point", "site", "address",
                "destination", "description", "category", "code", "label"
            ],
            "latitude": [
                "latitude", "lat", "lat_deg", "y", "coord_y", "coord_lat", "geo_lat", 
                "pickup_latitude", "dropoff_latitude", "start_lat", "end_lat"
            ],
            "longitude": [
                "longitude", "lon", "lng", "long", "x", "coord_x", "coord_lon", "geo_lon", 
                "pickup_longitude", "dropoff_longitude", "start_lon", "end_lon"
            ],
            "demand": [
                "demand", "weight", "weight_kg", "qty", "quantity", "load", "volume", 
                "packages", "orders", "boxes", "units", "size", "parcel_count", "capacity_used", "value"
            ],
            "priority": [
                "priority", "prio", "urgency", "level", "rank", "importance", "tier", 
                "service_level", "sla", "class"
            ]
        }
        
        mapping = {}
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        
        for std_field, alias_list in aliases.items():
            matched = None
            for alias in alias_list:
                if alias in cols_lower:
                    matched = cols_lower[alias]
                    break
            mapping[std_field] = matched
            
        return mapping

    @classmethod
    def try_auto_map_and_parse(cls, df: pd.DataFrame) -> Tuple[Optional[List[LocationNode]], dict, List[str]]:
        """
        Attempts to automatically map columns using fuzzy alias detection and parse valid nodes.
        Returns: (nodes_or_None, mapping_dict, errors_list)
        """
        mapping = cls.detect_column_mapping(df)
        
        # If latitude and longitude are missing, auto-mapping cannot produce coordinates directly
        if not mapping.get("latitude") or not mapping.get("longitude"):
            return None, mapping, ["Coordinates (latitude, longitude) could not be automatically identified."]
            
        # Build renamed copy
        rename_dict = {}
        for std_name, src_name in mapping.items():
            if src_name:
                rename_dict[src_name] = std_name
                
        mapped_df = df.rename(columns=rename_dict).copy()
        
        # Fill missing location_id with default labels if needed
        if "location_id" not in mapped_df.columns:
            mapped_df["location_id"] = [f"Stop_{i}" for i in range(len(mapped_df))]
            
        # Fill missing demand with default 5.0
        if "demand" not in mapped_df.columns:
            mapped_df["demand"] = 5.0
            
        # Fill missing priority with default 3
        if "priority" not in mapped_df.columns:
            mapped_df["priority"] = 3
            
        nodes, errors = cls.validate_and_parse(mapped_df)
        if not errors:
            return nodes, mapping, []
        return None, mapping, errors

    @classmethod
    def adapt_any_dataframe(
        cls,
        df: pd.DataFrame,
        num_stops: int = 4,
        depot_lat: float = 17.3850,
        depot_lon: float = 78.4867,
        radius_km: float = 8.0,
        id_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        demand_col: Optional[str] = None,
        priority_col: Optional[str] = None,
        seed: int = 42
    ) -> Tuple[List[LocationNode], List[str]]:
        """
        Universal CSV Adapter: Converts ANY uploaded DataFrame (even non-logistics or survey data)
        into a valid, calibrated Logistics Network suitable for QAOA and Classical solvers.
        """
        notes: List[str] = []
        n_stops = max(3, min(num_stops, len(df) if len(df) >= 3 else 3))
        np.random.seed(seed)
        
        # Take the first n_stops rows from df
        sub_df = df.iloc[:n_stops].copy() if len(df) >= n_stops else df.copy()
        while len(sub_df) < n_stops:
            sub_df = pd.concat([sub_df, df.iloc[:(n_stops - len(sub_df))]], ignore_index=True)
            
        nodes: List[LocationNode] = []
        
        # 1. Row 0 is ALWAYS the Depot
        depot_id = "DEPOT"
        if id_col and id_col in sub_df.columns:
            raw_depot_name = str(sub_df.iloc[0][id_col]).strip()[:15].replace(" ", "_")
            depot_id = f"DEPOT_{raw_depot_name}" if raw_depot_name else "DEPOT"
            
        depot = LocationNode(
            id=depot_id,
            latitude=round(depot_lat, 4),
            longitude=round(depot_lon, 4),
            demand=0.0,
            priority=0,
            is_depot=True
        )
        nodes.append(depot)
        notes.append(f"Designated stop 0 ('{depot_id}') as the central DEPOT at ({depot_lat:.4f}, {depot_lon:.4f}).")
        
        # 2. Rows 1 .. n_stops-1 are Customer Stops
        KM_PER_DEG_LAT = 111.0
        deg_lon_factor = KM_PER_DEG_LAT * np.cos(np.radians(depot_lat))
        
        for idx in range(1, n_stops):
            row = sub_df.iloc[idx]
            
            # Stop ID
            if id_col and id_col in sub_df.columns:
                raw_name = str(row[id_col]).strip()
                # Clean and truncate long strings
                clean_name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-"))[:12]
                stop_id = f"C{idx}_{clean_name}" if clean_name else f"C{idx}"
            else:
                stop_id = f"C{idx}"
                
            # Coordinates
            has_valid_coords = False
            if lat_col and lon_col and lat_col in sub_df.columns and lon_col in sub_df.columns:
                try:
                    c_lat = float(row[lat_col])
                    c_lon = float(row[lon_col])
                    if -90.0 <= c_lat <= 90.0 and -180.0 <= c_lon <= 180.0 and not (np.isnan(c_lat) or np.isnan(c_lon)):
                        has_valid_coords = True
                        stop_lat, stop_lon = c_lat, c_lon
                except (ValueError, TypeError):
                    has_valid_coords = False
                    
            if not has_valid_coords:
                # Synthesize clustered coordinate around depot
                angle = (2.0 * np.pi * (idx - 1) / (n_stops - 1)) + (np.random.uniform(-0.3, 0.3))
                dist = np.random.uniform(0.35, 0.95) * radius_km
                stop_lat = depot_lat + (dist * np.sin(angle)) / KM_PER_DEG_LAT
                stop_lon = depot_lon + (dist * np.cos(angle)) / deg_lon_factor
                
            # Demand
            stop_demand = float(np.random.randint(2, 9))
            if demand_col and demand_col in sub_df.columns:
                try:
                    raw_dem = float(row[demand_col])
                    if not np.isnan(raw_dem) and raw_dem > 0:
                        # Scale if too large (e.g. enterprise revenue in millions)
                        if raw_dem > 25.0:
                            stop_demand = round(2.0 + (raw_dem % 12.0), 1)
                        else:
                            stop_demand = round(raw_dem, 1)
                except (ValueError, TypeError):
                    pass
                    
            # Priority
            stop_priority = int(np.random.randint(1, 6))
            if priority_col and priority_col in sub_df.columns:
                try:
                    raw_prio = int(float(row[priority_col]))
                    if 1 <= raw_prio <= 5:
                        stop_priority = raw_prio
                    elif raw_prio > 5:
                        stop_priority = int(1 + (raw_prio % 5))
                except (ValueError, TypeError):
                    pass
                    
            node = LocationNode(
                id=stop_id,
                latitude=round(stop_lat, 4),
                longitude=round(stop_lon, 4),
                demand=stop_demand,
                priority=stop_priority,
                is_depot=False
            )
            nodes.append(node)
            
        notes.append(f"Successfully converted {n_stops - 1} records into customer delivery stops.")
        return nodes, notes

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
