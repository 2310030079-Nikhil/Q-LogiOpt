"""
Q-LogiOpt: Global Configuration and Default Parameters
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration parameters."""
    APP_NAME: str = "Q-LogiOpt"
    APP_SUBTITLE: str = "Quantum Approximate Optimization for Intelligent Logistics Route and Delivery Optimization"
    VERSION: str = "1.0.0"
    
    # Problem size limits for simulation feasibility on local hardware
    MIN_LOCATIONS: int = 3
    DEFAULT_LOCATIONS: int = 4
    MAX_LOCATIONS_SIMULATION: int = 8
    MAX_LOCATIONS_EXACT: int = 9
    
    # Quantum defaults
    DEFAULT_QAOA_DEPTH: int = 1
    MAX_QAOA_DEPTH: int = 3
    DEFAULT_SHOTS: int = 1024
    ALLOWED_SHOTS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    
    # Default optimization weights
    DEFAULT_DISTANCE_WEIGHT: float = 1.0
    DEFAULT_PRIORITY_WEIGHT: float = 0.5
    DEFAULT_CAPACITY_PENALTY: float = 10.0
    DEFAULT_PENALTY_LAMBDA: float = 100.0  # Automatic fallback if not computed from cost
    
    # UI styling
    PRIMARY_COLOR: str = "#00D4B2"     # Quantum Cyan
    SECONDARY_COLOR: str = "#7928CA"   # Quantum Violet
    BG_DARK: str = "#0E1117"
    TEXT_LIGHT: str = "#FAFAFA"
    SUCCESS_COLOR: str = "#00E676"
    WARNING_COLOR: str = "#FFB300"
    DANGER_COLOR: str = "#FF3D71"


CONFIG = AppConfig()
