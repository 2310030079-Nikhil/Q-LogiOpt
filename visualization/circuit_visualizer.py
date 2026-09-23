"""
Quantum Circuit Metrics and Visualization Helpers
"""
from typing import Dict
import pandas as pd
from quantum.circuit import CircuitMetrics


class CircuitVisualizer:
    """Formats circuit schematics and resource metrics for dashboard display."""
    
    @staticmethod
    def get_metrics_dataframe(metrics: CircuitMetrics) -> pd.DataFrame:
        """Converts CircuitMetrics into a clean presentation DataFrame."""
        return pd.DataFrame([
            {"Metric": "Qubit Count (N)", "Value": str(metrics.num_qubits)},
            {"Metric": "Circuit Depth", "Value": str(metrics.circuit_depth)},
            {"Metric": "Total Gate Count", "Value": str(metrics.total_gates)},
            {"Metric": "Two-Qubit Entangling Gates (RZZ/CX)", "Value": str(metrics.two_qubit_gates)},
            {"Metric": "QAOA Layer Depth (p)", "Value": str(metrics.qaoa_depth_p)},
        ])

    @staticmethod
    def get_gate_distribution_dataframe(metrics: CircuitMetrics) -> pd.DataFrame:
        """Returns detailed breakdown of individual quantum gate types."""
        return pd.DataFrame([
            {"Gate Type": gate, "Count": count}
            for gate, count in metrics.gate_counts.items()
        ])
