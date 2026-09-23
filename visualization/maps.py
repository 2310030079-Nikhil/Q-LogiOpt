"""
Plotly Geographic Route Visualization for Logistics Networks
"""
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
from data_processing.loader import LocationNode
from optimization.objective import RouteSolution


class RouteMapVisualizer:
    """Generates interactive Plotly geographic maps for logistics deliveries."""
    
    PALETTE = {
        "depot": "#FFD700",             # Bright Gold
        "customer": "#00E5FF",          # Cyan
        "classical": "#FF9100",         # Orange
        "qaoa_strict": "#00E676",       # Bright Green
        "qaoa_repaired": "#B388FF",     # Lavender Purple
        "exact": "#2979FF",             # Electric Blue
        "background": "#0E1117"
    }

    @classmethod
    def plot_routes(
        cls,
        nodes: List[LocationNode],
        solutions: Dict[str, RouteSolution],
        active_routes: Optional[List[str]] = None,
        title: str = "Q-LogiOpt: Delivery Route Network Map"
    ) -> go.Figure:
        """
        Plots depot, customer nodes with priority/demand badges, and overlays
        selected routing paths.
        """
        fig = go.Figure()
        
        # 1. Plot routes as lines
        route_colors = {
            "Nearest Neighbor": "#FF9100",
            "2-Opt": "#FF3D71",
            "Simulated Annealing": "#FFAB00",
            "Exact (Brute Force)": "#2979FF",
            "QAOA (Strict)": "#00E676",
            "QAOA (Repaired)": "#B388FF",
            "QAOA": "#00E676"
        }
        
        # Plot route paths
        for name, sol in solutions.items():
            if active_routes is not None and name not in active_routes:
                continue
                
            color = route_colors.get(name, "#FFFFFF")
            route_lats = [nodes[idx].latitude for idx in sol.route_indices]
            route_lons = [nodes[idx].longitude for idx in sol.route_indices]
            
            # Hover text along route
            hover_text = [
                f"{name}<br>Stop {k}: {nodes[idx].id} (Demand: {nodes[idx].demand})"
                for k, idx in enumerate(sol.route_indices)
            ]
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=route_lats,
                    lon=route_lons,
                    mode="lines+markers",
                    line=dict(width=3.5, color=color),
                    marker=dict(size=6, color=color),
                    name=f"{name} ({sol.total_distance:.1f} km)",
                    text=hover_text,
                    hoverinfo="text"
                )
            )

        # 2. Plot customer nodes
        cust_nodes = [n for n in nodes if not n.is_depot]
        if cust_nodes:
            cust_lats = [n.latitude for n in cust_nodes]
            cust_lons = [n.longitude for n in cust_nodes]
            cust_ids = [n.id for n in cust_nodes]
            cust_demands = [n.demand for n in cust_nodes]
            cust_prios = [n.priority for n in cust_nodes]
            
            hover_cust = [
                f"<b>{n.id}</b><br>Demand: {n.demand} units<br>Priority: {n.priority}/5<br>Lat: {n.latitude:.4f}<br>Lon: {n.longitude:.4f}"
                for n in cust_nodes
            ]
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=cust_lats,
                    lon=cust_lons,
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color=cust_prios,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Priority", x=1.02, len=0.6)
                    ),
                    text=cust_ids,
                    textposition="top right",
                    textfont=dict(color="#111827", size=13, family="Arial Black"),
                    name="Customer Locations",
                    hovertext=hover_cust,
                    hoverinfo="text"
                )
            )

        # 3. Plot Depot separately with distinct star/large marker
        depot_node = next((n for n in nodes if n.is_depot), nodes[0])
        fig.add_trace(
            go.Scattermapbox(
                lat=[depot_node.latitude],
                lon=[depot_node.longitude],
                mode="markers+text",
                marker=dict(size=22, color="#E53935", symbol="star"),
                text=[f"★ {depot_node.id}"],
                textposition="bottom center",
                textfont=dict(color="#B71C1C", size=14, family="Arial Black"),
                name="Central Depot",
                hovertext=[f"<b>CENTRAL DEPOT</b><br>Lat: {depot_node.latitude:.4f}<br>Lon: {depot_node.longitude:.4f}"],
                hoverinfo="text"
            )
        )

        # Compute centroid for map center
        center_lat = np.mean([n.latitude for n in nodes])
        center_lon = np.mean([n.longitude for n in nodes])
        
        fig.update_layout(
            title="",
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=11.5
            ),
            margin=dict(l=10, r=10, t=10, b=40),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(color="#FAFAFA", size=11)
            )
        )
        
        return fig
