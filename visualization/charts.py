"""
Analytical Plotly Charts for Quantum & Classical Benchmarking
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimization.objective import RouteSolution
from quantum.qubo import QuboProblem


class ChartVisualizer:
    """Generates analytical charts for optimization tracking and comparisons."""
    
    THEME_BG = "#0E1117"
    THEME_CARD = "#1A1D24"
    FONT_COLOR = "#FAFAFA"
    
    @classmethod
    def plot_convergence(
        cls,
        energy_history: List[float],
        title: str = "QAOA Variational Energy Convergence"
    ) -> go.Figure:
        """Plots the classical optimizer energy trajectory across iterations."""
        iterations = list(range(1, len(energy_history) + 1))
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=energy_history,
                mode="lines+markers",
                name="Energy <H_C>",
                line=dict(color="#00D4B2", width=3),
                marker=dict(size=6, color="#7928CA")
            )
        )
        
        # Highlight initial and minimum energy points
        min_idx = int(np.argmin(energy_history))
        fig.add_trace(
            go.Scatter(
                x=[iterations[min_idx]],
                y=[energy_history[min_idx]],
                mode="markers+text",
                name="Optimal Energy",
                marker=dict(size=12, color="#00E676", symbol="star"),
                text=[f"Min: {energy_history[min_idx]:.2f}"],
                textposition="bottom center",
                textfont=dict(color="#00E676", size=12)
            )
        )
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=cls.FONT_COLOR, size=16)),
            xaxis=dict(title="Optimizer Iteration", gridcolor="#2E3440", color=cls.FONT_COLOR),
            yaxis=dict(title="Expectation Value <H_C>", gridcolor="#2E3440", color=cls.FONT_COLOR),
            paper_bgcolor=cls.THEME_BG,
            plot_bgcolor=cls.THEME_CARD,
            font=dict(color=cls.FONT_COLOR),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig

    @classmethod
    def plot_energy_convergence(
        cls,
        energy_history: List[float],
        title: str = "QAOA Variational Energy Convergence"
    ) -> go.Figure:
        """Alias for plot_convergence."""
        return cls.plot_convergence(energy_history, title=title)

    @classmethod
    def plot_qubo_heatmap(cls, qubo: QuboProblem) -> go.Figure:
        """Visualizes the QUBO matrix as an interactive annotated heatmap."""
        matrix = qubo.qubo_matrix
        labels = qubo.variable_names
        
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Viridis",
                colorbar=dict(title="Weight", tickfont=dict(color=cls.FONT_COLOR)),
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"QUBO Matrix Q ({qubo.num_qubits}x{qubo.num_qubits} Qubits, λ={qubo.penalty_lambda:.1f})",
                font=dict(color=cls.FONT_COLOR, size=16)
            ),
            xaxis=dict(tickangle=-45, color=cls.FONT_COLOR),
            yaxis=dict(color=cls.FONT_COLOR, autorange="reversed"),
            paper_bgcolor=cls.THEME_BG,
            plot_bgcolor=cls.THEME_CARD,
            margin=dict(l=50, r=50, t=60, b=60)
        )
        return fig

    @classmethod
    def plot_measurement_distribution(
        cls,
        counts: Dict[str, int],
        top_k: int = 15,
        valid_bitstrings: Optional[set] = None,
        top_n: Optional[int] = None
    ) -> go.Figure:
        """
        Plots measurement probability distribution histogram, highlighting
        valid permutation bitstrings in green and invalid ones in red/purple.
        """
        k = top_n if top_n is not None else top_k
        total_shots = sum(counts.values())
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
        
        bitstrings = [item[0] for item in sorted_counts]
        probs = [(item[1] / total_shots) * 100.0 for item in sorted_counts]
        
        colors = []
        hover_info = []
        for b in bitstrings:
            is_valid = (valid_bitstrings is not None and b in valid_bitstrings)
            if is_valid:
                colors.append("#00E676")  # Green for valid
                hover_info.append("Status: FEASIBLE Permutation")
            else:
                colors.append("#FF3D71")  # Red for infeasible
                hover_info.append("Status: INFEASIBLE (Constraint Violation)")
                
        fig = go.Figure(
            data=[
                go.Bar(
                    x=bitstrings,
                    y=probs,
                    marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
                    text=[f"{p:.1f}%" for p in probs],
                    textposition="auto",
                    hovertext=hover_info
                )
            ]
        )
        
        fig.update_layout(
            title=dict(text=f"Measurement Distribution (Top {len(bitstrings)} Sampled Bitstrings)", font=dict(color=cls.FONT_COLOR, size=16)),
            xaxis=dict(title="Bitstring (Qubits N-1 ... 0)", tickangle=-45, color=cls.FONT_COLOR),
            yaxis=dict(title="Sampling Probability (%)", gridcolor="#2E3440", color=cls.FONT_COLOR),
            paper_bgcolor=cls.THEME_BG,
            plot_bgcolor=cls.THEME_CARD,
            font=dict(color=cls.FONT_COLOR),
            margin=dict(l=40, r=40, t=50, b=60)
        )
        return fig

    @classmethod
    def plot_algorithm_comparison(
        cls,
        solutions: Dict[str, RouteSolution]
    ) -> go.Figure:
        """Multi-metric comparison bar chart (Distance & Runtime ms)."""
        algos = list(solutions.keys())
        distances = [sol.total_distance for sol in solutions.values()]
        runtimes = [sol.runtime_ms for sol in solutions.values()]
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Route Distance (km)",
                x=algos,
                y=distances,
                marker_color="#00D4B2",
                text=[f"{d:.1f} km" for d in distances],
                textposition="auto"
            )
        )
        fig.add_trace(
            go.Bar(
                name="Runtime (ms)",
                x=algos,
                y=runtimes,
                marker_color="#7928CA",
                text=[f"{r:.1f} ms" for r in runtimes],
                textposition="auto"
            )
        )
        
        fig.update_layout(
            title=dict(text="Classical vs Quantum Solution Comparison", font=dict(color=cls.FONT_COLOR, size=16)),
            barmode="group",
            xaxis=dict(color=cls.FONT_COLOR),
            yaxis=dict(title="Value", gridcolor="#2E3440", color=cls.FONT_COLOR),
            paper_bgcolor=cls.THEME_BG,
            plot_bgcolor=cls.THEME_CARD,
            font=dict(color=cls.FONT_COLOR),
            legend=dict(font=dict(color=cls.FONT_COLOR)),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return fig
