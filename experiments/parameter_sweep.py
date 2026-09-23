"""
QLDO Parameter Sweep Engine: Systematic Hyperparameter Sensitivity Analysis
Supports Sweeps for Alpha (variance), Beta (concentration), Gamma (barrier), Epsilon, Depth p, and Size n.
"""
from typing import Callable, Dict, List, Optional
import pandas as pd
import numpy as np

from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from quantum.qubo import QuboBuilder
from quantum.hamiltonian import HamiltonianConverter
from quantum.qaoa_qldo import QLDOQAOAOptimizer


class ParameterSweepRunner:
    """Automates hyperparameter sensitivity and ablation studies."""

    @classmethod
    def sweep_alpha(
        cls,
        nodes,
        cost_matrix,
        distance_matrix,
        hamiltonian,
        alpha_values: List[float] = [0.0, 0.01, 0.05, 0.1, 0.5],
        p: int = 1,
        shots: int = 512,
        max_iterations: int = 20,
        progress_cb: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Experiment 4: Studies the effect of variance regularization alpha."""
        results = []
        tot = len(alpha_values)
        for idx, a in enumerate(alpha_values):
            if progress_cb:
                progress_cb(idx + 1, tot, f"Evaluating Alpha = {a} ({idx+1}/{tot})...")
            res = QLDOQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cost_matrix,
                distance_matrix=distance_matrix,
                hamiltonian=hamiltonian,
                p=p,
                alpha=a,
                beta=5.0,
                gamma=5.0,
                max_iterations=max_iterations,
                shots=shots
            )
            m = res.final_metrics
            c = res.final_components
            results.append({
                "Alpha": a,
                "J_QLDO": round(res.final_qldo, 2),
                "Expected Cost E[C]": round(c.expected_cost, 2),
                "Cost Variance Var(C)": round(c.variance, 2),
                "Feasible Prob P_F": f"{c.feasible_probability * 100:.1f}%",
                "Best Route Dist": round(m.best_route_cost, 2),
                "Best Route Prob": f"{m.best_route_probability * 100:.2f}%",
                "Runtime (ms)": round(m.total_runtime_ms, 1)
            })
        return pd.DataFrame(results)

    @classmethod
    def sweep_beta(
        cls,
        nodes,
        cost_matrix,
        distance_matrix,
        hamiltonian,
        beta_values: List[float] = [0.0, 1.0, 5.0, 10.0],
        p: int = 1,
        shots: int = 512,
        max_iterations: int = 20,
        progress_cb: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Experiment 5: Studies the effect of good-route concentration beta."""
        results = []
        tot = len(beta_values)
        for idx, b in enumerate(beta_values):
            if progress_cb:
                progress_cb(idx + 1, tot, f"Evaluating Beta = {b} ({idx+1}/{tot})...")
            res = QLDOQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cost_matrix,
                distance_matrix=distance_matrix,
                hamiltonian=hamiltonian,
                p=p,
                alpha=0.05,
                beta=b,
                gamma=5.0,
                max_iterations=max_iterations,
                shots=shots
            )
            m = res.final_metrics
            c = res.final_components
            results.append({
                "Beta": b,
                "J_QLDO": round(res.final_qldo, 2),
                "Good Concentration G": round(c.good_concentration, 6),
                "Expected Cost E[C]": round(c.expected_cost, 2),
                "Feasible Prob P_F": f"{c.feasible_probability * 100:.1f}%",
                "Best Route Prob P(r*)": f"{m.best_route_probability * 100:.2f}%",
                "Best Route Dist": round(m.best_route_cost, 2)
            })
        return pd.DataFrame(results)

    @classmethod
    def sweep_gamma_epsilon(
        cls,
        nodes,
        cost_matrix,
        distance_matrix,
        hamiltonian,
        gamma_values: List[float] = [0.0, 1.0, 5.0, 10.0],
        epsilon_values: List[float] = [1e-2, 1e-4, 1e-6],
        p: int = 1,
        shots: int = 512,
        max_iterations: int = 20,
        progress_cb: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Experiment 6: Studies the logarithmic feasibility barrier parameters gamma and epsilon."""
        results = []
        pairs = [(g, e) for g in gamma_values for e in epsilon_values]
        tot = len(pairs)
        for idx, (g, e) in enumerate(pairs):
            if progress_cb:
                progress_cb(idx + 1, tot, f"Evaluating Gamma={g}, Epsilon={e:.0e} ({idx+1}/{tot})...")
            res = QLDOQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cost_matrix,
                distance_matrix=distance_matrix,
                hamiltonian=hamiltonian,
                p=p,
                alpha=0.05,
                beta=5.0,
                gamma=g,
                epsilon=e,
                max_iterations=max_iterations,
                shots=shots
            )
            c = res.final_components
            results.append({
                "Gamma": g,
                "Epsilon": f"{e:.0e}",
                "J_QLDO": round(res.final_qldo, 2),
                "Barrier Value": round(c.feasibility_barrier, 2),
                "Feasible Prob P_F": f"{c.feasible_probability * 100:.1f}%",
                "Best Route Dist": round(res.final_metrics.best_route_cost, 2)
            })
        return pd.DataFrame(results)

    @classmethod
    def sweep_depth_p(
        cls,
        nodes,
        cost_matrix,
        distance_matrix,
        hamiltonian,
        p_values: List[int] = [1, 2, 3],
        shots: int = 512,
        max_iterations: int = 20,
        progress_cb: Optional[Callable] = None
    ) -> pd.DataFrame:
        """Experiment 3: Studies circuit depth p scaling in QLDO."""
        results = []
        tot = len(p_values)
        for idx, p in enumerate(p_values):
            if progress_cb:
                progress_cb(idx + 1, tot, f"Evaluating Depth p={p} ({idx+1}/{tot})...")
            res = QLDOQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cost_matrix,
                distance_matrix=distance_matrix,
                hamiltonian=hamiltonian,
                p=p,
                alpha=0.05,
                beta=5.0,
                gamma=5.0,
                max_iterations=max_iterations,
                shots=shots
            )
            m = res.final_metrics
            results.append({
                "Depth p": p,
                "J_QLDO": round(res.final_qldo, 2),
                "Feasible Prob P_F": f"{m.feasible_probability * 100:.1f}%",
                "Best Route Prob": f"{m.best_route_probability * 100:.2f}%",
                "Best Route Dist": round(m.best_route_cost, 2),
                "Circuit Depth": m.circuit_depth,
                "Runtime (ms)": round(m.total_runtime_ms, 1)
            })
        return pd.DataFrame(results)
