"""
QLDO-QAOA: Quantum Logistics Distribution Objective Variational Optimizer.
Drives QAOA circuit parameterization by minimizing J_QLDO:
J_QLDO(theta) = E[C] + alpha * Var(C) - beta * G - gamma * log(P_F + epsilon)
"""
from dataclasses import dataclass
import time
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_processing.loader import LocationNode
from quantum.hamiltonian import IsingHamiltonian
from quantum.circuit import QAOACircuitBuilder, CircuitMetrics
from quantum.simulator import QuantumSimulator, SimulationResult
from optimization.objective import RouteSolution
from optimization.route_decoder import RouteDecoder, DecodedQuantumState
from qldo.probability import ProbabilityDistribution
from qldo.feasibility import FeasibilityEngine
from qldo.concentration import ConcentrationEngine, GoodRouteDefinition
from qldo.objective import QLDOComponents, QLDOObjectiveEngine
from qldo.metrics import QLDOResearchMetrics, MetricsEvaluator


@dataclass
class QLDOIterationRecord:
    """Detailed telemetry record for a single optimization iteration."""
    iteration: int
    j_qldo: float
    expected_cost: float
    variance: float
    feasible_probability: float
    feasibility_barrier: float
    good_concentration: float
    gamma_params: List[float]
    beta_params: List[float]


@dataclass
class QLDOExecutionResult:
    """Complete summary of a QLDO-driven QAOA variational execution."""
    p: int
    optimal_gamma: List[float]
    optimal_beta: List[float]
    initial_qldo: float
    final_qldo: float
    final_components: QLDOComponents
    final_metrics: QLDOResearchMetrics
    iteration_history: pd.DataFrame
    best_feasible_solution: Optional[RouteSolution]
    best_repaired_solution: RouteSolution
    all_valid_states: List[DecodedQuantumState]
    decoded_states: List[DecodedQuantumState]
    simulation_result: SimulationResult
    circuit_metrics: CircuitMetrics
    circuit_ascii: str
    total_runtime_ms: float
    optimization_runtime_ms: float
    alpha: float
    beta: float
    gamma: float
    epsilon: float
    optimizer_name: str
    success: bool
    message: str


class QLDOQAOAOptimizer:
    """
    Executes QAOA parameter tuning using the Quantum Logistics Distribution Objective (QLDO).
    """

    @classmethod
    def optimize(
        cls,
        nodes: List[LocationNode],
        cost_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        hamiltonian: IsingHamiltonian,
        p: int = 1,
        optimizer_type: Literal["COBYLA", "Nelder-Mead", "BFGS"] = "COBYLA",
        max_iterations: int = 50,
        shots: int = 1024,
        alpha: float = 0.05,
        beta: float = 5.0,
        gamma: float = 5.0,
        epsilon: float = 1e-4,
        good_route_def: GoodRouteDefinition = GoodRouteDefinition.RELATIVE_BEST,
        percentile_q: float = 0.25,
        delta: float = 0.15,
        vehicle_capacity: float = 50.0,
        max_distance: float = 150.0,
        initial_params: Optional[np.ndarray] = None,
        seed: Optional[int] = 42,
        optimal_cost: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> QLDOExecutionResult:
        """
        Runs the full QLDO-QAOA closed-loop optimization:
        1. Evaluates parameterized ansatz circuit.
        2. Samples bitstring distribution via quantum simulator.
        3. Decodes candidate routes and evaluates physical feasibility.
        4. Calculates Expected Cost, Variance, Feasible Probability, Concentration, and Barrier.
        5. Classical optimizer updates angles to minimize J_QLDO.
        """
        start_total = time.perf_counter()
        simulator = QuantumSimulator(seed=seed)

        # 1. Parameter initialization
        if initial_params is not None:
            init_theta = np.array(initial_params, dtype=float)
        else:
            # Linear schedule ramp
            gammas_init = [0.1 * (l + 1) / p for l in range(p)]
            betas_init = [0.5 * (1.0 - (l / p)) for l in range(p)]
            init_theta = np.array(gammas_init + betas_init, dtype=float)

        history_records: List[QLDOIterationRecord] = []
        iter_counter = 0

        # Cache for bitstring costs across iterations to avoid re-evaluating unchanged bitstrings
        cost_cache: Dict[str, float] = {}
        feas_cache: Dict[str, bool] = {}

        def qldo_loss_function(theta: np.ndarray) -> float:
            nonlocal iter_counter
            iter_counter += 1
            curr_gamma = [float(g) for g in theta[:p]]
            curr_beta = [float(b) for b in theta[p:]]

            # Construct measurement circuit
            qc, _ = QAOACircuitBuilder.build_qaoa_circuit(
                hamiltonian=hamiltonian,
                p=p,
                gamma_params=curr_gamma,
                beta_params=curr_beta,
                include_measurements=True
            )

            # Sample counts
            sim_res = simulator.sample_shots(qc, shots=shots)
            probs = ProbabilityDistribution.counts_to_probabilities(sim_res.counts)

            # Fast evaluation of routes
            route_costs: Dict[str, float] = {}
            feasible_set: Set[str] = set()

            for bitstr in probs:
                if bitstr in cost_cache:
                    route_costs[bitstr] = cost_cache[bitstr]
                    if feas_cache.get(bitstr, False):
                        feasible_set.add(bitstr)
                else:
                    c_res = RouteDecoder.decode_single_bitstring(
                        bitstring=bitstr,
                        nodes=nodes,
                        distance_matrix=distance_matrix,
                        cost_matrix=cost_matrix,
                        vehicle_capacity=vehicle_capacity,
                        max_distance=max_distance
                    )
                    cost_val = c_res.strict_solution.total_cost if c_res.is_valid_permutation else (
                        c_res.repaired_solution.total_cost + 50.0  # soft penalty baseline
                    )
                    cost_cache[bitstr] = cost_val
                    is_feas = c_res.is_valid_permutation and (
                        c_res.strict_solution.is_feasible if c_res.strict_solution else False
                    )
                    feas_cache[bitstr] = is_feas
                    route_costs[bitstr] = cost_val
                    if is_feas:
                        feasible_set.add(bitstr)

            # Compute QLDO components
            components = QLDOObjectiveEngine.evaluate(
                counts_or_probs=probs,
                route_costs=route_costs,
                feasible_subspace=feasible_set,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epsilon=epsilon,
                good_route_def=good_route_def,
                percentile_q=percentile_q,
                delta=delta
            )

            # Track iteration history
            record = QLDOIterationRecord(
                iteration=iter_counter,
                j_qldo=round(components.total_qldo_objective, 4),
                expected_cost=round(components.expected_cost, 4),
                variance=round(components.variance, 4),
                feasible_probability=round(components.feasible_probability, 4),
                feasibility_barrier=round(components.feasibility_barrier, 4),
                good_concentration=round(components.good_concentration, 6),
                gamma_params=[round(g, 4) for g in curr_gamma],
                beta_params=[round(b, 4) for b in curr_beta]
            )
            history_records.append(record)

            if progress_callback:
                progress_callback(
                    iter_counter,
                    max_iterations,
                    f"Iteration {iter_counter}: J_QLDO={components.total_qldo_objective:.2f} (P_F={components.feasible_probability * 100:.1f}%)"
                )

            return components.total_qldo_objective

        # Initial evaluation
        initial_val = qldo_loss_function(init_theta)

        # Classical optimization
        opt_start = time.perf_counter()
        options = {"maxiter": max_iterations, "disp": False}
        if optimizer_type == "COBYLA":
            options["rhobeg"] = 0.5
            options["tol"] = 1e-3

        res = minimize(
            fun=qldo_loss_function,
            x0=init_theta,
            method=optimizer_type,
            options=options
        )
        opt_runtime_ms = (time.perf_counter() - opt_start) * 1000.0

        optimal_theta = res.x
        opt_gamma = [round(float(g), 4) for g in optimal_theta[:p]]
        opt_beta = [round(float(b), 4) for b in optimal_theta[p:]]

        # Final measurement circuit
        optimal_circuit, metrics = QAOACircuitBuilder.build_qaoa_circuit(
            hamiltonian=hamiltonian,
            p=p,
            gamma_params=opt_gamma,
            beta_params=opt_beta,
            include_measurements=True
        )
        circuit_ascii = QAOACircuitBuilder.draw_ascii_circuit(optimal_circuit)
        final_sim_result = simulator.sample_shots(optimal_circuit, shots=shots)

        # Full route decoding on final state
        final_decoded, final_summary = RouteDecoder.evaluate_measurement_counts(
            counts=final_sim_result.counts,
            nodes=nodes,
            distance_matrix=distance_matrix,
            cost_matrix=cost_matrix,
            vehicle_capacity=vehicle_capacity,
            max_distance=max_distance
        )

        final_probs = ProbabilityDistribution.counts_to_probabilities(final_sim_result.counts)
        final_costs = {s.bitstring: s.strict_solution.total_cost if s.is_valid_permutation else (s.repaired_solution.total_cost + 50.0) for s in final_decoded}
        final_feas_set = {s.bitstring for s in final_decoded if s.is_valid_permutation and (s.strict_solution.is_feasible if s.strict_solution else False)}

        final_components = QLDOObjectiveEngine.evaluate(
            counts_or_probs=final_probs,
            route_costs=final_costs,
            feasible_subspace=final_feas_set,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            good_route_def=good_route_def,
            percentile_q=percentile_q,
            delta=delta
        )

        good_routes = ConcentrationEngine.identify_good_routes(
            feasible_subspace=final_feas_set,
            route_costs=final_costs,
            definition=good_route_def,
            percentile_q=percentile_q,
            delta=delta
        )

        total_runtime_ms = (time.perf_counter() - start_total) * 1000.0

        # Multi-dimensional research metrics
        final_metrics = MetricsEvaluator.evaluate_metrics(
            probabilities=final_probs,
            route_costs=final_costs,
            feasible_subspace=final_feas_set,
            good_routes=good_routes,
            optimal_cost=optimal_cost,
            runtime_ms=total_runtime_ms,
            iterations=iter_counter,
            shots=shots,
            qubits=metrics.num_qubits,
            circuit_depth=metrics.circuit_depth
        )

        # Best solutions
        valid_states = [s for s in final_decoded if s.is_valid_permutation]
        best_feas = valid_states[0].strict_solution if valid_states else None
        if best_feas:
            best_feas.runtime_ms = round(total_runtime_ms, 2)
            best_feas.algorithm = "QLDO-QAOA (Strict)"

        best_repaired = final_decoded[0].repaired_solution
        best_repaired.runtime_ms = round(total_runtime_ms, 2)
        best_repaired.algorithm = "QLDO-QAOA (Repaired)"

        # Iteration history DataFrame
        df_history = pd.DataFrame([
            {
                "Iteration": r.iteration,
                "J_QLDO": r.j_qldo,
                "Expected Cost E[C]": r.expected_cost,
                "Variance Var(C)": r.variance,
                "Feasible Prob P_F": f"{r.feasible_probability * 100:.1f}%",
                "Barrier (-gamma*log)": r.feasibility_barrier,
                "Good Concentration G": r.good_concentration,
                "Gamma": str(r.gamma_params),
                "Beta": str(r.beta_params)
            }
            for r in history_records
        ])

        return QLDOExecutionResult(
            p=p,
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            initial_qldo=round(initial_val, 4),
            final_qldo=round(float(res.fun), 4),
            final_components=final_components,
            final_metrics=final_metrics,
            iteration_history=df_history,
            best_feasible_solution=best_feas,
            best_repaired_solution=best_repaired,
            all_valid_states=valid_states,
            decoded_states=final_decoded,
            simulation_result=final_sim_result,
            circuit_metrics=metrics,
            circuit_ascii=circuit_ascii,
            total_runtime_ms=round(total_runtime_ms, 2),
            optimization_runtime_ms=round(opt_runtime_ms, 2),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epsilon=epsilon,
            optimizer_name=optimizer_type,
            success=bool(res.success),
            message=str(res.message)
        )
