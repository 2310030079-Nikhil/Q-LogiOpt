"""
AQ-LogiQAOA: Adaptive Constraint-Aware Quantum Approximate Optimization Algorithm
for Intelligent Logistics Route and Delivery Optimization.
"""
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data_processing.loader import LocationNode
from quantum.qubo import QuboBuilder, QuboProblem
from quantum.hamiltonian import HamiltonianConverter, IsingHamiltonian
from quantum.circuit import QAOACircuitBuilder, CircuitMetrics
from quantum.simulator import QuantumSimulator, SimulationResult
from optimization.route_decoder import RouteDecoder, DecodedQuantumState
from optimization.objective import RouteSolution


@dataclass
class AdaptiveRoundRecord:
    """Telemetry record for a single outer adaptation round."""
    round_idx: int
    penalty_lambda: float
    energy: float
    feasibility_ratio_pct: float
    valid_shots_count: int
    best_feasible_distance: Optional[float]
    optimal_gamma: List[float]
    optimal_beta: List[float]


@dataclass
class AQLogiQAOAResult:
    """Complete results of the AQ-LogiQAOA execution."""
    p: int
    total_rounds: int
    initial_lambda: float
    final_lambda: float
    final_energy: float
    final_csr_pct: float
    best_feasible_solution: Optional[RouteSolution]
    best_repaired_solution: RouteSolution
    all_valid_states: List[DecodedQuantumState]
    decoded_states: List[DecodedQuantumState]
    simulation_result: SimulationResult
    circuit_metrics: CircuitMetrics
    circuit_ascii: str
    rounds_history: pd.DataFrame
    total_runtime_ms: float
    adaptation_rate_eta: float


class AQLogiQAOAOptimizer:
    """
    Implements AQ-LogiQAOA:
    A closed-loop adaptive framework that dynamically couples measured route feasibility (F_t)
    and solution quality (Q_t) to the constraint penalty parameter (lambda_{t+1}).
    
    Update rule:
    lambda_{t+1} = lambda_t * [1 + eta * (1 - F_t) - delta * F_t * Q_t]
    """
    
    @classmethod
    def optimize(
        cls,
        nodes: List[LocationNode],
        cost_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        p: int = 1,
        initial_lambda: float = 150.0,
        adaptation_rate_eta: float = 0.5,
        cooling_delta: float = 0.1,
        num_rounds: int = 4,
        inner_iterations: int = 20,
        shots: int = 1024,
        optimizer_type: str = "COBYLA",
        seed: Optional[int] = 42,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> AQLogiQAOAResult:
        """
        Executes the multi-round adaptive QAOA framework.
        """
        start_total = time.perf_counter()
        simulator = QuantumSimulator(seed=seed)
        
        current_lambda = float(initial_lambda)
        max_edge_cost = float(np.max(cost_matrix))
        lambda_min = max(10.0, max_edge_cost * 1.0)
        lambda_max = max(100.0, max_edge_cost * 15.0)
        
        rounds_data: List[AdaptiveRoundRecord] = []
        all_accumulated_valid: Dict[str, DecodedQuantumState] = {}
        
        best_overall_feasible: Optional[RouteSolution] = None
        best_overall_repaired: Optional[RouteSolution] = None
        
        # Determine baseline cost lower bound for Q_t calculation
        min_heuristic_cost = float(np.sum(np.min(cost_matrix + np.eye(len(nodes)) * 1e5, axis=1)))
        
        # Warm-starting initial angles
        current_gamma = [0.45 * (l + 1) / p for l in range(p)]
        current_beta = [0.75 * (1.0 - (l / p)) for l in range(p)]
        
        last_sim_result = None
        last_decoded = []
        last_metrics = None
        last_ascii = ""
        
        for r in range(1, num_rounds + 1):
            if progress_callback:
                progress_callback(
                    r - 1, num_rounds,
                    f"Round {r}/{num_rounds}: Simulating with adaptive λ = {current_lambda:.1f}..."
                )
                
            # 1. Build QUBO with current adaptive penalty lambda_t
            qubo = QuboBuilder.build_qubo(nodes, cost_matrix, penalty_lambda=current_lambda)
            ising = HamiltonianConverter.qubo_to_ising(qubo)
            
            # 2. Optimize variational angles (gamma, beta)
            def inner_objective(theta: np.ndarray) -> float:
                g = theta[:p]
                b = theta[p:]
                qc, _ = QAOACircuitBuilder.build_qaoa_circuit(
                    hamiltonian=ising, p=p, gamma_params=g, beta_params=b, include_measurements=False
                )
                return simulator.evaluate_expectation_statevector(
                    qc, ising.cost_operator, ising.offset
                )
                
            init_theta = np.array(current_gamma + current_beta, dtype=float)
            opt_res = minimize(
                inner_objective,
                x0=init_theta,
                method=optimizer_type,
                options={"maxiter": inner_iterations, "rhobeg": 0.25}
            )
            
            current_gamma = [round(float(g), 4) for g in opt_res.x[:p]]
            current_beta = [round(float(b), 4) for b in opt_res.x[p:]]
            round_energy = float(opt_res.fun)
            
            # 3. Construct measurement circuit and sample
            optimal_circuit, metrics = QAOACircuitBuilder.build_qaoa_circuit(
                hamiltonian=ising,
                p=p,
                gamma_params=current_gamma,
                beta_params=current_beta,
                include_measurements=True
            )
            last_metrics = metrics
            last_ascii = QAOACircuitBuilder.draw_ascii_circuit(optimal_circuit)
            
            sim_res = simulator.sample_shots(optimal_circuit, shots=shots)
            last_sim_result = sim_res
            
            # 4. Decode measurement bitstrings
            decoded_states, summary = RouteDecoder.evaluate_measurement_counts(
                counts=sim_res.counts,
                nodes=nodes,
                distance_matrix=distance_matrix,
                cost_matrix=cost_matrix
            )
            last_decoded = decoded_states
            
            # 5. Calculate Feasibility Ratio F_t
            csr_pct = summary["constraint_satisfaction_rate"]
            f_t = csr_pct / 100.0
            valid_shots_count = summary["valid_shots"]
            
            # 6. Extract valid permutation states
            round_valid = [s for s in decoded_states if s.is_valid_permutation]
            for s in round_valid:
                all_accumulated_valid[s.bitstring] = s
                
            round_best_dist = round_valid[0].strict_solution.total_distance if round_valid else None
            
            # Track overall best
            if round_valid:
                if best_overall_feasible is None or round_valid[0].strict_solution.total_distance < best_overall_feasible.total_distance:
                    best_overall_feasible = round_valid[0].strict_solution
                    best_overall_feasible.algorithm = "AQ-LogiQAOA (Strict)"
                    
            if best_overall_repaired is None or decoded_states[0].repaired_solution.total_distance < best_overall_repaired.total_distance:
                best_overall_repaired = decoded_states[0].repaired_solution
                best_overall_repaired.algorithm = "AQ-LogiQAOA (Repaired)"
                
            # 7. Calculate Route Quality Metric Q_t
            if round_valid and round_best_dist is not None and round_best_dist > 0:
                q_t = min(1.0, min_heuristic_cost / round_best_dist)
            else:
                q_t = 0.001
                
            # Record round telemetry
            rounds_data.append(AdaptiveRoundRecord(
                round_idx=r,
                penalty_lambda=round(current_lambda, 2),
                energy=round(round_energy, 2),
                feasibility_ratio_pct=round(csr_pct, 2),
                valid_shots_count=valid_shots_count,
                best_feasible_distance=round(round_best_dist, 2) if round_best_dist is not None else None,
                optimal_gamma=current_gamma,
                optimal_beta=current_beta
            ))
            
            # 8. Adaptive Penalty Update Rule
            # lambda_{t+1} = lambda_t * [1 + eta * (1 - F_t) - delta * F_t * Q_t]
            scaling_factor = 1.0 + adaptation_rate_eta * (1.0 - f_t) - cooling_delta * (f_t * q_t)
            next_lambda = current_lambda * scaling_factor
            current_lambda = float(np.clip(next_lambda, lambda_min, lambda_max))
            
        if progress_callback:
            progress_callback(num_rounds, num_rounds, "AQ-LogiQAOA optimization complete!")
            
        total_time_ms = (time.perf_counter() - start_total) * 1000.0
        if best_overall_feasible:
            best_overall_feasible.runtime_ms = round(total_time_ms, 2)
        if best_overall_repaired:
            best_overall_repaired.runtime_ms = round(total_time_ms, 2)
        
        # Format history dataframe
        df_history = pd.DataFrame([
            {
                "Round": rec.round_idx,
                "Penalty Lambda": rec.penalty_lambda,
                "Feasibility (CSR %)": f"{rec.feasibility_ratio_pct:.2f}%",
                "Valid Shots": f"{rec.valid_shots_count}/{shots}",
                "Best Feasible Dist (km)": f"{rec.best_feasible_distance:.2f}" if rec.best_feasible_distance else "None",
                "Energy <H_C>": rec.energy,
                "Optimal Gamma": str(rec.optimal_gamma),
                "Optimal Beta": str(rec.optimal_beta)
            }
            for rec in rounds_data
        ])
        
        return AQLogiQAOAResult(
            p=p,
            total_rounds=num_rounds,
            initial_lambda=initial_lambda,
            final_lambda=rounds_data[-1].penalty_lambda,
            final_energy=rounds_data[-1].energy,
            final_csr_pct=rounds_data[-1].feasibility_ratio_pct,
            best_feasible_solution=best_overall_feasible,
            best_repaired_solution=best_overall_repaired if best_overall_repaired else decoded_states[0].repaired_solution,
            all_valid_states=list(all_accumulated_valid.values()),
            decoded_states=last_decoded,
            simulation_result=last_sim_result,
            circuit_metrics=last_metrics,
            circuit_ascii=last_ascii,
            rounds_history=df_history,
            total_runtime_ms=round(total_time_ms, 2),
            adaptation_rate_eta=adaptation_rate_eta
        )
