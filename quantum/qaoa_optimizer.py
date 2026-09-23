"""
QAOA Hybrid Variational Optimization Engine
"""
from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Literal, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from quantum.hamiltonian import IsingHamiltonian
from quantum.circuit import QAOACircuitBuilder, CircuitMetrics
from quantum.simulator import QuantumSimulator, SimulationResult


@dataclass
class QAOAExecutionResult:
    """Complete summary of a QAOA optimization and simulation execution."""
    p: int
    optimal_gamma: List[float]
    optimal_beta: List[float]
    final_energy: float
    initial_energy: float
    energy_history: List[float]
    iteration_count: int
    optimizer_name: str
    circuit_metrics: CircuitMetrics
    circuit_ascii: str
    simulation_result: SimulationResult
    optimization_runtime_ms: float
    total_runtime_ms: float
    success: bool
    message: str


class QAOAOptimizer:
    """Executes the hybrid classical-quantum QAOA parameter optimization loop."""
    
    @classmethod
    def optimize(
        cls,
        hamiltonian: IsingHamiltonian,
        p: int = 1,
        optimizer_type: Literal["COBYLA", "Nelder-Mead", "BFGS"] = "COBYLA",
        max_iterations: int = 60,
        shots: int = 1024,
        initial_params: Optional[np.ndarray] = None,
        seed: Optional[int] = 42,
        callback: Optional[Callable[[int, float], None]] = None
    ) -> QAOAExecutionResult:
        """
        Runs the full QAOA pipeline:
        1. Formulates parameterized ansatz circuit of depth p.
        2. Iteratively tunes (gamma, beta) using classical optimizer to minimize <H_C>.
        3. Measures final state with AerSimulator using configured shot count.
        """
        start_total = time.perf_counter()
        simulator = QuantumSimulator(seed=seed)
        
        # 1. Parameter initialization
        if initial_params is not None:
            init_theta = np.array(initial_params, dtype=float)
        else:
            # Heuristic standard initialization: gamma in [0, pi], beta in [0, pi/2]
            # Linear ramp schedule (T-LQAOA heuristic)
            gammas_init = [0.1 * (l + 1) / p for l in range(p)]
            betas_init = [0.5 * (1.0 - (l / p)) for l in range(p)]
            init_theta = np.array(gammas_init + betas_init, dtype=float)
            
        energy_history: List[float] = []
        iter_counter = 0
        
        # 2. Objective function for classical optimizer
        def cost_function(theta: np.ndarray) -> float:
            nonlocal iter_counter
            iter_counter += 1
            gamma = theta[:p]
            beta = theta[p:]
            
            # Construct bound circuit
            qc, _ = QAOACircuitBuilder.build_qaoa_circuit(
                hamiltonian=hamiltonian,
                p=p,
                gamma_params=gamma,
                beta_params=beta,
                include_measurements=False
            )
            
            # Compute exact statevector expectation value <H_C> + offset
            energy = simulator.evaluate_expectation_statevector(
                circuit=qc,
                cost_operator=hamiltonian.cost_operator,
                offset=hamiltonian.offset
            )
            energy_history.append(float(energy))
            
            if callback is not None:
                callback(iter_counter, energy)
                
            return energy
            
        initial_energy = cost_function(init_theta)
        
        # 3. Run classical optimization
        opt_start = time.perf_counter()
        options = {"maxiter": max_iterations, "disp": False}
        if optimizer_type == "COBYLA":
            options["rhobeg"] = 0.5
            options["tol"] = 1e-3
            
        res = minimize(
            fun=cost_function,
            x0=init_theta,
            method=optimizer_type,
            options=options
        )
        opt_runtime_ms = (time.perf_counter() - opt_start) * 1000.0
        
        optimal_theta = res.x
        opt_gamma = [round(float(g), 4) for g in optimal_theta[:p]]
        opt_beta = [round(float(b), 4) for b in optimal_theta[p:]]
        final_energy = float(res.fun)
        
        # 4. Build final measurement circuit with optimal angles
        optimal_circuit, metrics = QAOACircuitBuilder.build_qaoa_circuit(
            hamiltonian=hamiltonian,
            p=p,
            gamma_params=opt_gamma,
            beta_params=opt_beta,
            include_measurements=True
        )
        circuit_ascii = QAOACircuitBuilder.draw_ascii_circuit(optimal_circuit)
        
        # 5. Measure candidate bitstrings using AerSimulator
        sim_result = simulator.sample_shots(optimal_circuit, shots=shots)
        total_runtime_ms = (time.perf_counter() - start_total) * 1000.0
        
        return QAOAExecutionResult(
            p=p,
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            final_energy=round(final_energy, 4),
            initial_energy=round(initial_energy, 4),
            energy_history=energy_history,
            iteration_count=len(energy_history),
            optimizer_name=optimizer_type,
            circuit_metrics=metrics,
            circuit_ascii=circuit_ascii,
            simulation_result=sim_result,
            optimization_runtime_ms=round(opt_runtime_ms, 3),
            total_runtime_ms=round(total_runtime_ms, 3),
            success=bool(res.success),
            message=str(res.message)
        )
