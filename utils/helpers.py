"""
Utility functions for state management and formatting
"""
import streamlit as st
import numpy as np
from data_processing import DataLoader, SyntheticDataGenerator, DistanceMatrixCalculator
from classical import NearestNeighborSolver, TwoOptSolver, SimulatedAnnealingSolver, ExactBruteForceSolver
from quantum import QuboBuilder, HamiltonianConverter


def initialize_session_state():
    """Initializes default Streamlit session state objects if not already present."""
    if "nodes" not in st.session_state:
        # Default synthetic dataset: 4 locations (1 depot, 3 customers)
        st.session_state.nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
        
    if "distance_metric" not in st.session_state:
        st.session_state.distance_metric = "haversine"
        
    if "distance_weight" not in st.session_state:
        st.session_state.distance_weight = 1.0
        
    if "priority_weight" not in st.session_state:
        st.session_state.priority_weight = 0.5
        
    if "vehicle_capacity" not in st.session_state:
        st.session_state.vehicle_capacity = 50.0
        
    if "max_distance" not in st.session_state:
        st.session_state.max_distance = 150.0
        
    if "penalty_lambda" not in st.session_state:
        st.session_state.penalty_lambda = 100.0
        
    # Recompute distance and cost matrices if not present
    if "distance_matrix" not in st.session_state or "cost_matrix" not in st.session_state:
        nodes = st.session_state.nodes
        dmat, labels = DistanceMatrixCalculator.compute_distance_matrix(
            nodes, metric=st.session_state.distance_metric
        )
        cmat = DistanceMatrixCalculator.compute_cost_matrix(
            nodes, dmat,
            distance_weight=st.session_state.distance_weight,
            priority_weight=st.session_state.priority_weight
        )
        st.session_state.distance_matrix = dmat
        st.session_state.cost_matrix = cmat
        st.session_state.labels = labels
    
    # Store solutions dict
    if "solutions" not in st.session_state:
        st.session_state.solutions = {}
        
    # Store QUBO and QAOA execution
    if "qubo" not in st.session_state:
        nodes = st.session_state.nodes
        cmat = st.session_state.cost_matrix
        st.session_state.qubo = QuboBuilder.build_qubo(nodes, cmat, penalty_lambda=st.session_state.penalty_lambda)
        
    if "qaoa_result" not in st.session_state:
        st.session_state.qaoa_result = None

    if "aq_res" not in st.session_state:
        st.session_state.aq_res = None
        
    if "decoded_states" not in st.session_state:
        st.session_state.decoded_states = None
        
    if "quantum_summary" not in st.session_state:
        st.session_state.quantum_summary = None

    # Pre-populate classical and baseline quantum results on first boot
    if not st.session_state.solutions:
        run_all_classical()
        if len(st.session_state.nodes) <= 5:
            run_baseline_quantum()


def run_all_classical():
    """Runs all classical optimization algorithms on current session dataset."""
    nodes = st.session_state.nodes
    dmat = st.session_state.distance_matrix
    cmat = st.session_state.cost_matrix
    cap = st.session_state.vehicle_capacity
    max_d = st.session_state.max_distance
    
    nn_sol = NearestNeighborSolver.solve(nodes, dmat, cmat, cap, max_d)
    two_opt_sol = TwoOptSolver.solve(nodes, dmat, cmat, vehicle_capacity=cap, max_distance=max_d)
    sa_sol = SimulatedAnnealingSolver.solve(nodes, dmat, cmat, vehicle_capacity=cap, max_distance=max_d)
    
    if "solutions" not in st.session_state:
        st.session_state.solutions = {}
        
    st.session_state.solutions["Nearest Neighbor"] = nn_sol
    st.session_state.solutions["2-Opt"] = two_opt_sol
    st.session_state.solutions["Simulated Annealing"] = sa_sol
    
    if len(nodes) <= 8:
        exact_sol = ExactBruteForceSolver.solve(nodes, dmat, cmat, cap, max_d)
        st.session_state.solutions["Exact (Brute Force)"] = exact_sol


def run_baseline_quantum():
    """
    Executes a fast baseline quantum simulation (AQ-LogiQAOA) for current problem,
    ensuring QAOA results, strictly feasible routes, and CSR telemetry are populated.
    """
    try:
        from quantum.aq_logi_qaoa import AQLogiQAOAOptimizer
        nodes = st.session_state.nodes
        dmat = st.session_state.distance_matrix
        cmat = st.session_state.cost_matrix
        pen_lambda = st.session_state.penalty_lambda

        if len(nodes) <= 5:
            aq_res = AQLogiQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cmat,
                distance_matrix=dmat,
                p=1,
                initial_lambda=pen_lambda,
                adaptation_rate_eta=0.5,
                num_rounds=2,
                shots=512
            )
            st.session_state.aq_res = aq_res
            st.session_state.active_quantum_mode = "AQ-LogiQAOA"
            st.session_state.quantum_summary = {
                "constraint_satisfaction_rate": aq_res.final_csr_pct,
                "valid_count": len(aq_res.all_valid_states),
                "total_shots": 512,
                "best_cost": aq_res.best_repaired_solution.total_cost
            }
            if "solutions" not in st.session_state:
                st.session_state.solutions = {}
            if aq_res.best_feasible_solution:
                st.session_state.solutions["QAOA (Strict)"] = aq_res.best_feasible_solution
            st.session_state.solutions["QAOA (Repaired)"] = aq_res.best_repaired_solution
    except Exception as e:
        print(f"Baseline quantum run skipped: {e}")


def update_dataset_and_recalculate(new_nodes):
    """
    Updates the dataset stops and triggers complete recalculation of:
    1. Distance Matrix (Haversine/Euclidean)
    2. Augmented Cost Matrix
    3. QUBO Matrix (Anchored Depot)
    4. Classical Baselines (NN, 2-Opt, SA, Exact)
    5. Clears stale quantum results
    """
    st.session_state.nodes = new_nodes
    
    # 1. Distance & Cost Matrices
    dmat, labels = DistanceMatrixCalculator.compute_distance_matrix(
        new_nodes, metric=st.session_state.distance_metric
    )
    cmat = DistanceMatrixCalculator.compute_cost_matrix(
        new_nodes, dmat,
        distance_weight=st.session_state.distance_weight,
        priority_weight=st.session_state.priority_weight
    )
    st.session_state.distance_matrix = dmat
    st.session_state.cost_matrix = cmat
    st.session_state.labels = labels
    
    # 2. Update recommended penalty lambda if current is too low
    max_edge_cost = float(np.max(cmat))
    rec_lambda = max(10.0, round(max_edge_cost * 2.0, 1))
    if st.session_state.penalty_lambda < rec_lambda:
        st.session_state.penalty_lambda = rec_lambda
        
    # 3. Rebuild QUBO
    st.session_state.qubo = QuboBuilder.build_qubo(
        new_nodes, cmat, penalty_lambda=st.session_state.penalty_lambda
    )
    
    # 4. Clear stale quantum results
    st.session_state.qaoa_result = None
    st.session_state.aq_res = None
    st.session_state.decoded_states = None
    st.session_state.quantum_summary = None
    
    # 5. Re-run classical algorithms
    run_all_classical()
    
    # 6. Re-run quantum if problem is compact (<= 5 stops)
    if len(new_nodes) <= 5:
        run_baseline_quantum()


def update_configuration_and_recalculate(
    vehicle_capacity: float,
    max_distance: float,
    distance_metric: str,
    distance_weight: float,
    priority_weight: float,
    penalty_lambda: float
):
    """
    Updates configuration parameters and recalculates matrices and baseline solvers.
    """
    st.session_state.vehicle_capacity = float(vehicle_capacity)
    st.session_state.max_distance = float(max_distance)
    st.session_state.distance_metric = distance_metric
    st.session_state.distance_weight = float(distance_weight)
    st.session_state.priority_weight = float(priority_weight)
    st.session_state.penalty_lambda = float(penalty_lambda)
    
    nodes = st.session_state.nodes
    
    # 1. Recompute distance and cost matrices
    dmat, labels = DistanceMatrixCalculator.compute_distance_matrix(
        nodes, metric=distance_metric
    )
    cmat = DistanceMatrixCalculator.compute_cost_matrix(
        nodes, dmat,
        distance_weight=float(distance_weight),
        priority_weight=float(priority_weight)
    )
    st.session_state.distance_matrix = dmat
    st.session_state.cost_matrix = cmat
    st.session_state.labels = labels
    
    # 2. Rebuild QUBO with new penalty
    st.session_state.qubo = QuboBuilder.build_qubo(
        nodes, cmat, penalty_lambda=float(penalty_lambda)
    )
    
    # 3. Clear stale quantum results
    st.session_state.qaoa_result = None
    st.session_state.aq_res = None
    st.session_state.decoded_states = None
    st.session_state.quantum_summary = None
    
    # 4. Re-run classical algorithms with updated capacities and matrices
    run_all_classical()

    # 5. Re-run quantum if problem is compact (<= 5 stops)
    if len(nodes) <= 5:
        run_baseline_quantum()
