"""
Unit and Integration Tests for QLDO (Quantum Logistics Distribution Objective) Engine
"""
import pytest
import numpy as np
from data_processing import SyntheticDataGenerator, DistanceMatrixCalculator
from quantum import QuboBuilder, HamiltonianConverter
from qldo import (
    ProbabilityDistribution,
    FeasibilityEngine,
    ConcentrationEngine,
    GoodRouteDefinition,
    QLDOObjectiveEngine,
    MetricsEvaluator
)
from quantum.qaoa_qldo import QLDOQAOAOptimizer


def test_probability_normalization():
    counts = {"000": 150, "001": 250, "010": 300, "111": 300}
    probs = ProbabilityDistribution.counts_to_probabilities(counts)
    total_p = sum(probs.values())
    assert np.isclose(total_p, 1.0, atol=1e-6)
    for p in probs.values():
        assert 0.0 <= p <= 1.0


def test_moments_calculation():
    probs = {"A": 0.5, "B": 0.5}
    costs = {"A": 10.0, "B": 30.0}
    exp_c, var_c, std_c = ProbabilityDistribution.compute_moments(probs, costs)
    # E[C] = 0.5*10 + 0.5*30 = 20.0
    assert np.isclose(exp_c, 20.0)
    # Var(C) = 0.5*(10-20)^2 + 0.5*(30-20)^2 = 0.5*100 + 0.5*100 = 100.0
    assert np.isclose(var_c, 100.0)
    assert np.isclose(std_c, 10.0)
    assert var_c >= 0.0


def test_feasibility_probability_and_barrier():
    probs = {"r1": 0.3, "r2": 0.4, "r3": 0.3}
    feas = {"r1", "r2"}
    p_f = FeasibilityEngine.compute_feasible_probability(probs, feas)
    assert np.isclose(p_f, 0.7)
    
    # Barrier properties
    b_high = FeasibilityEngine.compute_logarithmic_barrier(p_f=0.0, gamma=5.0, epsilon=1e-4)
    b_med = FeasibilityEngine.compute_logarithmic_barrier(p_f=0.5, gamma=5.0, epsilon=1e-4)
    b_low = FeasibilityEngine.compute_logarithmic_barrier(p_f=1.0, gamma=5.0, epsilon=1e-4)
    
    # Monotonic decrease: higher feasibility -> lower penalty barrier
    assert b_high > b_med > b_low
    assert np.isclose(b_high, -5.0 * np.log(1e-4))


def test_good_route_definitions_and_concentration():
    feas = {"r1", "r2", "r3"}
    costs = {"r1": 20.0, "r2": 22.0, "r3": 35.0}
    
    # Definition B: Within 15% of best (20 * 1.15 = 23.0) -> should include r1 and r2
    g_set_b = ConcentrationEngine.identify_good_routes(
        feas, costs, definition=GoodRouteDefinition.RELATIVE_BEST, delta=0.15
    )
    assert "r1" in g_set_b
    assert "r2" in g_set_b
    assert "r3" not in g_set_b
    
    # Concentration test: peaked vs diffuse
    peaked_probs = {"r1": 0.8, "r2": 0.1, "r3": 0.1}
    diffuse_probs = {"r1": 0.3, "r2": 0.3, "r3": 0.4}
    
    g_peaked = ConcentrationEngine.compute_concentration_metric(peaked_probs, g_set_b)
    g_diffuse = ConcentrationEngine.compute_concentration_metric(diffuse_probs, g_set_b)
    
    # Peaked: 0.8^2 + 0.1^2 = 0.65
    # Diffuse: 0.3^2 + 0.3^2 = 0.18
    assert g_peaked > g_diffuse
    assert g_peaked >= 0.0


def test_qldo_objective_exact_decomposition():
    probs = {"r1": 0.7, "r2": 0.2, "r3": 0.1}
    costs = {"r1": 15.0, "r2": 25.0, "r3": 50.0}
    feas = {"r1", "r2"}
    
    alpha = 0.1
    beta = 2.0
    gamma = 3.0
    epsilon = 1e-4
    
    comp = QLDOObjectiveEngine.evaluate(
        counts_or_probs=probs,
        route_costs=costs,
        feasible_subspace=feas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        epsilon=epsilon
    )
    
    # Verify manual reconstruction
    expected_j = (
        comp.expected_cost +
        (alpha * comp.variance) -
        (beta * comp.good_concentration) +
        comp.feasibility_barrier
    )
    assert np.isclose(comp.total_qldo_objective, expected_j, atol=1e-5)


def test_end_to_end_qldo_qaoa_optimization():
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes, metric="haversine")
    cmat = DistanceMatrixCalculator.compute_cost_matrix(nodes, dmat)
    qubo = QuboBuilder.build_qubo(nodes, cmat, penalty_lambda=100.0)
    ising = HamiltonianConverter.qubo_to_ising(qubo)
    
    res = QLDOQAOAOptimizer.optimize(
        nodes=nodes,
        cost_matrix=cmat,
        distance_matrix=dmat,
        hamiltonian=ising,
        p=1,
        optimizer_type="COBYLA",
        max_iterations=10,
        shots=256,
        alpha=0.05,
        beta=5.0,
        gamma=5.0
    )
    
    assert res.p == 1
    assert len(res.optimal_gamma) == 1
    assert len(res.optimal_beta) == 1
    assert res.final_components.feasible_probability >= 0.0
    assert res.final_metrics.best_route_cost > 0.0
    assert len(res.iteration_history) > 0
    assert res.best_repaired_solution is not None
