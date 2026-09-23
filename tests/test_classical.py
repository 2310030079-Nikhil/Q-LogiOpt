"""
Unit Tests for Classical Optimization Solvers
"""
import pytest
import numpy as np
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from classical.nearest_neighbor import NearestNeighborSolver
from classical.two_opt import TwoOptSolver
from classical.simulated_annealing import SimulatedAnnealingSolver
from classical.exact_solver import ExactBruteForceSolver


@pytest.fixture
def sample_problem():
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    return nodes, dmat


def test_nearest_neighbor(sample_problem):
    nodes, dmat = sample_problem
    sol = NearestNeighborSolver.solve(nodes, dmat)
    assert sol.is_feasible is True
    assert sol.route_indices[0] == 0
    assert sol.route_indices[-1] == 0
    assert len(sol.route_indices) == len(nodes) + 1
    assert sol.total_distance > 0.0


def test_two_opt_improvement(sample_problem):
    nodes, dmat = sample_problem
    nn_sol = NearestNeighborSolver.solve(nodes, dmat)
    two_opt_sol = TwoOptSolver.solve(nodes, dmat)
    assert two_opt_sol.is_feasible is True
    assert two_opt_sol.total_distance <= nn_sol.total_distance + 1e-6


def test_exact_brute_force(sample_problem):
    nodes, dmat = sample_problem
    ex_sol = ExactBruteForceSolver.solve(nodes, dmat)
    nn_sol = NearestNeighborSolver.solve(nodes, dmat)
    two_opt_sol = TwoOptSolver.solve(nodes, dmat)
    sa_sol = SimulatedAnnealingSolver.solve(nodes, dmat)
    
    assert ex_sol.is_feasible is True
    # Global optimum must be <= any heuristic
    assert ex_sol.total_distance <= nn_sol.total_distance + 1e-6
    assert ex_sol.total_distance <= two_opt_sol.total_distance + 1e-6
    assert ex_sol.total_distance <= sa_sol.total_distance + 1e-6


def test_edge_case_two_locations():
    """Verify solver handles 1 Depot + 1 Customer gracefully."""
    nodes = SyntheticDataGenerator.generate(num_customers=1, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    sol = NearestNeighborSolver.solve(nodes, dmat)
    assert sol.is_feasible is True
    assert sol.route_indices == [0, 1, 0]
