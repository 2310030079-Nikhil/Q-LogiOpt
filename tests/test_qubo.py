"""
Unit Tests for QUBO Formulation and Energy Matching
"""
import pytest
import numpy as np
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from classical.exact_solver import ExactBruteForceSolver
from quantum.qubo import QuboBuilder


def test_qubo_dimension_and_variable_count():
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    qubo = QuboBuilder.build_qubo(nodes, dmat)
    # 3 customers -> 3^2 = 9 variables
    assert qubo.num_qubits == 9
    assert qubo.qubo_matrix.shape == (9, 9)
    assert len(qubo.variable_names) == 9


def test_qubo_energy_exact_match():
    """Verify that a feasible tour's QUBO energy exactly matches physical route cost."""
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    ex = ExactBruteForceSolver.solve(nodes, dmat)
    qubo = QuboBuilder.build_qubo(nodes, dmat)
    
    # Exact tour format: [0, c_1, c_2, c_3, 0]
    # Build binary vector x
    m = 3
    x_opt = np.zeros(m * m, dtype=int)
    for s, cust_node in enumerate(ex.route_indices[1:-1]):
        u = cust_node - 1  # 0-indexed customer
        k = u * m + s
        x_opt[k] = 1
        
    energy = QuboBuilder.evaluate_qubo_energy(x_opt, qubo)
    assert abs(energy - ex.total_distance) < 1e-3


def test_invalid_solution_penalized():
    """Verify that all-zero vector or duplicate visits incur high penalty."""
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    qubo = QuboBuilder.build_qubo(nodes, dmat, penalty_multiplier=2.0)
    
    zero_x = np.zeros(9, dtype=int)
    energy_zero = QuboBuilder.evaluate_qubo_energy(zero_x, qubo)
    
    # Must be heavily penalized (offset is 2 * m * P)
    assert energy_zero > 100.0
