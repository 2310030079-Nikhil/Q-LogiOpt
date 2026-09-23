"""
Unit tests for automated experiment runners (QLDO Benchmark, Parameter Sweeps, and Noise Experiments)
"""
import pytest
import numpy as np

from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from quantum.qubo import QuboBuilder
from quantum.hamiltonian import HamiltonianConverter
from experiments.qldo_benchmark import QLDOBenchmarkRunner
from experiments.parameter_sweep import ParameterSweepRunner
from experiments.noise_experiment import NoiseExperimentRunner


@pytest.fixture
def small_instance():
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dist_mat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    cost_mat = DistanceMatrixCalculator.compute_cost_matrix(nodes, dist_mat)
    qubo = QuboBuilder.build_qubo(nodes, cost_mat, penalty_lambda=50.0)
    ising = HamiltonianConverter.qubo_to_ising(qubo)
    return nodes, dist_mat, cost_mat, ising


def test_qldo_benchmark_head_to_head(small_instance):
    """Verifies that the head-to-head comparison runner executes without error and returns all required keys."""
    nodes, dist_mat, cost_mat, ising = small_instance
    record, std_res, qldo_res = QLDOBenchmarkRunner.run_head_to_head_comparison(
        nodes=nodes,
        cost_matrix=cost_mat,
        distance_matrix=dist_mat,
        p=1,
        max_iterations=4,
        shots=256,
        seed=42
    )
    result_dict = record.to_dict()
    assert "Ground Truth (C*)" in result_dict
    assert "Std QAOA Best" in result_dict
    assert "QLDO Best" in result_dict
    assert "Std Feasible (P_F)" in result_dict
    assert "QLDO Feasible (P_F)" in result_dict
    assert record.num_locations == 4
    assert record.qubits == 9


def test_parameter_sweep_alpha(small_instance):
    """Verifies hyperparameter sweep over alpha."""
    nodes, dist_mat, cost_mat, ising = small_instance
    df = ParameterSweepRunner.sweep_alpha(
        nodes=nodes,
        cost_matrix=cost_mat,
        distance_matrix=dist_mat,
        hamiltonian=ising,
        alpha_values=[0.0, 0.05],
        p=1,
        shots=128,
        max_iterations=3
    )
    assert len(df) == 2
    assert "J_QLDO" in df.columns
    assert "Cost Variance Var(C)" in df.columns


def test_noise_experiment_evaluation(small_instance):
    """Verifies NISQ noisy simulation runner."""
    nodes, dist_mat, cost_mat, ising = small_instance
    ideal, noisy = NoiseExperimentRunner.evaluate_circuit_under_noise(
        hamiltonian=ising,
        opt_gamma=[0.1],
        opt_beta=[0.4],
        nodes=nodes,
        distance_matrix=dist_mat,
        cost_matrix=cost_mat,
        shots=128,
        seed=42
    )
    assert "mode" in ideal and ideal["mode"] == "Ideal Simulator"
    assert "mode" in noisy and noisy["mode"] == "Noisy Simulator (NISQ)"
    assert 0.0 <= ideal["feasible_prob"] <= 1.0
    assert 0.0 <= noisy["feasible_prob"] <= 1.0
