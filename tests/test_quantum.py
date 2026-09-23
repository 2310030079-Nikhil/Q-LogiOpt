"""
Unit Tests for Quantum Ising Conversion, QAOA Circuit, Simulator, and Decoder
"""
import pytest
import numpy as np
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from quantum.qubo import QuboBuilder
from quantum.hamiltonian import HamiltonianConverter
from quantum.circuit import QAOACircuitBuilder
from quantum.simulator import QuantumSimulator
from quantum.qaoa_optimizer import QAOAOptimizer
from optimization.route_decoder import RouteDecoder


@pytest.fixture
def small_quantum_system():
    # 2 customers -> 4 qubits (fast execution in tests)
    nodes = SyntheticDataGenerator.generate(num_customers=2, seed=42)
    dmat, _ = DistanceMatrixCalculator.compute_distance_matrix(nodes)
    qubo = QuboBuilder.build_qubo(nodes, dmat)
    ising = HamiltonianConverter.qubo_to_ising(qubo)
    return nodes, dmat, qubo, ising


def test_ising_hamiltonian_properties(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    assert ising.num_qubits == 4
    assert ising.cost_operator.num_qubits == 4
    assert ising.mixer_operator.num_qubits == 4


def test_qaoa_circuit_construction(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    qc, metrics = QAOACircuitBuilder.build_qaoa_circuit(
        hamiltonian=ising,
        p=1,
        gamma_params=[0.5],
        beta_params=[0.2],
        include_measurements=True
    )
    assert qc.num_qubits == 4
    assert metrics.circuit_depth > 0
    assert metrics.two_qubit_gates > 0
    assert "rzz" in metrics.gate_counts or "cx" in metrics.gate_counts


def test_statevector_expectation_evaluation(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    qc, _ = QAOACircuitBuilder.build_qaoa_circuit(
        hamiltonian=ising,
        p=1,
        gamma_params=[0.5],
        beta_params=[0.2],
        include_measurements=False
    )
    sim = QuantumSimulator(seed=42)
    exp_val = sim.evaluate_expectation_statevector(qc, ising.cost_operator, ising.offset)
    assert isinstance(exp_val, float)
    assert exp_val > 0.0


def test_qaoa_optimizer_run(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    res = QAOAOptimizer.optimize(
        hamiltonian=ising,
        p=1,
        optimizer_type="COBYLA",
        max_iterations=15,
        shots=256,
        seed=42
    )
    assert len(res.optimal_gamma) == 1
    assert len(res.optimal_beta) == 1
    assert res.iteration_count > 0
    assert len(res.energy_history) > 0
    assert res.simulation_result.shots == 256


def test_route_decoder_strict_and_repaired(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    # In 2-customer setting (m=2, N=4):
    # Customer 0 at step 0, Customer 1 at step 1:
    # matrix = [[1, 0], [0, 1]] -> linear: u=0,s=0 -> k=0; u=1,s=1 -> k=3
    # Ordered bits: index 0 is '1', index 1 is '0', index 2 is '0', index 3 is '1' -> '1001'
    # In Qiskit endian (qubit 3 is leftmost): '1001'
    state = RouteDecoder.decode_single_bitstring(
        bitstring="1001",
        nodes=nodes,
        distance_matrix=dmat
    )
    assert state.is_valid_permutation is True
    assert state.strict_tour is not None
    assert state.repaired_tour is not None
    assert state.strict_tour == [0, 1, 2, 0]


def test_aq_logi_qaoa_optimizer(small_quantum_system):
    nodes, dmat, qubo, ising = small_quantum_system
    cmat = dmat.copy()
    from quantum.aq_logi_qaoa import AQLogiQAOAOptimizer
    res = AQLogiQAOAOptimizer.optimize(
        nodes=nodes,
        cost_matrix=cmat,
        distance_matrix=dmat,
        p=1,
        initial_lambda=100.0,
        num_rounds=2,
        inner_iterations=10,
        shots=256,
        seed=42
    )
    assert res.total_rounds == 2
    assert res.final_lambda >= 100.0
    assert len(res.rounds_history) == 2
    assert res.best_repaired_solution is not None
