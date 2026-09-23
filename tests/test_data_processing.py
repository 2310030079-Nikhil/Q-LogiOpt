"""
Unit Tests for Data Processing and Distance Calculation
"""
import pytest
import numpy as np
from data_processing.loader import DataLoader, LocationNode
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator


def test_haversine_distance_known():
    """Verify Haversine distance between known coordinates."""
    # Distance between New York (40.7128, -74.0060) and London (51.5074, -0.1278) ~ 5570 km
    d = DistanceMatrixCalculator.haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < d < 5650


def test_synthetic_generator_output():
    """Verify synthetic generator generates exact counts and valid nodes."""
    nodes = SyntheticDataGenerator.generate(num_customers=4, seed=123)
    assert len(nodes) == 5
    assert nodes[0].id == "DEPOT"
    assert nodes[0].is_depot is True
    assert nodes[0].demand == 0.0
    for node in nodes[1:]:
        assert node.is_depot is False
        assert node.demand >= 1.0
        assert 1 <= node.priority <= 5


def test_distance_matrix_symmetry_and_diagonal():
    """Verify distance matrix is symmetric with zero diagonal."""
    nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
    dmat, labels = DistanceMatrixCalculator.compute_distance_matrix(nodes, metric="haversine")
    assert dmat.shape == (4, 4)
    assert len(labels) == 4
    for i in range(4):
        assert dmat[i, i] == 0.0
        for j in range(4):
            assert dmat[i, j] == dmat[j, i]
            assert dmat[i, j] >= 0.0


def test_loader_validation_errors():
    """Verify DataLoader catches invalid schemas."""
    import pandas as pd
    
    # Missing columns
    bad_df = pd.DataFrame({"latitude": [10.0], "longitude": [20.0]})
    nodes, errors = DataLoader.validate_and_parse(bad_df)
    assert len(errors) > 0
    assert "Missing required columns" in errors[0]
    
    # Missing depot
    no_depot_df = pd.DataFrame({
        "location_id": ["C1", "C2"],
        "latitude": [17.1, 17.2],
        "longitude": [78.1, 78.2],
        "demand": [5, 8],
        "priority": [2, 3]
    })
    nodes, errors = DataLoader.validate_and_parse(no_depot_df)
    assert any("Missing DEPOT" in e for e in errors)
