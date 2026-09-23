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


def test_auto_mapping_and_adaptation():
    """Verify DataLoader can adapt arbitrary non-logistics dataframes."""
    import pandas as pd
    
    # 1. Test arbitrary enterprise survey style dataframe (zero geographic columns)
    survey_df = pd.DataFrame({
        "Year": [2024, 2024, 2024, 2024, 2024],
        "Industry_name": ["Finance", "Manufacturing", "Retail", "Healthcare", "Agriculture"],
        "Value": [120.5, 85.0, 45.2, 90.1, 33.4],
        "Units": ["Dollars", "Dollars", "Dollars", "Dollars", "Dollars"]
    })
    
    adapted_nodes, notes = DataLoader.adapt_any_dataframe(
        df=survey_df,
        num_stops=4,
        id_col="Industry_name",
        demand_col="Value",
        depot_lat=17.3850,
        depot_lon=78.4867
    )
    assert len(adapted_nodes) == 4
    assert adapted_nodes[0].is_depot is True
    assert adapted_nodes[0].demand == 0.0
    for node in adapted_nodes[1:]:
        assert node.is_depot is False
        assert 1.0 <= node.demand <= 25.0
        assert 1 <= node.priority <= 5
        assert -90.0 <= node.latitude <= 90.0
        assert -180.0 <= node.longitude <= 180.0

    # 2. Test auto column alias mapping
    aliased_df = pd.DataFrame({
        "name": ["DEPOT", "ShopA", "ShopB"],
        "lat": [17.3850, 17.4000, 17.3700],
        "lon": [78.4867, 78.4800, 78.5000],
        "weight": [0, 5, 8],
        "urgency": [0, 3, 2]
    })
    mapped_nodes, mapping, errors = DataLoader.try_auto_map_and_parse(aliased_df)
    assert len(errors) == 0
    assert mapped_nodes is not None
    assert len(mapped_nodes) == 3
    assert mapped_nodes[0].id == "DEPOT"
