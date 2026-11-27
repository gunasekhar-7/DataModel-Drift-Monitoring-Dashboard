"""Tests for batch generation and scenario reproducibility."""

import sys

sys.path.insert(0, '.')

from scripts.generate_batches import (
    generate_all_scenarios,
    generate_scenario_batch,
    load_reference_data,
    load_scenarios_yaml,
    setup_scenarios,
)


def test_scenarios_loaded():
    """Test that all 10 scenarios are loaded from YAML."""
    scenarios = load_scenarios_yaml()
    assert len(scenarios) == 10, f"Expected 10 scenarios, got {len(scenarios)}"


def test_scenario_types():
    """Test that all scenario types are registered."""
    from data.simulator.scenarios import SCENARIO_REGISTRY
    from scripts.generate_batches import load_scenarios_yaml

    # Need to register scenarios first
    scenarios = load_scenarios_yaml()
    setup_scenarios(scenarios, seed=42)

    assert len(SCENARIO_REGISTRY) == 10, f"Expected 10 registered scenarios, got {len(SCENARIO_REGISTRY)}"


def test_scenario_reproducibility():
    """Test that batches generated with the same seed are reproducible."""

    reference_df = load_reference_data(
        filepath='data/raw/Telco-Customer-Churn.csv'
    )

    # Generate two batches with the same scenario and seed
    batch1 = generate_scenario_batch(
        scenario_name='sudden_numerical_shift',
        batch_size=50,
        seed=42,
        reference_df=reference_df
    )
    batch2 = generate_scenario_batch(
        scenario_name='sudden_numerical_shift',
        batch_size=50,
        seed=42,
        reference_df=reference_df
    )

    # Both should have the same batch_id and row_count
    assert batch1['batch_id'] == batch2['batch_id'], "Batch IDs should match with same seed"
    assert batch1['row_count'] == batch2['row_count'], "Row counts should match with same seed"


def test_batch_generation():
    """Test that batches can be generated for all scenarios."""
    from scripts.generate_batches import load_scenarios_yaml

    reference_df = load_reference_data(
        filepath='data/raw/Telco-Customer-Churn.csv'
    )
    scenarios = load_scenarios_yaml()
    setup_scenarios(scenarios, seed=42)

    # Generate batches for all scenarios
    results = generate_all_scenarios(batch_size=50, reference_df=reference_df)

    # All scenarios should complete without error
    for scenario_name, result in results.items():
        assert 'error' not in result, f"Scenario {scenario_name} failed: {result['error']}"
        assert result['row_count'] == 50, f"Expected 50 rows, got {result['row_count']}"
