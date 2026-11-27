"""Script for generating simulated production batches.

Generates batches from the reference Telco Churn dataset,
supporting scenario configurations for drift injection.
"""

import argparse
import logging
import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.core.config import settings
from data.simulator.scenarios import ScenarioConfig, generate_batch, register_scenario

# Set up logger
logger = logging.getLogger(__name__)


def load_reference_data(filepath: str = None) -> pd.DataFrame:
    """Load the reference dataset for batch generation.

    Args:
        filepath: Path to the reference CSV file. If None, uses the path from
            settings (which defaults to data/processed/processed_telco.csv).

    Returns:
        Reference DataFrame.
    """
    if filepath is None:
        filepath = settings.simulator_reference_data_path

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Reference data not found at {filepath}. "
            "Please set the SIMULATOR_REFERENCE_DATA_PATH environment variable "
            "or place the data at the expected location."
        )

    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def load_scenarios_yaml(filepath: str = None) -> dict[str, Any]:
    """Load scenario configurations from YAML file.

    Args:
        filepath: Path to the scenarios YAML file.

    Returns:
        Dictionary of scenario configurations.
    """
    if filepath is None:
        filepath = "data/simulator/scenarios.yaml"

    import yaml

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenarios configuration not found at {filepath}")

    with open(filepath, "r") as f:
        scenarios = yaml.safe_load(f)

    return scenarios


def setup_scenarios(scenarios: dict[str, Any], seed: int = 42) -> None:
    """Register all scenario configurations from YAML data.

    Args:
        scenarios: Dictionary of scenario configurations from YAML.
        seed: Base random seed for scenario reproducibility.
    """
    from data.simulator.scenarios import (
        CategoricalDistributionShift,
        ConceptDrift,
        GradualNumericalShift,
        MissingValueSpike,
        PredictionDistributionShift,
        SchemaChange,
        SeasonalShift,
        SuddenNumericalShift,
        UnexpectedCategoricalValues,
        UnitConversionError,
    )

    scenario_class_map = {
        "sudden_numerical_shift": SuddenNumericalShift,
        "gradual_numerical_shift": GradualNumericalShift,
        "categorical_distribution_shift": CategoricalDistributionShift,
        "missing_value_spike": MissingValueSpike,
        "schema_change": SchemaChange,
        "unexpected_categorical_values": UnexpectedCategoricalValues,
        "unit_conversion_error": UnitConversionError,
        "prediction_distribution_shift": PredictionDistributionShift,
        "seasonal_shift": SeasonalShift,
        "concept_drift": ConceptDrift,
    }

    for name, config in scenarios.items():
        scenario_type = config.get("scenario_type", "custom")
        scenario_class = scenario_class_map.get(scenario_type, ScenarioConfig)
        scenario_obj = scenario_class(
            name=name,
            description=config.get("description", ""),
            scenario_type=scenario_type,
            parameters=config,
            seed=seed,
        )
        register_scenario(scenario_obj)


def generate_scenario_batch(
    scenario_name: str,
    batch_size: int = 100,
    seed: int = 42,
    reference_df: pd.DataFrame = None,
) -> dict[str, Any]:
    """Generate a single batch with a specific scenario applied.

    Args:
        scenario_name: Name of the scenario to apply.
        batch_size: Number of rows in the batch.
        seed: Random seed for reproducibility.
        reference_df: Reference dataset to sample from.

    Returns:
        Batch dictionary with data and metadata.
    """
    if reference_df is None:
        reference_df = load_reference_data()

    # Register scenarios if not already done
    scenarios = load_scenarios_yaml()
    setup_scenarios(scenarios, seed=seed)

    # Generate the batch
    batch = generate_batch(
        reference_df=reference_df,
        scenario_name=scenario_name,
        batch_size=batch_size,
        seed=seed,
    )

    return batch


def generate_all_scenarios(
    batch_size: int = 100,
    seed: int = 42,
    reference_df: pd.DataFrame = None,
    output_dir: str = None,
) -> dict[str, dict[str, Any]]:
    """Generate batches for all scenarios.

    Args:
        batch_size: Number of rows per batch.
        seed: Base random seed.
        reference_df: Reference dataset.
        output_dir: Directory to save batch CSV files.

    Returns:
        Dictionary mapping scenario names to batch results.
    """
    if reference_df is None:
        reference_df = load_reference_data()

    if output_dir is None:
        output_dir = settings.simulator_batch_output_path

    os.makedirs(output_dir, exist_ok=True)

    scenarios = load_scenarios_yaml()
    setup_scenarios(scenarios, seed=seed)
    results = {}

    for scenario_name in scenarios.keys():
        try:
            batch = generate_scenario_batch(
                scenario_name=scenario_name,
                batch_size=batch_size,
                seed=seed,
                reference_df=reference_df,
            )

            # Save batch to CSV
            csv_path = os.path.join(output_dir, f"{scenario_name}.csv")
            batch["batch_data"].to_csv(csv_path, index=False)

            results[scenario_name] = {
                "batch_id": batch["batch_id"],
                "timestamp": batch["timestamp"],
                "scenario": batch["scenario"],
                "row_count": batch["row_count"],
                "output_path": csv_path,
            }

            logger.info(
                f"Generated scenario batch: {scenario_name}",
                extra={"batch_id": batch["batch_id"], "row_count": batch["row_count"]},
            )

        except Exception as e:
            logger.error(
                f"Failed to generate scenario {scenario_name}: {e!s}",
                extra={"scenario_name": scenario_name, "error": str(e)},
            )
            results[scenario_name] = {
                "error": str(e),
                "batch_id": None,
            }

    return results


def main():
    """CLI entry point for batch generation."""
    parser = argparse.ArgumentParser(
        description="Generate simulated production batches"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name to generate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows per batch (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default=None,
        help="Path to reference data CSV",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load reference data
    reference_df = load_reference_data(args.reference_data)

    if args.scenario == "all":
        # Generate all scenarios
        results = generate_all_scenarios(
            batch_size=args.batch_size,
            seed=args.seed,
            reference_df=reference_df,
        )

        print(f"\nGenerated {len(results)} scenario batches:\n")
        for scenario_name, result in results.items():
            if "error" not in result:
                print(
                    f"  {scenario_name}: "
                    f"batch_id={result['batch_id']}, "
                    f"rows={result['row_count']}, "
                    f"file={result['output_path']}"
                )
            else:
                print(f"  {scenario_name}: ERROR - {result['error']}")
    else:
        # Generate single scenario
        batch = generate_scenario_batch(
            scenario_name=args.scenario,
            batch_size=args.batch_size,
            seed=args.seed,
            reference_df=reference_df,
        )

        csv_path = os.path.join(
            settings.simulator_batch_output_path,
            f"{args.scenario}.csv",
        )
        batch["batch_data"].to_csv(csv_path, index=False)

        print(f"Scenario: {args.scenario}")
        print(f"Batch ID: {batch['batch_id']}")
        print(f"Timestamp: {batch['timestamp']}")
        print(f"Scenario: {batch['scenario']}")
        print(f"Row count: {batch['row_count']}")
        print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
