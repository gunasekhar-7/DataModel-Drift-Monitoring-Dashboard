"""Production data simulator for generating simulated batches.

This module generates simulated daily production batches from a reference dataset,
preserving the reference data distribution while supporting configurable scenario
configurations for drift injection (Phase 2+).
"""

import hashlib
import random
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings


class ScenarioConfig:
    """Configuration for a single drift scenario."""

    def __init__(
        self,
        name: str,
        description: str,
        scenario_type: str,
        parameters: dict[str, Any],
        seed: int | None = None,
    ):
        self.name = name
        self.description = description
        self.scenario_type = scenario_type
        self.parameters = parameters
        self.seed = seed or int(time.time())
        self.rng = random.Random(self.seed)

    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply the scenario to a copy of the dataframe.

        Args:
            df: Input dataframe to apply scenario to.
            **kwargs: Additional arguments for specific scenarios.

        Returns:
            DataFrame with scenario applied.
        """
        raise NotImplementedError("Subclasses must implement apply method")


class SuddenNumericalShift(ScenarioConfig):
    """Scenario: Sudden numerical shift in a feature's distribution.

    Shifts the mean of a numeric feature by a specified amount immediately.
    """

    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")
        shift = self.parameters.get("shift_amount", 0.0)

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df[col] = df[col] + shift
        return df


class GradualNumericalShift(ScenarioConfig):
    """Scenario: Gradual numerical shift in a feature's distribution.

    Shifts the mean gradually over multiple batches.
    """

    def apply(
        self,
        df: pd.DataFrame,
        current_batch: int = 1,
        total_batches: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")
        total_shift = self.parameters.get("total_shift", 0.0)

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        incremental_shift = total_shift * (current_batch / total_batches)
        df[col] = df[col] + incremental_shift

        return df


class CategoricalDistributionShift(ScenarioConfig):
    """Scenario: Categorical distribution shift.

    Changes the distribution of a categorical feature, simulating
    new categories or changed proportions.
    """

    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")
        new_categories = self.parameters.get("new_categories", [])

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        if new_categories:
            category_map = {
                old: new_categories[i % len(new_categories)]
                for i, old in enumerate(df[col].unique())
            }
            df[col] = df[col].map(category_map)

        return df


class MissingValueSpike(ScenarioConfig):
    """Scenario: Missing value spike in a feature.

    Introduces a high percentage of missing values in a feature.
    """

    def apply(
        self, df: pd.DataFrame, spike_percentage: float | None = None, **kwargs
    ) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        spike_pct = spike_percentage or self.parameters.get("spike_percentage", 0.5)

        n_rows = len(df)
        n_missing = int(n_rows * spike_pct)

        missing_indices = self.rng.sample(range(n_rows), n_missing)
        df.iloc[missing_indices, df.columns.get_loc(col)] = None

        return df


class SchemaChange(ScenarioConfig):
    """Scenario: Schema change - adding/removing columns.

    Simulates a change in the feature schema, such as a column being
    added or removed from the data pipeline.
    """

    def apply(
        self,
        df: pd.DataFrame,
        add_column: bool = True,
        remove_column: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        df = df.copy()

        if remove_column:
            col = self.parameters.get("column")
            if col in df.columns:
                df = df.drop(columns=[col])

        if add_column:
            col = self.parameters.get("column")
            default_value = self.parameters.get("default_value", None)
            df[col] = default_value

        return df


class UnexpectedCategoricalValues(ScenarioConfig):
    """Scenario: Unexpected categorical values.

    Introduces categorical values not seen in the reference dataset,
    simulating data pipeline changes or user input errors.
    """

    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")
        new_values = self.parameters.get("new_values", [])

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        if new_values:
            n_rows = len(df)
            n_new = self.rng.randint(1, min(len(new_values), max(1, n_rows // 10)))

            replace_indices = self.rng.sample(range(n_rows), n_new)
            for idx in replace_indices:
                df.at[df.index[idx], col] = self.rng.choice(new_values)

        return df


class UnitConversionError(ScenarioConfig):
    """Scenario: Unit conversion error.

    Simulates a factor-of-10 or other unit conversion error in numeric data.
    """

    def apply(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")
        factor = self.parameters.get("factor", 1.0)

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        df[col] = df[col] * factor
        return df


class PredictionDistributionShift(ScenarioConfig):
    """Scenario: Prediction distribution shift.

    Changes the model's prediction distribution, simulating a change
    in the relationship between features and target.
    """

    def apply(
        self, df: pd.DataFrame, model_predictions: list[float] = None, **kwargs
    ) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column", "prediction")

        if col not in df.columns:
            df[col] = 0.0

        shift_factor = self.parameters.get("shift_factor", 0.1)

        if col in df.columns:
            df[col] = df[col] + shift_factor

        return df


class SeasonalShift(ScenarioConfig):
    """Scenario: Seasonal shift.

    Applies seasonal patterns to numeric features, simulating
    expected seasonal variation in the data.
    """

    def apply(
        self, df: pd.DataFrame, season_month: int | None = None, **kwargs
    ) -> pd.DataFrame:
        df = df.copy()
        col = self.parameters.get("column")

        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        if season_month is None:
            season_month = datetime.now(timezone.utc).month

        amplitude = self.parameters.get("amplitude", 0.1)
        period = self.parameters.get("period", 12)

        t = np.linspace(0, 2 * np.pi * season_month / period, len(df))
        seasonal_component = amplitude * np.sin(t)

        df[col] = df[col] + seasonal_component
        return df


class ConceptDrift(ScenarioConfig):
    """Scenario: Concept drift.

    Changes the relationship between features and the target variable,
    simulating a fundamental change in the data generation process.
    """

    def apply(
        self, df: pd.DataFrame, change_magnitude: float = 0.3, **kwargs
    ) -> pd.DataFrame:
        df = df.copy()
        target_col = self.parameters.get("target_column", "churn")

        if target_col not in df.columns:
            for col in df.columns:
                if col.lower() == str(target_col).lower():
                    target_col = col
                    break
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found in dataframe")

        n_rows = len(df)
        n_flip = int(n_rows * change_magnitude)

        flip_indices = self.rng.sample(range(n_rows), n_flip)
        target_loc = df.columns.get_loc(target_col)
        current_vals = df.iloc[flip_indices, target_loc]
        if pd.api.types.is_numeric_dtype(current_vals):
            df.iloc[flip_indices, target_loc] = (1 - current_vals).astype(int)
        else:
            df.iloc[flip_indices, target_loc] = current_vals.apply(
                lambda v: "No" if str(v).lower() in ["yes", "1"] else "Yes"
            )

        return df


# Scenario registry
SCENARIO_REGISTRY: dict[str, ScenarioConfig] = {}


def register_scenario(config: ScenarioConfig) -> None:
    """Register a scenario configuration."""
    SCENARIO_REGISTRY[config.name] = config


def get_scenario(name: str) -> ScenarioConfig | None:
    """Get a registered scenario by name."""
    return SCENARIO_REGISTRY.get(name)


def list_scenarios() -> dict[str, str]:
    """List all registered scenario names and descriptions."""
    return {name: config.description for name, config in SCENARIO_REGISTRY.items()}


def generate_batch(
    reference_df: pd.DataFrame,
    scenario_name: str,
    batch_size: int | None = None,
    batch_id: str | None = None,
    **scenario_kwargs,
) -> dict[str, Any]:
    """Generate a simulated production batch with optional scenario application."""
    if batch_size is None:
        batch_size = min(len(reference_df), 1000)

    if batch_id is None:
        seed_value = scenario_kwargs.get("seed", settings.simulator_random_seed)
        seed_str = f"{seed_value}-{scenario_name}-{batch_size}"
        batch_id = f"batch-{hashlib.md5(seed_str.encode()).hexdigest()[:8]}"

    seed_value = scenario_kwargs.pop("seed", settings.simulator_random_seed)
    model_predictions = scenario_kwargs.pop("model_predictions", None)
    if isinstance(seed_value, int):
        rng = np.random.default_rng(seed_value)
    else:
        rng = seed_value

    batch_df = reference_df.sample(n=batch_size, random_state=rng)

    scenario = None
    if scenario_name and scenario_name in SCENARIO_REGISTRY:
        scenario = SCENARIO_REGISTRY[scenario_name]
        batch_df = scenario.apply(
            batch_df, model_predictions=model_predictions, **scenario_kwargs
        )
    elif scenario_name:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    result = {
        "batch_id": batch_id,
        "timestamp": timestamp,
        "reference_dataset": settings.simulator_reference_data_path,
        "scenario": scenario_name or "none",
        "row_count": len(batch_df),
        "batch_data": batch_df,
    }

    if scenario:
        result["scenario_details"] = {
            "name": scenario.name,
            "type": scenario.scenario_type,
            "seed": scenario.seed,
        }

    return result
