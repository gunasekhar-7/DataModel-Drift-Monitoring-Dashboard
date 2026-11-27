"""Generate a synthetic Telco Churn dataset for the Data & Model Drift Monitoring Platform.

This script creates a synthetic dataset similar to the Telco Customer Churn dataset
if the original is not available. The dataset contains customer information for churn prediction.
"""

import os
import random

import numpy as np
import pandas as pd


def generate_synthetic_telco(n_rows: int = 7000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic Telco Customer Churn dataset.

    Args:
        n_rows: Number of rows to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Synthetic DataFrame with Telco-like features.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    data = []

    for _ in range(n_rows):
        # Customer ID
        customer_id = f"customer_{random.randint(10000, 99999)}"

        # Demographics
        gender = random.choice(["Male", "Female"])
        senior_citizen = random.choice([0, 1])
        partner = random.choice(["Yes", "No"])
        dependents = random.choice(["Yes", "No"])

        # Tenure (months)
        tenure = random.randint(0, 72)

        # Contract
        contract = random.choice(["Month-to-month", "One year", "Two year", "Year contract"])

        # Payment method
        payment_method = random.choice([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

        # Monthly charges
        monthly_charges = round(random.uniform(18.25, 118.75), 2)

        # Total charges
        total_charges = round(monthly_charges * tenure * random.uniform(0.8, 1.2), 2)

        # Churn target (binary: 0 = stays, 1 = leaves)
        # Churn is more likely with certain factors
        churn_prob = 0.15  # Base rate
        if contract == "Month-to-month":
            churn_prob += 0.25
        if tenure < 12:
            churn_prob += 0.15
        if gender == "Female":
            churn_prob += 0.02
        if random.random() < churn_prob:
            churn = 1
        else:
            churn = 0

        data.append({
            "customer_id": customer_id,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "contract": contract,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn
        })

    df = pd.DataFrame(data)

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Generate the dataset
    df = generate_synthetic_telco(n_rows=7000)

    # Save to data/raw - use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data", "raw")
    output_path = os.path.join(data_dir, "Telco-Customer-Churn.csv")
    output_path = os.path.abspath(output_path)

    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Synthetic Telco dataset generated with {len(df)} rows")
    print(f"Saved to: {output_path}")
    print("\nClass distribution:")
    print(df["churn"].value_counts())
    print(f"\nSample columns: {list(df.columns[:5])}")
