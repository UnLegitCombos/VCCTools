import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def analyze_vlr_formula(current_csv="data/vlr_data.csv", previous_csv="data/old.csv"):
    """
    Analyzes VLR.gg rating formula by comparing current and previous data models.
    Outputs:
    - Model comparison (coefficients, R², MSE)
    - Correlation matrix
    - Full formula representation

    Args:
        current_csv: Path to the current/new CSV data
        previous_csv: Path to the previous/old CSV data
    """
    # Features and target
    feature_cols = [
        "kd",
        "adr",
        "kpr",
        "apr",
        "dpr",
        "fkpr",
        "fdpr",
        "cl_percent",
        "kast",
    ]
    target_col = "vlr_rating"

    # Load both datasets
    print(f"Loading data from {previous_csv} and {current_csv}...")

    previous_df = pd.read_csv(previous_csv).dropna(subset=feature_cols + [target_col])
    current_df = pd.read_csv(current_csv).dropna(subset=feature_cols + [target_col])

    print(f"Previous dataset: {len(previous_df)} rows")
    print(f"Current dataset: {len(current_df)} rows")

    # Fit Linear Models on full datasets (for formula extraction)
    print("\nFitting models on full datasets...")

    model_previous = LinearRegression().fit(
        previous_df[feature_cols], previous_df[target_col]
    )
    model_current = LinearRegression().fit(
        current_df[feature_cols], current_df[target_col]
    )

    # Calculate in-sample metrics (on full datasets)
    preds_previous_in = model_previous.predict(previous_df[feature_cols])
    preds_current_in = model_current.predict(current_df[feature_cols])

    r2_previous_in = r2_score(previous_df[target_col], preds_previous_in)
    mse_previous_in = mean_squared_error(previous_df[target_col], preds_previous_in)

    r2_current_in = r2_score(current_df[target_col], preds_current_in)
    mse_current_in = mean_squared_error(current_df[target_col], preds_current_in)

    print("\n--- In-Sample Model Performance ---")
    print(f"Previous Model: R² = {r2_previous_in:.4f}, MSE = {mse_previous_in:.4f}")
    print(f"Current Model: R² = {r2_current_in:.4f}, MSE = {mse_current_in:.4f}")

    # Create train/test split for out-of-sample evaluation
    print("\nCreating train/test splits for out-of-sample evaluation...")

    # Ensure we handle potential overlap between the two datasets
    print("Checking for overlap between datasets...")

    # Try to identify common records across datasets
    # This is a simple approach assuming the combination of these columns uniquely identifies a record
    id_columns = ["kd", "adr", "kpr", "apr", "dpr", "fkpr", "fdpr"]

    # Create temporary ID columns for comparison
    previous_df["temp_id"] = previous_df[id_columns].astype(str).agg("-".join, axis=1)
    current_df["temp_id"] = current_df[id_columns].astype(str).agg("-".join, axis=1)

    # Find common IDs and report overlap
    common_ids = set(previous_df["temp_id"]).intersection(set(current_df["temp_id"]))

    print(
        f"Found {len(common_ids)} overlapping records between datasets ({len(common_ids)/len(previous_df):.1%} of previous dataset)"
    )

    # Combine datasets for a shared test set, ensuring duplicates are removed
    combined_df = (
        pd.concat([previous_df, current_df])
        .drop_duplicates(subset=id_columns)
        .reset_index(drop=True)
    )

    # Remove temporary ID columns
    combined_df = combined_df.drop("temp_id", axis=1)

    # Split the combined dataset into train/test
    train_combined, test_combined = train_test_split(
        combined_df, test_size=0.2, random_state=42
    )

    # Train models on respective full datasets (not the combined training set)
    # Note: We're using the original datasets, not subsets of the combined dataset
    print("Training models on their respective full datasets...")

    # Train models on entire datasets (matches approach in testicolo.py)
    model_previous_full = LinearRegression().fit(
        previous_df[feature_cols], previous_df[target_col]
    )
    model_current_full = LinearRegression().fit(
        current_df[feature_cols], current_df[target_col]
    )

    # Test both models on the shared test set
    preds_previous_cv = model_previous_full.predict(test_combined[feature_cols])
    preds_current_cv = model_current_full.predict(test_combined[feature_cols])

    # Calculate metrics on shared test set
    r2_previous_cv = r2_score(test_combined[target_col], preds_previous_cv)
    mse_previous_cv = mean_squared_error(test_combined[target_col], preds_previous_cv)

    r2_current_cv = r2_score(test_combined[target_col], preds_current_cv)
    mse_current_cv = mean_squared_error(test_combined[target_col], preds_current_cv)

    print("\n--- Out-of-Sample Model Performance (Train/Test Split) ---")
    print(f"Previous Model: R² = {r2_previous_cv:.4f}, MSE = {mse_previous_cv:.4f}")
    print(f"Current Model: R² = {r2_current_cv:.4f}, MSE = {mse_current_cv:.4f}")

    # Display Coefficients Comparison
    print("\n--- Coefficients Comparison ---")
    coeff_df = pd.DataFrame(
        {
            "Features": feature_cols,
            "Previous Coefficients": model_previous_full.coef_,
            "Current Coefficients": model_current_full.coef_,
            "Difference": model_current_full.coef_ - model_previous_full.coef_,
        }
    )

    print(coeff_df)

    # Calculate and display correlation matrices
    print("\n--- Correlation Analysis ---")

    # Correlation matrix for current data
    corr_matrix = current_df[feature_cols + [target_col]].corr()

    print("Correlation Matrix for Current Dataset:")
    print(corr_matrix)

    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Features + Target")
    plt.tight_layout()
    plt.savefig("images/correlation_heatmap.png")
    print("Correlation heatmap saved as 'images/correlation_heatmap.png'")

    # Plotting Predicted vs Actual Ratings
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(test_combined[target_col], preds_previous_cv, alpha=0.6, color="blue")
    plt.plot(
        [test_combined[target_col].min(), test_combined[target_col].max()],
        [test_combined[target_col].min(), test_combined[target_col].max()],
        "r--",
    )
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Previous Model Predictions")

    plt.subplot(1, 2, 2)
    plt.scatter(test_combined[target_col], preds_current_cv, alpha=0.6, color="green")
    plt.plot(
        [test_combined[target_col].min(), test_combined[target_col].max()],
        [test_combined[target_col].min(), test_combined[target_col].max()],
        "r--",
    )
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Current Model Predictions")

    plt.tight_layout()
    plt.savefig("images/prediction_comparison.png")
    print("Prediction comparison plot saved as 'images/prediction_comparison.png'")

    # Generate full formula representation
    print("\n--- Full Formula Representation ---")
    intercept = model_current_full.intercept_
    coefs = model_current_full.coef_

    formula_str = f"rating = {intercept:.6f}"
    for feat, c in zip(feature_cols, coefs):
        formula_str += f" + ({c:.6f} * {feat})"

    print("Current Formula:")
    print(formula_str)

    # Create a prettier formatted version for spreadsheets
    print("\nSpreadsheet-Friendly Formula:")
    formula_parts = []
    for feat, c in zip(feature_cols, coefs):
        if c >= 0:
            formula_parts.append(f"+ {c:.6f} * {feat}")
        else:
            formula_parts.append(f"- {abs(c):.6f} * {feat}")

    sheet_formula = f"{intercept:.6f} " + " ".join(formula_parts)
    print(sheet_formula)


if __name__ == "__main__":
    analyze_vlr_formula("data/vlr_data.csv", "data/old.csv")
