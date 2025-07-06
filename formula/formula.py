import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def analyze_vlr_formula(current_csv="data/vlr_data.csv", previous_csv="data/old.csv"):
    """
    Analyze and compare VLR.gg rating formulas using current and previous datasets.
    Outputs concise model metrics, coefficients, and feature impact summaries.
    """
    # Features and target
    feature_cols = [
        "adr",
        "kpr",
        "apr",
        "dpr",
        "kast",
        "fkpr",
        "fdpr",
        "cl_percent",
    ]
    target_col = "vlr_rating"

    # Load both datasets
    previous_df = pd.read_csv(previous_csv).dropna(subset=feature_cols + [target_col])
    current_df = pd.read_csv(current_csv).dropna(subset=feature_cols + [target_col])

    # Fit Linear Models on full datasets (for formula extraction)

    model_previous = LinearRegression().fit(
        previous_df[feature_cols], previous_df[target_col]
    )
    model_current = LinearRegression().fit(
        current_df[feature_cols], current_df[target_col]
    )

    # Fit constrained models with equal and opposite FKPR/FDPR coefficients

    def fit_constrained_model(df, feature_cols, target_col):
        """Fit model with FKPR and FDPR having equal and opposite coefficients"""
        df_temp = df.copy()
        # Create net first engagement feature (FKPR - FDPR)
        df_temp["net_first_engagement"] = df_temp["fkpr"] - df_temp["fdpr"]

        # Use all features except individual FKPR/FDPR
        constrained_features = [
            f for f in feature_cols if f not in ["fkpr", "fdpr"]
        ] + ["net_first_engagement"]

        model = LinearRegression().fit(
            df_temp[constrained_features], df_temp[target_col]
        )

        # Extract coefficient for net first engagement
        net_coef_idx = constrained_features.index("net_first_engagement")
        fkpr_coef = model.coef_[net_coef_idx]
        fdpr_coef = -fkpr_coef

        # Build full coefficient array
        full_coefs = []
        full_intercept = model.intercept_

        for feature in feature_cols:
            if feature == "fkpr":
                full_coefs.append(fkpr_coef)
            elif feature == "fdpr":
                full_coefs.append(fdpr_coef)
            else:
                idx = constrained_features.index(feature)
                full_coefs.append(model.coef_[idx])

        return full_coefs, full_intercept, model

    def fit_custom_constrained_model(
        df, feature_cols, target_col, custom_constraints=None
    ):
        """
        Fit model with custom constraints on feature coefficients

        Args:
            df: DataFrame with data
            feature_cols: List of feature column names
            target_col: Target column name
            custom_constraints: Dict with feature names as keys and desired coefficients as values
                               e.g., {'adr': 0.005, 'fkpr': 0.2, 'fdpr': -0.2}

        Returns:
            full_coefs: List of coefficients for all features
            full_intercept: Model intercept
            adjusted_r2: R² of the constrained model
        """
        if custom_constraints is None:
            custom_constraints = {}

        # Start with the original target
        adjusted_y = df[target_col].values.copy()

        # Subtract the contribution of constrained features
        for feature, coef in custom_constraints.items():
            if feature in feature_cols:
                adjusted_y -= coef * df[feature].values

        # Get remaining features to fit
        remaining_features = [
            f for f in feature_cols if f not in custom_constraints.keys()
        ]

        if len(remaining_features) == 0:
            # All features are constrained, just return the constraints
            full_coefs = [custom_constraints.get(f, 0.0) for f in feature_cols]

            # Calculate intercept as mean of adjusted target
            full_intercept = adjusted_y.mean()

            # Calculate R²
            predictions = full_intercept + sum(
                custom_constraints[f] * df[f] for f in custom_constraints.keys()
            )
            adjusted_r2 = r2_score(df[target_col], predictions)

        else:
            # Fit model on remaining features
            X_remaining = df[remaining_features]
            model = LinearRegression().fit(X_remaining, adjusted_y)

            # Build full coefficient array
            full_coefs = []
            full_intercept = model.intercept_

            for feature in feature_cols:
                if feature in custom_constraints:
                    full_coefs.append(custom_constraints[feature])
                else:
                    idx = remaining_features.index(feature)
                    full_coefs.append(model.coef_[idx])

            # Calculate R² for the full constrained model
            predictions = full_intercept
            for i, feature in enumerate(feature_cols):
                predictions += full_coefs[i] * df[feature].values

            adjusted_r2 = r2_score(df[target_col], predictions)

        return full_coefs, full_intercept, adjusted_r2

    # Fit standard constrained models (FKPR = -FDPR)
    constrained_coefs_prev, constrained_intercept_prev, _ = fit_constrained_model(
        previous_df, feature_cols, target_col
    )
    constrained_coefs_curr, constrained_intercept_curr, _ = fit_constrained_model(
        current_df, feature_cols, target_col
    )

    # Fit custom constrained models with boosted ADR importance

    # Test different ADR coefficient values
    adr_boost_options = [0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.0030]
    custom_results = []

    for adr_coef in adr_boost_options:
        # Only constrain ADR, let FKPR/FDPR be optimized by the constrained model
        custom_constraints = {
            "adr": adr_coef,
        }

        # First get the constrained model coefficients (FKPR = -FDPR)
        constrained_coefs_temp, constrained_intercept_temp, _ = fit_constrained_model(
            current_df, feature_cols, target_col
        )
        
        # Then apply ADR constraint while keeping FKPR = -FDPR constraint
        custom_coefs, custom_intercept, custom_r2 = fit_custom_constrained_model(
            current_df, feature_cols, target_col, custom_constraints
        )
        
        # Manually enforce FKPR = -FDPR constraint on the result
        fkpr_idx = feature_cols.index("fkpr")
        fdpr_idx = feature_cols.index("fdpr")
        
        # Calculate the average magnitude and apply opposite signs
        avg_magnitude = (abs(custom_coefs[fkpr_idx]) + abs(custom_coefs[fdpr_idx])) / 2
        custom_coefs[fkpr_idx] = avg_magnitude
        custom_coefs[fdpr_idx] = -avg_magnitude
        
        # Recalculate R² with the enforced constraint
        custom_pred_temp = custom_intercept
        for i, feature in enumerate(feature_cols):
            custom_pred_temp += custom_coefs[i] * current_df[feature].values
        custom_r2 = r2_score(current_df[target_col], custom_pred_temp)

        custom_results.append(
            {
                "adr_coef": adr_coef,
                "coefficients": custom_coefs,
                "intercept": custom_intercept,
                "r2": custom_r2,
                "constraints": custom_constraints,
            }
        )

    # Find best ADR boost option
    best_custom = max(custom_results, key=lambda x: x["r2"])

    # Store the best custom constrained model
    custom_coefs_best = best_custom["coefficients"]
    custom_intercept_best = best_custom["intercept"]

    # Calculate in-sample metrics (on full datasets)
    preds_previous_in = model_previous.predict(previous_df[feature_cols])
    preds_current_in = model_current.predict(current_df[feature_cols])

    r2_previous_in = r2_score(previous_df[target_col], preds_previous_in)
    mse_previous_in = mean_squared_error(previous_df[target_col], preds_previous_in)

    r2_current_in = r2_score(current_df[target_col], preds_current_in)
    mse_current_in = mean_squared_error(current_df[target_col], preds_current_in)

    print(
        f"In-sample R²: prev={r2_previous_in:.3f}, curr={r2_current_in:.3f} | MSE: prev={mse_previous_in:.3f}, curr={mse_current_in:.3f}"
    )

    # Create train/test split for out-of-sample evaluation

    # Try to identify common records across datasets
    # This is a simple approach assuming the combination of these columns uniquely identifies a record
    id_columns = ["kd", "adr", "kpr", "apr", "dpr", "kast", "fkpr", "fdpr"]

    # Create temporary ID columns for comparison
    previous_df["temp_id"] = previous_df[id_columns].astype(str).agg("-".join, axis=1)
    current_df["temp_id"] = current_df[id_columns].astype(str).agg("-".join, axis=1)

    # Find common IDs and report overlap
    common_ids = set(previous_df["temp_id"]).intersection(set(current_df["temp_id"]))

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

    print(
        f"Out-of-sample R²: prev={r2_previous_cv:.3f}, curr={r2_current_cv:.3f} | MSE: prev={mse_previous_cv:.3f}, curr={mse_current_cv:.3f}"
    )

    # Display Coefficients Comparison
    # Print concise coefficient summary
    print("\nCoefficients (Current Model):")
    for f, c in zip(feature_cols, model_current_full.coef_):
        print(f"  {f}: {c:.4f}")
    print("\nConstrained (FKPR=-FDPR):")
    for f, c in zip(feature_cols, constrained_coefs_curr):
        print(f"  {f}: {c:.4f}")
    print(f"\nCustom (ADR={best_custom['adr_coef']:.3f}, FKPR=-FDPR):")
    for f, c in zip(feature_cols, custom_coefs_best):
        print(f"  {f}: {c:.4f}")

    # Calculate and display correlation matrices
    # Correlation matrix and heatmap
    corr_matrix = current_df[feature_cols + [target_col]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature/Target Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("formula/images/correlation_heatmap.png")

    # Predicted vs Actual Ratings plot
    plt.figure(figsize=(10, 5))
    plt.scatter(
        test_combined[target_col],
        preds_previous_cv,
        alpha=0.5,
        label="Prev",
        color="blue",
    )
    plt.scatter(
        test_combined[target_col],
        preds_current_cv,
        alpha=0.5,
        label="Curr",
        color="green",
    )
    plt.plot(
        [test_combined[target_col].min(), test_combined[target_col].max()],
        [test_combined[target_col].min(), test_combined[target_col].max()],
        "r--",
        lw=1,
    )
    plt.xlabel("Actual Ratings")
    plt.ylabel("Predicted Ratings")
    plt.legend()
    plt.title("Predicted vs Actual Ratings (Test Set)")
    plt.tight_layout()
    plt.savefig("formula/images/prediction_comparison.png")

    # Generate full formula representation
    # Print spreadsheet-friendly custom formula
    sheet_formula = f"{custom_intercept_best:.6f} " + " ".join(
        [
            (f"+ {c:.6f} * {f}" if c >= 0 else f"- {abs(c):.6f} * {f}")
            for f, c in zip(feature_cols, custom_coefs_best)
        ]
    )
    print("\nCustom formula (spreadsheet style):\n", sheet_formula)

    # Descriptive spreadsheet formula
    feature_descriptions = {
        "adr": "('Total Damage' / Rounds)",
        "kpr": "(Kills / Rounds)",
        "apr": "(Assists / Rounds)",
        "dpr": "(Deaths / Rounds)",
        "kast": "('KAST (counts)' / Rounds)",
        "fkpr": "('First Kills' / Rounds)",
        "fdpr": "('First Deaths' / Rounds)",
        "cl_percent": "IFERROR('Clutch wins' / 'Clutch attempts', 0)",
    }

    descriptive_formula = f"= {custom_intercept_best:.6f}"
    for f, c in zip(feature_cols, custom_coefs_best):
        sign = "+" if c >= 0 else "-"
        desc = feature_descriptions[f]
        descriptive_formula += f" {sign} {abs(c):.6f} * {desc}"

    print("\nDescriptive spreadsheet formula:")
    print(descriptive_formula)

    # Performance comparison
    # Concise performance summary
    unconstrained_pred = model_current_full.predict(current_df[feature_cols])
    constrained_pred = constrained_intercept_curr
    for i, feature in enumerate(feature_cols):
        constrained_pred += constrained_coefs_curr[i] * current_df[feature].values
    custom_pred = custom_intercept_best
    for i, feature in enumerate(feature_cols):
        custom_pred += custom_coefs_best[i] * current_df[feature].values
    unconstrained_r2 = r2_score(current_df[target_col], unconstrained_pred)
    constrained_r2 = r2_score(current_df[target_col], constrained_pred)
    custom_r2 = best_custom["r2"]
    print(
        f"\nR²: Unconstr={unconstrained_r2:.3f}, Standard={constrained_r2:.3f}, Custom={custom_r2:.3f}"
    )

    # ADR Impact comparison
    adr_impact_unconstrained = (
        model_current_full.coef_[feature_cols.index("adr")] * current_df["adr"].std()
    )
    adr_impact_custom = (
        custom_coefs_best[feature_cols.index("adr")] * current_df["adr"].std()
    )
    print(
        f"ADR std impact: Unconstr={abs(adr_impact_unconstrained):.4f}, Custom={abs(adr_impact_custom):.4f} (x{abs(adr_impact_custom)/abs(adr_impact_unconstrained):.2f})"
    )

    # Verify constraint
    fkpr_coef = constrained_coefs_curr[feature_cols.index("fkpr")]
    fdpr_coef = constrained_coefs_curr[feature_cols.index("fdpr")]
    custom_fkpr_coef = custom_coefs_best[feature_cols.index("fkpr")]
    custom_fdpr_coef = custom_coefs_best[feature_cols.index("fdpr")]

    # Constraint check
    print(
        f"FKPR+FDPR (standard): {fkpr_coef + fdpr_coef:.2e}, (custom): {custom_fkpr_coef + custom_fdpr_coef:.2e}"
    )

    # Feature Impact Analysis

    # Feature impact summary (custom model)
    def feature_impact(df, coefs, names):
        stats = []
        for i, f in enumerate(names):
            std = df[f].std()
            stats.append((f, coefs[i], std, coefs[i] * std))
        stats.sort(key=lambda x: abs(x[3]), reverse=True)
        print("\nCustom model: Feature std impacts:")
        for f, c, s, impact in stats:
            print(f"  {f}: {impact:+.4f} (coef={c:.4f}, std={s:.3f})")
        print("Top 3 impactful:", ", ".join(f for f, _, _, _ in stats[:3]))

    feature_impact(current_df, custom_coefs_best, feature_cols)
    return {
        "best_custom_result": best_custom,
        "adr_boost_factor": abs(adr_impact_custom) / abs(adr_impact_unconstrained),
    }


if __name__ == "__main__":
    results = analyze_vlr_formula("formula/data/vlr_data.csv", "formula/data/old3.csv")
