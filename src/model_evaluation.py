# src/model_evaluation.py
"""
Model evaluation & visualization script for PersonalExpenseForecasting.

Reads model results (CSV) and generates:
 - comparison bar charts (MAE, RMSE, R2, MAPE)
 - normalized heatmap across metrics
 - best-model summary & CSV ranking
 - saves outputs to project_output/evaluation/

Designed to be robust to slightly different paths / missing columns.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# ---------------------------
# Paths (adjustable)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root
CANDIDATE_PATHS = [
    BASE_DIR / "project_output" / "models" / "model_results.csv",
    BASE_DIR / "project_output" / "model_results.csv",
    BASE_DIR / "project_output" / "model_performance.csv",
    BASE_DIR / "project_output" / "model_comparison.csv",
    BASE_DIR / "reports" / "forecasts" / "model_comparison.csv",
    BASE_DIR / "project_output" / "models" / "model_comparison.csv",
]

OUTPUT_DIR = BASE_DIR / "project_output" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Utilities
# ---------------------------
def find_results_file():
    for p in CANDIDATE_PATHS:
        if p.exists():
            return p
    # as last resort search recursively for *_results*.csv in project_output
    for p in BASE_DIR.rglob("project_output/**/*.csv"):
        if "result" in p.name.lower() or "performance" in p.name.lower() or "comparison" in p.name.lower():
            return p
    return None


def safe_read_csv(path):
    # try a few encodings
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise IOError(f"Could not read CSV at {path} with common encodings.")


def ensure_numeric_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_df_for_heatmap(df_metrics):
    # normalize each metric to 0-1 range (higher=better where appropriate)
    norm = df_metrics.copy()
    for col in norm.columns:
        arr = norm[col].values.astype(float)
        # For R2 bigger is better: keep sign. For MAE/RMSE/MAPE, smaller is better -> invert after normalization.
        if col in ("MAE", "RMSE", "MAPE"):
            # invert: lower -> higher normalized
            maxv = np.nanmax(arr)
            minv = np.nanmin(arr)
            if maxv == minv:
                norm[col] = 1.0
            else:
                norm[col] = 1 - (arr - minv) / (maxv - minv)
        else:
            # R2 or other where larger is better
            maxv = np.nanmax(arr)
            minv = np.nanmin(arr)
            if maxv == minv:
                norm[col] = 1.0
            else:
                norm[col] = (arr - minv) / (maxv - minv)
    return norm


# ---------------------------
# Main evaluation flow
# ---------------------------
def main():
    print("🔹 Loading model performance results...")

    results_path = find_results_file()
    if results_path is None:
        print("❌ No model results CSV found. Looked for:")
        for p in CANDIDATE_PATHS:
            print("   -", p)
        sys.exit(1)

    print("✅ Found results file:", results_path)
    df = safe_read_csv(results_path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure model column
    model_col = None
    for candidate in ("Model", "model", "model_name", "Model Name"):
        if candidate in df.columns:
            model_col = candidate
            break
    if model_col is None:
        raise ValueError("Could not find model name column (Model/model/model_name).")

    # expected metric columns (case-sensitive as in your CSV): MAE, RMSE, R2, MAPE
    metric_cols = []
    for c in ("MAE", "RMSE", "R2", "MAPE"):
        if c in df.columns:
            metric_cols.append(c)

    if not metric_cols:
        raise ValueError("No metric columns found in results CSV. Expected at least one of: MAE, RMSE, R2, MAPE")

    # keep only required columns
    df_eval = df[[model_col] + metric_cols].copy()
    df_eval.rename(columns={model_col: "Model"}, inplace=True)
    df_eval = ensure_numeric_cols(df_eval, metric_cols)

    # basic cleaning: drop all-NaN rows
    df_eval.dropna(how="all", subset=metric_cols, inplace=True)

    if df_eval.empty:
        print("❌ No numeric metric rows found after parsing.")
        sys.exit(1)

    # show loaded table
    print(df_eval.to_string(index=False))

    # -------------------------------
    # ranking & summary
    # -------------------------------
    # rank by MAE then RMSE (if MAE exists), else by RMSE then MAE
    rank_df = df_eval.copy()
    if "MAE" in rank_df.columns:
        rank_df["MAE_Rank"] = rank_df["MAE"].rank(method="min", ascending=True)
    if "RMSE" in rank_df.columns:
        rank_df["RMSE_Rank"] = rank_df["RMSE"].rank(method="min", ascending=True)
    if "R2" in rank_df.columns:
        rank_df["R2_Rank"] = rank_df["R2"].rank(method="min", ascending=False)  # higher R2 better
    if "MAPE" in rank_df.columns:
        rank_df["MAPE_Rank"] = rank_df["MAPE"].rank(method="min", ascending=True)

    # overall score (sum of ranks where available)
    rank_cols = [c for c in rank_df.columns if c.endswith("_Rank")]
    if rank_cols:
        rank_df["Overall_Score"] = rank_df[rank_cols].sum(axis=1)
    else:
        rank_df["Overall_Score"] = 0

    rank_df.sort_values("Overall_Score", inplace=True)
    rank_df.reset_index(drop=True, inplace=True)

    # save ranking CSV
    rank_out = OUTPUT_DIR / "model_ranking.csv"
    rank_df.to_csv(rank_out, index=False, encoding="utf-8")
    print("✅ Saved model ranking ->", rank_out)

    # best model
    best_row = rank_df.iloc[0]
    best_summary_path = OUTPUT_DIR / "best_model_summary.txt"
    with open(best_summary_path, "w", encoding="utf-8") as f:
        f.write("Best model summary\n")
        f.write("------------------\n")
        for k, v in best_row.items():
            f.write(f"{k}: {v}\n")
    print("🏆 Best model:", best_row["Model"])
    print("✅ Best model summary saved ->", best_summary_path)

    # -------------------------------
    # Plots
    # -------------------------------
    def save_bar(metric, title=None):
        if metric not in df_eval.columns:
            return
        plt.figure(figsize=(8, max(4, 0.4 * len(df_eval))))
        order = df_eval.sort_values(metric, ascending=(metric != "R2"))["Model"]
        sns.barplot(x=metric, y="Model", data=df_eval, order=order)
        plt.title(title or f"{metric} comparison")
        plt.tight_layout()
        out = OUTPUT_DIR / f"{metric.lower()}_comparison.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print("Saved plot:", out)

    save_bar("RMSE", "RMSE comparison (lower is better)")
    save_bar("MAE", "MAE comparison (lower is better)")
    save_bar("R2", "R² comparison (higher is better)")
    save_bar("MAPE", "MAPE comparison (lower is better)")

    # Heatmap (normalize metrics so that higher is better for all)
    metrics_for_heatmap = [c for c in ("MAE", "RMSE", "R2", "MAPE") if c in df_eval.columns]
    if metrics_for_heatmap:
        hm = df_eval.set_index("Model")[metrics_for_heatmap]
        hm_norm = normalize_df_for_heatmap(hm)
        plt.figure(figsize=(max(6, 0.5 * len(hm_norm)), 6))
        sns.heatmap(hm_norm, annot=True, fmt=".2f", cmap="vlag", cbar_kws={"label": "normalized (0-1)"})
        plt.title("Normalized metric heatmap (higher = better)")
        plt.tight_layout()
        hm_out = OUTPUT_DIR / "metric_heatmap.png"
        plt.savefig(hm_out, dpi=150)
        plt.close()
        print("Saved heatmap:", hm_out)

    # Save a combined CSV summary for quick viewing
    summary_out = OUTPUT_DIR / "model_summary_table.csv"
    df_eval.to_csv(summary_out, index=False, encoding="utf-8")
    print("Saved summary table ->", summary_out)

    # -------------------------------
    # Optional analyses if additional columns present: category, horizon
    # -------------------------------
    extra_cols = [c for c in ("category", "Category", "Horizon", "horizon") if c in df.columns]
    if extra_cols:
        print("Note: results CSV contains extra columns:", extra_cols)
        # If category/horizon available, do more advanced breakdowns (not implemented generically here)
    else:
        print("No category/horizon breakdown present in the results CSV; skipping those analyses.")

    print("\n✅ Evaluation finished. All outputs are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
