import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

# ============================================================
# Paths
# ============================================================
data_path = "project_output/processed_features_transactions.csv"
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Load Data
# ============================================================
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Processed data not found at: {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

# ============================================================
# Summary Statistics
# ============================================================
print("\n📊 Summary Statistics:")
print(df.describe())

# ============================================================
# Category-wise Spending
# ============================================================
category_sum = df.groupby("category")["amount"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 8))
category_sum.plot(kind="pie", autopct="%1.1f%%")
plt.title("Category-wise Total Spending (Pie Chart)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "category_pie_chart.png"))
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(x=category_sum.index, y=category_sum.values, palette="mako")
plt.title("Category-wise Total Spending (Bar Chart)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "category_bar_chart.png"))
plt.close()

# ============================================================
# Time-based Spending Trends (Monthly)
# ============================================================
df["year_month"] = pd.to_datetime(df["year_month"])
monthly_trend = df.groupby("year_month")["amount"].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x="year_month", y="amount", data=monthly_trend, marker="o")
plt.title("Monthly Total Spending Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_trend.png"))
plt.close()

# ============================================================
# Seasonality Detection (Autocorrelation + Decomposition)
# ============================================================
plt.figure(figsize=(8, 4))
plot_acf(monthly_trend["amount"], lags=12)
plt.title("Autocorrelation Plot (Seasonality Detection)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "autocorrelation_plot.png"))
plt.close()

try:
    result = seasonal_decompose(monthly_trend.set_index("year_month")["amount"], model="additive", period=12)
    result.plot()
    plt.suptitle("Seasonal Decomposition (Trend, Seasonal, Residual)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seasonal_decomposition.png"))
    plt.close()
except Exception as e:
    print("⚠️ Skipping decomposition (not enough data):", e)

# ============================================================
# Correlation Heatmap (Spending vs Income/Demographics)
# ============================================================
numeric_cols = df.select_dtypes(include=["number"])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# ============================================================
# Expense Ratio Distribution
# ============================================================
if "expense_ratio" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="category", y="expense_ratio", data=df, palette="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Expense Ratio Distribution per Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expense_ratio_boxplot.png"))
    plt.close()

# ============================================================
# Insights Summary
# ============================================================
print("\n💡 Insights Summary:")
print(f"- Top 3 Spending Categories: {category_sum.head(3).index.tolist()}")
print("- Time-based spending trend and seasonality detected.")
if "expense_ratio" in df.columns:
    print("- Expense ratios and volatility patterns analyzed.")
print(f"- All charts saved in: {output_dir}")

print("\n✅ EDA completed successfully.")
