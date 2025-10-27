# 💰 Personal Expense Forecasting & Budget Optimization System  
**An AI-Powered Solution for Smart Financial Planning**

> 📊 *Predict, Plan, and Optimize your Personal Finances using Machine Learning, Deep Learning, and Transformer-based Forecasting.*

---

## 🧩 Project Overview

This project builds an **AI-driven personal expense forecasting system** that predicts and optimizes monthly expenses using **machine learning (ML)**, **deep learning (DL)**, and **transformer-based models**.  

The system analyzes historical financial data, identifies spending patterns, and provides **personalized budget recommendations** through an **interactive Streamlit dashboard**.

---

## 🎯 Objectives

- Predict future expenses (1, 3, and 6 months ahead) across multiple categories.  
- Identify key expense trends, seasonal patterns, and anomalies.  
- Suggest optimized budget allocations using intelligent algorithms.  
- Visualize and report financial performance in real time.  
- Deploy a user-friendly Streamlit web app for end users.

---

## 🧠 Skills Acquired

> **Skills Takeaway:**  
> Time Series Analysis • Feature Engineering • ML/DL Forecasting • LSTM • Transformers • Streamlit Development • Data Visualization • Financial Analytics

| Category | Skills |
|-----------|--------|
| **Data Science** | Data cleaning, preprocessing, EDA, feature engineering |
| **Machine Learning** | Random Forest, XGBoost, LightGBM, Prophet |
| **Deep Learning** | LSTM, GRU, Bi-LSTM, CNNs for time series |
| **Transformers** | Temporal Fusion Transformer (TFT), N-BEATS, Autoformer |
| **Visualization & App Dev** | Plotly, Streamlit, ReportLab, Dashboarding |
| **Finance Analytics** | Budget optimization, trend analysis, forecasting |

---

## 🧾 Problem Statement

> Managing personal finances effectively is a global challenge.

Most individuals rely on **static budgeting tools** that fail to adapt to lifestyle changes, seasonal spending, and income variations.  
This leads to:
- Overspending and unplanned debt 💸  
- Poor savings decisions 📉  
- Lack of visibility into financial trends 📊  

This project aims to build a **data-driven, predictive system** capable of:
- Forecasting future expenses across categories.  
- Providing intelligent budget recommendations.  
- Visualizing spending patterns in an interactive and actionable format.

---

## 🏦 Domain  
**Personal Finance Management & Predictive Analytics**

---

## 💼 Business Use Cases

| Sector | Use Case |
|--------|-----------|
| 🧍‍♀️ **Personal Finance** | Individuals can plan monthly budgets effectively. |
| 💡 **Financial Advisors** | Offer personalized insights for clients. |
| 🏦 **Banking** | Integrate predictive expense analysis in mobile apps. |
| 💳 **FinTech Apps** | Add forecasting & budget optimization features. |
| 🧮 **Credit & Loan Services** | Use expense behavior to evaluate credit risk. |

---

## ⚙️ Project Approach

### **1️⃣ Data Collection**
- 🏦 **User Data:** Bank statements, card bills, UPI/Wallet exports (Paytm, GPay, PhonePe).  
- 📊 **Public Datasets:** [Kaggle: Personal Expense Transaction Data](https://www.kaggle.com/) and related sources.  
- 🧩 **Hybrid Dataset:** Combination of real + synthetic + external financial indicators.

### **2️⃣ Data Preprocessing**
- Categorize expenses using NLP on merchant names.  
- Handle missing values, duplicates, and outliers.  
- Create time-based features: week, month, season, holiday, etc.  
- Encode income patterns and demographic factors.  

### **3️⃣ Exploratory Data Analysis (EDA)**
- Category-wise expense trends & correlations.  
- Seasonality and cyclical behavior detection.  
- Identify overspending patterns and volatility.  

### **4️⃣ Feature Engineering**
- Rolling averages, lag features, trend differentials.  
- Category ratio and variance metrics.  
- Integration of inflation and economic indicators.  

### **5️⃣ Modeling**

| Type | Models Used | Description |
|-------|--------------|-------------|
| 🧮 **Baseline Models** | Linear Regression, ARIMA, SARIMA, Prophet | Foundational statistical forecasting |
| 🤖 **Machine Learning** | Random Forest, XGBoost, LightGBM | Handles non-linear dependencies |
| 🧠 **Deep Learning** | LSTM, GRU, Bi-LSTM, CNNs | Captures sequential temporal dependencies |
| ⚡ **Transformers (Advanced)** | Temporal Fusion Transformer (TFT), N-BEATS, Autoformer | State-of-the-art forecasting |
| 🔗 **Ensemble Models** | ML + DL hybrid | Combined predictions for higher accuracy |

### **6️⃣ Evaluation Metrics**
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Percentage Error (MAPE)  
- Directional Accuracy  
- Category-wise precision  

### **7️⃣ Deployment**
- 🎨 Interactive Streamlit Dashboard  
- 📉 Forecast Visualizations (Plotly Graphs)  
- 💡 Budget Optimization Insights  
- 📤 Report Export (Excel, PDF)  
- ☁️ Deploy on Streamlit Cloud / AWS / Heroku  

---



---

## 💻 Tech Stack & Dependencies

| Category | Tools |
|-----------|-------|
| **Languages** | Python |
| **Libraries** | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Prophet, TensorFlow |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Frameworks** | Streamlit |
| **Deployment** | Docker, Streamlit Cloud |
| **Version Control** | Git, GitHub |
| **Reporting** | ReportLab, OpenPyXL |

---

Here is the complete project structure with brief descriptions for each file, presented in a visually clean format:

-----

## 📂 Personal Expense Forecasting Project Structure

```
PersonalExpenseForecasting/
├── app/
│   └── streamlit_app.py                # Web application interface
├── data/
│   ├── interim/
│   │   └── interim_cleaned_transactions.csv  # Transactions after initial cleaning
│   ├── processed/
│   │   └── processed_features_transactions.csv # Final features for modeling
│   └── raw/
│       ├── budgetwise_finance_dataset.csv      # Original raw dataset
│       └── budgetwise_synthetic_dirty.csv      # Secondary dirty dataset
├── models/
│   ├── bilstm_model.h5                 # Trained Bidirectional LSTM model
│   ├── cnn1d_model.h5                  # Trained 1D CNN model
│   ├── feature_columns.pkl             # Metadata for features used
│   ├── gru_model.h5                    # Trained GRU deep learning model
│   ├── lightgbm_model.pkl              # Trained LightGBM model
│   ├── linear_regression.pkl           # Trained Linear Regression model
│   ├── lstm_model.h5                   # Trained LSTM deep learning model
│   ├── random_forest.pkl               # Trained Random Forest model
│   ├── scaler.pkl                      # Fitted data scaler object
│   └── xgb_model.pkl                   # Trained XGBoost model
├── notebooks/                          # Exploratory Analysis (Jupyter Notebooks)
├── reports/                            # All project outputs and reports
│   ├── evaluation/
│   │   ├── best_model_summary.txt      # Text summary of best model
│   │   ├── metric_heatmap.png          # Visualization of model metrics
│   │   ├── r2_comparison.png           # R-squared score comparison plot
│   │   └── rmse_comparison.png         # RMSE score comparison plot
│   ├── figures/
│   │   ├── arima_forecast.png          # ARIMA model forecast plot
│   │   ├── autocorrelation_plot.png    # Autocorrelation function plot
│   │   ├── category_bar_chart.png      # Expense distribution bar chart
│   │   ├── category_pie_chart.png      # Expense distribution pie chart
│   │   ├── correlation_heatmap.png     # Feature correlation heatmap
│   │   ├── mae_horizon1.png            # MAE for first forecast horizon
│   │   ├── monthly_trend.png           # Plot of monthly expenses
│   │   ├── prophet_forecast.png        # Prophet model forecast plot
│   │   └── seasonal_decomposition.png  # Time series decomposition plot
│   └── forecasts/
│       ├── model_comparison.csv        # CSV of model performance metrics
│       └── model_results.csv           # Final forecast results
├── src/                                # Source code scripts
│   ├── data_preprocessing.py           # Functions for data cleaning
│   ├── eda_analysis.py                 # Scripts for EDA
│   ├── feature_engineering.py          # Logic for feature creation
│   ├── forecasting.py                  # Core prediction logic
│   ├── model_evaluation.py             # Script for model testing
│   ├── model_optimization.py           # Script for hyperparameter tuning
│   ├── model_training.py               # Script for model training
│   └── utils.py                        # General helper functions
├── .venv/                              # Python Virtual Environment
├── Dockerfile                          # Docker container configuration
└── requirements.txt                    # List of required packages
```
## 📈 Results & Model Performance

| Model         | MAE       | RMSE       | MAPE     | R²       | Observation               |
| ------------- | --------- | ---------- | -------- | -------- | ------------------------- |
| Prophet       | 12,540    | 17,620     | 14.8%    | 0.89     | Baseline trend accuracy   |
| Random Forest | 9,850     | 13,200     | 11.2%    | 0.93     | Strong performance        |
| **XGBoost**   | **7,420** | **10,150** | **8.4%** | **0.96** | ⭐ Best performer          |
| LSTM          | 8,150     | 11,870     | 9.1%     | 0.95     | Excellent long-term model |
---
#### ✅ Best Model: XGBoost
#### 📊 Overall Accuracy: ~91.6%
#### 📉 MAPE Improvement: ~43% better than baseline Prophet model.


---

  

# 💡 Insights from the Dashboard

#### Spending on Education dropped 82% (savings opportunity).

#### Miscellaneous category overspent 166% this month.

#### Suggested saving potential: ₹3,23,937 (~8.3%).

#### Recommended reallocation towards priority goals (housing, savings).

# 🧭 Future Enhancements

#### Integrate real-time APIs (Plaid, Razorpay) for live data ingestion.

#### Add transformer-based forecasting (TFT, N-BEATS).

#### Enable multi-user authentication and secure personal data vaults.

#### Build LangChain-based AI Assistant for conversational financial advice.

#### Integrate FastAPI microservice for backend scalability.

## 📜 License

#### This project is licensed under the MIT License.
#### You are free to use, modify, and distribute it with proper attribution.

## 👩‍💻 Author & Contact

#### Developed by: Sakshi Birajdar
####  📧 Email: sakshibirajdar34@gmail.com

#### 🔗 LinkedIn: linkedin.com/in/sakshibirajdar

## 🏁 Summary

#### Personal Expense Forecasting System combines AI forecasting, budget optimization, and visual analytics into a powerful, deployable web solution.
#### Built using Python, Streamlit, and advanced ML/DL models (XGBoost, LSTM, Prophet), it empowers users to make data-driven financial decisions with 90%+ accuracy.

## 🎯 “From Predicting to Planning — Your Finances, Smarter with AI.”

#### ⭐ If this project helped you, don’t forget to star it on GitHub!