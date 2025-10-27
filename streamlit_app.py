# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import os
import numpy as np
import difflib
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import requests
from streamlit_lottie import st_lottie

# -------------------------------
# 🌟 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Personal Expense Forecasting Dashboard",
    layout="wide",
    page_icon="💰"
)

# -------------------------------
# 🌈 CUSTOM THEME / CSS
# -------------------------------
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #f7f9fc;
        font-family: "Poppins", sans-serif;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #2b2b2b;
    }
    /* Success color */
    .stSuccess {
        background-color: #e7f9ed !important;
        border-left: 5px solid #3CCF4E !important;
    }
    /* Buttons */
    div.stDownloadButton button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stDownloadButton button:hover {
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🎬 LOAD LOTTIE ANIMATION
# -------------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_finance = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# -------------------------------
# 🎬 ANIMATED HEADER
# -------------------------------
col_anim, col_title = st.columns([1, 3])
with col_anim:
    st_lottie(lottie_finance, height=120, key="finance_anim")
with col_title:
    st.title("💰 Personal Expense Forecasting Dashboard")
    st.markdown("### Track • Forecast • Optimize Your Expenses")

# -------------------------------
# 📂 FILE UPLOAD
# -------------------------------
with st.sidebar:
    st.header("📂 Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    st.subheader("📈 Dashboard Sections")
    section = st.radio(
        "Select a view:",
        ["Overview", "Forecasting (AI Models)", "Budget Optimization", "Model Comparison", "Export Reports"]
    )

# -------------------------------
# 🔠 CATEGORY CLEANUP FUNCTION
# -------------------------------
def clean_category(cat):
    if pd.isna(cat):
        return "Uncategorized"
    cat = str(cat).strip().title()
    master_categories = [
        "Food", "Rent", "Travel", "Utilities", "Health",
        "Education", "Entertainment", "Savings", "Salary",
        "Bonus", "Investment", "Freelance", "Others", "Misc"
    ]
    match = difflib.get_close_matches(cat, master_categories, n=1, cutoff=0.6)
    return match[0] if match else "Others"

# -------------------------------
# LOAD DATA
# -------------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
else:
    if section != "Model Comparison":
        st.info("Please upload your expense file to get started.")
        st.stop()

# -------------------------------
# 🧹 CLEANING + NORMALIZATION
# -------------------------------
if section != "Model Comparison":
    df.columns = df.columns.str.strip().str.lower()
    if 'amount' not in df.columns:
        st.error("The dataset must contain an 'amount' column.")
        st.stop()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
    elif 'year_month' not in df.columns:
        df['year_month'] = [f"2025-{(i%12)+1:02d}" for i in range(len(df))]

    if 'category' not in df.columns:
        df['category'] = 'General'

    with st.spinner("🧠 Cleaning and normalizing categories..."):
        df['category'] = df['category'].apply(clean_category)
    st.success("✅ Categories standardized successfully!")

# -------------------------------
# 1️⃣ OVERVIEW SECTION
# -------------------------------
if section == "Overview":
    st.header("📊 Spending Breakdown Overview")
    col1, col2 = st.columns(2)

    category_sum = df.groupby('category')['amount'].sum().sort_values(ascending=False).reset_index()
    with col1:
        fig1 = px.bar(category_sum, x='category', y='amount', title="Total Spending by Category",
                      color='amount', color_continuous_scale="viridis")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.pie(category_sum, values='amount', names='category',
                      title="Expense Share by Category", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    monthly = df.groupby('year_month')['amount'].sum().reset_index()
    fig3 = px.area(monthly, x='year_month', y='amount',
                   title="Monthly Expense Trend", markers=True,
                   color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# 2️⃣ FORECASTING (AI MODELS)
# -------------------------------
elif section == "Forecasting (AI Models)":
    st.header("🔮 AI Forecasted Expenses")

    monthly = df.groupby('year_month')['amount'].sum().reset_index()
    monthly['year_month'] = pd.to_datetime(monthly['year_month'])
    monthly = monthly.sort_values('year_month')

    model_choice = st.selectbox("Select Forecasting Model", ["Moving Average", "Prophet", "XGBoost"])

    # Prepare for metrics
    results_list = []

    if model_choice == "Moving Average":
        monthly['forecast'] = monthly['amount'].rolling(3).mean().shift(1)
        monthly['forecast'].fillna(method='bfill', inplace=True)

    elif model_choice == "Prophet":
        prophet_df = monthly.rename(columns={'year_month': 'ds', 'amount': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)
        forecast = forecast[['ds', 'yhat']].rename(columns={'ds': 'year_month', 'yhat': 'forecast'})
        monthly = pd.merge(monthly, forecast, on='year_month', how='outer')

    elif model_choice == "XGBoost":
        monthly['month_num'] = np.arange(len(monthly))
        X = monthly[['month_num']]
        y = monthly['amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        monthly['forecast'] = model.predict(X)

    # -------------------------------
    # 📉 Plot
    # -------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['year_month'], y=monthly['amount'],
                             mode='lines+markers', name='Actual', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=monthly['year_month'], y=monthly['forecast'],
                             mode='lines+markers', name='Forecast', line=dict(dash='dash', color='orange')))
    fig.update_layout(title=f"Expense Forecast ({model_choice})",
                      xaxis_title="Month", yaxis_title="Amount")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # 📊 Model Metrics + AI Summary
    # -------------------------------
    valid = monthly.dropna(subset=['forecast', 'amount'])
    if not valid.empty:
        mae = mean_absolute_error(valid['amount'], valid['forecast'])
        rmse = np.sqrt(mean_squared_error(valid['amount'], valid['forecast']))
        r2 = r2_score(valid['amount'], valid['forecast'])
        mape = np.mean(np.abs((valid['amount'] - valid['forecast']) / valid['amount'])) * 100

        st.metric("📅 Next Month Forecast", f"{monthly['forecast'].iloc[-1]:,.0f} ₹")
        st.write(f"**MAE:** {mae:,.0f} | **RMSE:** {rmse:,.0f} | **R²:** {r2:.3f} | **MAPE:** {mape:.2f}%")

        # 🧠 AI Insights Summary
        diff = monthly['forecast'].iloc[-1] - monthly['amount'].iloc[-2]
        st.markdown("### 🧠 Smart Insights")
        if diff > 0:
            st.info(f"📈 Your expenses are projected to increase by **₹{abs(diff):,.0f}** next month. "
                    f"Consider reviewing high-cost categories to stay within budget.")
        else:
            st.success(f"📉 Great! Your expenses may decrease by **₹{abs(diff):,.0f}** next month. "
                       f"Keep maintaining your current saving habits!")

        # Save model results
        os.makedirs("project_output", exist_ok=True)
        results = pd.DataFrame([{
            "Model": model_choice,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }])
        results.to_csv(f"project_output/{model_choice}_results.csv", index=False)

        # Merge into global comparison file
        combined_path = "project_output/model_results.csv"
        if os.path.exists(combined_path):
            old = pd.read_csv(combined_path)
            df_all = pd.concat([old, results]).drop_duplicates(subset=['Model'], keep='last')
        else:
            df_all = results
        df_all.to_csv(combined_path, index=False)
        st.success("✅ Forecast and performance metrics saved!")

# -------------------------------
# 3️⃣ BUDGET OPTIMIZATION
# -------------------------------
elif section == "Budget Optimization":
    st.header("💡 Budget Optimization Recommendations")
    category_sum = df.groupby('category')['amount'].sum().reset_index()
    total = category_sum['amount'].sum()
    category_sum['percent'] = (category_sum['amount'] / total) * 100

    st.dataframe(category_sum.style.format({'amount': '{:,.0f}', 'percent': '{:.2f}%'}))

    st.subheader("🔻 Suggested Budget Limits (10% Cut)")
    category_sum['suggested_limit'] = category_sum['amount'] * 0.9
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=category_sum['category'], y=category_sum['amount'], name='Current'))
    fig4.add_trace(go.Bar(x=category_sum['category'], y=category_sum['suggested_limit'], name='Suggested'))
    fig4.update_layout(barmode='group', title="Current vs Suggested Spending")
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# 4️⃣ MODEL COMPARISON
# -------------------------------
elif section == "Model Comparison":
    st.header("🤖 Model Performance Comparison Dashboard")

    results_path = "project_output/model_results.csv"
    if not os.path.exists(results_path):
        st.error("⚠️ Model results not found! Please run Forecasting first.")
        st.stop()

    df_results = pd.read_csv(results_path)
    st.subheader("📋 Model Performance Summary")
    st.dataframe(df_results.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.2f}"}))

    best_model = df_results.loc[df_results['R2'].idxmax(), 'Model']
    st.success(f"🏆 Best Performing Model: **{best_model}**")

    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    for metric in metrics:
        ascending = False if metric == 'R2' else True
        fig = px.bar(df_results.sort_values(metric, ascending=ascending),
                     x='Model', y=metric, color='Model', title=f"{metric} Comparison",
                     text_auto='.2s', color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5️⃣ EXPORT REPORTS
# -------------------------------
elif section == "Export Reports":
    st.header("📤 Export Forecast Report")

    def export_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Report')
        return output.getvalue()

    def export_pdf(df):
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("Expense Forecast Report", styles['Title']), Spacer(1, 20)]
        for _, row in df.iterrows():
            story.append(Paragraph(f"{row['category']}: ₹{row['amount']:,.0f}", styles['Normal']))
        doc.build(story)
        return output.getvalue()

    st.download_button(
        label="📊 Download Excel Report",
        data=export_excel(df),
        file_name="expense_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=export_pdf(df),
        file_name="expense_report.pdf",
        mime="application/pdf"
    )
    st.success("✅ Reports ready for download!")

# -------------------------------
# 🏁 END + ABOUT SECTION
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Prophet, XGBoost & Plotly | © 2025 Personal Expense Forecasting System")
st.markdown("""
<div style='text-align:center;'>
    <b>👩‍💻 Developed by:</b> Sakshi Birajdar <br>
    <a href='https://github.com/sakshibirajdar09' target='_blank'>GitHub</a> |
    <a href='https://www.linkedin.com/in/sakshibirajdar/' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
