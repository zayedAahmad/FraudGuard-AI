import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Fraud Guard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #ffffff;
    color: #111827;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #111827;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #6b7280 !important;
    border-bottom: 2px solid transparent !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #22c55e !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background-color: #22c55e;
    color: white !important;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #16a34a;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #f9fafb;
    border: 1px dashed #d1d5db;
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}

/* Alerts */
.stSuccess {
    background-color: #dcfce7 !important;
    color: #166534 !important;
}

.stError {
    background-color: #fee2e2 !important;
    color: #991b1b !important;
}

.stWarning {
    background-color: #fef3c7 !important;
    color: #92400e !important;
}

/* Small note */
.small-note {
    color: #6b7280;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# Session with retries
def get_api_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    return session

session = get_api_session()

# Sidebar

with st.sidebar:
    st.title("Fraud Guard Settings")
    api_url = st.text_input("API URL", "http://127.0.0.1:8004/predict")
    st.markdown("---")
    st.info("This dashboard uses XGBoost + SMOTE for fraud detection.")
    st.markdown("### Tips")
    st.markdown("- Use **Random sample** for more realistic batch results")
    st.markdown("- Start with **50–100 rows** for faster testing")

# Header
st.title("Fraud Guard AI")
st.markdown("### Enterprise Fraud Monitoring Dashboard")
st.markdown(
    '<p class="small-note">Real-time fraud scoring, batch analysis, and visual insights in one place.</p>',
    unsafe_allow_html=True
)

tabs = st.tabs(["Single Transaction", "Batch Analysis", "Insights"])

#  Single Transaction
with tabs[0]:
    st.header("Single Transaction Check")

    try:
        sample = pd.read_csv("data/raw/creditcard.csv").drop("Class", axis=1).iloc[0]
        input_data = {}

        with st.expander("Edit Transaction Features", expanded=True):
            cols = st.columns(4)

            for i, feature in enumerate(sample.index):
                input_data[feature] = cols[i % 4].number_input(
                    feature,
                    value=float(sample[feature]),
                    format="%.4f",
                    key=f"single_{feature}"
                )

        if st.button("Run Prediction", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                response = session.post(api_url, json={"data": input_data})

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("---")
                    st.subheader("Live Decision")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Fraud Probability", result["fraud_probability"])
                    m2.metric("Fraud Flag", "FRAUD" if result["is_fraud"] else "CLEAN")
                    m3.metric("Risk Level", result["risk_level"])
                    m4.metric("Recommendation", result["recommendation"])

                    probability = float(result["fraud_probability"])

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            title={"text": "Fraud Probability %"},
                            gauge={
                                "axis": {"range": [None, 100]},
                                "bar": {
                                    "color": "#ef4444" if probability >= 0.25 else "#22c55e"
                                },
                                "steps": [
                                    {"range": [0, 25], "color": "#1f2937"},
                                    {"range": [25, 60], "color": "#78350f"},
                                    {"range": [60, 100], "color": "#450a0a"}
                                ]
                            }
                        )
                    )
                    gauge.update_layout(
                        paper_bgcolor="#0f172a",
                        font={"color": "white", "size": 16}
                    )

                    st.plotly_chart(gauge, use_container_width=True)

                    if result["is_fraud"]:
                        st.error("Suspicious transaction detected. Review is recommended.")
                    else:
                        st.success("Transaction looks safe.")

                else:
                    st.error(f"API error: {response.status_code}")
                    st.write(response.text)

    except Exception as e:
        st.error(f"Initialization error: {e}")

#  Batch Analysis
with tabs[1]:
    st.header("Batch Transaction Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        mode = st.radio(
            "Select data mode",
            ["Top rows", "Random sample"],
            horizontal=True
        )

        max_limit = min(len(df), 300)
        rows_to_analyze = st.slider(
            "Rows to analyze",
            min_value=5,
            max_value=max_limit,
            value=min(50, max_limit)
        )

        if st.button("Run Batch Prediction", use_container_width=True):
            if mode == "Top rows":
                batch_df = df.head(rows_to_analyze).copy()
            else:
                batch_df = df.sample(n=rows_to_analyze, random_state=42).copy()

            results = []
            progress = st.progress(0)
            status = st.empty()

            total_rows = len(batch_df)

            for i, (_, row) in enumerate(batch_df.iterrows()):
                row_data = row.to_dict()
                response = session.post(api_url, json={"data": row_data})

                if response.status_code == 200:
                    result = response.json()
                    row_result = {**row_data, **result}
                    row_result["fraud_probability"] = float(row_result["fraud_probability"])
                    results.append(row_result)

                progress.progress((i + 1) / total_rows)
                status.text(f"Processing row {i + 1} of {total_rows}")

            res_df = pd.DataFrame(results)
            st.session_state["res_df"] = res_df

            st.markdown("---")
            st.subheader("Monitoring Summary")

            fraud_alerts = int(res_df["is_fraud"].sum())
            avg_risk = float(res_df["fraud_probability"].mean())
            high_risk = int((res_df["risk_level"] == "High").sum())
            total_tx = len(res_df)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", total_tx)
            c2.metric("Fraud Alerts", fraud_alerts)
            c3.metric("High Risk", high_risk)
            c4.metric("Average Risk", f"{avg_risk:.4f}")

            if fraud_alerts > 0:
                st.error(f"{fraud_alerts} suspicious transactions were flagged.")
            else:
                st.success("No suspicious transactions were flagged in this batch.")

            st.markdown("---")
            st.subheader("Results Table")

            show_only_flagged = st.checkbox("Show only flagged transactions")

            display_df = res_df[res_df["is_fraud"] == True] if show_only_flagged else res_df

            st.dataframe(
                display_df.style.background_gradient(
                    subset=["fraud_probability"],
                    cmap="Reds"
                ),
                use_container_width=True
            )

            csv_data = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=" Download Results",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

# 
#  Insights
with tabs[2]:
    st.header("Insights & Charts")

    if "res_df" in st.session_state:
        res_df = st.session_state["res_df"]

        col1, col2 = st.columns(2)

        with col1:
            fig_donut = px.pie(
                res_df,
                names="risk_level",
                hole=0.55,
                title="Risk Level Distribution",
                color="risk_level",
                color_discrete_map={
                    "Low": "#22c55e",
                    "Medium": "#f59e0b",
                    "High": "#ef4444"
                }
            )
            fig_donut.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font={"color":"#111827"}
                
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            fig_scatter = px.scatter(
                res_df,
                x="Amount",
                y="fraud_probability",
                color="is_fraud",
                size="Amount",
                title="Amount vs Fraud Probability",
                color_discrete_map={True: "#ef4444", False: "#22c55e"}
            )
            fig_scatter.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font={"color":"#111827"}
                
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Top High-Risk Transactions")
        top_risky = res_df.sort_values(by="fraud_probability", ascending=False).head(10)
        st.dataframe(
            top_risky.style.background_gradient(subset=["fraud_probability"], cmap="Reds"),
            use_container_width=True
        )

    else:
        st.warning("Run batch analysis first to unlock insights.")

st.caption("Built with XGBoost, SMOTE, FastAPI, Streamlit, and Plotly")