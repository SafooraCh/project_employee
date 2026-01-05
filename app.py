# =============================================================================
# EMPLOYEE PRODUCTIVITY PREDICTION APP
# Streamlit Course Project
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# AVAILABLE MODELS (ADD MORE IF YOU HAVE)
# =============================================================================
MODEL_FILES = {
    "Linear Regression": "linear_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Support Vector Machine": "svm_model.pkl"
}

SCALER_FILE = "scaler.pkl"
METADATA_FILE = "model_metadata.pkl"

# =============================================================================
# LOAD MODEL SAFELY
# =============================================================================
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(SCALER_FILE)
        metadata = joblib.load(METADATA_FILE)

        if "features" in metadata:
            feature_names = metadata["features"]
        else:
            feature_names = None

        return model, scaler, feature_names, metadata

    except Exception as e:
        st.error("❌ Failed to load model files")
        st.exception(e)
        return None, None, None, None

# =============================================================================
# LOAD DATASET
# =============================================================================
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("employee_productivity.csv")
    except:
        st.warning("⚠ Dataset not found. Using sample data.")
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.randint(22, 60, 100),
            "experience_years": np.random.randint(0, 30, 100),
            "training_hours": np.random.randint(10, 200, 100),
            "projects_completed": np.random.randint(1, 50, 100),
            "productivity_score": np.random.uniform(50, 100, 100)
        })

df = load_dataset()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📈 Data Explorer", "🎯 Make Prediction", "📊 Model Performance"]
)

# MODEL SELECTION
st.sidebar.markdown("### 🤖 Select ML Model")
selected_model_name = st.sidebar.selectbox(
    "Choose model:",
    list(MODEL_FILES.keys())
)

model_path = MODEL_FILES[selected_model_name]
model, scaler, feature_names, metadata = load_model(model_path)

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎯 Employee Productivity Prediction System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Employees", len(df))
    col2.metric("Features", len(feature_names) if feature_names else "N/A")
    col3.metric(
        "Avg Productivity",
        round(df.select_dtypes(np.number).iloc[:, -1].mean(), 2)
    )

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# DATA EXPLORER
# =============================================================================
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    feature = st.selectbox("Select Feature", numeric_cols)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feature, nbins=30, title="Histogram")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, y=feature, title="Box Plot")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Scatter Plot")
    x_axis = st.selectbox("X Axis", numeric_cols)
    y_axis = st.selectbox("Y Axis", numeric_cols, index=1)

    fig = px.scatter(df, x=x_axis, y=y_axis)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PREDICTION PAGE
# =============================================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Predict Employee Productivity")

    st.info(f"🤖 Selected Model: **{selected_model_name}**")

    if model is None or scaler is None or feature_names is None:
        st.error("Model not properly loaded.")
    else:
        input_data = {}

        for feature in feature_names:
            if feature in df.columns:
                input_data[feature] = st.slider(
                    feature,
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].mean())
                )
            else:
                input_data[feature] = st.number_input(feature, value=0.0)

        if st.button("🚀 Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]

                st.success(
                    f"✅ **Predicted Productivity Score:** {prediction:.2f}"
                )

            except Exception as e:
                st.error("Prediction failed")
                st.exception(e)

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    if metadata:
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", round(metadata.get("r2_score", 0), 4))
        col2.metric("RMSE", round(metadata.get("rmse", 0), 4))
        col3.metric("MAE", round(metadata.get("mae", 0), 4))

        st.subheader("📋 Features Used")
        st.write(metadata.get("features", []))
    else:
        st.error("Metadata not available")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>Employee Productivity Prediction App | Course Project</p>",
    unsafe_allow_html=True
)
