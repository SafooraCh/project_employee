"""
================================================================================
EMPLOYEE PRODUCTIVITY PREDICTION APP
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# LOAD MODEL FILES (FIXED)
# =============================================================================
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("final_model.pkl")
        scaler = joblib.load("scaler.pkl")
        metadata = joblib.load("model_metadata.pkl")

        # Get feature names safely
        if metadata and "features" in metadata:
            feature_names = metadata["features"]
        elif hasattr(scaler, "feature_names_in_"):
            feature_names = list(scaler.feature_names_in_)
        else:
            feature_names = None

        return model, scaler, feature_names, metadata

    except Exception as e:
        st.error("❌ Model files could not be loaded")
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
        st.warning("Dataset not found. Using sample data.")
        np.random.seed(42)
        return pd.DataFrame({
            "age": np.random.randint(22, 60, 100),
            "experience_years": np.random.randint(0, 30, 100),
            "training_hours": np.random.randint(10, 200, 100),
            "projects_completed": np.random.randint(1, 50, 100),
            "productivity_score": np.random.uniform(50, 100, 100)
        })

model, scaler, feature_names, metadata = load_model_files()
df = load_dataset()

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📈 Data Explorer", "🎯 Make Prediction", "📊 Model Performance"]
)

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎯 Employee Productivity Prediction System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Employees", len(df))
    col2.metric("Features", len(feature_names) if feature_names else "N/A")

    if "productivity_score" in df.columns:
        col3.metric("Avg Productivity", round(df["productivity_score"].mean(), 2))
    else:
        col3.metric("Avg Target", "N/A")

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# DATA EXPLORER
# =============================================================================
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Select Feature", numeric_cols)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df, x=selected_col), use_container_width=True)
        with col2:
            st.plotly_chart(px.box(df, y=selected_col), use_container_width=True)

        st.subheader("📊 Scatter Plot")
        x_axis = st.selectbox("X Axis", numeric_cols)
        y_axis = st.selectbox("Y Axis", numeric_cols, index=1)

        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)

# =============================================================================
# PREDICTION
# =============================================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Predict Productivity")

    if model is None or scaler is None or feature_names is None:
        st.error("⚠️ Model not ready. Check model files.")
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

        if st.button("🚀 Predict Productivity"):
            try:
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]

                st.success(f"✅ Predicted Productivity Score: **{prediction:.2f}**")

            except Exception as e:
                st.error("❌ Prediction failed")
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
