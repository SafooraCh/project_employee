# ==============================================================================
# EMPLOYEE PRODUCTIVITY STREAMLIT APP - FULL VERSION FOR KAGGLE
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(page_title="Employee Productivity App",
                   page_icon="📊", layout="wide")

st.title("📊 Employee Productivity Analysis - Full Version")
st.markdown("---")

# ==============================================================================
# LOAD DATA FUNCTION
# ==============================================================================
@st.cache_data
def load_data():
    path = "/kaggle/input/employee-productivity/employee_productivity.csv"
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at {path}")
        return None

# ==============================================================================
# LOAD MODEL FUNCTION
# ==============================================================================
@st.cache_resource
def load_model():
    try:
        with open('/kaggle/working/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('/kaggle/working/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('/kaggle/working/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except:
        st.warning("⚠️ Model not found. Run the notebook first to train and save it.")
        return None, None, None

# ==============================================================================
# LOAD DATA
# ==============================================================================
df = load_data()
model, scaler, encoders = load_model()  # load model for prediction

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Select a page:", ["🏠 Home & EDA", "📈 Visualizations", "🤖 Predict"])

# ==============================================================================
# PAGE 1: HOME & EDA
# ==============================================================================
if page == "🏠 Home & EDA":
    if df is not None:
        st.header("👋 Home & Exploratory Data Analysis")

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("ℹ️ Dataset Info")
        buffer = []
        df.info(buf=buffer)
        info_str = "\n".join(buffer)
        st.text(info_str)

        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("❓ Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values})
            missing_df = missing_df[missing_df["Missing"] > 0]
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values")

        st.subheader("🔢 Column Types")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write(f"Numerical columns: {num_cols}")
        st.write(f"Categorical columns: {cat_cols}")

# ==============================================================================
# PAGE 2: VISUALIZATIONS
# ==============================================================================
elif page == "📈 Visualizations":
    if df is not None:
        st.header("📈 Data Visualizations")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        st.subheader("🔢 Numerical Columns - Histogram & Boxplot")
        for col in num_cols:
            st.markdown(f"### {col}")
            fig1 = px.histogram(df, x=col, nbins=30, title=f'Distribution of {col}')
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.box(df, y=col, title=f'Boxplot of {col}')
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📝 Categorical Columns - Bar Charts")
        for col in cat_cols:
            st.markdown(f"### {col}")
            counts = df[col].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values, title=f'Top values in {col}')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: PREDICTION
# ==============================================================================
elif page == "🤖 Predict":
    if df is not None and model is not None:
        st.header("🎯 Make Predictions")
        st.markdown("Enter values below to predict employee productivity")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        num_cols = num_cols[:-1]  # remove target column

        input_data = {}
        col1, col2 = st.columns(2)
        for i, col in enumerate(num_cols):
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()),
                                                  help=f"Min: {df[col].min()}, Max: {df[col].max()}")

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            try:
                pred = model.predict(input_df)
                st.success(f"✅ Prediction: {pred[0]:.2f}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Model not found. Run the notebook first to train and save the model.")
