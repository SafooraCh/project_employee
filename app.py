# =============================================================================
# EMPLOYEE PRODUCTIVITY PREDICTION APP (100% ERROR-FREE)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="wide"
)

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
            "age": np.random.randint(22, 60, 200),
            "experience": np.random.randint(0, 30, 200),
            "training_hours": np.random.randint(10, 200, 200),
            "department": np.random.choice(["HR", "IT", "Sales"], 200),
            "productivity": np.random.uniform(50, 100, 200)
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

# ================= TARGET SELECTION =================
st.sidebar.markdown("### 🎯 Select Target Column")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

target_col = st.sidebar.selectbox(
    "Target column:",
    numeric_cols,
    index=len(numeric_cols) - 1
)

# ================= MODEL SELECTION =================
st.sidebar.markdown("### 🤖 Select ML Model")

model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["Linear Regression", "Random Forest", "Support Vector Machine"]
)

# =============================================================================
# SPLIT FEATURES / TARGET
# =============================================================================
X = df.drop(columns=[target_col])
y = df[target_col]

# Detect column types
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# =============================================================================
# PREPROCESSING PIPELINE (KEY FIX)
# =============================================================================
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# =============================================================================
# MODEL SELECTION
# =============================================================================
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

else:
    model = SVR(kernel="rbf")

# FULL PIPELINE
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# =============================================================================
# EVALUATION
# =============================================================================
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎯 Employee Productivity Prediction System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Features", X.shape[1])
    col3.metric("Avg Target", round(y.mean(), 2))

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# DATA EXPLORER
# =============================================================================
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")

    col = st.selectbox("Select Column", df.columns)

    if df[col].dtype != "object":
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)
    else:
        st.plotly_chart(px.bar(df[col].value_counts()), use_container_width=True)

# =============================================================================
# PREDICTION
# =============================================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Make Prediction")
    st.info(f"🤖 Model: **{model_choice}** | 🎯 Target: **{target_col}**")

    input_data = {}

    for col in X.columns:
        if col in numeric_features:
            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
        else:
            input_data[col] = st.selectbox(
                col,
                df[col].unique()
            )

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]

        st.success(f"✅ Predicted Value: **{prediction:.2f}**")

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", round(r2, 4))
    col2.metric("RMSE", round(rmse, 4))
    col3.metric("MAE", round(mae, 4))

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>Employee Productivity Prediction App | Course Project</p>",
    unsafe_allow_html=True
)
