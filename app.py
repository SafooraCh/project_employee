# =============================================================================
# EMPLOYEE PRODUCTIVITY PREDICTION APP
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            "experience_years": np.random.randint(0, 30, 200),
            "training_hours": np.random.randint(10, 200, 200),
            "projects_completed": np.random.randint(1, 50, 200),
            "productivity_score": np.random.uniform(50, 100, 200)
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

st.sidebar.markdown("### 🤖 Select ML Algorithm")
model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["Linear Regression", "Random Forest", "Support Vector Machine"]
)

# =============================================================================
# PREPARE DATA
# =============================================================================
X = df.drop(columns=["productivity_score"])
y = df["productivity_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# TRAIN MODEL (BASED ON USER SELECTION)
# =============================================================================
if model_choice == "Linear Regression":
    model = LinearRegression()

elif model_choice == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

elif model_choice == "Support Vector Machine":
    model = SVR(kernel="rbf")

model.fit(X_train_scaled, y_train)

# =============================================================================
# EVALUATE MODEL
# =============================================================================
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎯 Employee Productivity Prediction System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Employees", len(df))
    col2.metric("Features", X.shape[1])
    col3.metric("Avg Productivity", round(y.mean(), 2))

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
        st.plotly_chart(
            px.histogram(df, x=feature, nbins=30),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.box(df, y=feature),
            use_container_width=True
        )

    st.subheader("📊 Scatter Plot")
    x_axis = st.selectbox("X Axis", numeric_cols)
    y_axis = st.selectbox("Y Axis", numeric_cols, index=1)

    st.plotly_chart(
        px.scatter(df, x=x_axis, y=y_axis),
        use_container_width=True
    )

# =============================================================================
# PREDICTION
# =============================================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Predict Employee Productivity")
    st.info(f"🤖 Selected Model: **{model_choice}**")

    input_data = {}

    for feature in X.columns:
        input_data[feature] = st.slider(
            feature,
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(
            f"✅ Predicted Productivity Score: **{prediction:.2f}**"
        )

# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", round(r2, 4))
    col2.metric("RMSE", round(rmse, 4))
    col3.metric("MAE", round(mae, 4))

    st.subheader("📋 Features Used")
    st.write(list(X.columns))

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>Employee Productivity Prediction App | Course Project</p>",
    unsafe_allow_html=True
)
