# ==============================================================================
# STREAMLIT APP - KAGGLE READY FOR EMPLOYEE PRODUCTIVITY
# ==============================================================================

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Employee Productivity App",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# TITLE
# ==============================================================================
st.title("📊 Employee Productivity Analysis")
st.markdown("---")

# ==============================================================================
# LOAD DATA FUNCTION
# ==============================================================================
@st.cache_data
def load_data():
    """Load the dataset from Kaggle input folder"""
    dataset_path = '/kaggle/input/employee-productivity/employee_productivity.csv'
    try:
        df = pd.read_csv(dataset_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at path: {dataset_path}")
        return None

# ==============================================================================
# LOAD MODEL FUNCTION
# ==============================================================================
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and encoders from Kaggle working folder"""
    try:
        with open('/kaggle/working/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('/kaggle/working/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('/kaggle/working/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except FileNotFoundError:
        st.warning("⚠️ Model files not found! Run the Kaggle notebook first to train and save the model.")
        return None, None, None

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.header("📱 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Data Explorer", "🤖 Make Prediction", "📈 Visualizations"]
)

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
if page == "🏠 Home":
    st.header("Welcome! 👋")
    st.markdown("""
    ### What can you do here?
    - 📊 **Explore Data**: View statistics and information about the dataset
    - 🤖 **Make Predictions**: Predict employee productivity
    - 📈 **Visualize Data**: View charts and correlations
    """)

    df = load_data()
    if df is not None:
        st.markdown("---")
        st.subheader("📋 Quick Dataset Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Rows", len(df))
        with col2:
            st.metric("📝 Total Columns", len(df.columns))
        with col3:
            st.metric("💾 Dataset Size", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        st.markdown("---")
        st.subheader("👀 Preview of Data")
        st.dataframe(df.head(10), use_container_width=True)

# ==============================================================================
# PAGE 2: DATA EXPLORER
# ==============================================================================
elif page == "📊 Data Explorer":
    st.header("Data Explorer 🔍")
    df = load_data()
    if df is not None:
        tab1, tab2 = st.tabs(["📊 Basic Statistics", "🔍 Column Details"])
        with tab1:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("Missing Values")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                missing_df = pd.DataFrame({'Column': missing.index, 'Missing': missing.values})
                missing_df = missing_df[missing_df['Missing'] > 0]
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values!")

        with tab2:
            st.subheader("Analyze Each Column")
            column = st.selectbox("Choose a column:", df.columns)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Column Information:**")
                st.write(f"- Data Type: {df[column].dtype}")
                st.write(f"- Unique Values: {df[column].nunique()}")
                st.write(f"- Missing Values: {df[column].isnull().sum()}")
                if df[column].dtype in ['int64', 'float64']:
                    st.write(f"- Mean: {df[column].mean():.2f}")
                    st.write(f"- Min: {df[column].min():.2f}")
                    st.write(f"- Max: {df[column].max():.2f}")
            with col2:
                st.write("**Visual:**")
                if df[column].dtype in ['int64', 'float64']:
                    fig = px.histogram(df, x=column, title=f'Distribution of {column}')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = df[column].value_counts().head(10)
                    fig = px.bar(x=counts.index, y=counts.values, title=f'Top values in {column}')
                    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: MAKE PREDICTION
# ==============================================================================
elif page == "🤖 Make Prediction":
    st.header("Make a Prediction 🎯")
    model, scaler, encoders = load_model()
    df = load_data()
    if model is not None and df is not None:
        st.markdown("### Enter values to predict:")

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = numerical_cols[:-1]  # Remove target

        input_data = {}
        col1, col2 = st.columns(2)
        for idx, col in enumerate(numerical_cols):
            with col1 if idx % 2 == 0 else col2:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(df[col].mean()),
                    help=f"Range: {df[col].min():.2f} to {df[col].max():.2f}"
                )

        if st.button("🎯 Predict Now!", type="primary"):
            input_df = pd.DataFrame([input_data])
            try:
                prediction = model.predict(input_df)
                st.success("✅ Prediction Complete!")
                st.markdown(f"## **Predicted Value: {prediction[0]:.2f}**")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please run the Kaggle notebook first to train the model!")

# ==============================================================================
# PAGE 4: VISUALIZATIONS
# ==============================================================================
elif page == "📈 Visualizations":
    st.header("Data Visualizations 📊")
    df = load_data()
    if df is not None:
        viz_type = st.selectbox(
            "Select chart type:",
            ["📊 Histogram", "📈 Line Chart", "🎯 Scatter Plot", "📦 Box Plot", "🔥 Heatmap"]
        )
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if viz_type == "📊 Histogram":
            column = st.selectbox("Select column:", numerical_cols)
            fig = px.histogram(df, x=column, title=f'Distribution of {column}')
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "📈 Line Chart":
            column = st.selectbox("Select column:", numerical_cols)
            fig = px.line(df, y=column, title=f'Line Chart of {column}')
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "🎯 Scatter Plot":
            col1, col2 = st.columns(2)
            with col1: x_col = st.selectbox("X-axis:", numerical_cols)
            with col2: y_col = st.selectbox("Y-axis:", numerical_cols)
            fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "📦 Box Plot":
            column = st.selectbox("Select column:", numerical_cols)
            fig = px.box(df, y=column, title=f'Box Plot of {column}')
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "🔥 Heatmap":
            corr = df[numerical_cols].corr()
            fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Employee Productivity Analysis | Course Project 2024</p>
    </div>
""", unsafe_allow_html=True)
