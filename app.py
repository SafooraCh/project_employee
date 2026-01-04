"""
================================================================================
EMPLOYEE PRODUCTIVITY PREDICTION APP
Streamlit Application for Course Project
================================================================================
This app allows users to:
1. Explore the employee productivity dataset
2. View model performance
3. Make predictions for new employees
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# =================================================================================
# PAGE CONFIGURATION
# =================================================================================
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================================
# CUSTOM CSS FOR BETTER STYLING
# =================================================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2E86AB;
    }
    h2 {
        color: #A23B72;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# =================================================================================
# LOAD MODEL AND DATA
# =================================================================================
@st.cache_resource
def load_model_files():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('final_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, scaler, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please make sure all .pkl files are in the same directory as app.py")
        return None, None, None, None

@st.cache_data
def load_dataset():
    """Load the original dataset for exploration"""
    try:
        # Try to load the dataset
        df = pd.read_csv('employee_productivity.csv')
        return df
    except:
        # If file not found, create sample data
        st.warning("Dataset file not found. Using sample data for demonstration.")
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'age': np.random.randint(22, 60, 100),
            'experience_years': np.random.randint(0, 30, 100),
            'training_hours': np.random.randint(10, 200, 100),
            'projects_completed': np.random.randint(1, 50, 100),
            'productivity_score': np.random.uniform(50, 100, 100)
        })
        return sample_data

# Load everything
model, scaler, feature_names, metadata = load_model_files()
df = load_dataset()

# =================================================================================
# SIDEBAR NAVIGATION
# =================================================================================
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📈 Data Explorer", "🎯 Make Prediction", "📊 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **About This App:**
    
    This application predicts employee productivity based on various factors.
    
    Built with Streamlit for the course project.
""")

# =================================================================================
# PAGE 1: HOME
# =================================================================================
if page == "🏠 Home":
    # Title
    st.title("🎯 Employee Productivity Prediction System")
    st.markdown("### Welcome to the Employee Productivity Analysis Platform")
    
    st.markdown("---")
    
    # Overview section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset</h3>
            <p>Explore employee data with various features and visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 ML Model</h3>
            <p>Trained machine learning model for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Predictions</h3>
            <p>Get instant productivity predictions for employees</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display key metrics
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        if 'productivity_score' in df.columns:
            st.metric("Avg Productivity", f"{df['productivity_score'].mean():.2f}")
        else:
            # Use the last numeric column as productivity
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.metric("Avg Target", f"{df[numeric_cols[-1]].mean():.2f}")
    
    with col3:
        st.metric("Features Used", len(feature_names) if feature_names else "N/A")
    
    with col4:
        if metadata:
            st.metric("Model Accuracy", f"{metadata.get('r2_score', 0):.2%}")
    
    st.markdown("---")
    
    # Model information
    if metadata:
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {metadata.get('model_name', 'N/A')}")
            st.write(f"**Target Variable:** {metadata.get('target', 'N/A')}")
            st.write(f"**Number of Features:** {len(metadata.get('features', []))}")
        
        with col2:
            st.write(f"**R² Score:** {metadata.get('r2_score', 0):.4f}")
            st.write(f"**RMSE:** {metadata.get('rmse', 0):.4f}")
            st.write(f"**MAE:** {metadata.get('mae', 0):.4f}")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.subheader("🚀 How to Use This App")
    
    st.markdown("""
    1. **📈 Data Explorer**: View and analyze the dataset with interactive visualizations
    2. **🎯 Make Prediction**: Enter employee details to predict productivity
    3. **📊 Model Performance**: See detailed model evaluation metrics
    
    Use the sidebar to navigate between different sections!
    """)

# =================================================================================
# PAGE 2: DATA EXPLORER
# =================================================================================
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("### Explore the Employee Productivity Dataset")
    
    st.markdown("---")
    
    # Dataset Overview
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Distributions", "🔗 Correlations", "📋 Statistics"])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("---")
        
        st.subheader("First 20 Rows")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Distributions")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Select column to visualize
            selected_col = st.selectbox("Select a feature to visualize:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, 
                                 title=f"Distribution of {selected_col}",
                                 nbins=30,
                                 color_discrete_sequence=['#2E86AB'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col,
                           title=f"Box Plot of {selected_col}",
                           color_discrete_sequence=['#A23B72'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader(f"Statistics for {selected_col}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[selected_col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[selected_col].median():.2f}")
            with col3:
                st.metric("Min", f"{df[selected_col].min():.2f}")
            with col4:
                st.metric("Max", f"{df[selected_col].max():.2f}")
        else:
            st.warning("No numeric columns found in the dataset.")
    
    with tab3:
        st.subheader("Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix,
                          text_auto='.2f',
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          title="Correlation Heatmap")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            st.markdown("---")
            st.subheader("Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis:", numeric_cols, key='scatter_x')
            with col2:
                y_axis = st.selectbox("Select Y-axis:", numeric_cols, key='scatter_y', 
                                    index=min(1, len(numeric_cols)-1))
            
            fig = px.scatter(df, x=x_axis, y=y_axis,
                           title=f"{y_axis} vs {x_axis}",
                           trendline="ols",
                           color_discrete_sequence=['#F18F01'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    with tab4:
        st.subheader("Statistical Summary")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.write("**Numeric Features Summary:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Categorical columns summary
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            st.write("**Categorical Features Summary:**")
            for col in cat_cols:
                st.write(f"**{col}:**")
                st.write(df[col].value_counts())
                st.markdown("---")

# =================================================================================
# PAGE 3: MAKE PREDICTION
# =================================================================================
elif page == "🎯 Make Prediction":
    st.title("🎯 Make Productivity Prediction")
    st.markdown("### Enter employee details to predict productivity")
    
    if model is None or scaler is None or feature_names is None:
        st.error("⚠️ Model files not loaded. Please ensure all .pkl files are present.")
    else:
        st.markdown("---")
        
        # Create input form
        st.subheader("📝 Enter Employee Information")
        
        # Get numeric columns from original data for reference
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create input fields based on feature names
        input_data = {}
        
        # Organize inputs in columns
        col1, col2 = st.columns(2)
        
        # Split features between two columns
        mid_point = len(feature_names) // 2
        
        with col1:
            for i, feature in enumerate(feature_names[:mid_point]):
                # Check if feature exists in original data to get min/max
                if feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    # Default numeric input
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        key=f"input_{feature}"
                    )
        
        with col2:
            for i, feature in enumerate(feature_names[mid_point:]):
                if feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    
                    input_data[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        key=f"input_{feature}"
                    )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🚀 Predict Productivity", type="primary"):
            try:
                # Create dataframe from input
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct column order
                input_df = input_df[feature_names]
                
                # Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                # Show prediction in a nice card
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 30px; background-color: #e8f5e9; 
                                border-radius: 10px; border: 2px solid #4CAF50;'>
                        <h2 style='color: #2E7D32;'>Predicted Productivity Score</h2>
                        <h1 style='color: #1B5E20; font-size: 60px;'>{prediction:.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Performance category
                if prediction >= 80:
                    performance = "Excellent Performance"
                    color = "green"
                    emoji = "🌟"
                elif prediction >= 65:
                    performance = "Good Performance"
                    color = "blue"
                    emoji = "👍"
                elif prediction >= 50:
                    performance = "Average Performance"
                    color = "orange"
                    emoji = "📊"
                else:
                    performance = "Needs Improvement"
                    color = "red"
                    emoji = "📈"
                
                st.markdown(f"### {emoji} Category: :{color}[{performance}]")
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                
                if prediction < 60:
                    st.markdown("""
                    - 📚 Consider additional training programs
                    - 👥 Increase mentorship and guidance
                    - 🎯 Set clear, achievable goals
                    - 💪 Provide regular feedback and support
                    """)
                elif prediction < 80:
                    st.markdown("""
                    - ✅ Maintain current performance standards
                    - 📈 Look for growth opportunities
                    - 🎓 Encourage skill development
                    - 🤝 Foster team collaboration
                    """)
                else:
                    st.markdown("""
                    - 🏆 Excellent performance! Keep it up!
                    - 👔 Consider for leadership roles
                    - 🎯 Challenge with complex projects
                    - 🌟 Recognize and reward achievements
                    """)
                
                # Show input summary
                st.markdown("---")
                st.subheader("📋 Input Summary")
                input_summary = pd.DataFrame([input_data]).T
                input_summary.columns = ['Value']
                st.dataframe(input_summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please check if all input values are valid.")

# =================================================================================
# PAGE 4: MODEL PERFORMANCE
# =================================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    st.markdown("### Detailed evaluation metrics and insights")
    
    if metadata is None:
        st.error("⚠️ Model metadata not loaded.")
    else:
        st.markdown("---")
        
        # Performance metrics
        st.subheader("🎯 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{metadata.get('r2_score', 0):.4f}")
            st.caption("Closer to 1.0 is better")
        
        with col2:
            st.metric("RMSE", f"{metadata.get('rmse', 0):.4f}")
            st.caption("Lower is better")
        
        with col3:
            st.metric("MAE", f"{metadata.get('mae', 0):.4f}")
            st.caption("Lower is better")
        
        with col4:
            accuracy_percent = metadata.get('r2_score', 0) * 100
            st.metric("Accuracy", f"{accuracy_percent:.2f}%")
            st.caption("Overall accuracy")
        
        st.markdown("---")
        
        # Model information
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {metadata.get('model_name', 'N/A')}")
            st.write(f"**Target Variable:** {metadata.get('target', 'N/A')}")
            st.write(f"**Number of Features:** {len(metadata.get('features', []))}")
        
        with col2:
            st.write("**What the metrics mean:**")
            st.write("- **R² Score**: How well the model explains the data (0-1)")
            st.write("- **RMSE**: Average prediction error")
            st.write("- **MAE**: Mean absolute error in predictions")
        
        st.markdown("---")
        
        # Feature list
        st.subheader("📋 Features Used in Model")
        
        if 'features' in metadata:
            features_df = pd.DataFrame({
                'Feature Name': metadata['features'],
                'Feature Number': range(1, len(metadata['features']) + 1)
            })
            st.dataframe(features_df, use_container_width=True)
        
        st.markdown("---")
        
        # Model interpretation
        st.subheader("📖 Model Interpretation")
        
        r2 = metadata.get('r2_score', 0)
        
        if r2 >= 0.9:
            interpretation = "🌟 Excellent! The model explains over 90% of the variance in the data."
        elif r2 >= 0.8:
            interpretation = "👍 Very Good! The model performs well with good predictive power."
        elif r2 >= 0.7:
            interpretation = "✅ Good! The model shows reasonable predictive capability."
        elif r2 >= 0.6:
            interpretation = "📊 Acceptable! The model has moderate predictive power."
        else:
            interpretation = "⚠️ Fair! The model might benefit from more data or features."
        
        st.info(interpretation)
        
        st.markdown("---")
        
        # Tips
        st.subheader("💡 Tips for Better Predictions")
        st.markdown("""
        1. **More Data**: Collect more employee records for better accuracy
        2. **Feature Engineering**: Create new meaningful features from existing data
        3. **Regular Updates**: Retrain the model periodically with new data
        4. **Data Quality**: Ensure input data is clean and accurate
        5. **Domain Knowledge**: Incorporate insights from HR experts
        """)

# =================================================================================
# FOOTER
# =================================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Employee Productivity Prediction System</strong></p>
        <p>Course Project | Data Science & Machine Learning | 2024</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)