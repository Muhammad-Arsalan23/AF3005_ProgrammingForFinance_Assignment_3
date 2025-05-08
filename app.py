import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime
import uuid

# Set page configuration
st.set_page_config(
    page_title="FinanceML Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent theme
st.markdown("""
<style>
:root {
    --primary: #1E88E5;
    --secondary: #FFD54F;
    --background: #121212;
    --surface: #1E1E1E;
    --text: #E0E0E0;
}

.main {
    background-color: var(--background);
}

h1, h2, h3 {
    color: var(--secondary) !important;
    font-family: 'Arial', sans-serif;
}

.stButton>button {
    background-color: var(--primary);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: var(--secondary);
    color: black;
}

.stTextInput>div>div>input,
.stSelectbox>div>div>select {
    background-color: var(--surface);
    color: var(--text);
    border-radius: 5px;
}

.stSidebar {
    background-color: var(--surface);
}

.stSpinner {
    color: var(--primary);
}
</style>
""", unsafe_allow_html=True)

# Custom Plotly template
plotly_template = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#E0E0E0"},
        "colorway": ["#1E88E5", "#FFD54F", "#EF5350"]
    }
}

# Sidebar navigation
st.sidebar.title("FinanceML Analytics")
page = st.sidebar.radio("Navigate", ["Welcome", "ML Pipeline"])

# Helper functions
def load_kaggle_data(file):
    """Load and validate Kragle dataset"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("Uploaded dataset is empty!")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def fetch_yahoo_data(symbol):
    """Fetch stock data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period="1y", progress=False)
        if df.empty:
            st.error("No data found for the given symbol!")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching Yahoo Finance data: {str(e)}")
        return None

def preprocess_data(df, features):
    """Preprocess data by handling missing values and scaling"""
    try:
        imputer = SimpleImputer(strategy='mean')
        df[features] = imputer.fit_transform(df[features])
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        return df.dropna()
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def feature_importance(model, features):
    """Calculate and visualize feature importance"""
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        fig = px.bar(x=features, y=importance, title="Feature Importance",
                    labels={"x": "Features", "y": "Importance"},
                    template=plotly_template)
        return fig
    return None

# Welcome Page
if page == "Welcome":
    st.markdown("<h1 style='text-align: center;'>Welcome to FinanceML Analytics 📈</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color: var(--text);'>A cutting-edge platform for financial data analysis using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display finance-themed GIF
    st.image("https://media.giphy.com/media/3o7TKsQ8kEkf5bZ6p2/giphy.gif", width=300)
    
    st.markdown("### Get Started")
    st.markdown("""
    - Upload a Kragle dataset or fetch real-time stock data from Yahoo Finance.
    - Follow the ML pipeline to preprocess, train, and evaluate models.
    - Visualize results with interactive Plotly charts.
    """)

# ML Pipeline Page
else:
    st.markdown("<h1>Interactive ML Pipeline</h1>", unsafe_allow_html=True)
    
    # Sidebar for data input
    with st.sidebar:
        st.subheader("Data Source")
        data_source = st.radio("Select Source", ["Kragle Dataset", "Yahoo Finance"])
        
        if data_source == "Kragle Dataset":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        else:
            symbol = st.text_input("Stock Symbol", "AAPL")
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.selectbox("Choose Model", ["Linear Regression", "Logistic Regression", "K-Means Clustering"])
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'features' not in st.session_state:
        st.session_state.features = []
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

    # Step 1: Load Data
    st.subheader("Step 1: Load Data")
    if st.button("Load Data"):
        if data_source == "Kragle Dataset" and uploaded_file:
            st.session_state.df = load_kaggle_data(uploaded_file)
        elif data_source == "Yahoo Finance" and symbol:
            st.session_state.df = fetch_yahoo_data(symbol)
        
        if st.session_state.df is not None:
            st.success("Data loaded successfully!")
            st.dataframe(st.session_state.df.head())
    
    # Step 2: Preprocessing
    if st.session_state.df is not None:
        st.subheader("Step 2: Preprocessing")
        st.write(f"Missing values:\n{st.session_state.df.isnull().sum()}")
        
        if st.button("Preprocess Data"):
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            st.session_state.df = preprocess_data(st.session_state.df, numeric_cols)
            if st.session_state.df is not None:
                st.success("Data preprocessed successfully!")
                st.dataframe(st.session_state.df.head())
    
    # Step 3: Feature Engineering
    if st.session_state.df is not None:
        st.subheader("Step 3: Feature Engineering")
        st.session_state.features = st.multiselect("Select Features", st.session_state.df.columns)
        
        if model_type != "K-Means Clustering":
            st.session_state.target = st.selectbox("Select Target", st.session_state.df.columns)
        
        if st.button("Confirm Features"):
            if st.session_state.features and (st.session_state.target or model_type == "K-Means Clustering"):
                st.success("Features selected successfully!")
            else:
                st.error("Please select features and target (if applicable)!")
    
    # Step 4: Train/Test Split
    if st.session_state.features:
        st.subheader("Step 4: Train/Test Split")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        if st.button("Split Data"):
            try:
                X = st.session_state.df[st.session_state.features]
                if model_type != "K-Means Clustering":
                    y = st.session_state.df[st.session_state.target]
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42)
                else:
                    st.session_state.X_train = X
                    st.session_state.X_test = X
                
                st.success("Data split successfully!")
                
                # Visualize split
                sizes = [len(st.session_state.X_train), len(st.session_state.X_test)]
                fig = px.pie(values=sizes, names=["Training", "Testing"], title="Train/Test Split",
                            template=plotly_template)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
    
    # Step 5: Model Training
    if st.session_state.X_train is not None:
        st.subheader("Step 5: Model Training")
        if st.button("Train Model"):
            try:
                if model_type == "Linear Regression":
                    st.session_state.model = LinearRegression()
                    st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                elif model_type == "Logistic Regression":
                    st.session_state.model = LogisticRegression()
                    st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                else:  # K-Means Clustering
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    st.session_state.model = KMeans(n_clusters=n_clusters, random_state=42)
                    st.session_state.model.fit(st.session_state.X_train)
                
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Step 6: Evaluation
    if st.session_state.model is not None:
        st.subheader("Step 6: Evaluation")
        if st.button("Evaluate Model"):
            try:
                if model_type == "Linear Regression":
                    st.session_state.predictions = st.session_state.model.predict(st.session_state.X_test)
                    mae = mean_absolute_error(st.session_state.y_test, st.session_state.predictions)
                    rmse = np.sqrt(mean_squared_error(st.session_state.y_test, st.session_state.predictions))
                    r2 = r2_score(st.session_state.y_test, st.session_state.predictions)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MAE", f"{mae:.2f}")
                    col2.metric("RMSE", f"{rmse:.2f}")
                    col3.metric("R²", f"{r2:.2f}")
                
                elif model_type == "Logistic Regression":
                    st.session_state.predictions = st.session_state.model.predict(st.session_state.X_test)
                    accuracy = accuracy_score(st.session_state.y_test, st.session_state.predictions)
                    st.metric("Accuracy", f"{accuracy:.2f}")
                
                else:  # K-Means Clustering
                    st.session_state.predictions = st.session_state.model.predict(st.session_state.X_test)
                
                st.success("Model evaluated successfully!")
                
                # Feature importance visualization
                if model_type in ["Linear Regression", "Logistic Regression"]:
                    fig = feature_importance(st.session_state.model, st.session_state.features)
                    if fig:
                        st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")
    
    # Step 7: Results Visualization
    if st.session_state.predictions is not None:
        st.subheader("Step 7: Results Visualization")
        if st.button("Visualize Results"):
            try:
                if model_type in ["Linear Regression", "Logistic Regression"]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=st.session_state.y_test, name="Actual", mode="lines",
                                           line=dict(color="#1E88E5")))
                    fig.add_trace(go.Scatter(y=st.session_state.predictions, name="Predicted", mode="lines",
                                           line=dict(color="#FFD54F")))
                    fig.update_layout(title="Actual vs Predicted", template=plotly_template)
                    st.plotly_chart(fig)
                
                else:  # K-Means Clustering
                    df_viz = st.session_state.X_test.copy()
                    df_viz['Cluster'] = st.session_state.predictions
                    if len(st.session_state.features) >= 2:
                        fig = px.scatter(df_viz, x=st.session_state.features[0], y=st.session_state.features[1],
                                       color='Cluster', title="Cluster Visualization",
                                       template=plotly_template)
                        st.plotly_chart(fig)
                
                # Download results
                results_df = pd.DataFrame({
                    'Predictions': st.session_state.predictions
                })
                if model_type != "K-Means Clustering":
                    results_df['Actual'] = st.session_state.y_test.values
                
                st.download_button(
                    label="Download Results",
                    data=results_df.to_csv(index=False),
                    file_name="ml_results.csv",
                    mime="text/csv"
                )
                
                st.success("Results visualized successfully!")
            except Exception as e:
                st.error(f"Error visualizing results: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: var(--text);'>"
            "FinanceML Analytics | Built for AF3005 Assignment 3</div>", unsafe_allow_html=True)
