import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tpot import TPOTClassifier, TPOTRegressor
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier

# Set up the Streamlit app
st.title("Machine Learning App with Genetic Programming")

st.sidebar.title("Upload or Select Dataset")
data_source = st.sidebar.radio("Choose data source:", ("Upload", "Pre-existing"))

df = None  # Initialize df to avoid NameError

# Handle data upload or selection
if data_source == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset successfully loaded.")
    else:
        st.write("Please upload a valid CSV file.")
else:
    dataset_name = st.sidebar.selectbox("Select a dataset", ["Iris", "Breast Cancer", "Wine", "Diabetes", "Boston Housing"])
    if dataset_name == "Iris":
        df = datasets.load_iris(as_frame=True).frame
    elif dataset_name == "Breast Cancer":
        df = datasets.load_breast_cancer(as_frame=True).frame
    elif dataset_name == "Wine":
        df = datasets.load_wine(as_frame=True).frame
    elif dataset_name == "Diabetes":
        df = datasets.load_diabetes(as_frame=True).frame
    elif dataset_name == "Boston Housing":
        df = datasets.load_boston(as_frame=True).frame

# Check if df is defined before attempting to use it
if df is not None:
    st.write("### Dataset Preview", df.head())

    # Feature selection and target
    st.sidebar.subheader("Features & Target")

    # Option to select all features
    select_all = st.sidebar.checkbox("Select all features")

    if select_all:
        features = df.columns[:-1].tolist()
    else:
        features = st.sidebar.multiselect("Select features", df.columns[:-1])

    target = st.sidebar.selectbox("Select target", df.columns)

    X = df[features]
    y = df[target]

    # Data Visualization for Outliers
    st.sidebar.subheader("Data Visualization")
    visualize_features = st.sidebar.multiselect("Select features to visualize", features)

    if visualize_features:
        for feature in visualize_features:
            st.write(f"### Distribution of {feature}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[feature], ax=ax)
            st.pyplot(fig)

    # Data Transformation
    st.sidebar.subheader("Data Transformation")
    transformation = st.sidebar.selectbox("Select transformation technique", ["None", "Normalization (Min-Max Scaling)", "Standardization (Z-Score)", "Robust Scaling"])

    if transformation == "Normalization (Min-Max Scaling)":
        scaler = MinMaxScaler()
        X[features] = scaler.fit_transform(X[features])
    elif transformation == "Standardization (Z-Score)":
        scaler = StandardScaler()
        X[features] = scaler.fit_transform(X[features])
    elif transformation == "Robust Scaling":
        scaler = RobustScaler()
        X[features] = scaler.fit_transform(X[features])

    # Model selection
    st.sidebar.subheader("Choose Model")
    model_name = st.sidebar.selectbox("Model", [
        "Decision Tree Classifier", 
        "Random Forest Classifier", 
        "SVM Classifier", 
        "Logistic Regression", 
        "Naive Bayes", 
        "K-Nearest Neighbors Classifier", 
        "KMeans", 
        "Linear Regression", 
        "Decision Tree Regressor", 
        "Random Forest Regressor", 
        "SVM Regressor", 
        "K-Nearest Neighbors Regressor", 
        "TPOT Classifier (Genetic Algorithm)", 
        "TPOT Regressor (Genetic Algorithm)", 
        "Symbolic Classifier (Genetic Programming)", 
        "Symbolic Regressor (Genetic Programming)"
    ])

    # Hyperparameter selection
    st.sidebar.subheader("Hyperparameter Selection")

    # Initialize the model
    if model_name == "Decision Tree Classifier":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif model_name == "Random Forest Classifier":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "SVM Classifier":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVC(C=C, kernel=kernel)
    elif model_name == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "K-Nearest Neighbors Classifier":
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "KMeans":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree Regressor":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_name == "Random Forest Regressor":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "SVM Regressor":
        C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        model = SVR(C=C, kernel=kernel)
    elif model_name == "K-Nearest Neighbors Regressor":
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_name == "TPOT Classifier (Genetic Algorithm)":
        generations = st.sidebar.slider("Generations", 5, 50, 10)
        population_size = st.sidebar.slider("Population Size", 20, 100, 50)
        model = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=42)
    elif model_name == "TPOT Regressor (Genetic Algorithm)":
        generations = st.sidebar.slider("Generations", 5, 50, 10)
        population_size = st.sidebar.slider("Population Size", 20, 100, 50)
        model = TPOTRegressor(generations=generations, population_size=population_size, verbosity=2, random_state=42)
    elif model_name == "Symbolic Classifier (Genetic Programming)":
        generations = st.sidebar.slider("Generations", 10, 100, 20)
        population_size = st.sidebar.slider("Population Size", 100, 1000, 500)
        model = SymbolicClassifier(generations=generations, population_size=population_size, verbose=1, random_state=42)
    elif model_name == "Symbolic Regressor (Genetic Programming)":
        generations = st.sidebar.slider("Generations", 10, 100, 20)
        population_size = st.sidebar.slider("Population Size", 100, 1000, 500)
        model = SymbolicRegressor(generations=generations, population_size=population_size, verbose=1, random_state=42)

    # Train-test split and model training
    st.sidebar.subheader("Model Parameters")
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
    k_folds = st.sidebar.slider("Number of Folds for Cross-Validation", 2, 10, 5)

    X_train, X_test
