import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Database functions
def init_database():
    conn = sqlite3.connect('creditcard_transactions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time REAL,
            v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL, v6 REAL, v7 REAL, v8 REAL, v9 REAL,
            v10 REAL, v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL,
            v20 REAL, v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,
            amount REAL,
            prediction INTEGER,
            actual_class INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(transaction_data, prediction, actual_class=None):
    conn = sqlite3.connect('creditcard_transactions.db')
    cursor = conn.cursor()
    
    columns = ['time'] + [f'v{i}' for i in range(1, 29)] + ['amount']
    placeholders = ','.join(['?'] * len(columns))
    
    cursor.execute(f'''
        INSERT INTO transactions ({','.join(columns)}, prediction, actual_class)
        VALUES ({placeholders}, ?, ?)
    ''', list(transaction_data) + [prediction, actual_class])
    
    conn.commit()
    conn.close()

def get_all_transactions():
    conn = sqlite3.connect('creditcard_transactions.db')
    df = pd.read_sql('SELECT * FROM transactions ORDER BY timestamp DESC', conn)
    conn.close()
    return df

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Initialize database
init_database()

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "View History", "Model Performance"])

# Load and train model - with demo data
@st.cache_resource
def load_data_and_train_model():
    try:
        # First try to load actual data
        df = pd.read_csv("creditcard.csv")
        st.success("Loaded creditcard.csv successfully!")
    except Exception as e:
        # Create demo data if file not found
        st.warning("creditcard.csv not found. Using demo data.")
        np.random.seed(42)
        n_samples = 408
        
        # Create demo dataset
        data = {}
        data['Time'] = np.random.normal(0, 1, n_samples)
        data['Amount'] = np.random.uniform(1, 1000, n_samples)
        
        # Create V1-V28 features
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        # Create target variable (Class)
        data['Class'] = np.concatenate([np.zeros(204), np.ones(204)])
        
        df = pd.DataFrame(data)
    
    df = df.dropna()
    
    # Balance the dataset as in your original code
    legitimate = df[df["Class"] == 0]
    fraudulent = df[df["Class"] == 1]
    legitimate_sample = legitimate.sample(n=len(fraudulent), random_state=42)
    balanced_df = pd.concat([legitimate_sample, fraudulent], axis=0)
    
    X = balanced_df.drop("Class", axis=1)
    y = balanced_df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, balanced_df.columns.tolist()

# Load model
model, X_test, y_test, feature_columns = load_data_and_train_model()

# Home Page
if page == "Home":
    st.header("Welcome to Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 About This System")
        st.write("""
        This machine learning system helps detect fraudulent credit card transactions 
        using Logistic Regression algorithm.
        
        **Features:**
        - Real-time fraud prediction
        - Transaction history tracking
        - Model performance analytics
        - Easy-to-use interface
        """)
    
    with col2:
        st.subheader("📊 Quick Stats")
        if model is not None:
            ypred = model.predict(X_test)
            accuracy = accuracy_score(y_test, ypred)
            
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            st.metric("Test Samples", len(X_test))
            st.metric("Features Used", len(feature_columns) - 1)

# Make Prediction Page
elif page == "Make Prediction":
    st.header("🔍 Make New Prediction")
    
    if model is None:
        st.error("Model not loaded. Please check your data file.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Transaction Details")
            
            amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0, step=10.0)
            time = st.number_input("Time", value=0.0, step=1.0)
            
            st.subheader("Feature Values")
            v1 = st.slider("V1", -5.0, 5.0, 0.0, 0.1)
            v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.1)
            v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.1)
            v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.1)
            v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.1)
            
        with col2:
            st.subheader("Prediction")
            
            if st.button("Check for Fraud", type="primary", use_container_width=True):
                # Prepare input data (set other V features to 0 for demo)
                input_data = [time, v1, v2, v3, v4, v5] + [0.0] * 23 + [amount]
                
                # Make prediction
                prediction = model.predict([input_data])[0]
                probability = model.predict_proba([input_data])[0]
                
                # Display results
                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED!")
                    st.metric("Fraud Probability", f"{probability[1]:.2%}")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                    st.metric("Legitimate Probability", f"{probability[0]:.2%}")
                
                # Save to database
                save_prediction(input_data, int(prediction))
                st.info("Prediction saved to database!")

# View History Page
elif page == "View History":
    st.header("📋 Prediction History")
    
    try:
        history_df = get_all_transactions()
        
        if len(history_df) > 0:
            st.subheader("Recent Predictions")
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(history_df))
            with col2:
                fraud_count = len(history_df[history_df['prediction'] == 1])
                st.metric("Fraud Predictions", fraud_count)
            with col3:
                fraud_rate = fraud_count / len(history_df) if len(history_df) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2%}")
            
            # Display data
            st.dataframe(history_df, use_container_width=True)
            
            # Download option
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv,
                file_name="fraud_predictions_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No predictions made yet. Go to 'Make Prediction' to get started!")
            
    except Exception as e:
        st.error(f"Error loading history: {e}")

# Model Performance Page
elif page == "Model Performance":
    st.header("📈 Model Performance")
    
    if model is not None and X_test is not None and y_test is not None:
        # Make predictions
        ypred = model.predict(X_test)
        yprob = model.predict_proba(X_test)[:, 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y_test, ypred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            accuracy = accuracy_score(y_test, ypred)
            st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, ypred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Fraud Detection System")