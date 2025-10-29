import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# EMBEDDED CSS - No external file needed
st.markdown("""
<style>
:root {
  --primary: #3b82f6;
  --primary-dark: #1d4ed8;
  --secondary: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
  --success: #10b981;
  --dark: #1f2937;
  --light: #f8fafc;
  --gray: #6b7280;
}

.stApp {
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  font-family: 'Inter', sans-serif;
}

.main-header {
  font-size: 2.5rem !important;
  font-weight: 800 !important;
  background: linear-gradient(45deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-align: center;
  margin-bottom: 0.5rem !important;
}

.sub-header {
  font-size: 1.1rem !important;
  color: var(--gray) !important;
  text-align: center;
  margin-bottom: 2rem !important;
  font-weight: 400;
}

.glass-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  padding: 1.5rem;
  margin-bottom: 1rem;
}

.metric-card {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: white;
  padding: 1rem;
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(59, 130, 246, 0.3);
}

.metric-value {
  font-size: 2rem !important;
  font-weight: 800 !important;
  margin-bottom: 0.5rem !important;
}

.metric-label {
  font-size: 0.875rem !important;
  opacity: 0.9;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stButton button {
  background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
  color: white !important;
  border: none !important;
  padding: 0.75rem 2rem !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}

.stButton button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
}

.stNumberInput input, .stTextInput input {
  border-radius: 12px !important;
  border: 2px solid #e5e7eb !important;
  padding: 0.75rem 1rem !important;
  transition: all 0.3s ease !important;
}

.stNumberInput input:focus, .stTextInput input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

.sidebar-nav {
  background: linear-gradient(135deg, var(--dark), #374151);
  padding: 1.5rem 1rem;
  border-radius: 0 15px 15px 0;
}

.nav-item {
  padding: 0.75rem;
  margin: 0.25rem 0;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  color: white;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.nav-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.fade-in {
  animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.stAlert {
  border-radius: 12px !important;
  border: none !important;
}
</style>
""", unsafe_allow_html=True)

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
    page_title="FraudShield Pro • AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_database()

# Modern Header with Hero Section
st.markdown("""
<div class="fade-in">
    <h1 class="main-header">🛡️ FraudShield Pro</h1>
    <p class="sub-header">Advanced AI-Powered Credit Card Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)

# Navigation with Icons
st.sidebar.markdown("""
<div class="sidebar-nav">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h3 style="color: white; margin-bottom: 0.5rem;">🔒 FraudShield</h3>
        <div style="height: 2px; background: linear-gradient(45deg, #3b82f6, #10b981); margin: 0 auto; width: 50px;"></div>
    </div>
""", unsafe_allow_html=True)

# Navigation Items
page = st.sidebar.radio("Navigation", [
    "🏠 Dashboard", 
    "🔍 Fraud Scanner", 
    "📊 Analytics", 
    "📈 Performance",
    "📁 Data Upload"
])

# Quick Stats in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")

try:
    history_df = get_all_transactions()
    total_predictions = len(history_df)
    fraud_count = len(history_df[history_df['prediction'] == 1])
    
    st.sidebar.markdown(f"""
    <div class="metric-card" style="margin: 0.5rem 0;">
        <div style="font-size: 0.875rem; opacity: 0.9;">Total Scans</div>
        <div style="font-size: 1.25rem; font-weight: 700;">{total_predictions}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="metric-card" style="margin: 0.5rem 0;">
        <div style="font-size: 0.875rem; opacity: 0.9;">Fraud Detected</div>
        <div style="font-size: 1.25rem; font-weight: 700;">{fraud_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
except:
    st.sidebar.markdown("""
    <div class="metric-card" style="margin: 0.5rem 0;">
        <div style="font-size: 0.875rem; opacity: 0.9;">Total Scans</div>
        <div style="font-size: 1.25rem; font-weight: 700;">0</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def load_data_and_train_model():
    # Try to load uploaded CSV first
    if 'uploaded_df' in st.session_state and st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.success("✅ Using uploaded dataset!")
    else:
        # Generate demo data
        st.info("📁 Using demo data. Upload your CSV for real analysis.")
        np.random.seed(42)
        n_samples = 2000
        
        data = {}
        data['Time'] = np.random.normal(0, 1, n_samples)
        data['Amount'] = np.random.uniform(1, 5000, n_samples)
        
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        fraud_indices = np.random.choice(n_samples, size=200, replace=False)
        data['Class'] = np.zeros(n_samples)
        data['Class'][fraud_indices] = 1
        
        df = pd.DataFrame(data)
    
    df = df.dropna()
    
    # Balance the dataset
    legitimate = df[df["Class"] == 0]
    fraudulent = df[df["Class"] == 1]
    
    if len(fraudulent) > 0:
        legitimate_sample = legitimate.sample(n=min(len(fraudulent)*2, len(legitimate)), random_state=42)
        balanced_df = pd.concat([legitimate_sample, fraudulent], axis=0)
    else:
        balanced_df = df
    
    X = balanced_df.drop("Class", axis=1)
    y = balanced_df["Class"]
    
    if len(X) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test, balanced_df.columns.tolist()
    else:
        return None, None, None, None

# Load model
model, X_test, y_test, feature_columns = load_data_and_train_model()

# DASHBOARD PAGE
if page == "🏠 Dashboard":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🛡️</div>
            <div class="metric-label">Active Protection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">⚡</div>
            <div class="metric-label">Real-time Scan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🎯</div>
            <div class="metric-label">High Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔒</div>
            <div class="metric-label">Secure</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Transactions
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📋 Recent Transactions")
    
    try:
        recent_tx = get_all_transactions().head(10)
        if not recent_tx.empty:
            st.dataframe(recent_tx[['timestamp', 'amount', 'prediction']].style.format({
                'amount': '₹{:.2f}',
                'prediction': lambda x: '✅ Legitimate' if x == 0 else '🚨 Fraud'
            }), use_container_width=True)
        else:
            st.info("No transactions yet. Use the Fraud Scanner to analyze transactions.")
    except:
        st.info("No transactions yet. Use the Fraud Scanner to analyze transactions.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# FRAUD SCANNER PAGE
elif page == "🔍 Fraud Scanner":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 🔍 Fraud Detection Scanner")
    
    if model is None:
        st.error("❌ Model not trained. Please upload data first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Details")
            amount = st.number_input("💰 Transaction Amount", min_value=0.0, max_value=100000.0, value=100.0)
            time = st.number_input("⏰ Time", value=0.0)
            
        with col2:
            st.subheader("Feature Values")
            # Generate random features for demo
            if st.button("🎲 Generate Random Transaction"):
                np.random.seed(None)
                features = [np.random.normal(0, 1) for _ in range(28)]
                for i, feat in enumerate(features, 1):
                    if f'v{i}' in st.session_state:
                        st.session_state[f'v{i}'] = float(feat)
            
            # Display first 5 features for simplicity
            v1 = st.number_input("V1", value=0.0, key="v1")
            v2 = st.number_input("V2", value=0.0, key="v2")
            v3 = st.number_input("V3", value=0.0, key="v3")
            v4 = st.number_input("V4", value=0.0, key="v4")
            v5 = st.number_input("V5", value=0.0, key="v5")
        
        if st.button("🔍 Scan Transaction", type="primary"):
            # Create feature array
            features = [time, v1, v2, v3, v4, v5] + [0.0] * 23  # Pad with zeros for demo
            features.append(amount)
            
            # Make prediction
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            # Save to database
            save_prediction(features, prediction)
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 0:
                    st.success("""
                    ## ✅ LEGITIMATE TRANSACTION
                    **Status:** Safe to proceed
                    **Confidence:** {:.1f}%
                    """.format(probability[0] * 100))
                else:
                    st.error("""
                    ## 🚨 FRAUD DETECTED
                    **Status:** Transaction Blocked
                    **Confidence:** {:.1f}%
                    """.format(probability[1] * 100))
            
            with col2:
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Transaction Analytics")
    
    try:
        df = get_all_transactions()
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Fraud distribution
                fraud_count = len(df[df['prediction'] == 1])
                legit_count = len(df[df['prediction'] == 0])
                
                fig1 = px.pie(
                    values=[legit_count, fraud_count],
                    names=['Legitimate', 'Fraud'],
                    title="Transaction Distribution"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Amount distribution
                fig2 = px.histogram(
                    df, 
                    x='amount',
                    color=df['prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate'),
                    title="Transaction Amount Distribution",
                    barmode='overlay'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time series of transactions
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_tx = df.groupby([df['timestamp'].dt.date, 'prediction']).size().reset_index(name='count')
            daily_tx['type'] = daily_tx['prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate')
            
            fig3 = px.line(
                daily_tx,
                x='timestamp',
                y='count',
                color='type',
                title="Daily Transaction Trends"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.info("No transaction data available. Use the Fraud Scanner first.")
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("No transaction data available yet.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# PERFORMANCE PAGE
elif page == "📈 Performance":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Model Performance")
    
    if model is not None and X_test is not None and y_test is not None:
        # Make predictions
        y_pred = model.predict(X_test)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.1%}")
        
        with col2:
            precision = precision_score(y_test, y_pred)
            st.metric("Precision", f"{precision:.1%}")
        
        with col3:
            recall = recall_score(y_test, y_pred)
            st.metric("Recall", f"{recall:.1%}")
        
        with col4:
            f1 = f1_score(y_test, y_pred)
            st.metric("F1-Score", f"{f1:.1%}")
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format({
                'precision': '{:.2f}',
                'recall': '{:.2f}',
                'f1-score': '{:.2f}',
                'support': '{:.0f}'
            }), use_container_width=True)
    
    else:
        st.warning("Model not trained or test data not available. Upload data first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# DATA UPLOAD PAGE
elif page == "📁 Data Upload":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📁 Upload Your Dataset")
    
    st.info("""
    **Expected CSV Format:**
    - Columns: Time, V1, V2, ..., V28, Amount, Class
    - Class: 0 (Legitimate), 1 (Fraud)
    - First row should be headers
    """)
    
    uploaded_file = st.file_uploader("creditcard.csv", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df
            
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", len(df))
            
            with col2:
                fraud_count = len(df[df['Class'] == 1]) if 'Class' in df.columns else 'N/A'
                st.metric("Fraud Cases", fraud_count)
            
            with col3:
                st.metric("Columns", len(df.columns))
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Retrain model button
            if st.button("🔄 Retrain Model with New Data", type="primary"):
                # Clear cache to retrain model
                st.cache_resource.clear()
                st.rerun()
                
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Modern Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🛡️ <b>FraudShield Pro</b> • Enterprise Fraud Detection • Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)