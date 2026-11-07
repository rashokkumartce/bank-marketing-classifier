import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Bank Marketing Classification Model")
st.markdown("### Predict Customer Response to Marketing Campaign")
st.divider()

# Sidebar - Model Selection
st.sidebar.title("⚙️ Configuration")
st.sidebar.divider()

# Load model results
model_results = pd.read_csv('model_results.csv')
model_names = model_results['Model'].tolist()

selected_model = st.sidebar.selectbox(
    "🤖 Select Classification Model",
    options=model_names,
    help="Choose which trained model to use for predictions"
)

st.sidebar.divider()

# Load the selected model
@st.cache_resource
def load_model(model_name):
    model_path = f"model/{model_name.lower().replace(' ', '_')}.joblib"
    return joblib.load(model_path)

model = load_model(selected_model)

# Load feature encoders
def get_feature_encoders():
    """Get encoders for categorical features"""
    encoders = {}
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    return categorical_features

categorical_features = get_feature_encoders()

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🔮 Make Predictions", "📈 Model Comparison"])

# TAB 1: Model Metrics
with tab1:
    st.header("Model Performance Metrics")
    
    # Get selected model metrics
    selected_metrics = model_results[model_results['Model'] == selected_model].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    with col2:
        st.metric("AUC Score", f"{selected_metrics['AUC Score']:.4f}")
    with col3:
        st.metric("Precision", f"{selected_metrics['Precision']:.4f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recall", f"{selected_metrics['Recall']:.4f}")
    with col2:
        st.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
    with col3:
        st.metric("MCC", f"{selected_metrics['MCC']:.4f}")
    
    st.divider()
    st.info(f"📌 **Selected Model:** {selected_model}")
    st.success(f"✅ **Accuracy:** {selected_metrics['Accuracy']:.2%} of predictions are correct")

# TAB 2: Make Predictions
with tab2:
    st.header("Make Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Options")
        input_method = st.radio(
            "How would you like to input data?",
            ["📝 Manual Input", "📤 Upload CSV"],
            help="Choose between entering data manually or uploading a CSV file"
        )
    
    # Initialize test data
    test_data = pd.read_csv('test_data.csv')
    
    if input_method == "📤 Upload CSV":
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV should contain all required features for prediction"
        )
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"✅ Uploaded {len(input_df)} rows")
            
            if st.button("🔮 Generate Predictions", key="predict_csv"):
                try:
                    # Encode categorical variables
                    input_df_encoded = input_df.copy()
                    
                    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
                    for col in categorical_cols:
                        if col in input_df_encoded.columns:
                            le = LabelEncoder()
                            input_df_encoded[col] = le.fit_transform(input_df_encoded[col])
                    
                    # Make predictions
                    predictions = model.predict(input_df_encoded)
                    probabilities = model.predict_proba(input_df_encoded)
                    
                    # Display results
                    results_df = pd.DataFrame({
                        'Prediction': ['Yes' if p == 1 else 'No' for p in predictions],
                        'Probability_No': probabilities[:, 0],
                        'Probability_Yes': probabilities[:, 1]
                    })
                    
                    st.subheader("Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    yes_count = (predictions == 1).sum()
                    no_count = (predictions == 0).sum()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted 'Yes'", yes_count)
                    with col2:
                        st.metric("Predicted 'No'", no_count)
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
    
    else:  # Manual input
        st.subheader("Manual Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            balance = st.number_input("Balance (€)", min_value=-10000, max_value=250000, value=1000)
            duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=300)
            campaign = st.number_input("Number of Contacts", min_value=1, max_value=56, value=2)
        
        with col2:
            job = st.selectbox("Job Type", ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student'])
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
            education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'])
            contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
        
        if st.button("🔮 Make Prediction", key="predict_manual"):
            try:
                # Prepare data
                input_data = {
                    'age': age, 'job': job, 'marital': marital, 'education': education,
                    'default': 'no', 'balance': balance, 'housing': 'yes', 'loan': 'no',
                    'contact': contact, 'day': 1, 'month': 'jan', 'duration': duration,
                    'campaign': campaign, 'pdays': -1, 'previous': 0, 'poutcome': 'nonexistent',
                    'emp.var.rate': -1.1, 'cons.price.idx': 93.994, 'cons.conf.idx': -46.2,
                    'euribor3m': 0.762, 'nr.employed': 5191.0
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
                le_dict = {}
                for col in categorical_cols:
                    if col in input_df.columns:
                        le = LabelEncoder()
                        input_df[col] = le.fit_transform(input_df[col])
                        le_dict[col] = le
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                st.divider()
                st.subheader("🎯 Prediction Result")
                
                if prediction == 1:
                    st.success(f"✅ **PREDICTION: Customer will ACCEPT the campaign**")
                    st.metric("Confidence", f"{probability[1]:.2%}")
                else:
                    st.warning(f"❌ **PREDICTION: Customer will REJECT the campaign**")
                    st.metric("Confidence", f"{probability[0]:.2%}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("P(No)", f"{probability[0]:.4f}")
                with col2:
                    st.metric("P(Yes)", f"{probability[1]:.4f}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# TAB 3: Model Comparison
with tab3:
    st.header("Model Performance Comparison")
    
    # Display comparison table
    st.subheader("All Models - Metrics Comparison")
    st.dataframe(model_results, use_container_width=True, hide_index=True)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(model_results['Model'], model_results['Accuracy'], color='#1f77b4')
        ax.set_xlabel('Accuracy Score')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlim([0, 1])
        for i, v in enumerate(model_results['Accuracy']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(model_results['Model'], model_results['F1 Score'], color='#ff7f0e')
        ax.set_xlabel('F1 Score')
        ax.set_title('Model F1 Score Comparison')
        ax.set_xlim([0, 1])
        for i, v in enumerate(model_results['F1 Score']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.divider()
    
    # Key insights
    st.subheader("📊 Key Insights")
    best_accuracy = model_results.loc[model_results['Accuracy'].idxmax()]
    best_f1 = model_results.loc[model_results['F1 Score'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Accuracy", f"{best_accuracy['Model']}", f"{best_accuracy['Accuracy']:.4f}")
    with col2:
        st.metric("Best AUC", f"{model_results.loc[model_results['AUC Score'].idxmax(), 'Model']}", f"{model_results['AUC Score'].max():.4f}")
    with col3:
        st.metric("Best F1 Score", f"{best_f1['Model']}", f"{best_f1['F1 Score']:.4f}")

st.divider()
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
    <p>🏦 Bank Marketing Classification Model | BITS Pilani M.Tech AIML Assignment</p>
    <p>All models trained on Bank Marketing dataset with 5,000 samples and 20 features</p>
    </div>
""", unsafe_allow_html=True)
