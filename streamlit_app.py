import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = Path('./best_credit_model.pkl')
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['model_name']
    else:
        st.error("Model file not found! Please ensure 'best_credit_model.pkl' is in the current directory.")
        return None, None

# Title and description
st.title("🏦 Credit Scoring System")
st.markdown("**Predict credit approval using machine learning**")

# Load model
model, model_name = load_model()

if model is not None:
    st.success(f"✅ Model loaded successfully: {model_name}")
    
    # Sidebar for input features
    st.sidebar.header("📋 Enter Customer Information")
    
    # Personal Information
    st.sidebar.subheader("👤 Personal Details")
    age = st.sidebar.slider("Age", min_value=18, max_value=80, value=35)
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["single", "married", "divorced", "widowed"])
    region = st.sidebar.selectbox("Region", ["Andijan", "Bukhara", "Namangan", "Samarkand", "Tashkent"])
    education_level = st.sidebar.selectbox("Education Level", ["high_school", "bachelor", "master", "phd"])
    
    # Employment Information
    st.sidebar.subheader("💼 Employment Details")
    employment_years = st.sidebar.slider("Years of Employment", min_value=0, max_value=45, value=10)
    
    # Financial Information
    st.sidebar.subheader("💰 Financial Details")
    income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=50000, value=10000)
    monthly_expenses = st.sidebar.number_input("Monthly Expenses", min_value=500, max_value=20000, value=2500)
    savings_balance = st.sidebar.number_input("Savings Balance", min_value=0, max_value=100000, value=5000)
    loan_amount = st.sidebar.number_input("Requested Loan Amount", min_value=1000, max_value=100000, value=15000)
    past_due = st.sidebar.slider("Past Due Payments (months)", min_value=0, max_value=6, value=0)
    
    # Feature Engineering (same as training)
    def create_features(data):
        # Calculate ratios
        data['debt_to_income_ratio'] = data['loan_amount'] / data['income']
        data['expense_to_income_ratio'] = data['monthly_expenses'] / data['income']
        data['savings_to_income_ratio'] = data['savings_balance'] / data['income']
        
        # Age category
        if data['age'] <= 30:
            data['age_category'] = 1  # Young
        elif data['age'] <= 40:
            data['age_category'] = 2  # Adult
        elif data['age'] <= 50:
            data['age_category'] = 3  # Middle_aged
        else:
            data['age_category'] = 4  # Senior
        
        # Experience level
        if data['employment_years'] <= 5:
            data['experience_level'] = 1  # Entry
        elif data['employment_years'] <= 15:
            data['experience_level'] = 2  # Mid
        elif data['employment_years'] <= 25:
            data['experience_level'] = 3  # Senior
        else:
            data['experience_level'] = 4  # Expert
        
        # Credit category (we'll set a default since we don't have credit_score input)
        # You might want to add credit_score as an input or use a default
        data['credit_category'] = 3  # Good (default)
        
        # Risk score bins (default)
        data['risk_score_bins'] = 2  # Low (default)
        
        return data
    
    # Encoding function
    def encode_features(data):
        # Gender encoding
        data['gender'] = 1 if data['gender'] == 'male' else 0
        
        # Education encoding
        education_mapping = {'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
        data['education_level'] = education_mapping[data['education_level']]
        
        # Marital status encoding
        marital_mapping = {'divorced': 0, 'married': 1, 'single': 2, 'widowed': 3}
        data['marital_status_encoded'] = marital_mapping[data['marital_status']]
        
        # Region encoding (one-hot)
        regions = ['Andijan', 'Bukhara', 'Namangan', 'Samarkand', 'Tashkent']
        for r in regions:
            data[f'region_{r}'] = 1 if data['region'] == r else 0
        
        # Remove original categorical columns
        data.pop('marital_status', None)
        data.pop('region', None)
        
        return data
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Credit Approval", type="primary"):
        # Create input data dictionary
        input_data = {
            'age': age,
            'gender': gender,
            'education_level': education_level,
            'employment_years': employment_years,
            'income': income,
            'loan_amount': loan_amount,
            'past_due': past_due,
            'monthly_expenses': monthly_expenses,
            'savings_balance': savings_balance,
            'marital_status': marital_status,
            'region': region
        }
        
        # Feature engineering
        input_data = create_features(input_data)
        
        # Encoding
        input_data = encode_features(input_data)
        
        # Expected feature order (from your training)
        expected_features = [
            'age', 'gender', 'education_level', 'employment_years', 'income',
            'loan_amount', 'past_due', 'monthly_expenses', 'savings_balance',
            'risk_score_bins', 'debt_to_income_ratio', 'expense_to_income_ratio',
            'savings_to_income_ratio', 'age_category', 'experience_level',
            'credit_category', 'marital_status_encoded', 'region_Andijan',
            'region_Bukhara', 'region_Namangan', 'region_Samarkand', 'region_Tashkent'
        ]
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[expected_features]
        
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **CREDIT APPROVED**")
                else:
                    st.error("❌ **CREDIT REJECTED**")
                
                st.metric("Approval Probability", f"{prediction_proba[1]:.2%}")
                st.metric("Rejection Probability", f"{prediction_proba[0]:.2%}")
            
            with col2:
                # Create a gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction_proba[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability %"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display input summary
            st.header("📋 Customer Profile Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Personal Info")
                st.write(f"**Age:** {age} years")
                st.write(f"**Gender:** {gender.title()}")
                st.write(f"**Marital Status:** {marital_status.title()}")
                st.write(f"**Region:** {region}")
                st.write(f"**Education:** {education_level.replace('_', ' ').title()}")
            
            with col2:
                st.subheader("Employment")
                st.write(f"**Experience:** {employment_years} years")
                
                st.subheader("Financial Profile")
                st.write(f"**Monthly Income:** ${income:,.2f}")
                st.write(f"**Monthly Expenses:** ${monthly_expenses:,.2f}")
                st.write(f"**Savings:** ${savings_balance:,.2f}")
            
            with col3:
                st.subheader("Loan Details")
                st.write(f"**Requested Amount:** ${loan_amount:,.2f}")
                st.write(f"**Past Due Payments:** {past_due} months")
                
                st.subheader("Financial Ratios")
                st.write(f"**Debt-to-Income:** {input_data['debt_to_income_ratio']:.2f}")
                st.write(f"**Expense-to-Income:** {input_data['expense_to_income_ratio']:.2f}")
                st.write(f"**Savings-to-Income:** {input_data['savings_to_income_ratio']:.2f}")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.error("Please check if the model file is compatible with the input features.")

    # Model information
    st.header("🤖 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Model Type:** {model_name}
        **Performance Metrics:**
        - ROC AUC: 96.18%
        - Accuracy: 88.25%
        - Precision: 90.16%
        - Recall: 94.67%
        """)
    
    with col2:
        st.info("""
        **Key Features Used:**
        - Credit Category
        - Risk Score Bins  
        - Debt-to-Income Ratio
        - Expense-to-Income Ratio
        - Employment Years
        - Financial Profile
        """)

else:
    st.error("❌ Unable to load the model. Please check if 'best_credit_model.pkl' exists in the current directory.")