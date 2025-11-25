import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go

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
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data['model'], model_data['model_name']
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
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
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📋 Enter Customer Information")
        
        # Personal Information
        st.subheader("👤 Personal Details")
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["male", "female"])
        marital_status = st.selectbox("Marital Status", ["single", "married", "divorced", "widowed"])
        region = st.selectbox("Region", ["Andijan", "Bukhara", "Namangan", "Samarkand", "Tashkent"])
        education_level = st.selectbox("Education Level", ["high_school", "bachelor", "master", "phd"])
        
        # Employment Information
        st.subheader("💼 Employment Details")
        employment_years = st.slider("Years of Employment", min_value=0, max_value=45, value=10)
        
        # Financial Information
        st.subheader("💰 Financial Details")
        income = st.number_input("Monthly Income ($)", min_value=1000.0, max_value=50000.0, value=10000.0, step=100.0)
        monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=500.0, max_value=20000.0, value=2500.0, step=100.0)
        savings_balance = st.number_input("Savings Balance ($)", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=100.0)
        past_due = st.slider("Past Due Payments (months)", min_value=0, max_value=6, value=0)
        
        # Prediction button
        predict_button = st.button("🔮 Predict Credit Approval", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            # Enhanced Feature Engineering with Risk-Based Logic
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
                
                # IMPROVED CREDIT CATEGORY LOGIC
                # Calculate a synthetic credit score based on financial behavior
                credit_score = 850  # Start with excellent
                
                # Penalize based on debt-to-income ratio
                if data['debt_to_income_ratio'] > 8:
                    credit_score -= 300  # Very high debt
                elif data['debt_to_income_ratio'] > 5:
                    credit_score -= 200  # High debt
                elif data['debt_to_income_ratio'] > 3:
                    credit_score -= 100  # Moderate debt
                
                # Penalize based on expense-to-income ratio
                if data['expense_to_income_ratio'] > 0.9:
                    credit_score -= 150  # Living paycheck to paycheck
                elif data['expense_to_income_ratio'] > 0.8:
                    credit_score -= 100  # High expenses
                elif data['expense_to_income_ratio'] > 0.7:
                    credit_score -= 50   # Moderate expenses
                
                # Penalize for past due payments
                credit_score -= data['past_due'] * 50  # Each past due month = -50 points
                
                # Penalize for low savings
                if data['savings_to_income_ratio'] < 0.1:
                    credit_score -= 100  # Very low savings
                elif data['savings_to_income_ratio'] < 0.5:
                    credit_score -= 50   # Low savings
                
                # Penalize for short employment history
                if data['employment_years'] < 2:
                    credit_score -= 150  # Very short history
                elif data['employment_years'] < 5:
                    credit_score -= 75   # Short history
                
                # Penalize for low income
                if data['income'] < 3000:
                    credit_score -= 100  # Very low income
                elif data['income'] < 5000:
                    credit_score -= 50   # Low income
                
                # Ensure credit score is within bounds
                credit_score = max(300, min(850, credit_score))
                
                # Map credit score to category
                if credit_score >= 740:
                    data['credit_category'] = 4  # Excellent
                elif credit_score >= 670:
                    data['credit_category'] = 3  # Good
                elif credit_score >= 580:
                    data['credit_category'] = 2  # Fair
                else:
                    data['credit_category'] = 1  # Poor
                
                # IMPROVED RISK SCORE BINS LOGIC
                risk_score = 0  # Start with low risk
                
                # Add risk based on debt-to-income
                if data['debt_to_income_ratio'] > 8:
                    risk_score += 4
                elif data['debt_to_income_ratio'] > 5:
                    risk_score += 3
                elif data['debt_to_income_ratio'] > 3:
                    risk_score += 2
                elif data['debt_to_income_ratio'] > 2:
                    risk_score += 1
                
                # Add risk for past due payments
                risk_score += min(data['past_due'], 3)  # Cap at 3 additional points
                
                # Add risk for high expenses
                if data['expense_to_income_ratio'] > 0.85:
                    risk_score += 2
                elif data['expense_to_income_ratio'] > 0.75:
                    risk_score += 1
                
                # Add risk for low employment
                if data['employment_years'] < 2:
                    risk_score += 2
                elif data['employment_years'] < 5:
                    risk_score += 1
                
                # Map total risk score to bins (1=Very Low, 5=Very High)
                if risk_score >= 8:
                    data['risk_score_bins'] = 5  # Very High
                elif risk_score >= 6:
                    data['risk_score_bins'] = 4  # High
                elif risk_score >= 4:
                    data['risk_score_bins'] = 3  # Medium
                elif risk_score >= 2:
                    data['risk_score_bins'] = 2  # Low
                else:
                    data['risk_score_bins'] = 1  # Very Low
                
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
                
                return data
            
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
            
            # Feature engineering and encoding
            input_data = create_features(input_data)
            input_data = encode_features(input_data)
            
            # Expected feature order (based on your training data)
            expected_features = [
                'age', 'gender', 'education_level', 'employment_years', 'income',
                'loan_amount', 'past_due', 'monthly_expenses', 'savings_balance',
                'risk_score_bins', 'debt_to_income_ratio', 'expense_to_income_ratio',
                'savings_to_income_ratio', 'age_category', 'experience_level',
                'credit_category', 'marital_status_encoded', 'region_Andijan',
                'region_Bukhara', 'region_Namangan', 'region_Samarkand', 'region_Tashkent'
            ]
            
            # Create DataFrame with correct feature order
            feature_values = []
            for feature in expected_features:
                feature_values.append(input_data.get(feature, 0))
            
            input_df = pd.DataFrame([feature_values], columns=expected_features)
            
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display results
                st.header("📊 Prediction Results")
                
                # Result display
                if prediction == 1:
                    st.success("✅ **CREDIT APPROVED**")
                    st.balloons()
                else:
                    st.error("❌ **CREDIT REJECTED**")
                
                # Metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric("Approval Probability", f"{prediction_proba[1]:.1%}")
                with col_metric2:
                    st.metric("Rejection Probability", f"{prediction_proba[0]:.1%}")
                
                # Risk Assessment Display
                st.subheader("🔍 Risk Assessment Details")
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    st.write(f"**Credit Category:** {['Poor', 'Fair', 'Good', 'Excellent'][input_data['credit_category']-1]}")
                    st.write(f"**Risk Level:** {['Very Low', 'Low', 'Medium', 'High', 'Very High'][input_data['risk_score_bins']-1]}")
                
                with risk_col2:
                    st.write(f"**Debt-to-Income:** {input_data['debt_to_income_ratio']:.2f}")
                    st.write(f"**Expense-to-Income:** {input_data['expense_to_income_ratio']:.2f}")
                
                # Create a gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction_proba[1] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Approval Probability (%)"},
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
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display input summary
                st.header("📋 Customer Profile Summary")
                
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.subheader("Personal Info")
                    st.write(f"**Age:** {age} years")
                    st.write(f"**Gender:** {gender.title()}")
                    st.write(f"**Marital Status:** {marital_status.title()}")
                    st.write(f"**Region:** {region}")
                    st.write(f"**Education:** {education_level.replace('_', ' ').title()}")
                
                with col_summary2:
                    st.subheader("Employment & Financial")
                    st.write(f"**Experience:** {employment_years} years")
                    st.write(f"**Monthly Income:** ${income:,.2f}")
                    st.write(f"**Monthly Expenses:** ${monthly_expenses:,.2f}")
                    st.write(f"**Savings:** ${savings_balance:,.2f}")
                
                with col_summary3:
                    st.subheader("Loan & Risk Ratios")
                    st.write(f"**Requested Amount:** ${loan_amount:,.2f}")
                    st.write(f"**Past Due:** {past_due} months")
                    st.write(f"**Debt-to-Income:** {input_data['debt_to_income_ratio']:.2f}")
                    st.write(f"**Expense-to-Income:** {input_data['expense_to_income_ratio']:.2f}")
                    st.write(f"**Savings-to-Income:** {input_data['savings_to_income_ratio']:.2f}")
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("Please check if the model file is compatible with the input features.")
                # Debug information
                st.write("Input data shape:", input_df.shape)
                st.write("Features:", input_df.columns.tolist())
                st.write("Sample values:", input_df.iloc[0].tolist()[:5])
        
        else:
            st.info("👈 Enter customer information and click 'Predict Credit Approval' to see results")
    
    # Model information at the bottom
    st.markdown("---")
    st.header("🤖 Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info(f"""
        **Model Type:** {model_name}
        **Performance Metrics:**
        - ROC AUC: 96.18%
        - Accuracy: 88.25%
        - Precision: 90.16%
        - Recall: 94.67%
        """)
    
    with col_info2:
        st.info("""
        **Key Risk Factors:**
        - High Debt-to-Income Ratio (>5)
        - High Expense-to-Income (>0.8)
        - Past Due Payments (>2 months)
        - Low Employment History (<2 years)
        - Low Savings-to-Income (<0.1)
        """)

else:
    st.error("❌ Unable to load the model. Please check if 'best_credit_model.pkl' exists in the current directory.")