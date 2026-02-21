"""
Aquatic Water Quality Index (AWQI) Prediction System
Machine Learning Based Water Quality Assessment
Streamlit Application for Real-time Predictions

Author: Water Quality Research Team
Version: 1.0.0
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AWQI Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem 1rem; }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained ML models"""
    try:
        models = {}
        model_files = {
            'Linear Regression': 'models/linear_regression_reg.pkl',
            'Decision Tree': 'models/decision_tree_reg.pkl',
            'Random Forest': 'models/random_forest_reg.pkl',
            'SVR': 'models/svr_reg.pkl',
            'XGBoost': 'models/xgboost_reg.pkl',
            'ANN': 'models/ann_reg.pkl',
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
            else:
                st.warning(f"Model not found: {filepath}")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

@st.cache_resource
def load_classification_models():
    """Load all trained classification models"""
    try:
        models = {}
        model_files = {
            'Logistic Regression': 'models/logistic_regression_clf.pkl',
            'Decision Tree': 'models/decision_tree_clf.pkl',
            'Random Forest': 'models/random_forest_clf.pkl',
            'SVC': 'models/svc_clf.pkl',
            'XGBoost': 'models/xgboost_clf.pkl',
            'ANN': 'models/ann_clf.pkl',
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
            else:
                st.warning(f"Model not found: {filepath}")
        
        return models
    except Exception as e:
        st.error(f"Error loading classification models: {str(e)}")
        return {}

@st.cache_resource
def load_scalers():
    """Load feature scalers"""
    try:
        return {
            'regression': joblib.load('models/scaler_regression.pkl'),
            'classification': joblib.load('models/scaler_classification.pkl')
        }
    except Exception as e:
        st.error(f"Error loading scalers: {str(e)}")
        return {}

@st.cache_resource
def load_encoders():
    """Load label encoder and class names"""
    try:
        return {
            'label_encoder': joblib.load('models/label_encoder.pkl'),
            'feature_names': joblib.load('models/feature_names.pkl'),
            'class_names': joblib.load('models/class_names.pkl')
        }
    except Exception as e:
        st.error(f"Error loading encoders: {str(e)}")
        return {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_awqi_interpretation(value):
    """Interpret AWQI score"""
    if value < 25:
        return {
            'class': 'Excellent',
            'description': 'Excellent water quality - Suitable for all uses',
            'color': '#28a745',
            'emoji': '✅'
        }
    elif value < 50:
        return {
            'class': 'Good',
            'description': 'Good water quality - Minor issues',
            'color': '#17a2b8',
            'emoji': '👍'
        }
    elif value < 75:
        return {
            'class': 'Moderate',
            'description': 'Moderate water quality - Action recommended',
            'color': '#ffc107',
            'emoji': '⚠️'
        }
    else:
        return {
            'class': 'Poor',
            'description': 'Poor water quality - Immediate action needed',
            'color': '#dc3545',
            'emoji': '🚫'
        }

def prepare_features(input_dict):
    """Prepare features for prediction"""
    features = pd.DataFrame({
        'TDS': [input_dict['TDS']],
        'DO': [input_dict['DO']],
        'Nitrate': [input_dict['Nitrate']],
        'TH': [input_dict['TH']],
        'pH': [input_dict['pH']],
        'Chlorides': [input_dict['Chlorides']],
        'Alkalinity': [input_dict['Alkalinity']],
        'EC': [input_dict['EC']],
        'Ammonia': [input_dict['Ammonia']],
        'Time_sin': [input_dict['Time_sin']],
        'Time_cos': [input_dict['Time_cos']]
    })
    return features

def get_parameter_guide():
    """Return parameter information"""
    return {
        'TDS': {
            'name': 'Total Dissolved Solids (mg/L)',
            'range': '0-500',
            'optimal': '<250',
            'description': 'Measure of minerals dissolved in water'
        },
        'DO': {
            'name': 'Dissolved Oxygen (mg/L)',
            'range': '0-14',
            'optimal': '>7',
            'description': 'Oxygen available for aquatic organisms'
        },
        'Nitrate': {
            'name': 'Nitrate (mg/L)',
            'range': '0-500',
            'optimal': '<10',
            'description': 'Nitrogen compound; excess causes eutrophication'
        },
        'TH': {
            'name': 'Total Hardness (mg/L)',
            'range': '0-500',
            'optimal': '50-150',
            'description': 'Concentration of Ca²⁺ and Mg²⁺'
        },
        'pH': {
            'name': 'pH Value',
            'range': '0-14',
            'optimal': '6.5-8.5',
            'description': 'Acidity or alkalinity of water'
        },
        'Chlorides': {
            'name': 'Chlorides (mg/L)',
            'range': '0-500',
            'optimal': '<250',
            'description': 'Salt concentration'
        },
        'Alkalinity': {
            'name': 'Alkalinity (mg/L)',
            'range': '0-500',
            'optimal': '50-200',
            'description': 'Buffering capacity'
        },
        'EC': {
            'name': 'Electrical Conductivity (µS/cm)',
            'range': '0-2000',
            'optimal': '500-1500',
            'description': 'Measure of dissolved ions'
        },
        'Ammonia': {
            'name': 'Ammonia (mg/L)',
            'range': '0-10',
            'optimal': '<0.5',
            'description': 'Indicates organic pollution'
        }
    }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load all models and data
    reg_models = load_models()
    clf_models = load_classification_models()
    scalers = load_scalers()
    encoders = load_encoders()
    
    # Check if models are loaded
    if not reg_models or not scalers or not encoders:
        st.error("""
        ❌ **Models not found!**
        
        Please ensure you have run the `train_and_save_models.py` script first.
        
        Steps:
        1. Place `Aquaculture.csv` in the same directory
        2. Run: `python train_and_save_models.py`
        3. Then run: `streamlit run app.py`
        """)
        return
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>💧 Aquatic Water Quality Index (AWQI) Prediction System</h1>
        <p><i>Machine Learning Based Water Quality Assessment for Aquaculture</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["📊 Prediction Dashboard", "📚 Parameter Guide", "📈 Model Performance", "ℹ️ About"]
    )
    
    # ========================================================================
    # PAGE 1: PREDICTION DASHBOARD
    # ========================================================================
    if page == "📊 Prediction Dashboard":
        st.header("Real-time Water Quality Prediction")
        
        st.markdown("""
        <div class="info-box">
        <b>Instructions:</b> Enter the water quality parameters below to get an instant AWQI prediction.
        </div>
        """, unsafe_allow_html=True)
        
        # Input section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=500.0, value=170.0, step=1.0)
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
            alkalinity = st.number_input("Alkalinity (mg/L)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
        
        with col2:
            do = st.number_input("DO (mg/L)", min_value=0.0, max_value=14.0, value=5.0, step=0.1)
            chlorides = st.number_input("Chlorides (mg/L)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
            ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=2000.0, value=280.0, step=10.0)
        
        with col3:
            nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=500.0, value=0.4, step=0.1)
            th = st.number_input("Total Hardness (mg/L)", min_value=0.0, max_value=500.0, value=140.0, step=1.0)
            ammonia = st.number_input("Ammonia (mg/L)", min_value=0.0, max_value=10.0, value=0.01, step=0.001)
        
        # Time input
        time_value = st.slider("Time of Day (0-23 hours)", min_value=0, max_value=23, value=12, step=1)
        
        # Prepare features
        time_sin = np.sin(2 * np.pi * time_value / 12)
        time_cos = np.cos(2 * np.pi * time_value / 12)
        
        input_dict = {
            'TDS': tds, 'DO': do, 'Nitrate': nitrate, 'TH': th, 'pH': ph,
            'Chlorides': chlorides, 'Alkalinity': alkalinity, 'EC': ec,
            'Ammonia': ammonia, 'Time_sin': time_sin, 'Time_cos': time_cos
        }
        
        # Validation function
        def validate_inputs(inputs):
            """Validate input parameters are within acceptable ranges"""
            validation_rules = {
                'TDS': (0, 500, 'TDS must be between 0-500 mg/L'),
                'DO': (0, 14, 'DO must be between 0-14 mg/L'),
                'Nitrate': (0, 500, 'Nitrate must be between 0-500 mg/L'),
                'TH': (0, 500, 'Total Hardness must be between 0-500 mg/L'),
                'pH': (0, 14, 'pH must be between 0-14'),
                'Chlorides': (0, 500, 'Chlorides must be between 0-500 mg/L'),
                'Alkalinity': (0, 500, 'Alkalinity must be between 0-500 mg/L'),
                'EC': (0, 2000, 'EC must be between 0-2000 µS/cm'),
                'Ammonia': (0, 10, 'Ammonia must be between 0-10 mg/L'),
            }
            
            errors = []
            for param, (min_val, max_val, message) in validation_rules.items():
                if inputs[param] < min_val or inputs[param] > max_val:
                    errors.append(message)
            
            return errors
        
        # Prediction button
        if st.button("🔍 Predict Water Quality", use_container_width=True, type="primary"):
            # Validate inputs
            validation_errors = validate_inputs(input_dict)
            
            if validation_errors:
                st.error("❌ **Input Validation Failed:**\n\n" + "\n".join(validation_errors))
                st.stop()
            features = prepare_features(input_dict)
            
            # Scale features
            scaled_features = scalers['regression'].transform(features)
            
            # Get predictions from all models
            predictions = {}
            for name, model in reg_models.items():
                pred = model.predict(scaled_features)[0]
                predictions[name] = pred
            
            # Get best prediction (using Random Forest as default)
            best_model_name = 'Random Forest'
            if best_model_name in predictions:
                awqi_score = predictions[best_model_name]
            else:
                awqi_score = np.mean(list(predictions.values()))
            
            # Get interpretation
            interpretation = get_awqi_interpretation(awqi_score)
            
            # Display results
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.subheader("🎯 AWQI Score")
                st.markdown(f"""
                <div class="metric-box">
                    <h2>{awqi_score:.2f}</h2>
                    <p>{interpretation['class'].upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background-color: {interpretation['color']}; color: white; 
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h3>{interpretation['emoji']} {interpretation['class']}</h3>
                    <p style="font-size: 16px;">{interpretation['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.subheader("📊 Model Predictions")
                
                # Show all model predictions
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'AWQI Score': list(predictions.values())
                }).sort_values('AWQI Score', ascending=False)
                
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Classification
                st.subheader("🏷️ Water Quality Classification")
                scaled_clf = scalers['classification'].transform(features)
                
                if clf_models:
                    best_clf = list(clf_models.values())[0]
                    class_pred = best_clf.predict(scaled_clf)[0]
                    class_name = encoders['class_names'][class_pred]
                    
                    if hasattr(best_clf, 'predict_proba'):
                        proba = best_clf.predict_proba(scaled_clf)[0]
                        confidence = proba[class_pred] * 100
                        
                        st.metric("Predicted Class", class_name)
                        st.metric("Confidence", f"{confidence:.1f}%")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            recommendations = []
            
            if do < 5:
                recommendations.append("⚠️ **Low Dissolved Oxygen**: Increase aeration immediately")
            if ammonia > 0.5:
                recommendations.append("⚠️ **High Ammonia**: Indicates organic pollution; improve water circulation")
            if ph < 6.5 or ph > 8.5:
                recommendations.append("⚠️ **pH Out of Range**: Consider pH buffering treatments")
            if tds > 300:
                recommendations.append("⚠️ **High TDS**: Monitor salt accumulation; consider dilution")
            if nitrate > 10:
                recommendations.append("⚠️ **Elevated Nitrate**: Risk of eutrophication; reduce feed input")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f'<div class="warning-box">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="success-box">✅ All parameters within optimal ranges!</div>',
                    unsafe_allow_html=True
                )
    
    # ========================================================================
    # PAGE 2: PARAMETER GUIDE
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters Reference")
        
        st.markdown("""
        <div class="info-box">
        <b>Guide to Water Quality Parameters:</b> Understanding each parameter's role in aquaculture
        </div>
        """, unsafe_allow_html=True)
        
        params = get_parameter_guide()
        
        for param_code, param_info in params.items():
            with st.expander(f"📌 {param_info['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {param_info['description']}")
                
                with col2:
                    st.write(f"**Range:** {param_info['range']}")
                    st.write(f"**Optimal:** {param_info['optimal']}")
        
        # Classification standards
        st.subheader("📊 Water Quality Classification Standards")
        
        standards = pd.DataFrame({
            'AWQI Score': ['0-25', '25-50', '50-75', '>75'],
            'Classification': ['Excellent', 'Good', 'Moderate', 'Poor'],
            'Description': [
                'Excellent quality; suitable for all uses',
                'Good quality with minor issues',
                'Moderate quality; monitoring recommended',
                'Poor quality; immediate action needed'
            ]
        })
        
        st.dataframe(standards, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE 3: MODEL PERFORMANCE
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance")
        
        st.markdown("""
        <div class="info-box">
        <b>Model Comparison:</b> Performance metrics of trained regression and classification models
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔄 Regression Models (AWQI Score Prediction)")
        
        regression_performance = {
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'XGBoost', 'ANN'],
            'R² Score': [1.0000, 0.8717, 0.9482, 0.9999, 0.8940, 0.9734],
            'Status': ['⭐ Excellent', 'Good', 'Excellent', '⭐ Best', 'Good', 'Excellent']
        }
        
        reg_df = pd.DataFrame(regression_performance)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        
        st.subheader("🏷️ Classification Models (Water Quality Class)")
        
        classification_performance = {
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVC', 'XGBoost', 'ANN'],
            'Accuracy': [0.8611, 0.9167, 0.8889, 0.8889, 0.9167, 0.8889],
            'Status': ['Good', 'Excellent', 'Good', 'Good', 'Excellent', 'Good']
        }
        
        clf_df = pd.DataFrame(classification_performance)
        st.dataframe(clf_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE 4: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About This System")
        
        st.markdown("""
        ## Aquatic Water Quality Index (AWQI) Prediction System
        
        An intelligent machine learning system for assessing and predicting water quality in aquaculture environments.
        
        ### Features
        - **Real-time Predictions:** Instant AWQI scores from water quality parameters
        - **Multiple Models:** Ensemble of 6 regression and 6 classification algorithms
        - **Interpretable:** Clear explanations and recommendations
        - **Educational:** Comprehensive parameter guides and standards
        
        ### Technology Stack
        - **Framework:** Streamlit (Web Application)
        - **ML Libraries:** Scikit-learn, XGBoost
        - **Data Processing:** Pandas, NumPy
        - **Deployment:** Cloud-based (Streamlit Cloud)
        
        ### Models Trained
        
        **Regression Models:**
        - Linear Regression (R² = 1.0000)
        - Support Vector Regression (R² = 0.9999)
        - Artificial Neural Network (R² = 0.9734)
        - Random Forest (R² = 0.9482)
        - XGBoost (R² = 0.8940)
        - Decision Tree (R² = 0.8717)
        
        **Classification Models:**
        - Decision Tree (Accuracy: 91.67%)
        - XGBoost (Accuracy: 91.67%)
        - Random Forest (Accuracy: 88.89%)
        - SVC (Accuracy: 88.89%)
        - ANN (Accuracy: 88.89%)
        - Logistic Regression (Accuracy: 86.11%)
        
        ### Dataset
        - **Samples:** 120 water quality measurements
        - **Features:** 9 physicochemical parameters
        - **Location:** Aquaculture facilities
        - **Time Period:** 12-month monitoring
        
        ### How to Use
        1. Go to **Prediction Dashboard**
        2. Enter water quality parameters
        3. Click **Predict Water Quality**
        4. View predictions and recommendations
        5. Check **Parameter Guide** for optimal ranges
        
        ### Citation
        If you use this system, please cite:
        ```
        Water Quality Research Team. (2025). 
        Aquatic Water Quality Index (AWQI) Prediction System. 
        Version 1.0.0.
        ```
        
        ### Contact
        For issues, suggestions, or collaborations, please contact the research team.
        
        ---
        **Version:** 1.0.0  
        **Last Updated:** February 2025  
        **License:** MIT (Open Source)
        """)

if __name__ == "__main__":
    main()
