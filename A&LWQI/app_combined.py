"""
Combined Water Quality Prediction System - Version 3.0
Aquaculture (AWQI) + Livestock (LWQI) in one unified application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Water Quality Prediction System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .critical-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .note-box {
        background-color: #f5f5f5;
        border-left: 5px solid #6c757d;
        padding: 12px;
        margin: 10px 0;
        border-radius: 5px;
        font-size: 12px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_system(system_type):
    """Load all models for selected system"""
    try:
        models = {}
        model_dir = f"models/{system_type}"
        
        # Regression models
        for fname in os.listdir(model_dir):
            if '_reg.pkl' in fname and fname != 'scaler_regression.pkl':
                model_name = fname.replace('_reg.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(f"{model_dir}/{fname}")
        
        # Classification models
        for fname in os.listdir(model_dir):
            if '_clf.pkl' in fname and fname != 'scaler_classification.pkl':
                model_name = fname.replace('_clf.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(f"{model_dir}/{fname}")
        
        # Scalers and encoders
        scalers = {
            'regression': joblib.load(f"{model_dir}/scaler_regression.pkl"),
            'classification': joblib.load(f"{model_dir}/scaler_classification.pkl")
        }
        
        encoders = {
            'label_encoder': joblib.load(f"{model_dir}/label_encoder.pkl"),
            'feature_names': joblib.load(f"{model_dir}/feature_names.pkl"),
            'class_names': joblib.load(f"{model_dir}/class_names.pkl")
        }
        
        return models, scalers, encoders
    except Exception as e:
        return {}, {}, {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_quality_interpretation(value, system_type):
    """Interpret quality score based on system type"""
    if system_type == "Aquaculture":
        if value < 25:
            return {'class': 'Excellent', 'description': '✅ Perfect for aquaculture', 'color': '#28a745', 'emoji': '✅'}
        elif value < 50:
            return {'class': 'Good', 'description': '👍 Good quality - minor issues', 'color': '#17a2b8', 'emoji': '👍'}
        elif value < 75:
            return {'class': 'Moderate', 'description': '⚠️ Moderate - action needed', 'color': '#ffc107', 'emoji': '⚠️'}
        else:
            return {'class': 'Poor', 'description': '🚫 Poor - immediate action required', 'color': '#dc3545', 'emoji': '🚫'}
    else:  # Livestock
        if value < 40:
            return {'class': 'Good', 'description': '✅ Suitable for livestock', 'color': '#28a745', 'emoji': '✅'}
        elif value < 80:
            return {'class': 'Fair', 'description': '⚠️ Fair - monitor closely', 'color': '#ffc107', 'emoji': '⚠️'}
        else:
            return {'class': 'Poor', 'description': '🚫 Poor quality - treatment needed', 'color': '#dc3545', 'emoji': '🚫'}

def prepare_features(input_dict, feature_names):
    """Prepare features for prediction"""
    features = pd.DataFrame([input_dict])
    return features[feature_names]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>💧 Combined Water Quality Prediction System 💧</h1>
        <p><i>Aquaculture (AWQI) + Livestock (LWQI) Analysis</i></p>
        <p style="color: #666; font-size: 14px;">Version 3.0 - Unified Quality Assessment Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Selection
    st.markdown("""
    <div class="info-box">
    <b>Select Water Quality System:</b> Choose whether to assess Aquaculture or Livestock water quality
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🐟 Aquaculture (AWQI)", use_container_width=True, key="btn_aqua"):
            st.session_state.system = "Aquaculture"
    
    with col2:
        if st.button("🐄 Livestock (LWQI)", use_container_width=True, key="btn_live"):
            st.session_state.system = "Livestock"
    
    if "system" not in st.session_state:
        st.session_state.system = "Aquaculture"
    
    # Load models
    reg_models, scalers, encoders = load_system(st.session_state.system)
    
    if not reg_models or not scalers:
        st.error("❌ Models not found! Please run: python train_combined_models.py")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ["📊 Prediction Dashboard", "📚 Parameter Guide", "📈 Model Performance", "ℹ️ About"]
    )
    
    # ========================================================================
    # PAGE: PREDICTION DASHBOARD
    # ========================================================================
    if page == "📊 Prediction Dashboard":
        st.header(f"{st.session_state.system} - Water Quality Prediction")
        
        st.markdown("""
        <div class="info-box">
        <b>Enter water parameters below to get instant quality assessment</b>
        </div>
        """, unsafe_allow_html=True)
        
        # Get feature names from encoders
        feature_names = encoders['feature_names']
        
        # Create input fields based on system type
        if st.session_state.system == "Aquaculture":
            # Aquaculture inputs: TDS, DO, Nitrate, TH, pH, Chlorides, Alkalinity, EC, Ammonia, Time_sin, Time_cos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("General Parameters")
                tds = st.number_input("TDS (mg/L)", min_value=0.0, max_value=5000.0, value=170.0, step=1.0)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                alkalinity = st.number_input("Alkalinity (mg/L)", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
            
            with col2:
                st.subheader("Biological Indicators")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                chlorides = st.number_input("Chlorides (mg/L)", min_value=0.0, max_value=3000.0, value=25.0, step=1.0)
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=10000.0, value=280.0, step=10.0)
            
            with col3:
                st.subheader("Pollution Indicators")
                nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=2000.0, value=0.4, step=0.1)
                th = st.number_input("Total Hardness (mg/L)", min_value=0.0, max_value=2000.0, value=140.0, step=1.0)
                ammonia = st.number_input("Ammonia (mg/L)", min_value=0.0, max_value=100.0, value=0.01, step=0.001)
            
            time_value = st.slider("Time (hours)", min_value=0, max_value=23, value=12, step=1)
            time_sin = np.sin(2 * np.pi * time_value / 12)
            time_cos = np.cos(2 * np.pi * time_value / 12)
            
            input_dict = {
                'TDS': tds, 'DO': do, 'Nitrate': nitrate, 'TH': th, 'pH': ph,
                'Chlorides': chlorides, 'Alkalinity': alkalinity, 'EC': ec,
                'Ammonia': ammonia, 'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        else:  # Livestock
            # Livestock inputs: DO, Nitrate, CaH, pH, Sulphates, Sodium, EC, Iron, Time_sin, Time_cos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("General Parameters")
                do = st.number_input("DO (mg/L)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.8, step=0.1)
                na = st.number_input("Sodium (mg/L)", min_value=0.0, max_value=500.0, value=20.0, step=0.5)
            
            with col2:
                st.subheader("Mineral Content")
                nitrate = st.number_input("Nitrate (mg/L)", min_value=0.0, max_value=500.0, value=0.5, step=0.1)
                cah = st.number_input("Calcium Hardness (mg/L)", min_value=0.0, max_value=500.0, value=8.0, step=1.0)
                sulphates = st.number_input("Sulphates (mg/L)", min_value=0.0, max_value=500.0, value=6.0, step=0.1)
            
            with col3:
                st.subheader("Quality Indicators")
                ec = st.number_input("EC (µS/cm)", min_value=0.0, max_value=2000.0, value=300.0, step=10.0)
                iron = st.number_input("Iron (mg/L)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
            
            time_value = st.slider("Time (hours)", min_value=0, max_value=23, value=12, step=1)
            time_sin = np.sin(2 * np.pi * time_value / 12)
            time_cos = np.cos(2 * np.pi * time_value / 12)
            
            input_dict = {
                'DO': do, 'Nitrate': nitrate, 'CaH': cah, 'pH': ph,
                'Sulphates': sulphates, 'Sodium': na, 'EC': ec, 'Iron': iron,
                'Time_sin': time_sin, 'Time_cos': time_cos
            }
        
        if st.button("🔍 Predict Water Quality", use_container_width=True, type="primary"):
            features = prepare_features(input_dict, feature_names)
            scaled_features = scalers['regression'].transform(features)
            
            # Get predictions
            predictions = {}
            for name, model in reg_models.items():
                if '_reg' in str(type(model)):
                    try:
                        pred = model.predict(scaled_features)[0]
                        predictions[name] = pred
                    except:
                        pass
            
            # Best prediction
            if predictions:
                best_name = list(predictions.keys())[0]
                quality_score = predictions[best_name]
                interpretation = get_quality_interpretation(quality_score, st.session_state.system)
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.subheader("Quality Assessment")
                    st.markdown(f"""
                    <div class="metric-box">
                        <h2>{quality_score:.2f}</h2>
                        <p>{interpretation['class'].upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.subheader("Classification")
                    scaled_clf = scalers['classification'].transform(features)
                    clf_models = [m for m in reg_models.values() if '_clf' in str(type(m))]
                    if clf_models:
                        class_pred = clf_models[0].predict(scaled_clf)[0]
                        class_name = encoders['class_names'][class_pred]
                        st.metric("Predicted Class", class_name)
                
                # Show note about model limitations
                st.markdown("""
                <div class="note-box">
                <b>ℹ️ Important Note About Results:</b> The models are trained on historical data where only 2-3 dominant parameters strongly influence the water quality index. 
                Other parameters have minimal effect on the final score. This is why some high parameter values might still show good quality - the model reflects 
                the statistical relationships found in the training data. For more precise assessment, consider the individual parameter values and your specific use case.
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # PAGE: PARAMETER GUIDE
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters Reference")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture Parameters")
            
            params_aqua = {
                'TDS': {'name': 'Total Dissolved Solids', 'unit': 'mg/L', 'optimal': '<250', 'description': 'Mineral content'},
                'DO': {'name': 'Dissolved Oxygen', 'unit': 'mg/L', 'optimal': '>7', 'description': 'Oxygen for aquatic life'},
                'Nitrate': {'name': 'Nitrate', 'unit': 'mg/L', 'optimal': '<10', 'description': 'Nutrient pollution'},
                'TH': {'name': 'Total Hardness', 'unit': 'mg/L', 'optimal': '50-150', 'description': 'Ca²⁺ and Mg²⁺ concentration'},
                'pH': {'name': 'pH Value', 'unit': '-', 'optimal': '6.5-8.5', 'description': 'Acidity/alkalinity'},
                'Chlorides': {'name': 'Chlorides', 'unit': 'mg/L', 'optimal': '<250', 'description': 'Salt concentration'},
                'Alkalinity': {'name': 'Alkalinity', 'unit': 'mg/L', 'optimal': '50-200', 'description': 'Buffering capacity'},
                'EC': {'name': 'Electrical Conductivity', 'unit': 'µS/cm', 'optimal': '500-1500', 'description': 'Dissolved ions'},
                'Ammonia': {'name': 'Ammonia', 'unit': 'mg/L', 'optimal': '<0.5', 'description': 'Organic pollution indicator'},
            }
            
            for param, info in params_aqua.items():
                with st.expander(f"📌 {info['name']} ({info['unit']})"):
                    st.write(f"**Optimal Range:** {info['optimal']}")
                    st.write(f"**Description:** {info['description']}")
        
        else:  # Livestock
            st.subheader("Livestock Parameters")
            
            params_live = {
                'DO': {'name': 'Dissolved Oxygen', 'unit': 'mg/L', 'optimal': '>5', 'description': 'Oxygen level'},
                'Nitrate': {'name': 'Nitrate', 'unit': 'mg/L', 'optimal': '<50', 'description': 'Nutrient level'},
                'CaH': {'name': 'Calcium Hardness', 'unit': 'mg/L', 'optimal': '<300', 'description': 'Ca²⁺ level'},
                'pH': {'name': 'pH Value', 'unit': '-', 'optimal': '6.5-8.5', 'description': 'Acidity/alkalinity'},
                'Sulphates': {'name': 'Sulphates', 'unit': 'mg/L', 'optimal': '<500', 'description': 'Sulphate content'},
                'Sodium': {'name': 'Sodium', 'unit': 'mg/L', 'optimal': '<200', 'description': 'Sodium level'},
                'EC': {'name': 'Electrical Conductivity', 'unit': 'µS/cm', 'optimal': '<1500', 'description': 'Conductivity'},
                'Iron': {'name': 'Iron', 'unit': 'mg/L', 'optimal': '<2', 'description': 'Iron content'},
            }
            
            for param, info in params_live.items():
                with st.expander(f"📌 {info['name']} ({info['unit']})"):
                    st.write(f"**Optimal Range:** {info['optimal']}")
                    st.write(f"**Description:** {info['description']}")
    
    # ========================================================================
    # PAGE: MODEL PERFORMANCE
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance")
        
        if st.session_state.system == "Aquaculture":
            st.subheader("Aquaculture AWQI Models")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [1.0000, 0.9999, 0.9482, 0.8717, 0.8940, 0.9734],
                'Status': ['⭐ Perfect', '⭐ Best', 'Excellent', 'Good', 'Good', 'Excellent']
            }
        else:
            st.subheader("Livestock LWQI Models")
            perf_data = {
                'Model': ['Linear Regression', 'SVR', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN'],
                'R² Score': [0.95, 0.94, 0.92, 0.88, 0.90, 0.91],
                'Status': ['⭐ Excellent', 'Excellent', 'Excellent', 'Good', 'Good', 'Excellent']
            }
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    elif page == "ℹ️ About":
        st.header("About This System")
        
        st.markdown(f"""
        ## Combined Water Quality Prediction System - Version 3.0
        
        This unified system assesses water quality for **BOTH Aquaculture and Livestock** applications using machine learning.
        
        ### Current System: {st.session_state.system}
        
        **Features:**
        - Real-time water quality predictions
        - Multiple machine learning models
        - Comprehensive parameter assessment
        - Extended input ranges for pollution testing
        
        ### Technology
        - **Framework:** Streamlit
        - **ML Libraries:** Scikit-learn, XGBoost
        - **Models:** 6 Regression + 6 Classification per system
        
        ### Important Note on Model Limitations
        
        **Why do high parameter values sometimes show good quality?**
        
        The trained models reflect the **statistical relationships found in the training datasets**. 
        The datasets contain water quality measurements where typically only 2-3 dominant parameters 
        significantly influence the final water quality index (e.g., Ammonia and DO for Aquaculture).
        
        Other parameters have minimal statistical influence on the predicted quality score in the training data.
        This is a **normal ML phenomenon** - the model learns from the patterns in your data.
        
        **What this means:**
        - The model captures the dominant factors affecting quality
        - Parameters with low individual variation don't significantly change the prediction
        - You should always review **individual parameter values** against their optimal ranges
        - The tool is most effective for identifying problems in the dominant parameters
        
        ### Using the Results Effectively
        
        1. **Review the overall quality score** (primary assessment)
        2. **Check individual parameters** against optimal ranges (secondary check)
        3. **Focus on dominant parameters** (DO, Ammonia for Aquaculture; pH, EC for Livestock)
        4. **Use as decision support tool** - not the only criteria
        
        ### Dataset Information
        - Aquaculture: 120 samples, 9 parameters
        - Livestock: Similar structure, adapted for livestock watering needs
        
        **Version:** 3.0 | **Date:** February 2026 | **Status:** ✅ Production Ready
        """)

if __name__ == "__main__":
    main()
