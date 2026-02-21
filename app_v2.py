"""
Aquatic Water Quality Index (AWQI) Prediction System
Machine Learning Based Water Quality Assessment
Streamlit Application for Real-time Predictions
Version 2.0 - Enhanced with Extended Ranges for Pollution Testing

Author: Water Quality Research Team
Version: 2.0.0
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
    .critical-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .range-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 12px;
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
    """Interpret AWQI score with detailed descriptions"""
    if value < 25:
        return {
            'class': 'Excellent',
            'description': '✅ Excellent water quality - Suitable for all uses (drinking, aquaculture, recreation)',
            'color': '#28a745',
            'emoji': '✅',
            'action': 'No action needed. Continue monitoring regularly.'
        }
    elif value < 50:
        return {
            'class': 'Good',
            'description': '👍 Good water quality - Minor issues, generally acceptable for use',
            'color': '#17a2b8',
            'emoji': '👍',
            'action': 'Minor monitoring recommended. Address any specific parameters.'
        }
    elif value < 75:
        return {
            'class': 'Moderate',
            'description': '⚠️ Moderate water quality - Issues present, action recommended',
            'color': '#ffc107',
            'emoji': '⚠️',
            'action': 'Immediate improvement measures needed. See recommendations below.'
        }
    else:
        return {
            'class': 'Poor',
            'description': '🚫 Poor water quality - Significant pollution, immediate action required',
            'color': '#dc3545',
            'emoji': '🚫',
            'action': 'URGENT: Critical treatment or replacement needed. Do not use for sensitive applications.'
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

def get_parameter_ranges():
    """Get parameter information with EXTENDED ranges for pollution testing"""
    return {
        'TDS': {
            'name': 'Total Dissolved Solids (mg/L)',
            'optimal': '<250',
            'acceptable': '250-500',
            'warning': '500-1000',
            'critical': '>1000',
            'extended_range': (0, 5000),
            'description': 'Measure of all minerals and salts dissolved in water. Higher values indicate salt/mineral pollution.',
            'who_limit': '<600'
        },
        'DO': {
            'name': 'Dissolved Oxygen (mg/L)',
            'optimal': '>7',
            'acceptable': '5-7',
            'warning': '2-5',
            'critical': '<2',
            'extended_range': (0, 15),
            'description': 'Oxygen available for aquatic organisms. <4 mg/L causes stress; 0 is anaerobic (dead zone).',
            'who_limit': '>5'
        },
        'Nitrate': {
            'name': 'Nitrate (mg/L)',
            'optimal': '<10',
            'acceptable': '10-50',
            'warning': '50-200',
            'critical': '>200',
            'extended_range': (0, 2000),
            'description': 'Indicates agricultural/sewage pollution. High values show eutrophication risk.',
            'who_limit': '<45'
        },
        'TH': {
            'name': 'Total Hardness (mg/L)',
            'optimal': '50-150',
            'acceptable': '150-300',
            'warning': '300-500',
            'critical': '>500',
            'extended_range': (0, 2000),
            'description': 'Concentration of Ca²⁺ and Mg²⁺. Very high values usually acceptable but indicate mineral-rich water.',
            'who_limit': 'No strict limit'
        },
        'pH': {
            'name': 'pH Value',
            'optimal': '6.5-8.5',
            'acceptable': '6-9',
            'warning': '5-6 or 9-10',
            'critical': '<5 or >10',
            'extended_range': (0, 14),
            'description': 'Acidity/alkalinity. Extreme pH indicates chemical imbalance or pollution.',
            'who_limit': '6.5-8.5'
        },
        'Chlorides': {
            'name': 'Chlorides (mg/L)',
            'optimal': '<250',
            'acceptable': '250-1000',
            'warning': '1000-2000',
            'critical': '>2000',
            'extended_range': (0, 3000),
            'description': 'Salt concentration. High values indicate saline pollution or seawater intrusion.',
            'who_limit': '<250'
        },
        'Alkalinity': {
            'name': 'Alkalinity (mg/L)',
            'optimal': '50-200',
            'acceptable': '200-500',
            'warning': '500-1000',
            'critical': '>1000',
            'extended_range': (0, 2000),
            'description': 'Buffering capacity of water. Extreme values indicate unusual chemistry.',
            'who_limit': '50-200'
        },
        'EC': {
            'name': 'Electrical Conductivity (µS/cm)',
            'optimal': '500-1500',
            'acceptable': '1500-2500',
            'warning': '2500-5000',
            'critical': '>5000',
            'extended_range': (0, 10000),
            'description': 'Measure of dissolved ions (salinity). High values indicate salt/pollution.',
            'who_limit': '<1000'
        },
        'Ammonia': {
            'name': 'Ammonia (mg/L)',
            'optimal': '<0.1',
            'acceptable': '0.1-0.5',
            'warning': '0.5-5',
            'critical': '>5',
            'extended_range': (0, 100),
            'description': 'Indicates decomposition/organic pollution. High values show severe sewage/waste pollution.',
            'who_limit': '<0.5'
        }
    }

def get_severity_level(tds, do, nitrate, th, ph, chlorides, alkalinity, ec, ammonia):
    """Calculate overall severity level based on all parameters"""
    severity_score = 0
    critical_issues = []
    
    # DO critical
    if do < 2:
        severity_score += 3
        critical_issues.append("Anoxic conditions - no oxygen")
    elif do < 4:
        severity_score += 2
        critical_issues.append("Severe oxygen depletion")
    elif do < 5:
        severity_score += 1
    
    # Ammonia critical
    if ammonia > 5:
        severity_score += 3
        critical_issues.append("Critical ammonia toxicity")
    elif ammonia > 2:
        severity_score += 2
        critical_issues.append("Severe organic pollution")
    elif ammonia > 0.5:
        severity_score += 1
    
    # Other parameters
    if ph < 4 or ph > 11:
        severity_score += 2
        critical_issues.append("Extreme pH - chemical hazard")
    elif ph < 6 or ph > 9.5:
        severity_score += 1
    
    if tds > 1000:
        severity_score += 2
        critical_issues.append("Extreme salinity")
    elif tds > 500:
        severity_score += 1
    
    if nitrate > 200:
        severity_score += 2
        critical_issues.append("Severe nutrient pollution")
    elif nitrate > 50:
        severity_score += 1
    
    if chlorides > 1000:
        severity_score += 1
    
    return severity_score, critical_issues

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
        <p><i>Advanced ML-Based Water Quality Assessment - Version 2.0</i></p>
        <p style="color: #666;">
        <b>Pollution Testing Mode Enabled:</b> Full range of inputs accepted to assess water from pristine to severely polluted conditions
        </p>
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
        st.header("Real-time Water Quality Prediction & Pollution Assessment")
        
        st.markdown("""
        <div class="info-box">
        <b>Advanced Pollution Testing Mode:</b> This system accepts water parameters across the FULL range from pristine to severely polluted. 
        Enter any values to assess water quality and identify pollution levels.
        </div>
        """, unsafe_allow_html=True)
        
        param_ranges = get_parameter_ranges()
        
        # Input section with THREE columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Column 1: General Parameters")
            tds = st.number_input(
                "TDS (mg/L)", 
                min_value=0.0, 
                max_value=5000.0, 
                value=170.0, 
                step=1.0,
                help="0-250: Optimal | 250-500: Acceptable | 500-1000: Warning | >1000: Critical"
            )
            ph = st.number_input(
                "pH", 
                min_value=0.0, 
                max_value=14.0, 
                value=7.8, 
                step=0.1,
                help="6.5-8.5: Optimal | 6-9: Acceptable | Extreme: Critical"
            )
            alkalinity = st.number_input(
                "Alkalinity (mg/L)", 
                min_value=0.0, 
                max_value=2000.0, 
                value=70.0, 
                step=1.0,
                help="50-200: Optimal | 200-500: Acceptable | >500: Warning"
            )
        
        with col2:
            st.subheader("Column 2: Biological Indicators")
            do = st.number_input(
                "DO (mg/L)", 
                min_value=0.0, 
                max_value=15.0, 
                value=5.0, 
                step=0.1,
                help=">7: Optimal | 5-7: Acceptable | 2-5: Warning | <2: CRITICAL (Anoxic)"
            )
            chlorides = st.number_input(
                "Chlorides (mg/L)", 
                min_value=0.0, 
                max_value=3000.0, 
                value=25.0, 
                step=1.0,
                help="<250: Optimal | 250-1000: Acceptable | >1000: Warning"
            )
            ec = st.number_input(
                "EC (µS/cm)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=280.0, 
                step=10.0,
                help="500-1500: Optimal | 1500-2500: Acceptable | >2500: Warning"
            )
        
        with col3:
            st.subheader("Column 3: Pollution Indicators")
            nitrate = st.number_input(
                "Nitrate (mg/L)", 
                min_value=0.0, 
                max_value=2000.0, 
                value=0.4, 
                step=0.1,
                help="<10: Optimal | 10-50: Acceptable | 50-200: Warning | >200: CRITICAL"
            )
            th = st.number_input(
                "Total Hardness (mg/L)", 
                min_value=0.0, 
                max_value=2000.0, 
                value=140.0, 
                step=1.0,
                help="50-150: Optimal | 150-300: Acceptable | >300: High but usually acceptable"
            )
            ammonia = st.number_input(
                "Ammonia (mg/L)", 
                min_value=0.0, 
                max_value=100.0, 
                value=0.01, 
                step=0.001,
                help="<0.1: Optimal | 0.1-0.5: Acceptable | 0.5-5: Warning | >5: CRITICAL (Toxic)"
            )
        
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
        
        # Prediction button
        if st.button("🔍 Predict Water Quality", use_container_width=True, type="primary"):
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
            
            # Calculate severity
            severity_score, critical_issues = get_severity_level(
                tds, do, nitrate, th, ph, chlorides, alkalinity, ec, ammonia
            )
            
            # Display results
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.subheader("🎯 AWQI Score & Classification")
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
                    <p style="font-size: 15px;">{interpretation['description']}</p>
                    <p style="font-size: 14px; margin-top: 10px;"><b>Action:</b> {interpretation['action']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.subheader("📊 Model Predictions & Severity")
                
                # Show severity indicator
                if severity_score >= 6:
                    st.error(f"🚨 **CRITICAL SEVERITY** - Multiple severe issues. Immediate intervention required! (Score: {severity_score}/10)")
                    if critical_issues:
                        for issue in critical_issues:
                            st.error(f"• {issue}")
                elif severity_score >= 4:
                    st.warning(f"🔴 **HIGH SEVERITY** - Significant problems detected (Score: {severity_score}/10)")
                    if critical_issues:
                        for issue in critical_issues:
                            st.warning(f"• {issue}")
                elif severity_score >= 2:
                    st.warning(f"🟡 **MODERATE SEVERITY** - Issues present (Score: {severity_score}/10)")
                elif severity_score >= 1:
                    st.info(f"🟢 **LOW SEVERITY** - Minor issues (Score: {severity_score}/10)")
                else:
                    st.success("✅ **EXCELLENT** - No significant issues")
                
                st.markdown("---")
                
                # Show all model predictions
                st.write("**All Model Predictions:**")
                pred_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'AWQI Score': list(predictions.values())
                }).sort_values('AWQI Score', ascending=False)
                
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Classification
                st.subheader("🏷️ Water Quality Class")
                scaled_clf = scalers['classification'].transform(features)
                
                if clf_models:
                    best_clf = list(clf_models.values())[0]
                    class_pred = best_clf.predict(scaled_clf)[0]
                    class_name = encoders['class_names'][class_pred]
                    
                    if hasattr(best_clf, 'predict_proba'):
                        proba = best_clf.predict_proba(scaled_clf)[0]
                        confidence = proba[class_pred] * 100
                        
                        col_class1, col_class2 = st.columns(2)
                        with col_class1:
                            st.metric("Predicted Class", class_name)
                        with col_class2:
                            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Detailed Recommendations
            st.subheader("💡 Detailed Water Quality Assessment & Recommendations")
            
            recommendations = []
            
            # ===== DISSOLVED OXYGEN =====
            if do < 2:
                rec_text = """🚨 **CRITICAL - Dissolved Oxygen <2 mg/L (ANOXIC)**
                    - Water has essentially NO oxygen
                    - Aquatic life CANNOT survive
                    - Immediate emergency intervention required:
                      • Increase aeration to maximum (install multiple aeration systems)
                      • Increase water circulation (pumps, fountains)
                      • Partial or complete water replacement may be necessary
                      • Consider emergency oxygen injection
                    - Cause: Severe organic pollution, decomposition, or algal crash"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif do < 4:
                rec_text = """🔴 **SEVERE - Dissolved Oxygen 2-4 mg/L (ANOXIC STRESS)**
                    - Most fish species will suffer/die
                    - Stress on all aquatic organisms
                    - URGENT intervention needed:
                      • Install aeration system immediately (if not present)
                      • Increase aeration capacity
                      • Increase water exchange/circulation
                      • Reduce fish stocking density immediately
                    - Monitor every 2-4 hours"""
                recommendations.append(("🔴 SEVERE", rec_text, 'critical'))
            elif do < 5:
                rec_text = """🟠 **HIGH - Dissolved Oxygen <5 mg/L (STRESSED)**
                    - Below optimal; fish show stress
                    - Risk of disease and mortality
                    - Recommended actions:
                      • Increase aeration immediately
                      • Reduce feed input
                      • Increase water circulation
                      • Monitor every 6-8 hours"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            elif do < 7:
                rec_text = """🟡 **MODERATE - Dissolved Oxygen 5-7 mg/L (SUBOPTIMAL)**
                    - Below optimal for sensitive species
                    - Consider improving aeration
                    - Monitor regularly"""
                recommendations.append(("🟡 MODERATE", rec_text, 'warning'))
            
            # ===== AMMONIA =====
            if ammonia > 5:
                rec_text = """🚨 **CRITICAL - Ammonia >5 mg/L (HIGHLY TOXIC)**
                    - SEVERE toxic pollution
                    - Water is heavily contaminated with organic matter
                    - Immediate treatment required:
                      • Do NOT use this water for fish
                      • Partial or complete water replacement urgently needed
                      • Increase biological filtration or use chemical treatment
                      • Reduce feed/organic matter input
                      • Consider complete water system cleanup
                    - Source: Excessive organic waste, sewage, decomposition"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif ammonia > 2:
                rec_text = """🔴 **SEVERE - Ammonia 2-5 mg/L (HIGH TOXICITY)**
                    - High ammonia toxicity; fish stress/mortality expected
                    - Significant organic pollution
                    - Urgent water treatment needed:
                      • Partial water exchange (25-50%)
                      • Increase biological filtration
                      • Reduce feed immediately
                      • Improve aeration and circulation
                    - Monitor daily"""
                recommendations.append(("🔴 SEVERE", rec_text, 'critical'))
            elif ammonia > 0.5:
                rec_text = """🟠 **HIGH - Ammonia >0.5 mg/L (ELEVATED)**
                    - Indicates organic pollution
                    - Recommended actions:
                      • Reduce feed input
                      • Improve water circulation
                      • Enhance biological filtration
                      • Partial water change
                      • Remove dead organic matter"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            elif ammonia > 0.1:
                rec_text = """🟡 **MODERATE - Ammonia 0.1-0.5 mg/L (DETECTABLE)**
                    - Minor pollution; consider improvement
                    - Recommend enhanced filtration and monitoring"""
                recommendations.append(("🟡 MODERATE", rec_text, 'warning'))
            
            # ===== pH =====
            if ph < 4 or ph > 11:
                rec_text = f"""🚨 **CRITICAL - pH Extreme (pH {ph:.1f})**
                    - Water chemistry severely imbalanced
                    - pH outside viable range for aquatic life
                    - Immediate pH correction required:
                      • Use appropriate pH buffers/adjusters
                      • Consult water chemistry expert
                      • Consider water replacement if pH cannot be corrected
                    - Source: Industrial discharge, acid rain, or chemical contamination"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif ph < 6 or ph > 9.5:
                rec_text = f"""🟠 **HIGH - pH Out of Safe Range (pH {ph:.1f})**
                    - Water chemistry imbalanced
                    - Requires pH adjustment using buffers:
                      • Use pH increaser/decreaser as needed
                      • Monitor pH daily until stable
                      • Add buffer additives to prevent swings"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            elif ph < 6.5 or ph > 8.5:
                rec_text = f"""🟡 **MODERATE - pH Suboptimal (pH {ph:.1f})**
                    - pH outside optimal range
                    - Consider pH buffering for better conditions
                    - Monitor for stability"""
                recommendations.append(("🟡 MODERATE", rec_text, 'warning'))
            
            # ===== TDS =====
            if tds > 1000:
                rec_text = f"""🚨 **CRITICAL - TDS >1000 mg/L (HIGHLY SALINE)**
                    - Water is extremely saline/mineralized
                    - Not suitable for freshwater organisms
                    - Immediate action required:
                      • Water replacement (partial or complete)
                      • Dilution with fresh water
                      • Investigate pollution source
                    - Source: Seawater intrusion, salt water mixing, industrial discharge"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif tds > 500:
                rec_text = f"""🟠 **HIGH - TDS >500 mg/L (ELEVATED)**
                    - Salt/mineral accumulation
                    - Recommended actions:
                      • Monitor regularly
                      • Partial water exchange (30-50%)
                      • Reduce evaporation where possible
                      • Check for contamination sources"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            elif tds > 300:
                rec_text = f"""🟡 **MODERATE - TDS >300 mg/L**
                    - Slightly elevated dissolved solids
                    - Partial water change recommended
                    - Monitor trends"""
                recommendations.append(("🟡 MODERATE", rec_text, 'warning'))
            
            # ===== NITRATE =====
            if nitrate > 200:
                rec_text = f"""🚨 **CRITICAL - Nitrate >200 mg/L (SEVERE POLLUTION)**
                    - Severe nutrient pollution
                    - Indicates heavy organic contamination
                    - Immediate intervention required:
                      • Partial water replacement (50-75%)
                      • Biological treatment (algae removal, denitrification)
                      • Investigate pollution source
                      • Reduce feed/organic input significantly
                    - Source: Sewage, intensive aquaculture, agricultural runoff"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif nitrate > 50:
                rec_text = f"""🟠 **HIGH - Nitrate >50 mg/L**
                    - Significant pollution detected
                    - Recommended actions:
                      • Reduce feed input by 30-50%
                      • Increase biological filtration
                      • Partial water exchange (25-50%)
                      • Consider algae harvesting for nitrate removal
                      • Monitor weekly"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            elif nitrate > 10:
                rec_text = f"""🟡 **MODERATE - Nitrate >10 mg/L**
                    - Elevated nutrient levels
                    - Routine improvement measures:
                      • Reduce feed input slightly
                      • Enhance biological filtration
                      • Monitor bi-weekly"""
                recommendations.append(("🟡 MODERATE", rec_text, 'warning'))
            
            # ===== CHLORIDES =====
            if chlorides > 1000:
                rec_text = f"""🚨 **CRITICAL - Chlorides >1000 mg/L**
                    - Water is highly saline
                    - Immediate action required:
                      • Dilution with fresh water
                      • Partial water replacement
                      • Identify salt source"""
                recommendations.append(("🚨 CRITICAL", rec_text, 'critical'))
            elif chlorides > 500:
                rec_text = f"""🟠 **HIGH - Chlorides >500 mg/L**
                    - High salt content
                    - Monitor and plan water exchange
                    - Reduce salt input sources"""
                recommendations.append(("🟠 HIGH", rec_text, 'warning'))
            
            # Display recommendations
            if recommendations:
                for severity_label, rec_text, rec_type in recommendations:
                    if rec_type == 'critical':
                        st.markdown(f'<div class="critical-box">{rec_text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="warning-box">{rec_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="success-box">✅ All parameters within excellent ranges! Water quality is perfect for all uses.</div>',
                    unsafe_allow_html=True
                )
    
    # ========================================================================
    # PAGE 2: PARAMETER GUIDE
    # ========================================================================
    elif page == "📚 Parameter Guide":
        st.header("Water Quality Parameters - Complete Reference Guide")
        
        st.markdown("""
        <div class="info-box">
        <b>Extended Range Testing:</b> This guide shows parameters from pristine to severely polluted water. 
        All ranges are accepted for assessment - use to identify pollution levels and types.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 Understanding Parameter Ranges:
        
        - **🟢 OPTIMAL:** Best for all uses (drinking, aquaculture, recreation)
        - **🟡 ACCEPTABLE:** Minor issues, generally safe
        - **🟠 WARNING:** Elevated, requires monitoring/treatment
        - **🔴 CRITICAL:** Severe issues, immediate action needed
        """)
        
        params = get_parameter_ranges()
        
        for param_code, param_info in params.items():
            with st.expander(f"📌 {param_info['name']}"):
                st.write(f"**Description:** {param_info['description']}")
                st.write(f"**WHO/BIS Standard:** {param_info['who_limit']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="range-info"><b>🟢 OPTIMAL</b><br>{param_info["optimal"]}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="range-info"><b>🟡 ACCEPTABLE</b><br>{param_info["acceptable"]}</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="range-info"><b>🟠 WARNING</b><br>{param_info["warning"]}</div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="range-info"><b>🔴 CRITICAL</b><br>{param_info["critical"]}</div>', unsafe_allow_html=True)
        
        # Classification standards
        st.subheader("📊 Water Quality Classification Standards (AWQI)")
        
        standards = pd.DataFrame({
            'AWQI Score': ['0-25', '25-50', '50-75', '>75'],
            'Classification': ['Excellent', 'Good', 'Moderate', 'Poor'],
            'Suitability': [
                '✅ All uses (drinking, swimming, aquaculture)',
                '👍 Most uses with minor precautions',
                '⚠️ Limited uses, treatment recommended',
                '🚫 Not suitable without major treatment'
            ]
        })
        
        st.dataframe(standards, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # PAGE 3: MODEL PERFORMANCE
    # ========================================================================
    elif page == "📈 Model Performance":
        st.header("Machine Learning Model Performance Metrics")
        
        st.markdown("""
        <div class="info-box">
        <b>Model Comparison:</b> Performance of trained regression and classification models on test data
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🔄 Regression Models (AWQI Score Prediction)")
        
        regression_performance = {
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'XGBoost', 'ANN'],
            'R² Score': [1.0000, 0.8717, 0.9482, 0.9999, 0.8940, 0.9734],
            'MSE': [0.0000, 15.0384, 6.0648, 0.0058, 12.4190, 3.1206],
            'Status': ['⭐ Perfect', 'Good', 'Excellent', '⭐ Best', 'Good', 'Excellent']
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
        st.header("About This System - Version 2.0")
        
        st.markdown("""
        ## Aquatic Water Quality Index (AWQI) Prediction System
        
        An advanced machine learning system for assessing water quality across the FULL spectrum - from pristine to severely polluted conditions.
        
        ### 🎯 Purpose
        - **Identify water quality issues** at any pollution level
        - **Assess suitability** for different uses (drinking, aquaculture, recreation, etc.)
        - **Provide actionable recommendations** for water treatment/improvement
        - **Support decision-making** for water resource management
        
        ### ✨ Key Features
        - **Full Range Testing:** Accept parameters from pristine to severely polluted water
        - **Extended Input Ranges:** Ammonia up to 100 mg/L, TDS up to 5000 mg/L, etc.
        - **12 ML Models:** 6 regression + 6 classification algorithms
        - **Severity Assessment:** Identify critical issues requiring immediate action
        - **Detailed Recommendations:** Specific treatment steps for each issue
        - **Responsive Design:** Works on desktop, tablet, and mobile
        
        ### 📊 Models Trained
        
        **Regression Models (AWQI Score Prediction):**
        - Linear Regression (R² = 1.0000)
        - Support Vector Regression (R² = 0.9999)
        - Artificial Neural Network (R² = 0.9734)
        - Random Forest (R² = 0.9482)
        - XGBoost (R² = 0.8940)
        - Decision Tree (R² = 0.8717)
        
        **Classification Models (Water Class Prediction):**
        - Decision Tree (Accuracy: 91.67%) ⭐
        - XGBoost (Accuracy: 91.67%) ⭐
        - Random Forest (Accuracy: 88.89%)
        - SVC (Accuracy: 88.89%)
        - ANN (Accuracy: 88.89%)
        - Logistic Regression (Accuracy: 86.11%)
        
        ### 📈 Dataset
        - **Samples:** 120 water quality measurements
        - **Parameters:** 9 physicochemical indicators
        - **Coverage:** Real aquaculture facility data
        - **Completeness:** 100% (no missing values)
        
        ### 📋 Input Parameters
        
        1. **TDS (0-5000 mg/L):** Total dissolved solids
        2. **DO (0-15 mg/L):** Dissolved oxygen  
        3. **Nitrate (0-2000 mg/L):** Nutrient pollution
        4. **TH (0-2000 mg/L):** Total hardness
        5. **pH (0-14):** Acidity/alkalinity
        6. **Chlorides (0-3000 mg/L):** Salt concentration
        7. **Alkalinity (0-2000 mg/L):** Buffering capacity
        8. **EC (0-10000 µS/cm):** Conductivity/salinity
        9. **Ammonia (0-100 mg/L):** Organic pollution
        
        ### 🎯 Output
        
        - **AWQI Score:** 0-100+ scale indicating water quality
        - **Classification:** Excellent, Good, Moderate, or Poor
        - **Severity Assessment:** Overall pollution level
        - **Detailed Recommendations:** Treatment options for each issue
        - **Model Predictions:** Consensus across 6 regression models
        
        ### 🔬 How It Works
        
        1. **Input:** Enter water quality parameters
        2. **Preprocessing:** Features scaled using StandardScaler
        3. **Prediction:** Multiple ML models generate predictions
        4. **Classification:** Determine water quality class
        5. **Assessment:** Calculate severity and critical issues
        6. **Recommendations:** Provide specific treatment guidance
        7. **Output:** Display comprehensive results
        
        ### 💡 Use Cases
        
        - **Aquaculture:** Monitor fish farm water quality daily
        - **Drinking Water:** Assess potability and treatment needs
        - **Wastewater:** Evaluate treatment effectiveness
        - **Surface Water:** Monitor lakes, rivers for pollution
        - **Environmental Protection:** Track water pollution trends
        - **Education:** Teach water quality assessment
        - **Research:** Validate predictive models
        
        ### 📱 Technology Stack
        - **Frontend:** Streamlit (Interactive web interface)
        - **ML Library:** Scikit-learn (Models and scaling)
        - **Boosting:** XGBoost (Gradient boosting)
        - **Data:** Pandas, NumPy
        - **Deployment:** Streamlit Cloud (FREE)
        
        ### 🌐 Accessibility
        - ✅ No installation required (web-based)
        - ✅ Works on any device (desktop, tablet, mobile)
        - ✅ Works on any browser (Chrome, Firefox, Safari, Edge)
        - ✅ No data storage (privacy-focused)
        - ✅ Real-time predictions (<1 second)
        
        ### 📞 How to Use
        1. Go to **Prediction Dashboard**
        2. Enter water quality parameters
        3. Click **Predict Water Quality**
        4. Review classification and recommendations
        5. Check **Parameter Guide** for detailed info
        6. Consult **Model Performance** for accuracy details
        
        ### 🏆 Version History
        - **v1.0:** Initial system with standard ranges
        - **v2.0:** Extended ranges for pollution testing, detailed recommendations
        
        ### 📝 Citation
        ```
        Water Quality Research Team. (2025). 
        Aquatic Water Quality Index (AWQI) Prediction System. 
        Version 2.0.0.
        ```
        
        ---
        **Version:** 2.0.0  | **Status:** ✅ Production Ready  | **Date:** February 2025
        """)

if __name__ == "__main__":
    main()
