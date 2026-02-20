# Aquatic Water Quality Index (AWQI) Prediction System

A machine learning-based web application for predicting and monitoring water quality in aquaculture environments using Streamlit.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen)

## 🌊 Overview

The AWQI Prediction System combines multiple machine learning algorithms to predict the Aquatic Water Quality Index and classify water quality into categories (Excellent, Good, Moderate, Poor). It provides real-time predictions and actionable recommendations for aquaculture facility managers.

### Key Features

- **Real-time AWQI Predictions** - Instant water quality scoring from 9 parameters
- **Multi-class Classification** - Categorize water quality into predefined classes
- **Batch Processing** - Analyze multiple water samples from CSV files
- **Parameter Guidance** - Comprehensive reference guide for all water quality indicators
- **Model Comparison** - Performance metrics for 6+ ML algorithms
- **Production Ready** - Fully tested and optimized for deployment

## 📊 Model Performance

### Regression Models (AWQI Score Prediction)

| Model | R² Score | MSE | Status |
|-------|----------|-----|--------|
| Linear Regression | 1.0000 | 0.0000 | ⭐ |
| SVR | 0.9999 | 0.0058 | ⭐ |
| ANN | 0.9734 | 3.1206 | Excellent |
| Random Forest | 0.9482 | 6.0648 | Excellent |
| XGBoost | 0.8940 | 12.4190 | Good |
| Decision Tree | 0.8717 | 15.0384 | Good |

### Classification Models (Water Quality Class)

| Model | Accuracy |
|-------|----------|
| Decision Tree | 91.67% |
| XGBoost | 91.67% |
| Random Forest | 88.89% |
| SVC | 88.89% |
| ANN | 88.89% |
| Logistic Regression | 86.11% |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/awqi-prediction-system.git
cd awqi-prediction-system
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train and save models**
```bash
python train_models.py
```

This will create a `models/` directory with all trained models.

5. **Run the application**
```bash
streamlit run app.py
```

Access the app at: **http://localhost:8501**

## 📁 Project Structure

```
awqi-prediction-system/
├── app.py                      # Main Streamlit application
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── Aquaculture.csv            # Training dataset
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── models/                    # Trained models directory
│   ├── linear_regression_reg.pkl
│   ├── decision_tree_reg.pkl
│   ├── random_forest_reg.pkl
│   ├── svr_reg.pkl
│   ├── xgboost_reg.pkl
│   ├── mlpregressor_reg.pkl
│   ├── logistic_regression_clf.pkl
│   ├── decision_tree_clf.pkl
│   ├── random_forest_clf.pkl
│   ├── svc_clf.pkl
│   ├── xgboost_clf.pkl
│   ├── mlpclassifier_clf.pkl
│   ├── scaler_regression.pkl
│   ├── scaler_classification.pkl
│   └── label_encoder.pkl
└── .streamlit/
    └── config.toml            # Streamlit configuration (optional)
```

## 💾 Data Format

The Aquaculture.csv file should contain the following columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| Code | String | - | Sample identifier |
| TDS | Float | 0-500 | Total Dissolved Solids (mg/L) |
| DO | Float | 0-14 | Dissolved Oxygen (mg/L) |
| Nitrate | Float | 0-500 | Nitrate concentration (mg/L) |
| TH | Float | 0-500 | Total Hardness (mg/L) |
| pH | Float | 0-14 | pH value |
| Chlorides | Float | 0-500 | Chlorides concentration (mg/L) |
| Alkalinity | Float | 0-500 | Alkalinity (mg/L) |
| EC | Float | 0-2000 | Electrical Conductivity (µS/cm) |
| Ammonia | Float | 0-10 | Ammonia concentration (mg/L) |
| Seasons | Integer | 0-3 | Season (0=Winter, 1=Spring, 2=Summer, 3=Fall) |
| Time | Integer | 0-23 | Time of day (24-hour format) |
| AWQI | Float | 0-100 | Aquatic Water Quality Index (target variable) |

## 🌐 Deployment to Streamlit Cloud

### Step-by-Step Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial AWQI deployment"
git push origin main
```

2. **Create Models** (Important!)
   - Run `python train_models.py` locally first
   - This creates the `models/` directory with all trained models
   - Commit the `models/` directory to GitHub

3. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Click "New app"
   - Select repository: `your-username/awqi-prediction-system`
   - Select branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

4. **Access Your App**
   - Your app will be live at: `https://your-app-name.streamlit.app`

## 📖 Usage Guide

### Single Sample Prediction

1. Open the application
2. Navigate to "📊 Prediction Dashboard"
3. Enter water quality parameters
4. Click "🔍 Predict Water Quality"
5. View AWQI score and quality classification

### Batch Analysis

1. Prepare a CSV file with water quality data
2. Select "Batch Analysis" tab
3. Upload your CSV file
4. Click "🔄 Process Batch"
5. Download results as CSV

### Parameter Information

1. Go to "📚 Parameter Guide"
2. Expand each parameter for detailed information
3. View optimal ranges

### Model Performance

1. Navigate to "📈 Model Performance"
2. View regression and classification metrics

## 🔄 Model Training

Run the training script:

```bash
python train_models.py
```

This script:
1. Loads the Aquaculture.csv dataset (120 samples)
2. Prepares features and performs train-test split
3. Trains 6 regression models
4. Trains 6 classification models
5. Saves all models to `models/` directory

**Training Time:** ~2-5 minutes on standard hardware

## 📈 Key Features (SHAP Analysis)

Most important features for AWQI prediction:

1. **Ammonia (NH₃-N)** - Dominant predictor (indicates pollution)
2. **Dissolved Oxygen (DO)** - Critical parameter (aquatic life)
3. **pH** - Moderate importance (chemical equilibria)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push and open PR

## 📝 License

MIT License - See LICENSE file for details

## 📚 References

- Tyagi, S., et al. (2013). Water quality assessment parameters
- Horton, R. K. (1965). Water quality index system
- Liou, S. M., et al. (2003). Water quality index for Taiwan
- Reichstein, M., et al. (2019). Deep learning for Earth systems

## 🐛 Troubleshooting

### Models Not Found Error
```
Error: No such file or directory: models/...
Solution: Run `python train_models.py` first
```

### ModuleNotFoundError
```
Error: No module named 'streamlit'
Solution: Run `pip install -r requirements.txt`
```

### Port Already in Use
```
Error: Address already in use
Solution: streamlit run app.py --server.port=8502
```

## 📧 Support

- Check GitHub Issues
- Review documentation
- Contact development team

---

**Version:** 1.0.0  
**Last Updated:** February 2025  
**Status:** ✅ Production Ready
