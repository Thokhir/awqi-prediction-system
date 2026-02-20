"""
AWQI Model Training and Saving Script
This script trains all ML models and saves them for Streamlit deployment
Run this once to generate the trained models
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AWQI MODEL TRAINING AND SAVING")
print("=" * 80)

# Create models directory
os.makedirs("models", exist_ok=True)

# Load the dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv('Aquaculture.csv')
print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Feature Engineering
print("\n[2/5] Engineering features...")
df['Time_sin'] = np.sin(2 * np.pi * df['Time'] / 12)
df['Time_cos'] = np.cos(2 * np.pi * df['Time'] / 12)

# Prepare features and target for regression
X = df.drop(['AWQI', 'Code', 'Time', 'Seasons'], axis=1)
y_reg = df['AWQI']

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"✓ Training set: {X_train_reg_scaled.shape}")
print(f"✓ Test set: {X_test_reg_scaled.shape}")

# Train Regression Models
print("\n[3/5] Training regression models...")
reg_models = {
    'Linear Regression': {'model': LinearRegression(), 'param_grid': {}},
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'param_grid': {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVR': {
        'model': SVR(),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPRegressor(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

best_reg_models = {}
for name, info in reg_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    if param_grid:
        gs = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        gs.fit(X_train_reg_scaled, y_train_reg)
        best_model = gs.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train_reg_scaled, y_train_reg)
    
    best_reg_models[name] = best_model
    y_pred = best_model.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    print(f"  ✓ {name:20s} - R²: {r2:.4f}, MSE: {mse:.4f}")

# Classification Models
print("\n[4/5] Training classification models...")

# Create classification target
bins = [0, 25, 50, float('inf')]
labels = ['Excellent', 'Good', 'Moderate']
df['AWQI_class'] = pd.cut(df['AWQI'], bins=bins, labels=labels, right=False)

X_clf = X
y_clf = df['AWQI_class']
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf_encoded, test_size=0.3, random_state=42
)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

clf_models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'param_grid': {'C': [0.1, 1, 10]}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'max_depth': [None, 10, 20]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {'n_estimators': [50, 100], 'max_depth': [None, 10]}
    },
    'SVC': {
        'model': SVC(random_state=42, class_weight='balanced'),
        'param_grid': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    },
    'ANN': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'param_grid': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }
}

best_clf_models = {}
for name, info in clf_models.items():
    model = info['model']
    param_grid = info['param_grid']
    
    gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    gs.fit(X_train_clf_scaled, y_train_clf)
    best_model = gs.best_estimator_
    
    best_clf_models[name] = best_model
    y_pred = best_model.predict(X_test_clf_scaled)
    acc = accuracy_score(y_test_clf, y_pred)
    print(f"  ✓ {name:20s} - Accuracy: {acc:.4f}")

# Save all models
print("\n[5/5] Saving models...")

# Save regression models
for name, model in best_reg_models.items():
    filename = f"models/{name.replace(' ', '_').lower()}_reg.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ Saved: {filename}")

# Save classification models
for name, model in best_clf_models.items():
    filename = f"models/{name.replace(' ', '_').lower()}_clf.pkl"
    joblib.dump(model, filename)
    print(f"  ✓ Saved: {filename}")

# Save scalers
joblib.dump(scaler_reg, "models/scaler_regression.pkl")
joblib.dump(scaler_clf, "models/scaler_classification.pkl")
print(f"  ✓ Saved: models/scaler_regression.pkl")
print(f"  ✓ Saved: models/scaler_classification.pkl")

# Save label encoder
joblib.dump(le, "models/label_encoder.pkl")
print(f"  ✓ Saved: models/label_encoder.pkl")

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")
print(f"  ✓ Saved: models/feature_names.pkl")

# Save class names
joblib.dump(le.classes_, "models/class_names.pkl")
print(f"  ✓ Saved: models/class_names.pkl")

print("\n" + "=" * 80)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 80)
print(f"\nModels saved in: {os.path.abspath('models')}/")
print(f"Ready for Streamlit deployment!")
