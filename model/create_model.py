#!/usr/bin/env python3
"""
Quick script to create a model.pkl file for the REST API
Uses the sample credit card data to train a basic model
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Try XGBoost, else fall back to RandomForest
USE_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    USE_XGB = False
    print("XGBoost not available, using RandomForest")

def create_model():
    # Load sample data
    try:
        df = pd.read_csv("data-source/creditcard_sample.csv")
    except FileNotFoundError:
        print("Sample data not found. Please ensure data-source/creditcard_sample.csv exists")
        return False
    
    print(f"Loaded {len(df)} samples")
    
    # Prepare data
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].astype(int)
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train supervised model
    if USE_XGB:
        sup_model = XGBClassifier(
            n_estimators=100,  # Smaller for quick training
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            tree_method="hist",
            scale_pos_weight=(y_train.value_counts()[0] / max(1, y_train.value_counts()[1])),
            eval_metric="logloss",
        )
    else:
        sup_model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    
    print("Training supervised model...")
    sup_model.fit(X_train, y_train)
    
    # Train anomaly detector
    print("Training anomaly detector...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.005,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    
    # Get anomaly score range for scaling
    anom_raw = -iso.score_samples(X_train)
    iso_min = anom_raw.min()
    iso_max = anom_raw.max()
    
    # Create model bundle
    bundle = {
        "sup_model": sup_model,
        "iso_model": iso,
        "feature_names": feature_names,
        "iso_min": iso_min,
        "iso_max": iso_max,
    }
    
    # Save bundle
    joblib.dump(bundle, "model.pkl")
    print("✅ Model bundle saved as model.pkl")
    print(f"Features: {len(feature_names)}")
    print(f"Model type: {'XGBoost' if USE_XGB else 'RandomForest'}")
    
    return True

if __name__ == "__main__":
    success = create_model()
    if success:
        print("\n🚀 Ready to run REST API:")
        print("uvicorn api:app --host 0.0.0.0 --port 8080")
    else:
        print("❌ Failed to create model. Please check the data file.")