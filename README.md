# Credit Card Fraud Detection with Explainable AI (XAI)

A comprehensive fraud detection system that combines supervised machine learning with anomaly detection, enhanced with SHAP (SHapley Additive exPlanations) for model interpretability and explainable AI capabilities.

## 🚀 Features

- **Hybrid Detection Model**: Combines XGBoost/Random Forest with Isolation Forest for enhanced fraud detection
- **Explainable AI**: SHAP integration for both global and local model explanations
- **Interactive Web Interface**: Streamlit-based dashboard for model training, evaluation, and batch scoring
- **Flexible Model Selection**: Automatic fallback from XGBoost to Random Forest if XGBoost is unavailable
- **Batch Scoring**: Score new transactions with detailed explanations
- **Comprehensive Metrics**: ROC-AUC, PR-AUC, confusion matrices, and precision-recall curves

## 📁 Project Structure

```
├── app.py                          # Streamlit web application
├── fraud_shap_xai.py              # Standalone fraud detection script with SHAP
├── data-source/
│   ├── creditcard.csv              # Full credit card dataset
│   └── creditcard_sample.csv       # Sample dataset for testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Installation

1. **Clone the repository** (or download the files)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If XGBoost installation fails, the system will automatically fall back to Random Forest.

3. **Prepare your data**:
   - Ensure your dataset has a `Class` column (1 = fraud, 0 = legitimate)
   - The system works best with the Kaggle Credit Card Fraud Detection dataset
   - Sample data is provided in `data-source/creditcard_sample.csv`

## 🚀 Usage

### Web Interface (Recommended)

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The web interface provides:
- **Data Upload**: Upload your credit card dataset
- **Model Configuration**: Adjust weights, thresholds, and contamination rates
- **Real-time Training**: Train models with your data
- **Interactive Visualizations**: SHAP plots, confusion matrices, PR curves
- **Batch Scoring**: Score new transactions with explanations
- **Downloadable Results**: Export scored data with fraud predictions and reasons

### Command Line Interface

Run the standalone script for batch processing:

```bash
# Set the path to your dataset (optional, defaults to creditcard.csv)
export CREDITCARD_CSV="data-source/creditcard.csv"

# Run the fraud detection script
python fraud_shap_xai.py
```

This will:
- Train the hybrid fraud detection model
- Generate SHAP visualizations (saved as PNG files)
- Create local explanations for top suspicious transactions
- Output performance metrics

## 📊 Model Architecture

### Hybrid Approach
The system combines two complementary approaches:

1. **Supervised Learning** (XGBoost/Random Forest)
   - Learns from labeled fraud examples
   - Optimized for class imbalance with appropriate weights
   - Provides probability scores for known fraud patterns

2. **Anomaly Detection** (Isolation Forest)
   - Detects novel, unseen fraud patterns
   - Identifies outliers in transaction behavior
   - Complements supervised learning for zero-day fraud

### Scoring Formula
```
Hybrid Score = w_sup × Supervised_Probability + w_anom × Scaled_Anomaly_Score
```

Where weights can be adjusted based on your fraud detection strategy.

## 🔍 Explainable AI Features

### Global Explanations
- **Feature Importance**: Which features matter most for fraud detection
- **SHAP Summary Plots**: How each feature contributes across all predictions
- **Beeswarm Plots**: Distribution of feature impacts

### Local Explanations
- **Waterfall Plots**: Step-by-step explanation for individual predictions
- **Force Plots**: Visual breakdown of prediction reasoning
- **Feature Contributions**: Quantified impact of each feature on specific decisions

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **PR-AUC**: Area under the Precision-Recall curve (important for imbalanced data)
- **Confusion Matrix**: True/False positives and negatives at chosen threshold
- **Precision-Recall Curves**: Compare supervised vs. hybrid model performance

## ⚙️ Configuration Options

### Model Parameters
- **Test Size**: Proportion of data used for testing (default: 20%)
- **Random Seed**: For reproducible results
- **Contamination Rate**: Expected fraud rate for anomaly detection (default: 0.5%)

### Hybrid Model Weights
- **Supervised Weight**: Emphasis on learned patterns (default: 0.7)
- **Anomaly Weight**: Emphasis on outlier detection (default: 0.3)
- **Decision Threshold**: Cutoff for fraud classification (default: 0.9)

## 📋 Data Requirements

Your dataset should:
- Contain a `Class` column with binary labels (1 = fraud, 0 = legitimate)
- Have numeric features (non-numeric columns are auto-encoded)
- Be reasonably balanced or the system will handle imbalance automatically
- Follow the structure of the Kaggle Credit Card Fraud Detection dataset

## 🔧 Troubleshooting

### XGBoost Installation Issues
If XGBoost fails to install:
1. The system automatically falls back to Random Forest
2. Remove `xgboost` from `requirements.txt` if needed
3. Performance will be similar with Random Forest

### Memory Issues
For large datasets:
1. Reduce the background sample size in SHAP calculations
2. Use the `max_display` parameter to limit visualization complexity
3. Process data in smaller batches

### Performance Optimization
- Use the contamination parameter to match your expected fraud rate
- Adjust hybrid weights based on your fraud detection priorities
- Tune the decision threshold based on your precision/recall requirements

## 🏃 Run Instructions

```bash
# Create & activate env
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Export a trained model bundle from the dashboard (Download model.pkl)
# Run REST API (in folder containing model.pkl)
uvicorn api:app --host 0.0.0.0 --port 8080

# Example request
curl -X POST http://localhost:8080/score \
-H 'Content-Type: application/json' \
-d '{"blend": 0.7,"threshold": 0.9,"records": [{"V1": 0.1, "V2": -1.2, "V3": 0.05, "Amount": 123.4}]}'
```

---

## 📝 Notes

* **Persistence:** Models/data can be saved & loaded with joblib; bundle stored in `model.pkl`.
* **Threshold Sweep:** Helps pick an operating point balancing fraud catch vs false positives.
* **REST API:** Exposes a `/predict` endpoint for integration with payment systems.
* **SHAP:** Explains supervised part; anomaly score is shown alongside for context.
* **Deployment:** Run Streamlit for analysts, FastAPI for integration.

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and metrics
- **streamlit**: Web interface framework
- **matplotlib**: Plotting and visualization
- **shap**: Model explainability and interpretability
- **xgboost**: Gradient boosting (optional, falls back to Random Forest)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the fraud detection system.

## 📄 License

This project is open source and available under standard licensing terms.

---

**Note**: This system is designed for educational and research purposes. For production fraud detection, additional security measures, data validation, and compliance considerations should be implemented.


This Streamlit app demonstrates a **hybrid fraud detection system** combining:

* **XGBoost** (or RandomForest fallback) for supervised detection
* **Isolation Forest** for anomaly detection
* **Ensemble scoring** (weighted blend)
* **SHAP explanations** for interpretability
* **Threshold sweeps** for optimal decision point selection
* **Model persistence (save/load)** for reproducibility
* **REST API wrapper** for deployment
