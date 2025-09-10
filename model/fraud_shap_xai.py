import os, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# 1) Load data
# -------------------------------------------------------------------
CSV_PATH = os.environ.get("CREDITCARD_CSV", "creditcard.csv")
df = pd.read_csv(CSV_PATH)

assert "Class" in df.columns, "Dataset must contain a 'Class' column (1 = fraud)."

X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Keep feature names for plots
feature_names = X.columns.tolist()

# -------------------------------------------------------------------
# 2) Train/valid split with strong class imbalance handling
# -------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------------------------------------------
# 3) Choose model (XGBoost if available; otherwise RandomForest)
# -------------------------------------------------------------------
use_xgb = True
try:
    from xgboost import XGBClassifier
except Exception:
    use_xgb = False

if use_xgb:
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric="logloss",
    )
else:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

model.fit(X_train, y_train)

# -------------------------------------------------------------------
# 4) Evaluate (ROC-AUC, PR-AUC, confusion matrix @ chosen threshold)
# -------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

proba_test = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, proba_test)
pr  = average_precision_score(y_test, proba_test)

# Choose threshold to favor recall at low FPR (typical for fraud)
thr = 0.90  # you can tune with PR curve; here we use a high threshold on proba
y_pred = (proba_test >= thr).astype(int)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {roc:.4f}")
print(f"PR-AUC : {pr:.4f}")
print("Confusion matrix @ threshold", thr, ":\n", cm)
print(classification_report(y_test, y_pred, digits=4))

# -------------------------------------------------------------------
# 5) SHAP explanations
# -------------------------------------------------------------------
import shap

# Use a small background sample for faster SHAP on big data
bg_size = min(1000, len(X_train))
background = shap.sample(X_train, bg_size, random_state=42)

if use_xgb:
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)  # shape: [n_rows, n_features]
else:
    # TreeExplainer also supports RF efficiently
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)

# -------------------------------------------------------------------
# 6) Global explanations (feature importance & beeswarm)
# -------------------------------------------------------------------
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=160)
plt.close()

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=160)
plt.close()

print("Saved global SHAP plots: 'shap_beeswarm.png', 'shap_importance_bar.png'")

# -------------------------------------------------------------------
# 7) Local explanation (single prediction)
# -------------------------------------------------------------------
# Pick a suspicious transaction (highest predicted probability)
idx_top = np.argsort(-proba_test)[:1][0]
x_row = X_test.iloc[[idx_top]]

# Explain that single prediction
sv_row = explainer.shap_values(x_row)

# Waterfall plot
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, sv_row[0], feature_names=feature_names, max_display=14, show=False
)
plt.tight_layout()
plt.savefig("shap_waterfall_top_case.png", dpi=160)
plt.close()

# Force plot (saved as HTML)
force = shap.force_plot(
    explainer.expected_value, sv_row[0], x_row, matplotlib=False, feature_names=feature_names
)
shap.save_html("shap_force_top_case.html", force)

# -------------------------------------------------------------------
# 8) Dependence plot for the top global feature
# -------------------------------------------------------------------
# Identify top feature by mean |SHAP|
mean_abs = np.abs(shap_values).mean(axis=0)
top_feat_idx = int(np.argsort(-mean_abs)[0])
top_feat = feature_names[top_feat_idx]

plt.figure(figsize=(8,6))
shap.dependence_plot(top_feat_idx, shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(f"shap_dependence_{top_feat}.png", dpi=160)
plt.close()

print(f"Saved local SHAP plots: 'shap_waterfall_top_case.png', 'shap_force_top_case.html', and dependence for '{top_feat}'.")
print("Done.")
