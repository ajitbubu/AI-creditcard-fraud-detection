# Streamlit Dashboard: Hybrid Fraud Detection (XGBoost + Anomaly + SHAP/XAI)

import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Try XGBoost, else fall back to RandomForest
USE_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    USE_XGB = False

import shap

st.set_page_config(page_title="Hybrid Fraud Detection XAI", layout="wide")
st.title("💳 Hybrid Credit Card Fraud Detection (XGBoost + Anomaly + SHAP)")

st.markdown(
    """
This demo blends **supervised learning (XGBoost/RandomForest)** with **unsupervised anomaly detection (Isolation Forest)**
into a **hybrid score**. It adds **model save/load**, a **threshold sweep tuner**, and download of **scored outputs**.

Upload your dataset (must include a `Class` column: 1=fraud, 0=legit), or load a pre-trained model bundle.
    """
)

# -----------------------------
# Sidebar: configuration
# -----------------------------
with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload training CSV (with `Class`)", type=["csv"])

    st.caption("— OR — load a pre-trained model (created via this app)")
    mdl_upload = st.file_uploader("Load model.pkl", type=["pkl"], key="mdl")

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    blend = st.slider("Blend weight (Supervised vs Anomaly)", 0.0, 1.0, 0.7, 0.05)
    threshold = st.slider("Decision threshold (on HYBRID score)", 0.0, 1.0, 0.90, 0.01)
    max_display = st.slider("Max SHAP features", 5, 30, 12)

    st.subheader("Anomaly settings")
    contamination = st.slider("Expected fraud rate (contamination)", 0.001, 0.05, 0.005, 0.001)

    st.subheader("Batch Scoring (optional)")
    scoring_csv = st.file_uploader("Upload CSV to score (no `Class` needed)", type=["csv"], key="score")

# -----------------------------
# Helpers
# -----------------------------

def numeric_encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.Categorical(out[c]).codes
    return out

# -----------------------------
# Load data / model bundle
# -----------------------------
if mdl_upload is not None:
    bundle = joblib.load(mdl_upload)
    sup_model = bundle["sup_model"]
    iso = bundle["iso_model"]
    feature_names = bundle["feature_names"]
    iso_min = bundle.get("iso_min", 0.0)
    iso_max = bundle.get("iso_max", 1.0)
    st.success("Loaded pre-trained model bundle.")

    # For SHAP background and evaluation, a small CSV is recommended
    if uploaded is None:
        st.info("Optional: upload a CSV to compute SHAP background and run threshold tuning.")
else:
    if uploaded is None:
        st.info("⬆️ Upload a dataset to train a new model, or load a model.pkl bundle.")
        st.stop()

    # Load dataset
    df = pd.read_csv(uploaded)
    if "Class" not in df.columns:
        st.error("Dataset must have a 'Class' column (1=fraud, 0=legit).")
        st.stop()

    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)
    feature_names = X.columns.tolist()

    # Auto-encode non-numeric
    X = numeric_encode(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=42
    )

    # Train supervised & anomaly models
    with st.spinner("Training supervised model…"):
        if USE_XGB:
            sup_model = XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, tree_method="hist", eval_metric="logloss",
                scale_pos_weight=(y_train.value_counts()[0] / max(1, y_train.value_counts()[1]))
            )
        else:
            sup_model = RandomForestClassifier(
                n_estimators=600, class_weight="balanced_subsample",
                n_jobs=-1, random_state=42
            )
        sup_model.fit(X_train, y_train)

    with st.spinner("Training anomaly detector…"):
        iso = IsolationForest(
            n_estimators=300, contamination=contamination,
            random_state=42, n_jobs=-1
        )
        iso.fit(X_train)

    # Fit scaling stats for anomaly score
    an_raw_train = -iso.score_samples(X_train)
    iso_min, iso_max = float(an_raw_train.min()), float(an_raw_train.max())

    # Offer bundle download
    bundle = {
        "sup_model": sup_model,
        "iso_model": iso,
        "feature_names": feature_names,
        "iso_min": iso_min,
        "iso_max": iso_max,
        "note": "Trained on uploaded dataset"
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    st.download_button(
        "⬇️ Download model bundle (model.pkl)",
        data=buf.getvalue(), file_name="model.pkl", mime="application/octet-stream"
    )

# -----------------------------
# If we have a dataset split, compute metrics and SHAP
# -----------------------------
if 'uploaded' in locals() and uploaded is not None:
    # Scores
    proba_test = sup_model.predict_proba(X_test)[:, 1]
    anom_raw = -iso.score_samples(X_test)
    anom_scaled = (anom_raw - iso_min) / (iso_max - iso_min + 1e-9)
    hybrid_score = blend * proba_test + (1 - blend) * anom_scaled

    # Metrics
    roc_sup = roc_auc_score(y_test, proba_test)
    pr_sup = average_precision_score(y_test, proba_test)
    roc_hyb = roc_auc_score(y_test, hybrid_score)
    pr_hyb = average_precision_score(y_test, hybrid_score)

    y_pred_hyb = (hybrid_score >= threshold).astype(int)
    cm_hyb = confusion_matrix(y_test, y_pred_hyb)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Supervised (tree)")
        st.metric("ROC-AUC", f"{roc_sup:.4f}")
        st.metric("PR-AUC", f"{pr_sup:.4f}")
    with col2:
        st.subheader("Hybrid (supervised + anomaly)")
        st.metric("ROC-AUC", f"{roc_hyb:.4f}")
        st.metric("PR-AUC", f"{pr_hyb:.4f}")
    with col3:
        st.subheader(f"Confusion @ thr={threshold:.2f}")
        cm_df = pd.DataFrame(cm_hyb, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]) 
        st.dataframe(cm_df)

    # Precision-Recall curves
    prec_sup, rec_sup, _ = precision_recall_curve(y_test, proba_test)
    prec_h, rec_h, _ = precision_recall_curve(y_test, hybrid_score)
    fig_pr, ax_pr = plt.subplots(figsize=(6,4))
    ax_pr.plot(rec_sup, prec_sup, label="Supervised")
    ax_pr.plot(rec_h, prec_h, label="Hybrid")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.set_title("PR Curves"); ax_pr.legend()
    st.pyplot(fig_pr)

    # Threshold sweep tuner (hybrid score)
    st.subheader("🔧 Threshold Sweep Tuner (Hybrid)")
    grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in grid:
        pred = (hybrid_score >= t).astype(int)
        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        rows.append((float(t), float(p), float(r), float(f1)))
    sweep = pd.DataFrame(rows, columns=["threshold","precision","recall","f1"])
    best_f1 = sweep.iloc[sweep["f1"].idxmax()]
    st.dataframe(sweep.style.format({"threshold":"{:.2f}","precision":"{:.3f}","recall":"{:.3f}","f1":"{:.3f}"}), use_container_width=True)
    st.caption(f"Best F1 @ thr={best_f1.threshold:.2f} → P={best_f1.precision:.3f}, R={best_f1.recall:.3f}")

    # SHAP global
    st.subheader("SHAP Global Explanations (Supervised part)")
    bg = X_train.sample(min(1000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(sup_model, data=bg, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)

    fig_bee = plt.figure(figsize=(9,6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=max_display)
    plt.tight_layout(); st.pyplot(fig_bee, clear_figure=True)

    fig_bar = plt.figure(figsize=(9,6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=max_display)
    plt.tight_layout(); st.pyplot(fig_bar, clear_figure=True)

    # Local SHAP
    st.subheader("Local SHAP Explanation")
    idx_default = int(np.argsort(-hybrid_score)[0])
    row_idx = st.number_input("Row index (test set)", 0, len(X_test)-1, idx_default)
    x_row = X_test.iloc[[row_idx]]
    sv_row = explainer.shap_values(x_row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Supervised proba", f"{proba_test[row_idx]:.3f}")
    c2.metric("Anomaly (scaled)", f"{anom_scaled[row_idx]:.3f}")
    c3.metric("Hybrid score", f"{hybrid_score[row_idx]:.3f}")

    fig_water = plt.figure(figsize=(8,6))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, sv_row[0], feature_names=feature_names, max_display=max_display, show=False
    )
    plt.tight_layout(); st.pyplot(fig_water, clear_figure=True)

    shap_tbl = (
        pd.DataFrame({"feature": feature_names, "value": x_row.iloc[0].values, "shap": sv_row[0]})
        .assign(abs_shap=lambda d: d["shap"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(max_display)
    )
    st.dataframe(shap_tbl, use_container_width=True)

# -----------------------------
# Batch scoring with reasons
# -----------------------------
st.subheader("Score a New CSV (optional)")
if scoring_csv is not None:
    new_df = pd.read_csv(scoring_csv)
    newX = new_df.copy()
    if "Class" in newX.columns:
        newX = newX.drop(columns=["Class"])  # ignore if present
    newX = numeric_encode(newX)

    proba_new = sup_model.predict_proba(newX)[:, 1]
    anom_raw_new = -iso.score_samples(newX)
    anom_scaled_new = (anom_raw_new - iso_min) / (iso_max - iso_min + 1e-9)
    hybrid_new = blend * proba_new + (1 - blend) * anom_scaled_new

    # SHAP values for reasons (supervised component)
    try:
        sv_new = shap.TreeExplainer(sup_model).shap_values(newX)
        k = min(3, max_display)
        reasons = []
        for i in range(len(newX)):
            row_vals = sv_new[i]
            top_idx = np.argsort(-np.abs(row_vals))[:k]
            parts = [f"{feature_names[j]}: {row_vals[j]:+.3f}" for j in top_idx]
            reasons.append("; ".join(parts))
    except Exception:
        reasons = [""] * len(newX)

    out = new_df.copy()
    out["supervised_proba"] = proba_new
    out["anomaly_scaled"] = anom_scaled_new
    out["hybrid_score"] = hybrid_new
    out["fraud_pred_hybrid"] = (hybrid_new >= threshold).astype(int)
    out["top_reasons_supervised"] = reasons

    st.download_button(
        "⬇️ Download scored CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="scored_hybrid_with_shap.csv",
        mime="text/csv",
    )
    st.dataframe(out.head(20), use_container_width=True)

import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, f1_score
)

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except Exception:
    USE_XGB = False

import shap
from fastapi import FastAPI

st.set_page_config(page_title="Hybrid Fraud Detection XAI", layout="wide")
st.title("💳 Hybrid Credit Card Fraud Detection (XGBoost + Anomaly + SHAP)")

st.markdown("""
This demo combines **supervised learning (XGBoost/RandomForest)** with **unsupervised anomaly detection (Isolation Forest)**.
It then explains predictions with **SHAP**. Models can be saved/loaded, thresholds tuned, and a REST API served.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    mdl_upload = st.file_uploader("Load model.pkl", type=["pkl"], key="mdl")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.9, 0.01)
    blend = st.slider("Blend weight (Supervised vs Anomaly)", 0.0, 1.0, 0.7, 0.05)
    max_display = st.slider("Max SHAP features", 5, 30, 12)
    contamination = st.slider("Anomaly contamination", 0.001, 0.05, 0.005, 0.001)
    threshold_sweep = st.checkbox("Run threshold sweep")

# Load or train model
if mdl_upload is not None:
    bundle = joblib.load(mdl_upload)
    sup_model = bundle["sup_model"]
    iso = bundle["iso_model"]
    feature_names = bundle["feature_names"]
    X_train, X_test, y_train, y_test = bundle["data"]
    st.success("Loaded pre-trained model bundle")
else:
    if uploaded is None:
        st.info("⬆️ Upload creditcard.csv or synthetic data to begin.")
        st.stop()
    df = pd.read_csv(uploaded)
    if "Class" not in df.columns:
        st.error("Dataset must have a 'Class' column.")
        st.stop()
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)
    if USE_XGB:
        sup_model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),
            eval_metric="logloss",
            random_state=42,
            tree_method="hist"
        )
    else:
        sup_model = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42)
    sup_model.fit(X_train, y_train)
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_train)
    # Save model bundle
    bundle = {"sup_model": sup_model, "iso_model": iso, "feature_names": X.columns.tolist(), "data": (X_train, X_test, y_train, y_test)}
    joblib.dump(bundle, "model.pkl")
    st.success("Model bundle saved as model.pkl")

# Scores
proba_test = sup_model.predict_proba(X_test)[:,1]
anom_raw = -iso.score_samples(X_test)
anom_scaled = (anom_raw - anom_raw.min())/(anom_raw.max()-anom_raw.min())
hybrid_score = blend * proba_test + (1-blend) * anom_scaled
y_pred = (hybrid_score >= threshold).astype(int)

roc = roc_auc_score(y_test, hybrid_score)
pr = average_precision_score(y_test, hybrid_score)
cm = confusion_matrix(y_test, y_pred)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("ROC-AUC", f"{roc:.4f}")
with c2:
    st.metric("PR-AUC", f"{pr:.4f}")
with c3:
    st.subheader(f"Confusion @ thr={threshold:.2f}")
    st.dataframe(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# Threshold sweep
if threshold_sweep:
    st.subheader("Threshold Sweep")
    grid = np.linspace(0.01,0.99,50)
    rows = []
    for t in grid:
        pred = (hybrid_score >= t).astype(int)
        rows.append((t, precision_score(y_test,pred,zero_division=0), recall_score(y_test,pred,zero_division=0), f1_score(y_test,pred,zero_division=0)))
    df_sweep = pd.DataFrame(rows, columns=["thr","precision","recall","f1"])
    st.line_chart(df_sweep.set_index("thr"))

# SHAP explanations
st.subheader("SHAP Global Explanations (Supervised model)")
explainer = shap.TreeExplainer(sup_model, X_train.sample(min(1000,len(X_train)), random_state=42))
shap_values = explainer.shap_values(X_test)

fig1 = plt.figure(figsize=(9,6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False, max_display=max_display)
st.pyplot(fig1, clear_figure=True)

fig2 = plt.figure(figsize=(9,6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False, max_display=max_display)
st.pyplot(fig2, clear_figure=True)

# Local explanation
st.subheader("Local SHAP Explanation")
idx = st.number_input("Row index (test set)", 0, len(X_test)-1, 0)
x_row = X_test.iloc[[idx]]
sv_row = explainer.shap_values(x_row)

fig3 = plt.figure(figsize=(8,6))
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, sv_row[0], feature_names=X.columns, max_display=max_display, show=False)
st.pyplot(fig3, clear_figure=True)

# REST API wrapper (FastAPI)
api = FastAPI()

@api.post("/predict")
def predict(transaction: dict):
    df_new = pd.DataFrame([transaction])
    proba = sup_model.predict_proba(df_new)[:,1][0]
    anom = -iso.score_samples(df_new)[0]
    anom_scaled = (anom - anom_raw.min())/(anom_raw.max()-anom_raw.min())
    hybrid = blend*proba + (1-blend)*anom_scaled
    return {"supervised_proba": float(proba), "anomaly_score": float(anom_scaled), "hybrid_score": float(hybrid)}
