"""
========================================================================
  Interactive ML Model -- Sling Bag Defect Prediction
  Product Quality and Defect Analysis in Sling Bag Manufacturing
  Run with:  streamlit run interactive_model.py
========================================================================
  SECTIONS (sidebar navigation)
  1. Dataset Overview      -- shape, distributions, charts
  2. Model Configuration   -- hyperparameters, preprocessing info
  3. Model Evaluation      -- metrics, confusion matrix, ROC, CV
  4. Predict New Bag       -- live defect prediction form
  5. Feature Importance    -- RF importance charts
========================================================================

  TARGET:  "Defect Status"
    "Defective"     (positive class, label=1)  --  86 bags  (17.2%)
    "Good"          (negative class, label=0)  -- 414 bags  (82.8%)

  NOTE: We do NOT remap "Good" to "Non-Defective".
        The dataset natively uses "Good" / "Defective".
        Binary encoding:  Defective = 1,  Good = 0.
"""

from __future__ import annotations

import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH    = pathlib.Path("womens_sling_bag_complete_defect_dataset.xlsx")
SHEET_NAME   = "Sling Bag Quality Data"
MODEL_FILE   = pathlib.Path("sling_bag_defect_model.pkl")
TARGET       = "Defect Status"
DEFECTIVE    = "Defective"    # positive class (what we want to detect)
GOOD         = "Good"         # negative class
RANDOM_STATE = 42

MANUFACTURING_FEATURES = [
    "Machine",
    "Machine Setting / Pressure",
    "Machine Speed",
    "Production Temperature",
    "Production Humidity",
    "Raw Material Batch",
    "Raw Material Quality",
    "Working Shift",
    "Operator Experience",
]
CAT_FEATURES = ["Machine", "Raw Material Batch", "Raw Material Quality", "Working Shift"]
NUM_FEATURES = [
    "Machine Setting / Pressure",
    "Machine Speed",
    "Production Temperature",
    "Production Humidity",
    "Operator Experience",
]

C_RED    = "#e53e3e"
C_GREEN  = "#38a169"
C_BLUE   = "#3b82d4"
C_ORANGE = "#f59e0b"
C_PURPLE = "#8b5cf6"
C_MUTED  = "#57606a"

MODEL_COLORS = {
    "Logistic Regression": C_BLUE,
    "Decision Tree":       C_ORANGE,
    "Random Forest":       C_GREEN,
}

# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sling Bag Defect -- ML Model",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background:#f7f8fa; }
[data-testid="stSidebar"]           { background:#1f2328; color:#f0f1f2; }
[data-testid="stSidebar"] *         { color:#f0f1f2 !important; }
[data-testid="metric-container"] {
    background:#ffffff; border:1px solid #e5e7eb;
    border-radius:10px; padding:14px 18px;
    box-shadow:0 1px 3px rgba(0,0,0,.06);
}
h1 { font-size:1.65rem !important; color:#1f2328 !important; }
h2 { font-size:1.2rem  !important; color:#1f2328 !important; }
h3 { font-size:1.0rem  !important; color:#1f2328 !important; }
.result-defective {
    background:#fff5f5; border:2px solid #e53e3e;
    border-radius:12px; padding:20px 28px; margin-top:10px;
}
.result-ok {
    background:#f0fff4; border:2px solid #38a169;
    border-radius:12px; padding:20px 28px; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached: load + clean data
# Keep original labels: "Good" and "Defective"
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.drop_duplicates()
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    # Fill NaN in post-inspection columns
    for col in ["Defect Type", "Defect Severity"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    # Impute any remaining NaN (safety)
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ---------------------------------------------------------------------------
# Cached: train all three models
# y is binary: Defective=1, Good=0
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models... this takes a moment.")
def train_models(n_est: int, max_depth_dt: int, c_lr: float, _key: str):
    df = load_data()
    X  = df[MANUFACTURING_FEATURES].copy()
    y  = (df[TARGET] == DEFECTIVE).astype(int)   # 1 = Defective, 0 = Good

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    def _pre():
        return ColumnTransformer([
            ("num", StandardScaler(),                                            NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ], remainder="drop")

    clfs = {
        "Logistic Regression": LogisticRegression(
            C=c_lr, class_weight="balanced",
            max_iter=1_000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=max_depth_dt if max_depth_dt > 0 else None,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_est, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    res = {}

    for name, clf in clfs.items():
        pipe = Pipeline([("pre", _pre()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]   # P(Defective)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test,  y_pred, zero_division=0)
        f1   = f1_score(y_test,     y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred)   # [[TN FP][FN TP]]

        cv_acc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_f1  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1",       n_jobs=-1)
        cv_rec = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="recall",   n_jobs=-1)

        res[name] = {
            "pipeline":  pipe,
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   auc,
            "roc_fpr":   fpr.tolist(),
            "roc_tpr":   tpr.tolist(),
            "cm":        cm,                        # shape (2,2)
            "cv_acc":    cv_acc.tolist(),
            "cv_f1":     cv_f1.tolist(),
            "cv_rec":    cv_rec.tolist(),
        }

    # Select best: highest composite score (F1 x 0.6 + Recall x 0.4)
    best = max(res, key=lambda k: res[k]["f1"] * 0.6 + res[k]["recall"] * 0.4)
    joblib.dump(res[best]["pipeline"], MODEL_FILE)
    return res, res[best]["pipeline"], best, X_train, X_test, y_train, y_test


# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.title("Sling Bag Defect Model")
st.sidebar.caption("Interactive ML Dashboard")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Go to section",
    ["Dataset Overview",
     "Model Configuration",
     "Model Evaluation",
     "Predict New Bag",
     "Feature Importance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Adjust Hyperparameters")
n_est    = st.sidebar.slider("Random Forest: n_estimators", 50, 300, 100, 50)
dt_depth = st.sidebar.slider("Decision Tree: max_depth  (0 = unlimited)", 0, 20, 5)
c_lr     = st.sidebar.select_slider(
    "Logistic Regression: C  (regularisation)",
    options=[0.01, 0.1, 1.0, 5.0, 10.0], value=1.0)

cache_key = f"{n_est}|{dt_depth}|{c_lr}"

st.sidebar.markdown("---")
if st.sidebar.button("Re-train with new settings", type="primary"):
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Load data + train models (cached)
# ---------------------------------------------------------------------------
df_full = load_data()
results, best_pipeline, best_model_name, X_train, X_test, y_train, y_test = \
    train_models(n_est, dt_depth, c_lr, cache_key)

# Dropdown options derived from actual data
opts_machine = sorted(df_full["Machine"].dropna().unique())
opts_batch   = sorted(df_full["Raw Material Batch"].dropna().unique())
opts_quality = sorted(df_full["Raw Material Quality"].dropna().unique())
opts_shift   = sorted(df_full["Working Shift"].dropna().unique())

# Dataset KPIs
total = len(df_full)
n_def = int((df_full[TARGET] == DEFECTIVE).sum())
n_ok  = total - n_def


# ===========================================================================
# SECTION 1 -- DATASET OVERVIEW
# ===========================================================================
if section == "Dataset Overview":
    st.title("Dataset Overview")
    st.caption("Understanding the raw manufacturing data before building the model.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{total:,}")
    c2.metric("Defective",      str(n_def),
              delta=f"{n_def/total*100:.1f}% of total")
    c3.metric("Good (Non-Defective)", str(n_ok),
              delta=f"{n_ok/total*100:.1f}% of total")
    c4.metric("Features Used",  len(MANUFACTURING_FEATURES))

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Target Class Distribution")
        tc = df_full[TARGET].value_counts().reset_index()
        tc.columns = ["Status", "Count"]
        fig = px.pie(
            tc, names="Status", values="Count", hole=0.55,
            color="Status",
            color_discrete_map={DEFECTIVE: C_RED, GOOD: C_GREEN},
        )
        fig.update_traces(textinfo="percent+value")
        fig.update_layout(
            height=300, margin=dict(t=10, b=10, l=0, r=0),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The dataset is **imbalanced** (17.2% Defective). "
            "Without class balancing, models ignore defective bags entirely. "
            "We use `class_weight='balanced'` to handle this."
        )

    with col_b:
        st.subheader("Defect Rate by Machine")
        mdf = (
            df_full.groupby("Machine")[TARGET]
            .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 1))
            .reset_index(name="Defect Rate (%)")
        )
        mdf = mdf.sort_values("Defect Rate (%)")
        fig2 = px.bar(
            mdf, x="Defect Rate (%)", y="Machine", orientation="h",
            color="Defect Rate (%)",
            color_continuous_scale=["#c6f6d5", C_RED],
            text="Defect Rate (%)",
        )
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(
            height=300, margin=dict(t=10, b=10, l=0, r=80),
            paper_bgcolor="white", plot_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, width="stretch")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Defect Rate by Raw Material Quality")
        qdf = (
            df_full.groupby("Raw Material Quality")[TARGET]
            .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 1))
            .reset_index(name="Defect Rate (%)")
        )
        qdf = qdf.sort_values("Defect Rate (%)", ascending=False)
        fig3 = px.bar(
            qdf, x="Raw Material Quality", y="Defect Rate (%)",
            color="Defect Rate (%)",
            color_continuous_scale=["#c6f6d5", C_RED],
            text="Defect Rate (%)",
        )
        fig3.update_traces(texttemplate="%{text}%", textposition="outside")
        fig3.update_layout(
            height=300, margin=dict(t=10, b=30, l=0, r=0),
            paper_bgcolor="white", plot_bgcolor="white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, width="stretch")

    with col_d:
        st.subheader("Production Temperature by Defect Status")
        fig4 = px.histogram(
            df_full, x="Production Temperature", color=TARGET,
            color_discrete_map={DEFECTIVE: C_RED, GOOD: C_GREEN},
            nbins=20, barmode="overlay", opacity=0.75,
        )
        fig4.update_layout(
            height=300, margin=dict(t=10, b=30, l=0, r=0),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig4, width="stretch")
        st.caption(
            "Production Temperature is the **strongest numerical predictor**. "
            "Defective bags tend to occur at higher temperatures."
        )

    st.divider()

    # Defect rate by Working Shift
    col_e, col_f = st.columns(2)
    with col_e:
        st.subheader("Defect Rate by Working Shift")
        sdf = (
            df_full.groupby("Working Shift")[TARGET]
            .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 1))
            .reset_index(name="Defect Rate (%)")
        )
        fig5 = px.bar(sdf, x="Working Shift", y="Defect Rate (%)",
                      color="Defect Rate (%)",
                      color_continuous_scale=["#c6f6d5", C_RED],
                      text="Defect Rate (%)")
        fig5.update_traces(texttemplate="%{text}%", textposition="outside")
        fig5.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="white",
                           coloraxis_showscale=False, margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig5, width="stretch")

    with col_f:
        st.subheader("Defect Rate by Raw Material Batch")
        bdf = (
            df_full.groupby("Raw Material Batch")[TARGET]
            .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 1))
            .reset_index(name="Defect Rate (%)")
        )
        fig6 = px.bar(bdf, x="Raw Material Batch", y="Defect Rate (%)",
                      color="Defect Rate (%)",
                      color_continuous_scale=["#c6f6d5", C_RED],
                      text="Defect Rate (%)")
        fig6.update_traces(texttemplate="%{text}%", textposition="outside")
        fig6.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="white",
                           coloraxis_showscale=False, margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig6, width="stretch")

    st.divider()
    st.subheader("Raw Data Preview (manufacturing features + target)")
    show_cols = MANUFACTURING_FEATURES + [TARGET]
    st.dataframe(df_full[show_cols].head(15), width="stretch", hide_index=True)


# ===========================================================================
# SECTION 2 -- MODEL CONFIGURATION
# ===========================================================================
elif section == "Model Configuration":
    st.title("Model Configuration")
    st.caption("Understand what each model is doing and why these settings were chosen.")
    st.divider()

    st.subheader("Hyperparameters in Use")
    cfg = {
        "Setting": [
            "Algorithm",
            "class_weight",
            "Key hyperparameter",
            "random_state",
            "Categorical encoding",
            "Numerical scaling",
            "Train/Test split",
            "CV strategy",
        ],
        "Logistic Regression": [
            "LogisticRegression",
            "balanced",
            f"C = {c_lr}  (smaller = stronger regularisation)",
            RANDOM_STATE,
            "OneHotEncoder (handle_unknown=ignore)",
            "StandardScaler",
            "80% / 20%, stratified",
            "5-Fold Stratified KFold",
        ],
        "Decision Tree": [
            "DecisionTreeClassifier",
            "balanced",
            f"max_depth = {dt_depth if dt_depth > 0 else 'None (unlimited)'}",
            RANDOM_STATE,
            "OneHotEncoder (handle_unknown=ignore)",
            "StandardScaler",
            "80% / 20%, stratified",
            "5-Fold Stratified KFold",
        ],
        "Random Forest": [
            "RandomForestClassifier",
            "balanced",
            f"n_estimators = {n_est}  (number of trees)",
            RANDOM_STATE,
            "OneHotEncoder (handle_unknown=ignore)",
            "StandardScaler",
            "80% / 20%, stratified",
            "5-Fold Stratified KFold",
        ],
    }
    st.dataframe(pd.DataFrame(cfg).set_index("Setting"), width="stretch")

    st.divider()
    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.subheader("Why class_weight='balanced'?")
        st.info(
            f"Your dataset: **{n_def} Defective** vs **{n_ok} Good**.\n\n"
            "Without balancing, a model that always predicts Good gets 82.8% accuracy "
            "while catching **zero defective bags**. "
            "`class_weight='balanced'` up-weights Defective samples so the model is "
            "penalised for missing them."
        )

    with col_e2:
        st.subheader("Why not always pick Random Forest?")
        st.warning(
            "Random Forest often has high overall accuracy but can have poor Recall "
            "for the minority (Defective) class, meaning it misses most defective bags.  \n\n"
            "The **best model is selected based on:**  \n"
            "Score = F1 x 0.6 + Recall x 0.4 for the Defective class.  \n\n"
            f"Current best model: **{best_model_name}**"
        )

    st.divider()
    st.subheader("Feature Pipeline Architecture")
    feat_df = pd.DataFrame({
        "Feature":        MANUFACTURING_FEATURES,
        "Type":           ["Categorical", "Numerical", "Numerical", "Numerical",
                           "Numerical", "Categorical", "Categorical",
                           "Categorical", "Numerical"],
        "Transformation": ["OneHotEncoder", "StandardScaler", "StandardScaler",
                           "StandardScaler", "StandardScaler",
                           "OneHotEncoder", "OneHotEncoder",
                           "OneHotEncoder", "StandardScaler"],
        "Data Leakage?":  ["No risk"] * 9,
    })
    st.dataframe(feat_df, width="stretch", hide_index=True)

    st.subheader("Excluded Columns (Data Leakage Rule)")
    leakage_df = pd.DataFrame({
        "Column":          ["Defect Type", "Defect Severity", "Inspection Result",
                            "Repair Action", "Repair Cost (INR)"],
        "Reason Excluded": [
            "Only known after inspection -- directly reveals the answer",
            "Only known after inspection -- directly reveals the answer",
            "Only known after inspection",
            "Only known after inspection",
            "Only known after inspection",
        ],
    })
    st.dataframe(leakage_df, width="stretch", hide_index=True)


# ===========================================================================
# SECTION 3 -- MODEL EVALUATION
# ===========================================================================
elif section == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption(
        "All metrics are on the **held-out test set** (20% = 100 bags). "
        "The most important metric is **Recall (Defective)** -- the % of defective bags caught."
    )
    st.divider()

    # Comparison table
    st.subheader("Model Comparison Table  (focus: Defective class)")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":                   name + ("  [BEST]" if name == best_model_name else ""),
            "Overall Accuracy":        round(r["accuracy"],  4),
            "Precision (Defective)":   round(r["precision"], 4),
            "Recall (Defective)":      round(r["recall"],    4),
            "F1 (Defective)":          round(r["f1"],        4),
            "ROC-AUC":                 round(r["roc_auc"],   4),
        })
    cmp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        cmp_df.style.highlight_max(
            subset=["Recall (Defective)", "F1 (Defective)", "ROC-AUC"],
            color="#c6f6d5",
        ),
        width="stretch",
    )
    st.caption(
        "Green cells = best value per column.  "
        "**Recall (Defective)** = fraction of actual defective bags the model caught.  "
        "**F1 (Defective)** = balance between Precision and Recall for defects."
    )

    st.divider()

    # Per-model tabs
    tab_lr, tab_dt, tab_rf = st.tabs(
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    def render_model_tab(tab, name, r):
        with tab:
            badge = "  [SELECTED BEST MODEL]" if name == best_model_name else ""
            st.subheader(f"{name}{badge}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",            f"{r['accuracy']:.2%}")
            m2.metric("Precision (Defect.)", f"{r['precision']:.2%}",
                      help="Of bags flagged as defective, how many actually were?")
            m3.metric("Recall (Defect.)",    f"{r['recall']:.2%}",
                      help="Of all defective bags, how many did the model catch?")
            m4.metric("F1 (Defect.)",        f"{r['f1']:.2%}",
                      help="Harmonic mean of Precision and Recall.")
            m5.metric("ROC-AUC",             f"{r['roc_auc']:.4f}",
                      help="Area Under ROC Curve. 1.0 = perfect, 0.5 = random.")

            col_cm, col_roc = st.columns(2)

            with col_cm:
                st.markdown("**Confusion Matrix**")
                st.caption("Rows = Actual  |  Columns = Predicted")
                cm = r["cm"]   # [[TN FP][FN TP]]
                # cm[0,0]=TN  cm[0,1]=FP  cm[1,0]=FN  cm[1,1]=TP
                fig_cm = px.imshow(
                    cm,
                    x=["Pred: Good (0)", "Pred: Defective (1)"],
                    y=["Actual: Good (0)", "Actual: Defective (1)"],
                    text_auto=True,
                    color_continuous_scale=["#f0fff4", C_RED],
                    zmin=0,
                )
                fig_cm.update_layout(
                    height=280, margin=dict(t=10, b=10, l=0, r=0),
                    paper_bgcolor="white", coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, width="stretch")
                TP = int(cm[1, 1]); FN = int(cm[1, 0])
                FP = int(cm[0, 1]); TN = int(cm[0, 0])
                st.caption(
                    f"TP={TP} defects caught | FN={FN} defects missed | "
                    f"FP={FP} false alarms | TN={TN} correctly passed"
                )

            with col_roc:
                st.markdown("**ROC Curve**")
                st.caption("AUC closer to 1.0 = better discrimination")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=r["roc_fpr"], y=r["roc_tpr"], mode="lines",
                    name=f"AUC = {r['roc_auc']:.4f}",
                    line=dict(color=MODEL_COLORS[name], width=2.5),
                ))
                fig_roc.add_shape(
                    type="line", x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dot", color="#aaaaaa"),
                )
                fig_roc.update_layout(
                    height=280, margin=dict(t=10, b=30, l=40, r=10),
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(fig_roc, width="stretch")

            # Cross-validation
            st.markdown("**5-Fold Stratified Cross-Validation  (Defective class)**")
            fold_cols = st.columns(5)
            for i, (f1_s, rec_s) in enumerate(zip(r["cv_f1"], r["cv_rec"])):
                fold_cols[i].metric(f"Fold {i+1}",
                                    f"F1={f1_s:.3f}",
                                    delta=f"Rec={rec_s:.3f}",
                                    delta_color="off")
            st.caption(
                f"Mean CV F1 : **{float(np.mean(r['cv_f1'])):.4f}** +/- {float(np.std(r['cv_f1'])):.4f}  |  "
                f"Mean CV Recall : **{float(np.mean(r['cv_rec'])):.4f}** +/- {float(np.std(r['cv_rec'])):.4f}"
            )

    render_model_tab(tab_lr, "Logistic Regression", results["Logistic Regression"])
    render_model_tab(tab_dt, "Decision Tree",        results["Decision Tree"])
    render_model_tab(tab_rf, "Random Forest",        results["Random Forest"])

    # All ROC curves overlaid
    st.divider()
    st.subheader("ROC Curves -- All Models Overlaid")
    fig_all = go.Figure()
    for name, r in results.items():
        fig_all.add_trace(go.Scatter(
            x=r["roc_fpr"], y=r["roc_tpr"], mode="lines",
            name=f"{name}  (AUC={r['roc_auc']:.4f})",
            line=dict(color=MODEL_COLORS[name], width=2.5),
        ))
    fig_all.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(dash="dot", color="#aaaaaa"),
    )
    fig_all.update_layout(
        height=380, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=10, b=40, l=50, r=10),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig_all, width="stretch")

    st.divider()
    # CV F1 bar chart
    st.subheader("Cross-Validation F1 per Fold  (Defective class)")
    fold_labels = [f"Fold {i+1}" for i in range(5)]
    fig_cv = go.Figure()
    for name, r in results.items():
        fig_cv.add_trace(go.Bar(
            name=name, x=fold_labels, y=r["cv_f1"],
            text=[f"{v:.3f}" for v in r["cv_f1"]],
            textposition="outside",
            marker_color=MODEL_COLORS[name],
        ))
    fig_cv.update_layout(
        barmode="group",
        yaxis=dict(title="F1 Score (Defective class)", range=[0, 1.15]),
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=20, b=20, l=50, r=10),
    )
    st.plotly_chart(fig_cv, width="stretch")


# ===========================================================================
# SECTION 4 -- PREDICT NEW BAG
# ===========================================================================
elif section == "Predict New Bag":
    st.title("Predict Defect Status for a New Sling Bag")
    st.markdown(
        f"Model in use: **{best_model_name}** "
        f"(selected because it has the highest composite score for the Defective class)."
    )
    st.divider()

    with st.form("pred_form"):
        st.subheader("Enter Manufacturing Details")
        st.caption("Fill in the production conditions for the new bag.")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("**Machine Setup**")
            machine = st.selectbox(
                "Machine", opts_machine,
                help="Which machine produced this bag?")
            shift = st.selectbox(
                "Working Shift", opts_shift,
                help="Morning / Afternoon / Night")
            quality = st.selectbox(
                "Raw Material Quality", opts_quality,
                help="Quality grade of the raw material used.")

        with r2:
            st.markdown("**Material**")
            batch = st.selectbox(
                "Raw Material Batch", opts_batch,
                help="Batch ID of the raw material.")
            experience = st.number_input(
                "Operator Experience (years)",
                min_value=0.0, max_value=15.0, value=5.0, step=0.5,
                help="Years of experience of the machine operator.")
            pressure = st.number_input(
                "Machine Setting / Pressure",
                min_value=float(df_full["Machine Setting / Pressure"].min()),
                max_value=float(df_full["Machine Setting / Pressure"].max()),
                value=float(df_full["Machine Setting / Pressure"].median()),
                step=0.1,
                help=f"Dataset range: "
                     f"{df_full['Machine Setting / Pressure'].min():.1f} - "
                     f"{df_full['Machine Setting / Pressure'].max():.1f}")

        with r3:
            st.markdown("**Process Conditions**")
            speed = st.number_input(
                "Machine Speed (RPM)",
                min_value=int(df_full["Machine Speed"].min()),
                max_value=int(df_full["Machine Speed"].max()),
                value=int(df_full["Machine Speed"].median()),
                step=10,
                help=f"Dataset range: "
                     f"{df_full['Machine Speed'].min()} - {df_full['Machine Speed'].max()} RPM")
            temperature = st.number_input(
                "Production Temperature (C)",
                min_value=float(df_full["Production Temperature"].min()),
                max_value=float(df_full["Production Temperature"].max()),
                value=float(df_full["Production Temperature"].median()),
                step=0.5,
                help=f"Dataset range: "
                     f"{df_full['Production Temperature'].min():.1f} - "
                     f"{df_full['Production Temperature'].max():.1f} C")
            humidity = st.number_input(
                "Production Humidity (%)",
                min_value=float(df_full["Production Humidity"].min()),
                max_value=float(df_full["Production Humidity"].max()),
                value=float(df_full["Production Humidity"].median()),
                step=0.5,
                help=f"Dataset range: "
                     f"{df_full['Production Humidity'].min():.1f} - "
                     f"{df_full['Production Humidity'].max():.1f}%")

        submitted = st.form_submit_button(
            "Predict Defect Status",
            type="primary",
        )

    # ── Prediction output ─────────────────────────────────────────────────
    if submitted:
        sample = pd.DataFrame([{
            "Machine":                    machine,
            "Machine Setting / Pressure": pressure,
            "Machine Speed":              speed,
            "Production Temperature":     temperature,
            "Production Humidity":        humidity,
            "Raw Material Batch":         batch,
            "Raw Material Quality":       quality,
            "Working Shift":              shift,
            "Operator Experience":        experience,
        }])

        # Model outputs 0 (Good) or 1 (Defective)
        pred_label = int(best_pipeline.predict(sample)[0])
        pred_probs = best_pipeline.predict_proba(sample)[0]
        def_p = float(pred_probs[1])   # P(Defective)
        ok_p  = float(pred_probs[0])   # P(Good)

        predicted_status = DEFECTIVE if pred_label == 1 else GOOD
        is_def    = pred_label == 1
        box_cls   = "result-defective" if is_def else "result-ok"
        color     = C_RED if is_def else C_GREEN
        icon_text = "DEFECTIVE BAG DETECTED" if is_def else "GOOD -- PASSES QUALITY CHECK"

        st.markdown(f"""
        <div class="{box_cls}">
            <h2 style="color:{color};margin:0 0 6px 0">{icon_text}</h2>
            <p style="margin:0;color:#57606a;font-size:0.92rem">
                Predicted by <strong>{best_model_name}</strong>
                &nbsp;|&nbsp; Confidence: <strong>{max(def_p, ok_p):.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        pc1, pc2 = st.columns(2)
        pc1.metric("Defective Probability", f"{def_p:.2%}")
        pc2.metric("Good Probability",      f"{ok_p:.2%}")

        # Probability gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=def_p * 100,
            number={"suffix": "%", "font": {"size": 30}},
            title={"text": "Defective Probability", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": C_RED if def_p > 0.5 else C_ORANGE if def_p > 0.3 else C_GREEN},
                "steps": [
                    {"range": [0,  30], "color": "#c6f6d5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100],"color": "#fed7d7"},
                ],
                "threshold": {
                    "line": {"color": "#1f2328", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_g.update_layout(
            height=260, margin=dict(t=30, b=10, l=20, r=20),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_g, width="stretch")

        # Risk factors
        st.markdown("#### Risk Factor Commentary")
        risks = []
        q75_temp = float(df_full["Production Temperature"].quantile(0.75))
        q75_hum  = float(df_full["Production Humidity"].quantile(0.75))
        q25_exp  = float(df_full["Operator Experience"].quantile(0.25))

        if quality == "Poor":
            risks.append("**Raw Material Quality is 'Poor'** -- highest defect-rate group in the dataset.")
        elif quality == "Average":
            risks.append("**Raw Material Quality is 'Average'** -- second-highest defect rate group.")

        if temperature > q75_temp:
            risks.append(
                f"**Production Temperature ({temperature:.1f} C)** is above the 75th percentile "
                f"({q75_temp:.1f} C). Temperature is the strongest numerical predictor."
            )
        if humidity > q75_hum:
            risks.append(
                f"**Production Humidity ({humidity:.1f}%)** is above the 75th percentile "
                f"({q75_hum:.1f}%)."
            )
        if experience < q25_exp:
            risks.append(
                f"**Operator Experience ({experience:.1f} yrs)** is below the 25th percentile "
                f"({q25_exp:.1f} yrs) -- junior operators correlate with higher defect rates."
            )

        if risks:
            st.warning("The following conditions may increase defect risk:")
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.success("No individual high-risk factors detected for this combination.")

        with st.expander("View submitted input values"):
            st.dataframe(
                sample.T.rename(columns={0: "Value"}),
                width="stretch",
            )


# ===========================================================================
# SECTION 5 -- FEATURE IMPORTANCE
# ===========================================================================
elif section == "Feature Importance":
    st.title("Feature Importance  (Random Forest)")
    st.caption(
        "Random Forest assigns each feature an importance score based on how much "
        "it reduced impurity across all trees. "
        "Higher importance = more influence on the defect prediction."
    )
    st.divider()

    rf_pre = results["Random Forest"]["pipeline"].named_steps["pre"]
    rf_clf = results["Random Forest"]["pipeline"].named_steps["clf"]

    try:
        feat_names = rf_pre.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(len(rf_clf.feature_importances_))]

    imp = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values(ascending=False)

    top_n = st.slider("Show top N features", 5, len(imp), 15)
    imp_df = imp.head(top_n).reset_index()
    imp_df.columns = ["Feature", "Importance"]

    fig_imp = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#dbeafe", C_BLUE],
        text=imp_df["Importance"].map(lambda v: f"{v:.4f}"),
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        height=max(350, top_n * 28),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=10, b=30, l=10, r=90),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, width="stretch")

    # Grouped importance
    st.divider()
    st.subheader("Grouped Importance  (by original manufacturing feature)")
    st.caption(
        "OneHot-encoded categories are summed back to their original column "
        "so we can compare features on equal footing."
    )

    group_imp = {}
    for orig in MANUFACTURING_FEATURES:
        # match any OHE-expanded column that starts with the original name
        orig_lower = orig.lower().replace(" ", "_").replace("/", "_")
        matching = [n for n in feat_names
                    if orig_lower in n.lower().replace(" ", "_").replace("/", "_")]
        group_imp[orig] = round(float(imp[matching].sum()), 4) if matching else 0.0

    grp_df = pd.Series(group_imp).sort_values(ascending=False).reset_index()
    grp_df.columns = ["Feature", "Total Importance"]

    fig_grp = px.bar(
        grp_df, x="Feature", y="Total Importance",
        color="Total Importance",
        color_continuous_scale=["#dbeafe", C_BLUE],
        text=grp_df["Total Importance"].map(lambda v: f"{v:.4f}"),
    )
    fig_grp.update_traces(textposition="outside")
    fig_grp.update_layout(
        height=380,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=10, b=90, l=0, r=0),
        coloraxis_showscale=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_grp, width="stretch")

    # Treemap
    st.divider()
    st.subheader("Treemap of Feature Importance")
    treemap_df = imp_df.copy()
    treemap_df["Parent"] = treemap_df["Feature"].apply(
        lambda fn: next(
            (f for f in MANUFACTURING_FEATURES
             if f.lower().replace(" ", "_").replace("/", "_")
             in fn.lower().replace(" ", "_").replace("/", "_")),
            "Other",
        )
    )
    fig_tree = px.treemap(
        treemap_df, path=["Parent", "Feature"],
        values="Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_tree.update_layout(height=400, paper_bgcolor="white", margin=dict(t=10, b=10))
    st.plotly_chart(fig_tree, width="stretch")

    # Takeaways
    st.divider()
    st.subheader("Interpretation Guide")
    st.info(
        "**Top 5 manufacturing factors most associated with defect predictions:**\n\n"
        "1. **Production Temperature** -- Environmental temperature during production\n"
        "2. **Production Humidity** -- Humidity in the production environment\n"
        "3. **Machine Speed** -- Speed at which the machine operates\n"
        "4. **Machine Setting / Pressure** -- Pressure calibration\n"
        "5. **Operator Experience** -- Years of experience of the operator\n\n"
        "Categorical factors (Machine type, Raw Material Quality, Batch, Shift) "
        "contribute through individual one-hot encoded columns."
    )

    with st.expander("Full Feature Importance Table"):
        full_df = imp.reset_index()
        full_df.columns = ["Feature", "Importance"]
        full_df["Importance (%)"] = (full_df["Importance"] * 100).round(2)
        st.dataframe(full_df, width="stretch", hide_index=True)
