"""
Sling Bag Manufacturing — Interactive Defect Analysis Dashboard
================================================================
Run with:  streamlit run app.py

Target column uses actual dataset values:
  "Good"      = non-defective  (414 bags)
  "Defective" = defective      ( 86 bags)
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
from scipy import stats
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH  = pathlib.Path("womens_sling_bag_complete_defect_dataset.xlsx")
MODEL_PATH = pathlib.Path("sling_bag_defect_model.pkl")
SHEET_NAME = "Sling Bag Quality Data"
TARGET     = "Defect Status"          # values: "Good" | "Defective"
DEFECTIVE  = "Defective"
GOOD       = "Good"

FEATURE_COLS = [
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
    "Machine Setting / Pressure", "Machine Speed",
    "Production Temperature", "Production Humidity",
    "Operator Experience",
]

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Colour palette
C_RED    = "#e53e3e"
C_GREEN  = "#38a169"
C_BLUE   = "#3b82d4"
C_ORANGE = "#f59e0b"
C_PURPLE = "#8b5cf6"
C_TEAL   = "#0d9488"
C_MUTED  = "#57606a"

MODEL_COLORS = {
    "Logistic Regression": C_BLUE,
    "Decision Tree":       C_ORANGE,
    "Random Forest":       C_GREEN,
}

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sling Bag Defect Dashboard",
    page_icon="👜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f8fa; }
[data-testid="stSidebar"] { background: #1f2328; color: #f0f1f2; }
[data-testid="stSidebar"] * { color: #f0f1f2 !important; }
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
h1 { font-size: 1.7rem !important; color: #1f2328 !important; }
h2 { font-size: 1.25rem !important; color: #1f2328 !important; }
h3 { font-size: 1.05rem !important; color: #1f2328 !important; }
hr { border-color: #e5e7eb; margin: 0.75rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached) — NO label remapping; keep original "Good"/"Defective"
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.drop_duplicates()

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Fill NaN in Defect Severity / Defect Type with "None" (non-defective bags)
    for col in ["Defect Type", "Defect Severity"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Operator experience bins (for EDA)
    df["Exp Bin"] = pd.cut(
        df["Operator Experience"],
        bins=4,
        labels=["Junior (0-3 yrs)", "Semi-Skilled (3-6 yrs)",
                "Skilled (6-9 yrs)", "Expert (9-12 yrs)"],
    )
    return df


# ---------------------------------------------------------------------------
# Model training (cached) — binary labels: Defective=1, Good=0
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training ML models (first run only)...")
def train_models(cache_key: int):
    """
    Trains Logistic Regression, Decision Tree, Random Forest.
    Returns (results_dict, best_pipeline, best_model_name).
    All metrics focus on the Defective class.
    """
    df = load_data()
    X = df[FEATURE_COLS].copy()
    y = (df[TARGET] == DEFECTIVE).astype(int)   # 1 = Defective, 0 = Good

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),                                              NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),   CAT_FEATURES),
        ],
        remainder="drop",
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=8, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in classifiers.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        roc  = roc_auc_score(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        cv_f1  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1",     n_jobs=-1)
        cv_rec = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="recall", n_jobs=-1)

        results[name] = {
            "pipeline":   pipe,
            "accuracy":   acc,
            "precision":  prec,
            "recall":     rec,
            "f1":         f1,
            "roc_auc":    roc,
            "cm":         cm,
            "roc_fpr":    fpr.tolist(),
            "roc_tpr":    tpr.tolist(),
            "cv_f1":      cv_f1.tolist(),
            "cv_rec":     cv_rec.tolist(),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_rec_mean":float(cv_rec.mean()),
        }

    # Select best model: weighted score prioritising Defective-class F1 + Recall
    def _score(name):
        r = results[name]
        return r["f1"] * 0.6 + r["recall"] * 0.4

    best_name = max(results, key=_score)
    best_pipe = results[best_name]["pipeline"]

    # Also save for external use
    joblib.dump(best_pipe, MODEL_PATH)

    # Attach feature importances (Random Forest only)
    rf_pipe = results["Random Forest"]["pipeline"]
    ohe_names = (
        rf_pipe.named_steps["preprocessor"]
               .named_transformers_["cat"]
               .get_feature_names_out(CAT_FEATURES)
               .tolist()
    )
    all_feat_names = NUM_FEATURES + ohe_names
    importances    = rf_pipe.named_steps["classifier"].feature_importances_
    results["Random Forest"]["feat_names"]    = all_feat_names
    results["Random Forest"]["feat_imp"]      = importances.tolist()

    return results, best_pipe, best_name


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="-apple-system, 'Segoe UI', system-ui, sans-serif", size=12),
    margin=dict(t=40, b=36, l=16, r=16),
)

def _fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, **_LAYOUT)
    return fig

COLOR_MAP = {DEFECTIVE: C_RED, GOOD: C_GREEN}

def _defect_rate_bar(group_col: str, fdf: pd.DataFrame,
                     color_scale: str = "Reds") -> go.Figure:
    gdf = (
        fdf.groupby(group_col)[TARGET]
        .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 2))
        .reset_index()
    )
    gdf.columns = [group_col, "Defect Rate (%)"]
    gdf = gdf.sort_values("Defect Rate (%)", ascending=False)
    fig = px.bar(
        gdf, x=group_col, y="Defect Rate (%)",
        color="Defect Rate (%)", color_continuous_scale=color_scale,
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return _fig(fig)


# ---------------------------------------------------------------------------
# Load data + train models
# ---------------------------------------------------------------------------
df_full = load_data()
_cache_key = int(DATA_PATH.stat().st_mtime)
model_results, best_pipeline, best_model_name = train_models(_cache_key)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Sling Bag QC")
st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

all_machines = sorted(df_full["Machine"].dropna().unique().tolist())
sel_machines = st.sidebar.multiselect("Machine", all_machines, default=all_machines)

all_shifts = sorted(df_full["Working Shift"].dropna().unique().tolist())
sel_shifts = st.sidebar.multiselect("Working Shift", all_shifts, default=all_shifts)

all_rmq = sorted(df_full["Raw Material Quality"].dropna().unique().tolist())
sel_rmq = st.sidebar.multiselect("Raw Material Quality", all_rmq, default=all_rmq)

sel_status = st.sidebar.multiselect(
    "Defect Status", [DEFECTIVE, GOOD], default=[DEFECTIVE, GOOD]
)

st.sidebar.markdown("---")

df: pd.DataFrame = df_full[
    df_full["Machine"].isin(sel_machines)
    & df_full["Working Shift"].isin(sel_shifts)
    & df_full["Raw Material Quality"].isin(sel_rmq)
    & df_full[TARGET].isin(sel_status)
].copy()

if df.empty:
    st.sidebar.error("No data matches the current filters.")

total       = len(df)
n_def       = int((df[TARGET] == DEFECTIVE).sum())
n_good      = int((df[TARGET] == GOOD).sum())
defect_pct  = n_def / total * 100 if total else 0.0

st.sidebar.markdown(f"**Showing:** {total:,} / {len(df_full):,} bags")

st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "EDA",
        "Statistical Analysis",
        "ML Models",
        "Feature Importance",
        "Predict New Bag",
    ],
)


# ===========================================================================
# SECTION 1 — OVERVIEW
# ===========================================================================
if section == "Overview":
    st.title("Manufacturing Quality Overview")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Bags",        f"{total:,}")
    k2.metric("Defective",         f"{n_def:,}",  delta=f"{defect_pct:.1f}%", delta_color="inverse")
    k3.metric("Good (Non-Defective)", f"{n_good:,}")
    k4.metric("Defect Rate",       f"{defect_pct:.1f}%")
    k5.metric("Unique Defect Types",
              str(df[df[TARGET] == DEFECTIVE]["Defect Type"]
                  .replace("None", pd.NA).dropna().nunique()))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Defective vs Good")
        sc = df[TARGET].value_counts().reset_index()
        sc.columns = ["Status", "Count"]
        fig_d = px.pie(sc, values="Count", names="Status", hole=0.52,
                       color="Status", color_discrete_map=COLOR_MAP)
        fig_d.update_traces(textinfo="percent+value", pull=[0.04, 0])
        fig_d.update_layout(legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(_fig(fig_d, 320), width='stretch', key="ov_donut")

    with c2:
        st.subheader("Defect Types (Defective bags only)")
        dt_df = (
            df[df[TARGET] == DEFECTIVE]["Defect Type"]
            .replace("None", pd.NA).dropna().value_counts().reset_index()
        )
        dt_df.columns = ["Defect Type", "Count"]
        fig_dt = px.bar(dt_df, x="Count", y="Defect Type", orientation="h",
                        color="Count", color_continuous_scale="Reds", text_auto=True)
        fig_dt.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(_fig(fig_dt, 320), width='stretch', key="ov_dtype")

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Defect Severity Distribution")
        sev_order = ["None", "Low", "Medium", "High", "Critical"]
        sev_df = df["Defect Severity"].value_counts().reindex(sev_order).dropna().reset_index()
        sev_df.columns = ["Severity", "Count"]
        fig_sev = px.bar(sev_df, x="Severity", y="Count", color="Severity",
                         color_discrete_map={
                             "None": C_GREEN, "Low": "#fbd38d",
                             "Medium": C_ORANGE, "High": C_RED, "Critical": "#7c0000"},
                         text_auto=True)
        fig_sev.update_layout(showlegend=False)
        st.plotly_chart(_fig(fig_sev, 300), width='stretch', key="ov_sev")

    with c4:
        st.subheader("Defect Rate by Working Shift")
        st.plotly_chart(_defect_rate_bar("Working Shift", df, "Oranges"),
                        width='stretch', key="ov_shift")

    st.divider()

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Defect Rate by Machine")
        st.plotly_chart(_defect_rate_bar("Machine", df, "Reds"),
                        width='stretch', key="ov_machine")

    with c6:
        st.subheader("Overall Defect Rate — Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=defect_pct,
            delta={"reference": 15, "suffix": "%",
                   "increasing": {"color": C_RED}, "decreasing": {"color": C_GREEN}},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 50], "ticksuffix": "%"},
                "bar": {"color": C_RED if defect_pct > 20 else C_ORANGE if defect_pct > 10 else C_GREEN},
                "steps": [
                    {"range": [0, 10],  "color": "#dcfce7"},
                    {"range": [10, 20], "color": "#fef9c3"},
                    {"range": [20, 50], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#1f2328", "width": 3},
                               "thickness": 0.78, "value": 15},
            },
            title={"text": "Defect Rate  (threshold 15%)", "font": {"size": 14}},
        ))
        st.plotly_chart(_fig(fig_g, 300), width='stretch', key="ov_gauge")

    with st.expander("View filtered data table"):
        show_cols = [c for c in [
            "Machine", "Working Shift", "Raw Material Quality",
            "Production Temperature", "Production Humidity", "Machine Speed",
            "Defect Status", "Defect Type", "Defect Severity",
        ] if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True), width='stretch', height=320)
        st.caption(f"{total:,} rows shown based on current filters.")


# ===========================================================================
# SECTION 2 — EDA
# ===========================================================================
elif section == "EDA":
    st.title("Exploratory Data Analysis")
    st.caption("All charts respond to the sidebar filters.")

    tab_mach, tab_mat, tab_ops, tab_dist, tab_corr = st.tabs([
        "Machine Conditions",
        "Raw Material",
        "Shift & Operator",
        "Distributions",
        "Correlation",
    ])

    with tab_mach:
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Defect Rate by Machine")
            st.plotly_chart(_defect_rate_bar("Machine", df, "Reds"),
                            width='stretch', key="eda_m1")
        with r2:
            st.subheader("Machine Pressure vs Defect Status")
            fig = px.box(df, x=TARGET, y="Machine Setting / Pressure",
                         color=TARGET, color_discrete_map=COLOR_MAP, points="outliers")
            st.plotly_chart(_fig(fig), width='stretch', key="eda_m2")

        r3, r4 = st.columns(2)
        with r3:
            st.subheader("Machine Speed vs Defect Status")
            fig = px.violin(df, x=TARGET, y="Machine Speed", color=TARGET,
                            box=True, points=False, color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_m3")
        with r4:
            st.subheader("Production Temperature vs Defect Status")
            fig = px.histogram(df, x="Production Temperature", color=TARGET,
                               barmode="overlay", opacity=0.72, nbins=25,
                               color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_m4")

        st.subheader("Temperature vs Humidity (coloured by Defect Status)")
        fig = px.scatter(df, x="Production Temperature", y="Production Humidity",
                         color=TARGET, opacity=0.65,
                         color_discrete_map=COLOR_MAP,
                         hover_data=["Machine", "Working Shift"])
        st.plotly_chart(_fig(fig, 380), width='stretch', key="eda_m5")

    with tab_mat:
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Defect Rate by Raw Material Quality")
            st.plotly_chart(_defect_rate_bar("Raw Material Quality", df, "YlOrRd"),
                            width='stretch', key="eda_rm1")
        with r2:
            st.subheader("Defect Rate by Raw Material Batch")
            st.plotly_chart(_defect_rate_bar("Raw Material Batch", df, "Oranges"),
                            width='stretch', key="eda_rm2")

        r3, r4 = st.columns(2)
        with r3:
            st.subheader("Production Humidity vs Defect Status")
            fig = px.box(df, x=TARGET, y="Production Humidity",
                         color=TARGET, points="outliers", color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_rm3")
        with r4:
            st.subheader("Stacked: Raw Material Quality x Defect Status")
            stk = df.groupby(["Raw Material Quality", TARGET]).size().reset_index(name="Count")
            fig = px.bar(stk, x="Raw Material Quality", y="Count", color=TARGET,
                         barmode="stack", color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_rm4")

    with tab_ops:
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Defect Rate by Working Shift")
            st.plotly_chart(_defect_rate_bar("Working Shift", df, "Purples"),
                            width='stretch', key="eda_op1")
        with r2:
            if "Exp Bin" in df.columns:
                st.subheader("Defect Rate by Operator Experience (Quartile Bins)")
                st.plotly_chart(_defect_rate_bar("Exp Bin", df, "Blues"),
                                width='stretch', key="eda_op2")

        st.subheader("Heatmap — Defect Rate by Machine x Working Shift")
        hm = (
            df.groupby(["Machine", "Working Shift"])[TARGET]
            .apply(lambda x: round((x == DEFECTIVE).mean() * 100, 1))
            .reset_index()
        )
        hm.columns = ["Machine", "Working Shift", "Defect Rate (%)"]
        hm_pivot = hm.pivot(index="Machine", columns="Working Shift",
                            values="Defect Rate (%)").fillna(0)
        fig_hm = px.imshow(hm_pivot, color_continuous_scale="RdYlGn_r",
                           text_auto=".1f", aspect="auto", zmin=0, zmax=40)
        fig_hm.update_coloraxes(colorbar_title="Defect %")
        st.plotly_chart(_fig(fig_hm, 360), width='stretch', key="eda_op3")

    with tab_dist:
        num_opts = [c for c in FEATURE_COLS
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        chosen = st.selectbox("Select numerical feature to explore", num_opts)
        dc1, dc2 = st.columns(2)
        with dc1:
            st.subheader(f"Distribution of {chosen}")
            fig = px.histogram(df, x=chosen, color=TARGET, barmode="overlay",
                               opacity=0.72, nbins=30, color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_dist1")
        with dc2:
            st.subheader(f"Box Plot of {chosen}")
            fig = px.box(df, x=TARGET, y=chosen, color=TARGET, points="outliers",
                         color_discrete_map=COLOR_MAP)
            st.plotly_chart(_fig(fig), width='stretch', key="eda_dist2")

        st.subheader("Summary Statistics by Defect Status")
        grp = df.groupby(TARGET)[num_opts].agg(["mean", "median", "std"])
        grp.columns = [f"{m} ({s})" for m, s in grp.columns]
        st.dataframe(grp.T.style.format("{:.3f}"), width='stretch')

    with tab_corr:
        num_all = df.select_dtypes(include="number").columns.tolist()
        if len(num_all) >= 2:
            corr = df[num_all].corr()
            fig_c = px.imshow(corr, color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
            st.plotly_chart(_fig(fig_c, 520), width='stretch', key="eda_corr")

        st.subheader("Scatter Matrix — Manufacturing Numerics")
        num_mfg = [c for c in FEATURE_COLS
                   if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        fig_pm = px.scatter_matrix(
            df.sample(min(300, len(df)), random_state=RANDOM_STATE),
            dimensions=num_mfg, color=TARGET,
            color_discrete_map=COLOR_MAP, opacity=0.5,
        )
        fig_pm.update_traces(diagonal_visible=False)
        st.plotly_chart(_fig(fig_pm, 560), width='stretch', key="eda_scatter")


# ===========================================================================
# SECTION 3 — STATISTICAL ANALYSIS
# ===========================================================================
elif section == "Statistical Analysis":
    st.title("Statistical Analysis")
    st.markdown(
        "**Chi-square tests** — association between categorical features and defect status.  \n"
        "**Welch's t-tests** — whether numerical means differ significantly between Good and Defective bags."
    )

    cat_feats = [c for c in FEATURE_COLS
                 if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    num_feats = [c for c in FEATURE_COLS
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Chi-square
    st.subheader("Chi-square Tests — Categorical Features")
    chi_rows = []
    for col in cat_feats:
        ct = pd.crosstab(df[col], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        chi_rows.append({
            "Feature": col, "Chi2": round(chi2, 3),
            "p-value": round(p, 4), "DoF": dof,
            "Significant (p<0.05)": "YES" if p < 0.05 else "No",
        })
    chi_df = pd.DataFrame(chi_rows).set_index("Feature")
    st.dataframe(chi_df, width='stretch')

    fig_chi = px.bar(
        pd.DataFrame(chi_rows).sort_values("Chi2"),
        x="Chi2", y="Feature", orientation="h",
        color="p-value", color_continuous_scale="RdYlGn_r", range_color=[0, 0.2],
        text="Chi2",
    )
    fig_chi.add_vline(x=3.84, line_dash="dash", line_color=C_MUTED,
                      annotation_text="p=0.05 threshold (1 DoF)")
    fig_chi.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(_fig(fig_chi, 300), width='stretch', key="stat_chi")

    st.divider()

    # t-tests
    st.subheader("Welch's t-tests — Numerical Features")
    def_sub  = df[df[TARGET] == DEFECTIVE]
    good_sub = df[df[TARGET] == GOOD]
    t_rows   = []
    for col in num_feats:
        g1, g2 = def_sub[col].dropna(), good_sub[col].dropna()
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        t_rows.append({
            "Feature": col,
            "Mean (Defective)": round(g1.mean(), 3),
            "Mean (Good)": round(g2.mean(), 3),
            "Difference": round(g1.mean() - g2.mean(), 3),
            "t-statistic": round(t_stat, 3),
            "p-value": round(p_val, 4),
            "Significant (p<0.05)": "YES" if p_val < 0.05 else "No",
        })
    t_df = pd.DataFrame(t_rows).set_index("Feature")
    st.dataframe(t_df, width='stretch')

    melted = pd.DataFrame(t_rows).melt(
        id_vars="Feature",
        value_vars=["Mean (Defective)", "Mean (Good)"],
        var_name="Group", value_name="Mean",
    )
    fig_means = px.bar(melted, x="Feature", y="Mean", color="Group", barmode="group",
                       color_discrete_map={"Mean (Defective)": C_RED, "Mean (Good)": C_GREEN},
                       text_auto=".2f")
    fig_means.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(_fig(fig_means, 340), width='stretch', key="stat_means")

    st.subheader("p-value Summary (all tests)")
    all_tests = (
        [{"Test": f"Chi2 — {r['Feature']}", "p-value": r["p-value"], "Type": "Chi-square"} for r in chi_rows]
      + [{"Test": f"t — {r['Feature']}",    "p-value": r["p-value"], "Type": "t-test"}     for r in t_rows]
    )
    pval_df = pd.DataFrame(all_tests).sort_values("p-value")
    fig_pv = px.bar(pval_df, x="Test", y="p-value", color="Type",
                    color_discrete_map={"Chi-square": C_BLUE, "t-test": C_PURPLE},
                    text_auto=".3f")
    fig_pv.add_hline(y=0.05, line_dash="dash", line_color=C_RED, annotation_text="alpha = 0.05")
    fig_pv.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(_fig(fig_pv, 360), width='stretch', key="stat_pval")


# ===========================================================================
# SECTION 4 — ML MODELS
# ===========================================================================
elif section == "ML Models":
    st.title("Machine Learning Models")
    st.markdown(
        "Three classifiers trained on **9 manufacturing features only** — "
        "post-inspection columns excluded to prevent data leakage."
    )

    tab_ov, tab_det, tab_roc, tab_cv, tab_cmp = st.tabs([
        "Overview",
        "Model Detail",
        "ROC Curves",
        "Cross-Validation",
        "Model Comparison",
    ])

    with tab_ov:
        for mname, res in model_results.items():
            badge = "  ★ Best Model" if mname == best_model_name else ""
            with st.expander(f"**{mname}**{badge}", expanded=(mname == best_model_name)):
                l, r = st.columns(2)
                with l:
                    st.metric("Accuracy",  f"{res['accuracy']:.2%}")
                    st.metric("Defective Recall",    f"{res['recall']:.2%}",
                              help="Most important — fraction of defective bags caught")
                    st.metric("Defective F1-Score",  f"{res['f1']:.2%}")
                with r:
                    st.metric("Defective Precision", f"{res['precision']:.2%}")
                    st.metric("ROC-AUC",   f"{res['roc_auc']:.4f}")
                    st.metric("CV F1 (mean)", f"{res['cv_f1_mean']:.4f}")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Train / Test Split", "80% / 20%")
        c2.metric("CV Strategy",        "5-Fold Stratified K-Fold")
        c3.metric("Features Used",      str(len(FEATURE_COLS)))
        st.info(
            "**Selection criterion:** Best model = highest composite score "
            "(F1 x 0.6 + Recall x 0.4) for the Defective class.  \n"
            f"**Selected:** `{best_model_name}`"
        )

    with tab_det:
        sel = st.selectbox("Select model to inspect",
                           list(model_results.keys()),
                           index=list(model_results.keys()).index(best_model_name),
                           key="ml_sel")
        res = model_results[sel]
        cm  = res["cm"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",         f"{res['accuracy']:.2%}")
        m2.metric("Defective Recall",  f"{res['recall']:.2%}")
        m3.metric("Defective F1",      f"{res['f1']:.2%}")
        m4.metric("Defective Precision", f"{res['precision']:.2%}")
        m5.metric("ROC-AUC",           f"{res['roc_auc']:.4f}")

        st.divider()
        col_cm, col_rep = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            labels = ["Good (0)", "Defective (1)"]
            fig_cm = px.imshow(cm, x=labels, y=labels,
                               color_continuous_scale="Blues",
                               text_auto=True, aspect="auto",
                               labels={"x": "Predicted", "y": "Actual"})
            st.plotly_chart(_fig(fig_cm, 340), width='stretch', key="ml_cm")
            TP, FN = int(cm[1, 1]), int(cm[1, 0])
            FP, TN = int(cm[0, 1]), int(cm[0, 0])
            st.caption(
                f"TP={TP} (defects caught) | FN={FN} (defects missed) | "
                f"FP={FP} (false alarms) | TN={TN} (correctly passed)"
            )

        with col_rep:
            st.subheader("Per-Class Metrics")
            y_bin   = (df_full[TARGET] == DEFECTIVE).astype(int)
            X_full  = df_full[FEATURE_COLS]
            y_pred_ = res["pipeline"].predict(X_full)
            rep_dict = classification_report(
                y_bin, y_pred_,
                target_names=["Good", "Defective"], output_dict=True
            )
            rep_df = (
                pd.DataFrame(rep_dict).T
                .loc[["Good", "Defective", "macro avg", "weighted avg"]]
            )
            fig_rep = px.bar(
                rep_df[["precision", "recall", "f1-score"]].reset_index().rename(columns={"index": "Class"}),
                x="Class",
                y=["precision", "recall", "f1-score"],
                barmode="group",
                text_auto=".2f",
                color_discrete_sequence=[C_BLUE, C_ORANGE, C_GREEN],
            )
            fig_rep.update_layout(yaxis_range=[0, 1.12])
            st.plotly_chart(_fig(fig_rep, 340), width='stretch', key="ml_rep")

    with tab_roc:
        st.subheader("ROC Curves — All Models")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="#9ca3af", width=1))
        for mname, res in model_results.items():
            fpr, tpr = res["roc_fpr"], res["roc_tpr"]
            label = f"{mname} (AUC={res['roc_auc']:.4f})"
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=label,
                line=dict(color=MODEL_COLORS.get(mname, C_PURPLE), width=2.5),
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
            legend=dict(x=0.55, y=0.12),
        )
        st.plotly_chart(_fig(fig_roc, 480), width='stretch', key="ml_roc")
        st.info(
            "AUC = 1.0 is a perfect classifier.  "
            "For defect detection, maximise **Recall** (catch all defective bags)."
        )

    with tab_cv:
        st.subheader("5-Fold Cross-Validation — Defective Class F1 & Recall")
        fold_labels = [f"Fold {i+1}" for i in range(5)]
        fig_cv = go.Figure()
        for mname, res in model_results.items():
            fig_cv.add_trace(go.Bar(
                name=f"{mname} — F1",
                x=fold_labels, y=res["cv_f1"],
                text=[f"{v:.3f}" for v in res["cv_f1"]],
                textposition="outside",
                marker_color=MODEL_COLORS.get(mname, C_PURPLE),
                opacity=0.85,
            ))
        fig_cv.update_layout(barmode="group",
                             yaxis=dict(title="F1 Score", range=[0, 1.15]))
        st.plotly_chart(_fig(fig_cv, 400), width='stretch', key="ml_cv")

        cv_summary = pd.DataFrame([{
            "Model": mname,
            "CV F1 Mean":    f"{res['cv_f1_mean']:.4f}",
            "CV F1 Std":     f"{np.std(res['cv_f1']):.4f}",
            "CV Recall Mean":f"{res['cv_rec_mean']:.4f}",
            "Best Model":    "YES" if mname == best_model_name else "",
        } for mname, res in model_results.items()]).set_index("Model")
        st.dataframe(cv_summary, width='stretch')

    with tab_cmp:
        st.subheader("All-Metric Comparison")
        cmp_rows = [{
            "Model":       mname,
            "Accuracy":    res["accuracy"],
            "Precision":   res["precision"],
            "Recall":      res["recall"],
            "F1-Score":    res["f1"],
            "ROC-AUC":     res["roc_auc"],
            "CV F1 Mean":  res["cv_f1_mean"],
        } for mname, res in model_results.items()]
        cmp_df = pd.DataFrame(cmp_rows).set_index("Model")
        palette = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_TEAL]
        fig_cmp = go.Figure()
        for i, col in enumerate(cmp_df.columns):
            fig_cmp.add_trace(go.Bar(
                name=col, x=list(cmp_df.index), y=cmp_df[col].tolist(),
                text=[f"{v:.3f}" for v in cmp_df[col].tolist()],
                textposition="outside",
                marker_color=palette[i % len(palette)],
            ))
        fig_cmp.update_layout(barmode="group",
                              yaxis=dict(title="Score", range=[0, 1.2]))
        st.plotly_chart(_fig(fig_cmp, 460), width='stretch', key="ml_cmp")

        disp = cmp_df.copy().applymap(lambda v: f"{v:.4f}")
        disp["Selected"] = ["YES  (best)" if n == best_model_name else "" for n in cmp_df.index]
        st.dataframe(disp, width='stretch')
        st.info(
            "**Selection criterion:** highest composite score (F1 x 0.6 + Recall x 0.4).  \n"
            f"**Selected model:** `{best_model_name}`"
        )


# ===========================================================================
# SECTION 5 — FEATURE IMPORTANCE
# ===========================================================================
elif section == "Feature Importance":
    st.title("Feature Importance — Random Forest Analysis")
    st.markdown(
        "Random Forest Gini importance shows which manufacturing conditions "
        "are most strongly associated with defect predictions."
    )

    rf_res = model_results.get("Random Forest")
    if rf_res and "feat_names" in rf_res:
        feat_names  = rf_res["feat_names"]
        importances = np.array(rf_res["feat_imp"])
        imp_series  = pd.Series(importances, index=feat_names).sort_values(ascending=False)

        top_n = st.slider("Show top N features", 5, min(35, len(imp_series)), 15)
        imp_df = imp_series.head(top_n).reset_index()
        imp_df.columns = ["Feature", "Importance"]

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
            text_auto=".4f",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(_fig(fig_imp, max(360, top_n * 26)), width='stretch', key="fi_bar")

        st.divider()

        # Grouped importance per original feature
        st.subheader("Importance Grouped by Manufacturing Factor")
        group_imp = {}
        for feat in FEATURE_COLS:
            key = feat.lower().replace(" ", "_").replace("/", "_")
            matching = [n for n in feat_names
                        if key in n.lower().replace(" ", "_").replace("/", "_")]
            group_imp[feat] = float(imp_series[matching].sum()) if matching else 0.0

        g_df = (
            pd.DataFrame(list(group_imp.items()), columns=["Feature", "Importance"])
            .sort_values("Importance", ascending=True)
        )
        fig_grp = px.bar(g_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Oranges",
                         text_auto=".4f")
        fig_grp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_fig(fig_grp, 360), width='stretch', key="fi_grp")

        # Treemap
        st.subheader("Treemap of Feature Importance")
        treemap_df = imp_df.copy()
        treemap_df["Parent"] = treemap_df["Feature"].apply(
            lambda fn: next(
                (f for f in FEATURE_COLS
                 if f.lower().replace(" ", "_").replace("/", "_")
                 in fn.lower().replace(" ", "_").replace("/", "_")),
                "Other",
            )
        )
        fig_tree = px.treemap(treemap_df, path=["Parent", "Feature"],
                              values="Importance", color="Importance",
                              color_continuous_scale="Teal")
        st.plotly_chart(_fig(fig_tree, 400), width='stretch', key="fi_tree")

        st.divider()
        st.subheader("Interpretation Guide")
        st.markdown("""
| Rank | Feature | What it means |
|---|---|---|
| 1 | Production Temperature | Higher/unusual temperatures are most linked to defects |
| 2 | Production Humidity | Environmental humidity significantly affects quality |
| 3 | Machine Speed | Speed outside optimal range increases defect risk |
| 4 | Machine Setting / Pressure | Pressure calibration is a key quality driver |
| 5 | Operator Experience | Less experienced operators correlate with more defects |

*Categorical features (Machine type, Material Batch, etc.) contribute through individual one-hot encoded columns.*
""")
    else:
        st.warning("Random Forest feature importance not available.")


# ===========================================================================
# SECTION 6 — PREDICT NEW BAG
# ===========================================================================
elif section == "Predict New Bag":
    st.title("Live Defect Prediction")
    st.markdown(
        f"Enter manufacturing parameters for a **new sling bag**.  \n"
        f"Prediction is made by **{best_model_name}** — selected as the best model "
        f"based on Defective class Recall and F1-Score."
    )

    def _opts(col: str) -> list:
        return sorted(df_full[col].dropna().unique().tolist())

    with st.form("pred_form", clear_on_submit=False):
        st.subheader("Manufacturing Parameters")
        r1c1, r1c2, r1c3 = st.columns(3)

        with r1c1:
            st.markdown("**Machine Setup**")
            machine       = st.selectbox("Machine",              _opts("Machine"))
            raw_mat_qual  = st.selectbox("Raw Material Quality", _opts("Raw Material Quality"))
            working_shift = st.selectbox("Working Shift",        _opts("Working Shift"))

        with r1c2:
            st.markdown("**Material**")
            raw_mat_batch = st.selectbox("Raw Material Batch",   _opts("Raw Material Batch"))
            op_exp = st.number_input(
                "Operator Experience (years)",
                min_value=float(df_full["Operator Experience"].min()),
                max_value=float(df_full["Operator Experience"].max()),
                value=float(df_full["Operator Experience"].median()),
                step=0.5,
            )

        with r1c3:
            st.markdown("**Process Conditions**")
            pressure = st.number_input(
                "Machine Setting / Pressure",
                min_value=float(df_full["Machine Setting / Pressure"].min()),
                max_value=float(df_full["Machine Setting / Pressure"].max()),
                value=float(df_full["Machine Setting / Pressure"].median()),
                step=0.1,
            )
            speed = st.number_input(
                "Machine Speed (RPM)",
                min_value=float(df_full["Machine Speed"].min()),
                max_value=float(df_full["Machine Speed"].max()),
                value=float(df_full["Machine Speed"].median()),
                step=10.0,
            )
            temperature = st.number_input(
                "Production Temperature (C)",
                min_value=float(df_full["Production Temperature"].min()),
                max_value=float(df_full["Production Temperature"].max()),
                value=float(df_full["Production Temperature"].median()),
                step=0.5,
            )
            humidity = st.number_input(
                "Production Humidity (%)",
                min_value=float(df_full["Production Humidity"].min()),
                max_value=float(df_full["Production Humidity"].max()),
                value=float(df_full["Production Humidity"].median()),
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Predict Defect Status", type="primary", width='stretch'
        )

    # ── Prediction output ─────────────────────────────────────────────────────
    if submitted:
        sample = pd.DataFrame([{
            "Machine":                    machine,
            "Machine Setting / Pressure": pressure,
            "Machine Speed":              speed,
            "Production Temperature":     temperature,
            "Production Humidity":        humidity,
            "Raw Material Batch":         raw_mat_batch,
            "Raw Material Quality":       raw_mat_qual,
            "Working Shift":              working_shift,
            "Operator Experience":        op_exp,
        }])

        # Model predicts 0 (Good) or 1 (Defective)
        pred_label = int(best_pipeline.predict(sample)[0])
        pred_probs = best_pipeline.predict_proba(sample)[0]
        defective_prob     = float(pred_probs[1])   # P(Defective)
        good_prob          = float(pred_probs[0])   # P(Good)
        predicted_status   = DEFECTIVE if pred_label == 1 else GOOD

        st.divider()

        if predicted_status == DEFECTIVE:
            st.error(f"## DEFECTIVE BAG DETECTED  (Confidence: {defective_prob:.1%})")
        else:
            st.success(f"## GOOD — BAG PASSES QUALITY CHECK  (Confidence: {good_prob:.1%})")

        # Gauge + probability metrics
        g_col, p_col1, p_col2 = st.columns([2, 1, 1])

        with g_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(defective_prob * 100, 1),
                number={"suffix": "%", "font": {"size": 46}},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": C_RED if defective_prob >= 0.5
                            else C_ORANGE if defective_prob >= 0.3 else C_GREEN},
                    "steps": [
                        {"range": [0,  30],  "color": "#dcfce7"},
                        {"range": [30, 60],  "color": "#fef9c3"},
                        {"range": [60, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#1f2328", "width": 3},
                                  "thickness": 0.78, "value": 50},
                },
                title={"text": "Defective Probability", "font": {"size": 15}},
            ))
            st.plotly_chart(_fig(fig_gauge, 320), width='stretch', key="pred_gauge")

        with p_col1:
            st.metric("Defective Probability", f"{defective_prob:.1%}")
            st.progress(defective_prob)
            st.metric("Good Probability",      f"{good_prob:.1%}")
            st.progress(good_prob)

        with p_col2:
            st.metric("Prediction",  predicted_status)
            st.metric("Model Used",  best_model_name)
            st.metric("Confidence",  f"{max(defective_prob, good_prob):.1%}")

        # Risk factors
        st.subheader("Risk Factor Assessment")
        risk_items = []

        if temperature > df_full["Production Temperature"].quantile(0.75):
            q75 = df_full["Production Temperature"].quantile(0.75)
            risk_items.append(
                f"**High temperature** ({temperature:.1f} C) — "
                f"above 75th percentile ({q75:.1f} C)"
            )
        if raw_mat_qual in ["Poor", "Average"]:
            risk_items.append(
                f"**Raw material quality is '{raw_mat_qual}'** — "
                "Poor and Average quality batches show higher defect rates"
            )
        if op_exp < df_full["Operator Experience"].quantile(0.25):
            q25 = df_full["Operator Experience"].quantile(0.25)
            risk_items.append(
                f"**Low operator experience** ({op_exp:.1f} yrs) — "
                f"below 25th percentile ({q25:.1f} yrs)"
            )
        if humidity > df_full["Production Humidity"].quantile(0.75):
            q75h = df_full["Production Humidity"].quantile(0.75)
            risk_items.append(
                f"**High humidity** ({humidity:.1f}%) — "
                f"above 75th percentile ({q75h:.1f}%)"
            )
        if speed > df_full["Machine Speed"].quantile(0.85):
            q85 = df_full["Machine Speed"].quantile(0.85)
            risk_items.append(
                f"**High machine speed** ({int(speed)} RPM) — "
                f"above 85th percentile ({int(q85)} RPM)"
            )

        if risk_items:
            st.warning("The following conditions may increase defect risk:")
            for item in risk_items:
                st.markdown(f"- {item}")
        else:
            st.success("No major risk factors detected for this production configuration.")

        with st.expander("View submitted parameters"):
            param_tbl = pd.DataFrame([{
                "Machine": machine,
                "Machine Setting / Pressure": pressure,
                "Machine Speed": speed,
                "Production Temperature": temperature,
                "Production Humidity": humidity,
                "Raw Material Batch": raw_mat_batch,
                "Raw Material Quality": raw_mat_qual,
                "Working Shift": working_shift,
                "Operator Experience": op_exp,
            }]).T.rename(columns={0: "Value"})
            st.dataframe(param_tbl, width='stretch')
