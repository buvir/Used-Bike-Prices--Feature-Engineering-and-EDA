# app.py
# Streamlit showcase for "Used Bike Prices — Feature Engineering & EDA"
# Run: streamlit run app.py
# Requires: streamlit, pandas, numpy, scikit-learn, joblib, (optional) matplotlib

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Config + paths
# ---------------------------
st.set_page_config(page_title="Used Bike Prices — EDA & Model", layout="wide")
MODEL_PATHS = [Path("artifacts/full_pipeline.joblib"), Path("artifacts/full_pipeline.pkl")]
CLEAN_PATH  = Path("artifacts/used_bikes_cleaned.csv")
RAW_PATH    = Path("data/bikes.csv")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data(show_spinner=False)
def load_local_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read {path}: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    # Try joblib first, then cloudpickle if available
    for p in MODEL_PATHS:
        if p.exists():
            try:
                return load(p), str(p)
            except Exception as e:
                st.warning(f"joblib load failed for {p}: {e}")
    # Optional: cloudpickle fallback
    try:
        import cloudpickle as cp
        for p in MODEL_PATHS:
            if p.exists():
                with open(p, "rb") as f:
                    return cp.load(f), str(p)
    except Exception:
        pass
    return None, None

def kpi_row(kpis: list[tuple[str, str, str]]):
    cols = st.columns(len(kpis))
    for c, (label, value, delta) in zip(cols, kpis):
        c.metric(label, value, delta)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5  # your preferred RMSE calc
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def quick_owner_map(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return np.where(s.str.contains("first"), "first",
           np.where(s.str.contains("second"), "second",
           np.where(s.str.contains("third"), "third", "fourth_or_more")))

def extract_brand_simple(name: pd.Series) -> pd.Series:
    return name.astype(str).str.strip().str.split().str[0].str.title()

# ---------------------------
# Sidebar: load model + data
# ---------------------------
st.sidebar.title("⚙️ Settings")

model, model_path = load_model()
if model is None:
    st.sidebar.error("Model not found. Export it from your notebook to artifacts/full_pipeline.joblib")
else:
    st.sidebar.success(f"Loaded model: {model_path}")

df = load_local_csv(CLEAN_PATH)
data_source = "artifacts/used_bikes_cleaned.csv"
if df is None:
    df = load_local_csv(RAW_PATH)
    data_source = "data/bikes.csv"
if df is None:
    st.sidebar.info("No local CSV found. Upload a CSV with raw columns (model_name, model_year, mileage, power, kms_driven, owner, location, price).")
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        data_source = f"Uploaded: {up.name}"

if df is None:
    st.stop()

st.sidebar.write(f"📄 Data source: `{data_source}`")
st.sidebar.write(f"Rows: {len(df):,}")

# ---------------------------
# Header + Summary
# ---------------------------
st.title("🏍️ Used Bike Prices — Feature Engineering & EDA")

st.markdown(
"""
**Conclusion (from your analysis)**

- Cleaning unified `mileage` → kmpl and `power` → bhp; `brand`/`cc` extracted; owner/location standardized.
- Price tends to **decrease** with **age** and **kilometers**, and **increase** with **power**/**cc**.
- **Owner category** and **brand** add strong categorical signal.
- Full raw→features pipeline runs cleanly (no NaNs). *Your last debug run reported in-sample RMSE ≈ ₹45k on ~7.8k rows.*
"""
)

st.divider()

# ---------------------------
# Tabs
# ---------------------------
tab_overview, tab_eda, tab_predict, tab_eval = st.tabs(["Overview", "EDA", "Predict", "Evaluate"])

with tab_overview:
    st.subheader("Project at a glance")
    left, right = st.columns([1,1])

    with left:
        st.markdown("**Columns detected:**")
        st.dataframe(pd.DataFrame({"column": df.columns}).head(30), use_container_width=True, hide_index=True)

    with right:
        # Missingness quick view
        miss = df.isna().sum().sort_values(ascending=False)
        miss_df = miss[miss > 0].to_frame("missing").reset_index(names="column")
        st.markdown("**Missing values (top):**")
        if len(miss_df):
            st.dataframe(miss_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected (top-level read).")

with tab_eda:
    st.subheader("Quick EDA")

    # Try to infer cleaned columns; otherwise create light derived ones
    cols_clean = {"price","bike_age","kms","power_bhp","mileage_kmpl","owner_cat","brand","cc"}
    have_clean = cols_clean.issubset(set(df.columns))

    dfe = df.copy()
    if not have_clean:
        # Minimal derivations for display only (does not change your training pipeline)
        if "owner_cat" not in dfe.columns and "owner" in dfe.columns:
            dfe["owner_cat"] = quick_owner_map(dfe["owner"])
        if "brand" not in dfe.columns and "model_name" in dfe.columns:
            dfe["brand"] = extract_brand_simple(dfe["model_name"])
        # age proxy if needed
        if "bike_age" not in dfe.columns and "model_year" in dfe.columns:
            dfe["bike_age"] = pd.Timestamp.now().year - pd.to_numeric(dfe["model_year"], errors="coerce")

    # KPIs if price present
    if "price" in dfe.columns:
        p = dfe["price"].dropna()
        if len(p):
            kpi_row([
                ("Median price", f"₹{p.median():,.0f}", ""),
                ("Mean price", f"₹{p.mean():,.0f}", ""),
                ("95th pct", f"₹{p.quantile(0.95):,.0f}", ""),
            ])

    # Owner vs price (mean)
    if {"owner_cat","price"}.issubset(set(dfe.columns)):
        grp = dfe.groupby("owner_cat")["price"].mean().sort_values(ascending=False).round(0)
        st.markdown("**Mean price by owner category**")
        st.dataframe(grp.to_frame("mean_price").reset_index(), use_container_width=True)

    # Top brands by count
    if "brand" in dfe.columns:
        topb = dfe["brand"].value_counts().head(15)
        st.markdown("**Top 15 brands by listing count**")
        st.dataframe(topb.to_frame(_
