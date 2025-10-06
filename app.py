# app.py (compat-safe)
# Streamlit showcase for "Used Bike Prices — Feature Engineering & EDA"
# Run: streamlit run app.py

import re
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
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
# Utilities (no modern type-hint syntax to avoid Py<3.10 errors)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_local_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning("Could not read {}: {}".format(path, e))
    return None

@st.cache_resource(show_spinner=False)
def load_model():
    # Try joblib first
    for p in MODEL_PATHS:
        if p.exists():
            try:
                return load(p), str(p)
            except Exception as e:
                st.warning("joblib load failed for {}: {}".format(p, e))
    # Optional: cloudpickle fallback if available
    try:
        import cloudpickle as cp
        for p in MODEL_PATHS:
            if p.exists():
                with open(p, "rb") as f:
                    return cp.load(f), str(p)
    except Exception:
        pass
    return None, None

def kpi_row(kpis):
    # kpis: list of (label, value, delta_str)
    cols = st.columns(len(kpis))
    for c, tpl in zip(cols, kpis):
        label, value, delta = tpl
        c.metric(label, value, delta)

def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def quick_owner_map(series):
    s = series.astype(str).str.lower()
    return np.where(s.str.contains("first"), "First Owner",
           np.where(s.str.contains("second"), "Second Owner",
           np.where(s.str.contains("third"), "Third Owner", "Fourth_or_more")))

def extract_brand_simple(series):
    return series.astype(str).str.strip().str.split().str[0].str.title()

# ---------------------------
# Sidebar: load model + data
# ---------------------------
st.sidebar.title("⚙️ Settings")

model, model_path = load_model()
if model is None:
    st.sidebar.error("Model not found. Export it to artifacts/full_pipeline.joblib")
else:
    st.sidebar.success("Loaded model: {}".format(model_path))

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
        data_source = "Uploaded: {}".format(up.name)

if df is None:
    st.stop()

st.sidebar.write("📄 Data source: `{}`".format(data_source))
st.sidebar.write("Rows: {:,}".format(len(df)))

# ---------------------------
# Header + Summary
# ---------------------------
st.title("🏍️ Used Bike Prices — Feature Engineering & EDA")

st.markdown(
"""
**Conclusion (from your analysis)**

- Listings rise through the 2000s, peak around **2017**, and taper off thereafter.
- **Bajaj** and **Royal Enfield** lead by volume, followed by **Hero** and **Yamaha**.
- The majority of bikes are **First Owner**.
- CC distribution is concentrated in lower/mid segments (e.g., **150–350 cc**); **≥1000 cc** is comparatively rare.
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
        miss_df = miss[miss > 0].to_frame("missing").reset_index()
        miss_df.columns = ["column", "missing"]
        st.markdown("**Missing values (top):**")
        if len(miss_df):
            st.dataframe(miss_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected (top-level read).")

with tab_eda:
    st.subheader("Quick EDA")

    # Try to infer cleaned columns; otherwise create light derived ones (display-only)
    dfe = df.copy()
    if "owner_cat" not in dfe.columns and "owner" in dfe.columns:
        dfe["owner_cat"] = quick_owner_map(dfe["owner"])
    if "brand" not in dfe.columns and "model_name" in dfe.columns:
        dfe["brand"] = extract_brand_simple(dfe["model_name"])
    if "bike_age" not in dfe.columns and "model_year" in dfe.columns:
        dfe["bike_age"] = pd.Timestamp.now().year - pd.to_numeric(dfe["model_year"], errors="coerce")

    # KPIs if price present
    if "price" in dfe.columns:
        p = pd.to_numeric(dfe["price"], errors="coerce").dropna()
        if len(p):
            kpi_row([
                ("Median price", "₹{:,.0f}".format(p.median()), ""),
                ("Mean price", "₹{:,.0f}".format(p.mean()), ""),
                ("95th pct", "₹{:,.0f}".format(p.quantile(0.95)), ""),
            ])

    # Owner vs price (mean)
    if "owner_cat" in dfe.columns and "price" in dfe.columns:
        grp = dfe.copy()
        grp["price"] = pd.to_numeric(grp["price"], errors="coerce")
        grp = grp.groupby("owner_cat")["price"].mean().sort_values(ascending=False).round(0)
        out = grp.reset_index()
        out.columns = ["owner_cat", "mean_price"]
        st.markdown("**Mean price by owner category**")
        st.dataframe(out, use_container_width=True)

    # Top brands by count (Top 15)
    if "brand" in dfe.columns:
        topb = dfe["brand"].value_counts().head(15)
        topb_df = topb.reset_index()
        topb_df.columns = ["brand", "count"]
        st.markdown("**Top 15 brands by listing count**")
        st.dataframe(topb_df, use_container_width=True)

with tab_predict:
    st.subheader("Predict a single listing")
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.text_input("Model name", value="Yamaha R15 150 cc")
        model_year = st.number_input("Model year", value=2019, step=1)
        mileage    = st.text_input("Mileage (e.g., '45 kmpl')", value="45 kmpl")
        power      = st.text_input("Power (e.g., '18 bhp')", value="18 bhp")
    with c2:
        kms_driven = st.text_input("Kms driven (e.g., '18000 Km')", value="18000 Km")
        owner      = st.text_input("Owner (e.g., 'First Owner')", value="First Owner")
        location   = st.text_input("Location", value="Chennai")

    sample = {
        "model_name": model_name,
        "model_year": model_year,
        "mileage": mileage,
        "power": power,
        "kms_driven": kms_driven,
        "owner": owner,
        "location": location
    }

    if st.button("Predict price", type="primary", use_container_width=False, disabled=(model is None)):
        if model is None:
            st.error("Model not loaded.")
        else:
            try:
                pred = float(model.predict(pd.DataFrame([sample]))[0])
                st.success("Predicted price: **₹{:,.0f}**".format(pred))
            except Exception as e:
                st.error("Prediction failed: {}".format(e))

    st.markdown("---")
    st.markdown("**Batch predict from CSV** (same raw columns as above)")
    up2 = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch")
    if up2 is not None and model is not None:
        try:
            df_new = pd.read_csv(up2)
            df_new["pred_price"] = model.predict(df_new)
            st.dataframe(df_new.head(20), use_container_width=True)
            st.download_button("⬇️ Download predictions CSV", df_new.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error("Batch prediction failed: {}".format(e))

with tab_eval:
    st.subheader("Evaluate on current data (quick check)")
    if model is None:
        st.info("Load a model to evaluate.")
    else:
        if "price" not in df.columns:
            st.info("No `price` column found in current data; cannot compute metrics.")
        else:
            dfeval = df.copy()
            y_true = pd.to_numeric(dfeval["price"], errors="coerce")
            mask = y_true.notna()
            try:
                pred   = model.predict(dfeval[mask])
                m = compute_metrics(y_true[mask], pred)
                kpi_row([
                    ("RMSE", "₹{:,.0f}".format(m["RMSE"]), ""),
                    ("MAE",  "₹{:,.0f}".format(m["MAE"]),  ""),
                    ("R²",   "{:.3f}".format(m["R2"]), "")
                ])
                st.caption("Note: Diagnostic on currently loaded data (not a strict hold-out test).")
            except Exception as e:
                st.error("Evaluation failed: {}".format(e))

st.divider()
st.caption("© Used Bike Prices — Feature Engineering & EDA | Streamlit demo")
