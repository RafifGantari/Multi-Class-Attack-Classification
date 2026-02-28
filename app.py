import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Network Attack Detection", layout="wide")

# ======================
# LOAD ARTIFACTS
# ======================
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    encoder = joblib.load("models/target_encoder.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, encoder, label_encoder, feature_cols

model, encoder, label_encoder, feature_cols = load_artifacts()

# ======================
# PREPROCESS FUNCTION
# ======================
def preprocess(df):
    df = df.copy()

    df = df.drop(columns=['src_ip', 'dst_ip'], errors='ignore')
    df = df.replace('-', np.nan)

    if 'dns_query' in df.columns:
        df["has_dns"] = df["dns_query"].notna().astype(int)
        df = df.drop(columns=['dns_query'], errors='ignore')

    binary_cols = ['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'T': 1, 'F': 0})

    if 'proto' in df.columns:
        df = pd.get_dummies(df, columns=['proto'], drop_first=False)

    target_cols = ['service', 'conn_state']
    existing_cols = [c for c in target_cols if c in df.columns]
    if len(existing_cols) > 0:
        df[existing_cols] = encoder.transform(df[existing_cols])

    df = df.reindex(columns=feature_cols, fill_value=0)

    return df

# ======================
# UI
# ======================
st.title("🚨 Network Attack Detection")
st.write("Upload CSV untuk prediksi")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df_input.head())

    try:
        X_processed = preprocess(df_input)
        preds = model.predict(X_processed)
        labels = label_encoder.inverse_transform(preds)

        result_df = df_input.copy()
        result_df["prediction"] = labels

        st.subheader("✅ Prediction Result")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Result",
            csv,
            "prediction.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")