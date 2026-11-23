import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #edf4ff, #f7efff, #e8fbff);
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(120deg, #3423a6, #813ce7, #00b8c4);
    padding: 35px;
    border-radius: 25px;
    color: white;
    box-shadow: 0 15px 35px rgba(74,55,180,0.25);
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 40px;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #f0edff;
    font-size: 15px;
    margin-bottom: 0;
}

.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 30px;
    padding: 6px 14px;
    font-size: 12px;
    margin-bottom: 12px;
}

.section-title {
    color: #30256f;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #747797;
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid #e5e3ff;
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 10px 28px rgba(65,49,135,0.08);
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(100deg, #4f35cf, #8b45e8, #00aeb9);
    color: white !important;
    border: none;
    border-radius: 13px;
    padding: 13px 20px;
    font-weight: 700;
    min-height: 48px;
    box-shadow: 0 6px 16px rgba(91,61,201,0.23);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(91,61,201,0.3);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
    border-color: #d9d7f4;
    background: #fcfbff;
}

.result-card {
    background: linear-gradient(135deg, #ffffff, #f0edff);
    border: 1px solid #dcd5ff;
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 12px 28px rgba(67,45,160,0.12);
}

.result-label {
    color: #77749a;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.result-value {
    color: #4f35cf;
    font-size: 30px;
    font-weight: 800;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.result-icon {
    font-size: 48px;
    margin-bottom: 8px;
}

.note {
    background: #fff8e6;
    border-left: 5px solid #f3b632;
    padding: 14px 17px;
    border-radius: 10px;
    color: #71551a;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #8888a5;
    font-size: 12px;
    padding: 25px 0 5px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    df = None

    for file in ["brain_tumor_dataset(3).csv", "brain_tumor_dataset.csv"]:
        try:
            df = pd.read_csv(file)
            break
        except FileNotFoundError:
            continue

    if df is None:
        raise FileNotFoundError(
            "Dataset CSV file ko app.py ke same folder mein rakho."
        )

    df.columns = df.columns.str.strip()

    target = "Tumor_Type"
    if target not in df.columns:
        raise ValueError("Dataset mein Tumor_Type column nahi mila.")

    df = df.dropna(subset=[target]).copy()

    drop_cols = [c for c in ["Patient_ID", target] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target].astype(str)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)

    return X, model, numeric_cols, categorical_cols


#HEADER 
st.markdown("""
<div class="hero">
    <!-- <div class="hero-pill">✦ AI-POWERED HEALTHCARE ANALYTICS</div> -->
    <h1>🧠 NeuroScan AI</h1>
    <p>Brain Tumor Prediction Dashboard</p>
    <p>Smart Brain Tumor Prediction with AI</p>
</div>
""", unsafe_allow_html=True)


#MODEL 
try:
    X, model, numeric_cols, categorical_cols = load_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()


# PATIENT FORM 
st.markdown(
    '<div class="section-title">👤 Patient Assessment</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Please fill in the patient and tumor details.</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    patient_input = {}

    st.markdown("### 🩺 Basic Information")
    c1, c2, c3 = st.columns(3)

    if "Age" in X.columns:
        with c1:
            median_age = pd.to_numeric(X["Age"], errors="coerce").median()
            patient_input["Age"] = st.number_input(
                "📅Patient Age",
                min_value=0,
                max_value=120,
                value=int(median_age) if pd.notna(median_age) else 30
            )

    if "Gender" in X.columns:
        with c2:
            opts = X["Gender"].dropna().astype(str).unique().tolist()
            patient_input["Gender"] = st.selectbox("⚧ Gender", opts)

    if "Location" in X.columns:
        with c3:
            opts = X["Location"].dropna().astype(str).unique().tolist()
            patient_input["Location"] = st.selectbox("📍 Tumor Location", opts)

    st.markdown("---")
    st.markdown("### 🧬 Tumor Details")
    c1, c2, c3 = st.columns(3)

    if "Tumor_Size" in X.columns:
        with c1:
            median_size = pd.to_numeric(X["Tumor_Size"], errors="coerce").median()
            patient_input["Tumor_Size"] = st.number_input(
                "📊 Tumor Size",
                min_value=0.0,
                value=float(median_size) if pd.notna(median_size) else 1.0,
                step=0.1
            )

    if "Histology" in X.columns:
        with c2:
            opts = X["Histology"].dropna().astype(str).unique().tolist()
            patient_input["Histology"] = st.selectbox("🧫 Histology", opts)

    if "Stage" in X.columns:
        with c3:
            opts = X["Stage"].dropna().astype(str).unique().tolist()
            patient_input["Stage"] = st.selectbox("📈 Tumor Stage", opts)

    st.markdown("---")
    st.markdown("### 🩻 Symptoms & MRI")
    c1, c2, c3 = st.columns(3)

    for col, label, container in [
        ("Symptom_1", "First Symptom", c1),
        ("Symptom_2", "Second Symptom", c2),
        ("Symptom_3", "Third Symptom", c3)
    ]:
        if col in X.columns:
            with container:
                opts = X[col].dropna().astype(str).unique().tolist()
                patient_input[col] = st.selectbox(f"🩺 {label}", opts)

    if "MRI_Result" in X.columns:
        opts = X["MRI_Result"].dropna().astype(str).unique().tolist()
        patient_input["MRI_Result"] = st.selectbox("🧲 MRI Result", opts)

    st.markdown("---")
    st.markdown("### 💊 Treatment & Medical History")
    c1, c2, c3 = st.columns(3)

    treatment_fields = [
        ("Radiation_Treatment", "☢️ Radiation Treatment"),
        ("Surgery_Performed", "🏥 Surgery Performed"),
        ("Chemotherapy", "💉 Chemotherapy"),
        ("Family_History", "👪 Family History"),
        ("Follow_Up_Required", "📅 Follow-up Required")
    ]

    for i, (col, label) in enumerate(treatment_fields):
        if col in X.columns:
            with [c1, c2, c3][i % 3]:
                opts = X[col].dropna().astype(str).unique().tolist()
                patient_input[col] = st.selectbox(label, opts)

    st.markdown("---")
    st.markdown("### 📋 Additional Measurements")
    c1, c2 = st.columns(2)

    for col, label, container in [
        ("Survival_Rate", "📊 Survival Rate", c1),
        ("Tumor_Growth_Rate", "📈 Tumor Growth Rate", c2)
    ]:
        if col in X.columns:
            with container:
                median_val = pd.to_numeric(X[col], errors="coerce").median()
                patient_input[col] = st.number_input(
                    label,
                    value=float(median_val) if pd.notna(median_val) else 0.0,
                    step=0.1
                )

    submitted = st.form_submit_button(
        "✨ ANALYZE & PREDICT",
        use_container_width=True
    )


# OUTPUT 
if submitted:
    input_df = pd.DataFrame([patient_input])
    input_df = input_df.reindex(columns=X.columns)

    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    try:
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">🧠</div>
                <div class="result-label">MODEL PREDICTION</div>
                <div class="result-value">{prediction}</div>
                <p style="color:#77749a;font-size:13px;">
                    Predicted category based on the entered information.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("### 📌 Patient Summary")
            st.write(f"**Age:** {patient_input.get('Age', 'N/A')}")
            st.write(f"**Location:** {patient_input.get('Location', 'N/A')}")
            st.write(f"**Stage:** {patient_input.get('Stage', 'N/A')}")
            st.write(f"**Histology:** {patient_input.get('Histology', 'N/A')}")
            st.write(f"**Tumor Size:** {patient_input.get('Tumor_Size', 'N/A')}")

            probabilities = model.predict_proba(input_df)[0]
            confidence = float(np.max(probabilities)) * 100

            st.markdown("### 📊 Model Confidence")
            st.progress(min(int(confidence), 100))
            st.write(f"**{confidence:.1f}%** model confidence")

        st.markdown("""
        <div class="note">
        ⚠️ <b>Medical Disclaimer:</b> This application is for educational purposes
        only. It is not a medical diagnosis tool. Model confidence does not
        represent clinical accuracy. Please consult a qualified medical professional.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")


st.markdown("""
<div class="footer">
    🧠 NeuroScan AI · Streamlit Machine Learning Project
</div>
""", unsafe_allow_html=True)