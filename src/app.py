import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f19; }
    h1, h2, h3, h4, label, p, span { color: #e5e7eb; }
    .block-container { padding-top: 2.5rem; max-width: 880px; }
    .stButton > button {
        background-color: #3b82f6;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #2563eb; color: #ffffff; }
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-churn { background-color: #2a1215; border: 1px solid #ef4444; }
    .result-stay { background-color: #0f2417; border: 1px solid #22c55e; }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .churn-text { color: #f87171; }
    .stay-text { color: #4ade80; }
    .prob-value { font-size: 2.4rem; font-weight: 800; color: #f8fafc; }
    .stProgress > div > div > div { background-color: #3b82f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Telco Churn Predictor")
st.caption("Preencha os dados do cliente e estime a probabilidade de churn.")

with st.sidebar:
    st.header("Configuracao")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    if st.button("Checar API"):
        try:
            h = requests.get(f"{api_url}/health", timeout=5).json()
            if h.get("model_loaded"):
                st.success("API ok, modelo carregado")
            else:
                st.warning("API respondeu, mas o modelo nao esta carregado.")
        except Exception as exc:
            st.error(f"Falha ao conectar: {exc}")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.radio("Senior Citizen", ["Yes", "No"], horizontal=True)
        Partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
        Dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)
        PhoneService = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    with col2:
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.divider()
    tenure = st.slider("Tenure (meses)", 0, 72, 12)
    MonthlyCharges = st.slider("Monthly Charges", 0.0, 120.0, 70.0)
    TotalCharges = st.slider("Total Charges", 0.0, 9000.0, 1400.0)

    submitted = st.form_submit_button("Prever Churn")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        st.error(f"Erro ao chamar a API: {exc}")
    else:
        prob = result["probability"]
        label = result["label"]
        is_churn = result["prediction"] == 1

        card_class = "result-churn" if is_churn else "result-stay"
        text_class = "churn-text" if is_churn else "stay-text"

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-title {text_class}">{label}</div>
                <div class="prob-value">{prob:.1%}</div>
                <div>Probabilidade de Churn</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))
