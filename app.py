import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

# ================== Page Setup ==================
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📊",
    layout="centered"
)

# ================== Custom CSS ==================
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(to right, #90e0ef, #ffffff);
    }

    /* Title */
    .title {
        color: #03045e;
        font-size: 52px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Subheader */
    .subheader {
        color: #023e8a;
        font-size: 22px;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Button */
    div.stButton > button:first-child {
        background-color: #03045e;
        color: white;
        height: 60px;
        width: 100%;
        border-radius: 12px;
        font-size: 22px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.08);
        background-color: #023e8a;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }

    /* Input boxes */
    div.stNumberInput > label, div.stSelectbox > label {
        font-weight: bold;
        color: #03045e;
        font-size: 20px;
    }
    .stNumberInput>div>input, select {
        border-radius: 10px;
        border: 2px solid #90e0ef;
        padding: 8px;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stNumberInput>div>input:focus, select:focus {
        border: 2px solid #023e8a;
        box-shadow: 0px 0px 10px rgba(0,48,94,0.4);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ================== Title ==================
st.markdown('<p class="title">📊 Sales Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter store data to get accurate predictions</p>', unsafe_allow_html=True)
st.markdown("---")

# ================== Load Model ==================
@st.cache_resource
def load_model():
    bundle = joblib.load("model_bundle.pkl")
    return bundle["model"], bundle["columns"]

model, columns = load_model()

# ================== Organize Inputs in Tabs ==================
tab1, tab2 = st.tabs(["Store Info", "Promotion Info"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        Store = st.number_input("Store", value=1)
        DayOfWeek = st.number_input("DayOfWeek", value=1)
        CompetitionDistance = st.number_input("CompetitionDistance", value=1000.0)
        Customers = st.number_input("Customers", value=500)
    with col2:
        StoreType = st.selectbox("StoreType", ['a','b','c','d'])
        Assortment = st.selectbox("Assortment", ['a','b','c'])
        Open = st.selectbox("Open", [0,1])
        StateHoliday = st.selectbox("StateHoliday", ['0','a','b','c'])

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        Promo = st.selectbox("Promo", [0,1])
        Promo2 = st.selectbox("Promo2", [0,1])
    with col2:
        Month = st.number_input("Month", value=1)
        Year = st.number_input("Year", value=2015)
        Day = st.number_input("Day", value=1)
        PromoInterval = st.selectbox("PromoInterval", ['None','Jan,Apr,Jul,Oct','Feb,May,Aug,Nov','Mar,Jun,Sept,Dec'])

st.markdown("---")

# ================== Predict Button ==================
if st.button("🚀 Predict"):
    with st.spinner("Calculating..."):
        time.sleep(1)

        input_data = pd.DataFrame({
            'Store': [Store],
            'DayOfWeek': [DayOfWeek],
            'Promo': [Promo],
            'SchoolHoliday': [0],
            'CompetitionDistance': [CompetitionDistance],
            'Promo2': [Promo2],
            'Month': [Month],
            'Year': [Year],
            'Day': [Day],
            'Customers': [Customers],
            'StoreType': [StoreType],
            'Assortment': [Assortment],
            'StateHoliday': [StateHoliday],
            'PromoInterval': [PromoInterval],
            'Open': [Open]
        })

        categorical_cols = {
            "StoreType": ['b','c','d'],
            "Assortment": ['a','b','c'],
            "StateHoliday": ['0','a','b','c'],
            "PromoInterval": ['None','Jan,Apr,Jul,Oct','Feb,May,Aug,Nov','Mar,Jun,Sept,Dec']
        }

        for col, categories in categorical_cols.items():
            for cat in categories:
                input_data[f"{col}_{cat}"] = (input_data[col] == cat).astype(int)
            input_data.drop(columns=[col], inplace=True)

        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[columns]

        try:
            prediction = model.predict(input_data)
            result = np.expm1(prediction[0])
            st.markdown("---")
            st.success(f"💰 Predicted Sales: {result:,.2f}")
        except Exception as e:
            st.error(f"❌ Model Error: {e}")