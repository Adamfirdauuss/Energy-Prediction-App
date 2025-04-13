import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="⚡ Energy Forecasting App", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/master/power%20Generation%20and%20consumption.csv"
    return pd.read_csv(url)

model = load_model()
df = load_data()

# Sidebar inputs
def user_inputs():
    st.sidebar.title("Adjust Energy Inputs")
    features = ['Natural Gas', 'Dammed Hydro', 'Lignite', 'River', 'Import Coal',
                'Wind', 'Solar', 'Fuel Oil', 'Geothermal', 'Asphaltite Coal',
                'Black Coal', 'Biomass', 'Naphta', 'LNG', 'Import-Export',
                'Waste Heat']
    inputs = {}
    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        inputs[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
    return pd.DataFrame([inputs])

# Layout
st.title("⚡ Energy Generation & Consumption Forecasting")
tabs = st.tabs(["🏠 Home", "📊 Forecast", "📈 Visual Insights"])

# Home Tab
with tabs[0]:
    st.header("📤 Uploaded Dataset Preview")
    st.write("This app allows you to forecast total energy generation and consumption based on energy sources.")
    st.dataframe(df.head(10))

# Forecast Tab
with tabs[1]:
    st.header("📊 Forecast")
    input_df = user_inputs()
    prediction = model.predict(input_df)[0]
    total, consumption = prediction

    st.subheader("🔮 Predicted Outputs:")
    col1, col2 = st.columns(2)
    col1.metric("Total Energy (MWh)", f"{total:.2f}")
    col2.metric("Consumption (MWh)", f"{consumption:.2f}")

# Visual Insights Tab
with tabs[2]:
    st.header("📈 Visual Insights from Dataset")

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr, cmap="crest", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Energy Sources")
    selected_feature = st.selectbox("Choose a feature", numeric_df.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, ax=ax2, color='skyblue')
    st.pyplot(fig2)

# Footer
st.markdown("""
---
🔗 Built by [Adamfirdauuss](https://github.com/Adamfirdauuss)  
🎯 Supporting SDG 7: Affordable and Clean Energy  
""")
