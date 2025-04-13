import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="⚡ Energy Forecasting", layout="wide")

# Set dark theme with custom CSS
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #1e1e1e;
        }
        .css-1d391kg, .css-1v3fvcr {
            background-color: #2e2e2e !important;
            color: white !important;
        }
        .css-1v0mbdj {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("linear_model.pkl")

# Load dataset for visualizations
@st.cache_data
def load_data():
    df = pd.read_csv("power Generation and consumption.csv", parse_dates=["Date_Time"])
    return df

df = load_data()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Forecast", "📈 Visual Insights"])

if page == "🏠 Home":
    st.title("⚡ Energy Generation & Consumption Forecasting")
    st.markdown("""
    This app uses historical electrical power generation and consumption data in Turkey (Jan 2020 - Dec 2022) to forecast future energy demands.
    
    **Use Cases:**
    - Predict energy demand to avoid blackouts
    - Optimize energy production schedules
    - Support smart grid automation and planning
    - Reduce costs with peak load shifting
    - Aid renewable energy forecasting
    """)

elif page == "📊 Forecast":
    st.title("📊 Forecast Energy Demand")

    # User inputs from sidebar
    def user_inputs():
        st.sidebar.header("Input Parameters")
        input_features = [col for col in df.columns if col not in ['Date_Time', 'Total (MWh)', 'Consumption (MWh)']]
        inputs = {}
        for feature in input_features:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            inputs[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)
        return pd.DataFrame([inputs])

    input_df = user_inputs()
    prediction = model.predict(input_df)

    st.subheader("Forecast Results")
    st.metric(label="Predicted Total Generation (MWh)", value=f"{prediction[0][0]:,.2f}")
    st.metric(label="Predicted Consumption (MWh)", value=f"{prediction[0][1]:,.2f}")

elif page == "📈 Visual Insights":
    st.title("📈 Visual Insights")

    selected_feature = st.selectbox("Select a Feature to Analyze:", [col for col in df.columns if col not in ['Date_Time']])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df['Date_Time'], y=df[selected_feature], ax=ax, color='cyan')
    ax.set_title(f"{selected_feature} Over Time", fontsize=16, color='white')
    ax.set_xlabel("Date", fontsize=12, color='white')
    ax.set_ylabel(selected_feature, fontsize=12, color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    st.pyplot(fig)

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        corr = df.drop(columns=["Date_Time"]).corr()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax2)
        st.pyplot(fig2)
