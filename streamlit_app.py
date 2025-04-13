import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/master/power%20Generation%20and%20consumption.csv'
    df = pd.read_csv(url)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], dayfirst=True)
    return df

df = load_data()
model = joblib.load("linear_model.pkl")

# Custom CSS for dark theme and navigation
st.markdown("""
    <style>
    body {
        background-color: #1e1e2f;
        color: white;
    }
    .main {
        background-color: #1e1e2f;
    }
    .stApp {
        background-color: #1e1e2f;
    }
    h1, h2, h3, h4, h5, h6, .css-1v0mbdj {
        color: white !important;
    }
    .nav-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
        background-color: #333;
        padding: 1rem;
        border-radius: 8px;
    }
    .nav-container a {
        color: #fff;
        text-decoration: none;
        font-size: 18px;
        padding: 0.5rem 1rem;
        border-radius: 4px;
    }
    .nav-container a:hover {
        background-color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
page = st.experimental_get_query_params().get("page", ["Home"])[0]
nav = st.markdown(f"""
    <div class="nav-container">
        <a href="?page=Home">🏠 Home</a>
        <a href="?page=Forecast">📊 Forecast</a>
        <a href="?page=Visual Insights">📈 Visual Insights</a>
    </div>
""", unsafe_allow_html=True)

# Pages
if page == "Home":
    st.title("⚡ Energy Generation & Consumption Forecasting in Turkey")
    st.markdown("""
        #### About
        This web application leverages historical electricity generation and consumption data from **Turkey (Jan 2020 - Dec 2022)**.

        It is designed for: 
        - 📉 Short Term Load Forecasting (STLF)
        - 🔍 Insights into renewable vs non-renewable energy sources
        - 🧠 Understanding which sources dominate energy supply

        #### Why it matters:
        - 🏷️ **Save costs** by optimizing generation planning
        - ⚙️ **Prevent over/under-generation** with reliable forecasts
        - 🌱 **Support sustainability** by tracking renewable growth
        - ⚡ **Improve grid stability** through accurate planning
        - 🧭 **Guide policy-making** with data-driven insights

        Use the navigation above to explore predictions or dive into trends.
    """)

elif page == "Forecast":
    st.title("📊 Forecast Energy Demand")
    st.markdown("""
    Use the controls below to simulate a scenario. The model will forecast:
    - Total Electricity Generation
    - Consumption (Demand)
    """)

    numeric_features = [col for col in df.columns if col not in ['Date_Time', 'Total (MWh)', 'Consumption (MWh)']]

    st.subheader("🔧 Adjust Energy Source Levels (MWh)")
    user_input = {}
    for feature in numeric_features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        user_input[feature] = st.slider(
            f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, step=1.0
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Energy Output"):
        prediction = model.predict(input_df)[0]
        st.success(f"🔌 Predicted Total Generation: {prediction[0]:,.2f} MWh")
        st.success(f"🔥 Predicted Consumption: {prediction[1]:,.2f} MWh")

elif page == "Visual Insights":
    st.title("📈 Interactive Energy Insights")
    st.markdown("""
    Visualize how energy sources relate to total generation and demand.
    """)

    energy_col = st.selectbox("Choose an energy type to compare:",
                               [col for col in df.columns if col not in ['Date_Time', 'Total (MWh)', 'Consumption (MWh)']])

    col1, col2 = st.columns(2)
    with col1:
        fig_total = px.line(df, x='Date_Time', y=['Total (MWh)', energy_col],
                            title=f'{energy_col} vs Total Generation')
        st.plotly_chart(fig_total, use_container_width=True)

    with col2:
        fig_cons = px.line(df, x='Date_Time', y=['Consumption (MWh)', energy_col],
                           title=f'{energy_col} vs Consumption')
        st.plotly_chart(fig_cons, use_container_width=True)


