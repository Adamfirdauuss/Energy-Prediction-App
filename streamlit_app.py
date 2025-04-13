import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(page_title="⚡ Energy Forecasting App", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme and nav bar
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .main, .block-container {
        background-color: #121212;
        color: #f0f0f0;
    }
    .css-1d391kg, .stSlider label, .stSelectbox label, .stRadio label, .stTextInput label {
        color: #f0f0f0 !important;
    }
    .css-145kmo2 a {
        color: #f0f0f0;
    }
    .css-1cpxqw2, .stMarkdown, .stHeading {
        color: #f0f0f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/master/power%20Generation%20and%20consumption.csv'
    return pd.read_csv(url)

df = load_data()

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("linear_model.pkl")

model = load_model()

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Forecast", "Visual Insight"],
    icons=["house", "bar-chart", "activity"],
    orientation="horizontal"
)

# Home Page
if selected == "Home":
    st.title("⚡ Energy Generation & Consumption Forecasting")
    st.markdown("""
    Welcome to the Energy Forecasting App! This tool helps predict **total electricity generation** and **consumption** in **Turkey** using historical data from **2020 to 2022**, provided by **EPIAS**.

    **Dataset Summary**:
    - Time Period: January 2020 to December 2022
    - Frequency: Hourly aggregated
    - Includes: Energy generation by type (Natural Gas, Geothermal, Solar, etc.) and total demand

    **Why it matters**:
    - Improve electricity demand planning
    - Support decision-making for renewable integration
    - Optimize costs and avoid overproduction
    - Help reduce CO₂ emissions
    - Enhance energy efficiency forecasting
    """)

# Forecast Page
elif selected == "Forecast":
    st.title("📊 Forecast")
    st.markdown("Use the sliders below to input values and predict Total Generation and Consumption.")

    features = df.drop(columns=["Date_Time", "Total (MWh)", "Consumption (MWh)"]).columns
    user_input = {}

    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        if min_val == max_val:
            max_val += 1  # Prevent slider crash
        user_input[feature] = st.slider(
            feature, float(min_val), float(max_val), float(mean_val)
        )

    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    st.subheader("🔮 Prediction Results")
    st.markdown(f"**Total Generation (MWh):** {prediction[0]:,.2f}")
    st.markdown(f"**Total Consumption (MWh):** {prediction[1]:,.2f}")

# Visual Insight Page
import pandas as pd
import plotly.express as px
import streamlit as st

# Assuming your dataframe is already loaded as 'df'

elif selected == "Visual Insight":
    st.title("📈 Visual Insights")
    st.markdown("Interactive charts based on energy source selection.")

    # Get list of energy sources from the dataset (excluding Date_Time and target columns)
    energy_sources = df.drop(columns=["Date_Time", "Total (MWh)", "Consumption (MWh)"]).columns.tolist()
    selected_source = st.selectbox("Choose an energy source to visualize:", energy_sources)

    # Convert Date_Time to datetime with the correct format
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format="%d.%m.%Y %H:%M")

    # Create the first chart for Total Energy Generation
    fig1 = px.line(df, x="Date_Time", y="Total (MWh)", title="Total Energy Generation Over Time", 
                   template="plotly_dark", line_shape="linear")
    fig1.update_traces(line=dict(color="#00BFFF"))
    fig1.update_layout(margin=dict(t=50, b=50), xaxis_title="Time", yaxis_title="Total Energy (MWh)")

    # Create the second chart for the selected energy source
    fig2 = px.line(df, x="Date_Time", y=selected_source, title=f"{selected_source} Energy Over Time", 
                   template="plotly_dark", line_shape="linear")
    fig2.update_traces(line=dict(color="#32CD32"))
    fig2.update_layout(margin=dict(t=50, b=50), xaxis_title="Time", yaxis_title=f"{selected_source} (MWh)")

    # Display both charts side by side using columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)
