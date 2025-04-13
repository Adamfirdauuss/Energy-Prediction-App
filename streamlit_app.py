import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set the dark theme
st.set_page_config(page_title="Energy Prediction App", layout="wide", initial_sidebar_state="collapsed")

# Load the dataset
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/refs/heads/master/power%20Generation%20and%20consumption.csv'
    df = pd.read_csv(url)
    
    # Ensure Date_Time column is properly cleaned and converted to datetime
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors='coerce')  # 'coerce' will turn invalid dates to NaT
    df.dropna(subset=["Date_Time"], inplace=True)  # Drop rows where Date_Time is NaT
    return df

df = load_data()

# Home Page
def home():
    st.title("⚡ Energy Generation & Consumption Forecasting")
    st.markdown("""
    #### About the Dataset
    This dataset consists of hourly electricity generation and consumption (demand) in Turkey.
    The dataset spans from January 2020 to December 2022. It includes generation by production type (natural gas, geothermal, solar, etc.) and total generation.
    
    The data was sourced from EPIAS and is ideal for Short-Term Load Forecasting (STLF), helping develop better day-ahead generator planning.
    """)

    st.markdown("""
    ### Key Features:
    - **Total Generation**: Total energy generation by all sources
    - **Consumption**: Energy consumption/demand across the country
    - **Energy Types**: Breakdown by generation type (e.g., solar, wind, natural gas)
    """)

# Forecast Page
def forecast():
    st.title("📊 Energy Forecasting")
    st.markdown("""
    This page allows you to forecast total energy generation and consumption based on historical data.
    You can interactively input values to predict future energy demand and generation across different sources.
    """)

    # Get user input for forecasting
    user_input = {}
    features = df.columns.difference(['Date_Time', 'Total (MWh)', 'Consumption (MWh)'])
    
    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        user_input[feature] = st.slider(f'{feature}', min_val, max_val, mean_val)

    # Convert user input to DataFrame for prediction
    input_df = pd.DataFrame([user_input])

    # Load the trained model
    @st.cache
    def load_model():
        return joblib.load("linear_model.pkl")  # Make sure you have this file

    model = load_model()

    # Predict based on input
    prediction = model.predict(input_df)
    st.write("Predicted Energy Generation and Consumption (MWh):")
    st.write(f"Total Generation: {prediction[0][0]:.2f} MWh")
    st.write(f"Total Consumption: {prediction[0][1]:.2f} MWh")

# Visual Insight Page
def visual_insights():
    st.title("📈 Visual Insights")

    # Visualize total energy generation and consumption over time
    st.markdown("""
    Explore how total energy generation and consumption have changed over time in Turkey.
    You can select different energy types for a more detailed comparison.
    """)

    # Plot total generation vs consumption
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Date_Time'], y=df['Total (MWh)'], mode='lines', name='Total Generation'))
    fig.add_trace(go.Scatter(x=df['Date_Time'], y=df['Consumption (MWh)'], mode='lines', name='Total Consumption'))

    fig.update_layout(title='Total Generation vs Consumption (2020-2022)', xaxis_title='Date', yaxis_title='Energy (MWh)')
    st.plotly_chart(fig)

    # Allow user to select energy type for comparison
    energy_type = st.selectbox('Select Energy Type to Compare', ['Natural Gas', 'Solar', 'Wind', 'Geothermal'])

    # Plot selected energy type
    fig_energy = go.Figure()

    fig_energy.add_trace(go.Scatter(x=df['Date_Time'], y=df[energy_type], mode='lines', name=energy_type))
    fig_energy.update_layout(title=f'{energy_type} Generation Over Time (2020-2022)', xaxis_title='Date', yaxis_title='Energy (MWh)')

    st.plotly_chart(fig_energy)

# Navigation
def main():
    st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        .css-1v0mbdj {
            color: #FFFFFF !important;
        }
        .streamlit-expanderHeader {
            color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    page = st.selectbox('Select Page', ['Home', 'Forecast', 'Visual Insights'])

    if page == 'Home':
        home()
    elif page == 'Forecast':
        forecast()
    elif page == 'Visual Insights':
        visual_insights()

# Run the app
if __name__ == "__main__":
    main()
