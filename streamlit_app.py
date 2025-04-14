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
# Navigation (moved to sidebar)
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Forecast", "Visual Insight"],
        icons=["house", "bar-chart", "activity"],
        orientation="vertical"
    )



# Home Page
# Home Page
if selected == "Home":
    # Hero banner with background image and title
    st.markdown("""
        <div style="background-image: url('https://images.unsplash.com/photo-1581091870622-1f168f15f65b?auto=format&fit=crop&w=1950&q=80');
                    background-size: cover; 
                    padding: 4rem; 
                    border-radius: 10px; 
                    margin-bottom: 2rem;">
            <h1 style="color: white; font-size: 3rem; text-shadow: 1px 1px 2px black;">⚡ Energy Generation & Consumption Forecasting</h1>
            <p style="color: white; font-size: 1.2rem; max-width: 700px; text-shadow: 1px 1px 2px black;">
                Gain insights and predict Turkey's electricity generation and consumption with historical data from 2020 to 2022, powered by machine learning.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Metrics summary
    col1, col2, col3 = st.columns(3)
    col1.metric("📅 Time Period", "Jan 2020 - Dec 2022")
    col2.metric("⏰ Data Frequency", "Hourly")
    col3.metric("⚙️ Features Used", "Energy by Type + Demand")

    # Expanders for more info
    with st.expander("📦 Dataset Summary"):
        st.markdown("""
        - **Time Period:** January 2020 to December 2022  
        - **Frequency:** Hourly aggregated  
        - **Source:** [EPIAS](https://www.epias.com.tr/en/)  
        - **Includes:**  
            - Energy generation by type: Natural Gas, Geothermal, Solar, Wind, Hydro, and more  
            - Total electricity demand  
        """)

    with st.expander("🌍 Why It Matters"):
        st.markdown("""
        - ⚡ Improve electricity demand planning  
        - 🌱 Support renewable energy decision-making  
        - 💸 Optimize production costs and prevent overgeneration  
        - 🧠 Enhance energy efficiency forecasting  
        - 🌍 Help reduce CO₂ emissions through better insights  
        """)

    st.markdown("---")
    st.info("Use the sidebar to navigate to Forecasting and Visual Insights.")


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
# Make sure to have an initial 'if' or 'elif' block before this
if selected == "Home":
    st.title("🏠 Home")
    st.markdown("Welcome to the Energy Generation and Consumption Forecasting app.")
    st.markdown("This app helps forecast and visualize Turkey's energy consumption and generation data.")

elif selected == "Forecast":
    st.title("📊 Forecast")
    st.markdown("Here, we predict energy demand based on the historical data of Turkey.")
    # Add your forecast logic here (e.g., prediction model, data inputs, etc.)

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
