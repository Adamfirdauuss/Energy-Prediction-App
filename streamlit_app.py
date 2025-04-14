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
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Sustainable_Development_Goal_7.svg/800px-Sustainable_Development_Goal_7.svg.png", width=100)

    with col2:
        st.title("⚡ Energy Generation & Consumption Forecasting")

    st.markdown("""
    Welcome to the **Energy Forecasting App** — a data-driven tool to predict **electricity generation** and **consumption** in **Turkey**, using historical hourly data from **EPIAS (2020–2022)**.

    This app serves the **UN SDG Goal 7: Affordable & Clean Energy**, contributing to smarter planning and sustainability.

    ---
    ### 🌍 About This Project
    - **Project Title:** Improving Energy Consumption Forecasting for Sustainable and Affordable Energy Solutions  
    - **Course:** Final Year Project  
    - **Student Name:** Adam Firdaus  
    - **Student ID:** TP068684  
    - **University:** Asia Pacific University of Technology & Innovation (APU)  
    - **Supervisor:** Mr. Ts Mohammad Namazee  
    - **Second Marker:** Dr. Vazeerudeen Abdul Hamed  

    ---
    ### 📌 Why It Matters:
    - Enhance energy efficiency & forecasting
    - Aid policy-making in renewable integration
    - Avoid overproduction & optimize costs
    - Reduce CO₂ emissions

    ---
    ### 📁 Dataset Overview:
    - Energy by generation type (Natural Gas, Solar, Wind, Hydro, etc.)
    - Target variables: Total Generation (MWh), Total Consumption (MWh)

    ---
    """)

    with st.expander("📚 University Information"):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://www.apu.edu.my/sites/default/files/APU_LOGO.jpg", width=100)
        with col2:
            st.markdown("""
            **Asia Pacific University of Technology & Innovation (APU)**  
            Technology Park Malaysia, Bukit Jalil  
            Kuala Lumpur, Malaysia  
            [Website](https://www.apu.edu.my/)
            """)

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>📊 Navigate to the top menu to access Forecasting and Visual Insights.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; font-size: 0.85rem; margin-top: 20px;'>Developed as part of a Final Year Project at APU. Powered by Python, Streamlit & Plotly.</div>", unsafe_allow_html=True)


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
