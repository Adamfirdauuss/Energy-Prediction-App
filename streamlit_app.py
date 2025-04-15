
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
df = pd.read_csv("power Generation and consumption.csv")  # replace with your actual path

if selected == "Home":
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Sustainable_Development_Goal_7.svg/800px-Sustainable_Development_Goal_7.svg.png", width=100)

    with col2:
        st.title("⚡ Energy Generation & Consumption Forecasting")

    st.markdown("""Welcome to the **Energy Forecasting App** — a smart platform to visualize and forecast **Turkey's electricity usage**.""")

    # 🔽 Expanders Section
    with st.expander("📌 About This Project"):
        st.markdown("""
        - **Project Title:** Improving Energy Consumption Forecasting for Sustainable and Affordable Energy Solutions  
        - **Course:** Final Year Project  
        - **Student Name:** Adam Firdaus  
        - **Student ID:** TP068684  
        - **University:** Asia Pacific University of Technology & Innovation (APU)  
        - **Supervisor:** Mr. Ts Mohammad Namazee  
        - **Second Marker:** Dr. Vazeerudeen Abdul Hamed  
        """)

    with st.expander("🌱 Why It Matters"):
        st.markdown("""
        - Enhance energy efficiency & planning  
        - Support decisions for renewable integration  
        - Avoid overproduction & optimize electricity costs  
        - Optimize production costs and prevent overgeneration  
        - Enhance energy efficiency forecasting  
        - Help reduce CO₂ emissions through better insights
        """)

    with st.expander("📊 Dataset Overview"):
        st.markdown("""
        - Data Source: EPIAS  
        - Period: Jan 2020 – Dec 2022  
        - Frequency: Hourly aggregated  
        - Features: Generation by type (e.g. Natural Gas, Hydro, Solar) & Total Demand 
        """)

    # 🔽 University Info
    with st.expander("🎓 University Information"):
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

       # 🔽 Interactive Pie Chart
    st.markdown("---")
    st.subheader("🔍 Total Energy Generation Breakdown by Type")

    # Assuming generation columns are like: 'Natural Gas', 'Solar', 'Hydro', etc.
    generation_types = ['Natural Gas', 'Solar', 'Hydro', 'Wind', 'Geothermal', 'Coal']  # customize based on your dataset
    
    generation_types = [
    'Natural Gas', 'Dammed Hydro', 'Lignite', 'River', 'Import Coal', 
    'Wind', 'Solar', 'Fuel Oil', 'Geothermal', 'Asphaltite Coal', 
    'Black Coal', 'Biomass', 'Naphta', 'LNG', 'Import-Export', 'Waste Heat'
    ]

    # Sum the energy generation by type
    total_by_type = df[generation_types].sum().reset_index()
    total_by_type.columns = ['Energy Type', 'Total (MWh)']

    # Create the pie chart
    fig = px.pie(
    total_by_type,
    names='Energy Type',
    values='Total (MWh)',
    title="Share of Total Energy Generated by Type",
    hole=0.4
    )
    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(total_by_type))  # adds hover + explode effect

    # Display the pie chart
    st.plotly_chart(fig, use_container_width=True)

    # 🔽 Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; font-size: 0.85rem;'>📘 Developed as part of a Final Year Project at APU. Powered by Python, Streamlit & Plotly.</div>", unsafe_allow_html=True)





  

# Forecast Page
elif selected == "Forecast":
    st.title("📊 Forecast: Energy Generation & Consumption")
    st.markdown("Adjust the energy source inputs below to predict Turkey’s total electricity generation and consumption.")

    # Get list of all input features
    features = df.drop(columns=["Date_Time", "Total (MWh)", "Consumption (MWh)"]).columns
    user_input = {}

    # Helper function to render sliders
    def render_slider(col, feature):
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        if min_val == max_val:
            max_val += 1  # Prevent slider crash
        user_input[feature] = col.slider(
            label=feature,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(mean_val),
        )

    st.subheader("🔧 Input Energy Sources")

    # Grouping features
    renewable = ["Dammed Hydro", "Wind", "Geothermal", "River", "Solar", "Biomass"]
    non_renewable = ["Natural Gas", "Import Coal", "Asphaltite Coal", "Naphta", "Lignite", "Fuel Oil", "Black Coal", "LNG"]
    other_sources = ["Import-Export", "Waste Heat"]

    # Render sliders
    with st.expander("🌱 Renewable Energy Sources"):
        cols = st.columns(2)
        for i, feature in enumerate(renewable):
            render_slider(cols[i % 2], feature)

    with st.expander("⚙️ Non-Renewable Energy Sources"):
        cols = st.columns(2)
        for i, feature in enumerate(non_renewable):
            render_slider(cols[i % 2], feature)

    with st.expander("📦 Import & Other Sources"):
        cols = st.columns(2)
        for i, feature in enumerate(other_sources):
            render_slider(cols[i % 2], feature)

    # Create input DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure feature order matches model
    try:
        input_df = input_df[model.feature_names_in_]
    except AttributeError:
        st.error("Model doesn't contain feature names. Ensure the model was trained on a DataFrame.")
        st.stop()
    except KeyError:
        st.error("Input features don't match model features. Please verify.")
        st.stop()

    # Prediction
    prediction = model.predict(input_df)[0]

    # Display results
    st.subheader("🔮 Prediction Results")

    

    # Text below results in light color for visibility
    # Text below results in white font and displayed side-by-side
    col1, col2 = st.columns(2)

    with col1:
    st.markdown(
        f"<h4 style='color:white;'>Total Generation (MWh)</h4><h2 style='color:white;'>{prediction[0]:,.2f}</h2>",
        unsafe_allow_html=True
    )

    with col2:
    st.markdown(
        f"<h4 style='color:white;'>Total Consumption (MWh)</h4><h2 style='color:white;'>{prediction[1]:,.2f}</h2>",
        unsafe_allow_html=True
    )



    # Improved Graph (Bar Chart)
    st.markdown("### 📈 Forecast Breakdown")
    data = pd.DataFrame({
        "Energy Type": ["Total Generation", "Total Consumption"],
        "MWh": [prediction[0], prediction[1]]
    })

    st.bar_chart(data.set_index("Energy Type"))

    # Optional: Add download button with prediction results
    download_data = input_df.copy()
    download_data['Total Generation (MWh)'] = prediction[0]
    download_data['Total Consumption (MWh)'] = prediction[1]

    st.download_button(
        label="📥 Download Forecast with Results",
        data=download_data.to_csv(index=False),
        file_name="energy_forecast_with_results.csv",
        mime="text/csv",
        help="Click to download the forecasted energy data with prediction results.",
        use_container_width=True
    )









# Visual Insight Page
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

    st.markdown("""
    <style>
    /* Force label text to white for sliders and date inputs */
    label, .stSlider label, .stDateInput label, .css-149m0r1, .css-81oif8 {
        color: white !important;
        font-weight: bold !important;
    }

    /* Optional: white text for all selectboxes, sliders, and inputs */
    .stSelectbox > label, .stTextInput > label {
        color: white !important;
    }

    /* Optional: make slider tick values white */
    .stSlider .css-1m3z3qc {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)



    # Get list of energy sources from the dataset (excluding Date_Time and target columns)
    energy_sources = df.drop(columns=["Date_Time", "Total (MWh)", "Consumption (MWh)"]).columns.tolist()
    selected_source = st.selectbox("Choose an energy source to visualize:", energy_sources)

    # Convert Date_Time to datetime with the correct format
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], format="%d.%m.%Y %H:%M")

    # Date range filter
    min_date = df["Date_Time"].min()
    max_date = df["Date_Time"].max()
    start_date, end_date = st.date_input("Select time range:", [min_date, max_date])

    # Filter data by selected date range
    mask = (df["Date_Time"] >= pd.to_datetime(start_date)) & (df["Date_Time"] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask].copy()

    # Add rolling average for the selected source
    filtered_df["Rolling Avg"] = filtered_df[selected_source].rolling(window=7).mean()

    # Insight Summary
    with st.expander("🔍 Quick Insight Summary"):
        st.markdown(
            f"""
            - **Most recent Total Energy Generation**: {df['Total (MWh)'].iloc[-1]:,.2f} MWh  
            - **Most recent Consumption**: {df['Consumption (MWh)'].iloc[-1]:,.2f} MWh  
            - **Selected Source Latest Output**: {df[selected_source].iloc[-1]:,.2f} MWh  
            """
        )

    # Create the first chart for Total Energy Generation
    fig1 = px.line(filtered_df, x="Date_Time", y="Total (MWh)", title="Total Energy Generation Over Time", 
                   template="plotly_dark", line_shape="linear")
    fig1.update_traces(line=dict(color="#00BFFF"))
    fig1.update_layout(margin=dict(t=50, b=50), xaxis_title="Time", yaxis_title="Total Energy (MWh)")

    # Create the second chart for the selected energy source with rolling average
    fig2 = px.line(filtered_df, x="Date_Time", y=[selected_source, "Rolling Avg"],
                   labels={"value": "Energy (MWh)", "variable": "Type"},
                   title=f"{selected_source} Energy Over Time with 7-Day Average",
                   template="plotly_dark")
    fig2.update_layout(margin=dict(t=50, b=50), xaxis_title="Time", yaxis_title=f"{selected_source} (MWh)")

    # Display both charts side by side using columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Download button for filtered data
    st.download_button("📥 Download Filtered Data (CSV)",
                       data=filtered_df.to_csv(index=False),
                       file_name=f"{selected_source}_visual_insight.csv",
                       mime="text/csv")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>Visualized with ❤️ using Streamlit and Plotly</p>", unsafe_allow_html=True)
