
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
        st.image("data:image/png;base64,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", width=100)

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
            st.image("data:image/jpeg;base64,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", width=100)
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

    # Create a Plotly bar chart
    import plotly.graph_objects as go

    bar_fig = go.Figure(
       data=[
         go.Bar(
            x=["Total Generation", "Total Consumption"],
            y=[prediction[0], prediction[1]],
            marker_color=["#1f77b4", "#ff7f0e"],  # Colors for the bars
            text=[f"{prediction[0]:,.2f}", f"{prediction[1]:,.2f}"],  # Add values on top of bars
            textposition="auto"  # Automatically position the text inside the bars
            )
        ]
    )

    # Update layout with title, axes, and styling
    bar_fig.update_layout(
     template="plotly_dark",  # Use dark template
     xaxis_title="Energy Type",  # X-axis label
     yaxis_title="MWh",  # Y-axis label
     title="Predicted Total Energy Generation and Consumption",  # Chart title
     title_font=dict(size=18),  # Title font size
     margin=dict(t=50, b=50),  # Margins for top and bottom
     height=450  # Set the height for the chart
    )

   # Display the interactive Plotly chart
    st.plotly_chart(bar_fig, use_container_width=True)

   # Optional: Add download button with prediction results
    download_data = input_df.copy()
    download_data['Total Generation (MWh)'] = prediction[0]
    download_data['Total Consumption (MWh)'] = prediction[1]

    # Apply custom styling to make the download button white with black text
    st.markdown("""
    <style>
    .stDownloadButton button {
        background-color: white !important;
        color: black !important;
        border: 2px solid black;
    }
    .stDownloadButton button:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

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

    # Apply custom styling to make the download button white with black text
    st.markdown("""
    <style>
    .stDownloadButton button {
        background-color: white !important;
        color: black !important;
        border: 2px solid black;
    }
    .stDownloadButton button:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Download button for filtered data
    st.download_button("📥 Download Filtered Data (CSV)",
                       data=filtered_df.to_csv(index=False),
                       file_name=f"{selected_source}_visual_insight.csv",
                       mime="text/csv")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:gray;'>Visualized with ❤️ using Streamlit and Plotly</p>", unsafe_allow_html=True)
