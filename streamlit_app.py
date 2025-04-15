
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
