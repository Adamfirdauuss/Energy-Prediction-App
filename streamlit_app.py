import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Set Streamlit's background theme to dark mode
st.set_page_config(page_title="Energy Forecasting App", page_icon="⚡", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
        }
        .css-1r6xtxe {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Information
st.title('⚡ Energy Generation & Consumption Forecasting')
st.info('This app forecasts energy generation and consumption in Turkey from historical data (2020-2022).')

# Load dataset
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/refs/heads/master/power%20Generation%20and%20consumption.csv'
    df = pd.read_csv(url)
    return df

df = load_data()
st.subheader('Dataset Preview:')
st.write(df.head())

# Preprocess the data for model
df_model = df.copy().dropna()
target_cols = ['Total (MWh)', 'Consumption (MWh)']
X = df_model.drop(columns=target_cols + ['Date_Time'])
y = df_model[target_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
@st.cache
def train_model():
    base_model = LinearRegression()
    multi_model = MultiOutputRegressor(base_model)
    multi_model.fit(X_train, y_train)
    return multi_model

multi_model = train_model()

# Make predictions
y_pred = multi_model.predict(X_test)

# Evaluation of the model
st.subheader('Model Evaluation:')
for i, target in enumerate(target_cols):
    y_true = y_test.iloc[:, i]
    y_hat = y_pred[:, i]

    mae = mean_absolute_error(y_true, y_hat)
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_hat)

    st.write(f"\n**Evaluation for: {target}**")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.4f}")

# Visual Insights: User Input for forecasting
st.sidebar.header('Forecasting Inputs')

# Sliders for user input
sliders = {}
for feature in X.columns:
    feature_min = float(df[feature].min())
    feature_max = float(df[feature].max())
    feature_mean = float(df[feature].mean())
    
    sliders[feature] = st.sidebar.slider(feature, feature_min, feature_max, feature_mean)

# Prepare user input for prediction
user_inputs = np.array([sliders[feature] for feature in X.columns]).reshape(1, -1)
user_pred = multi_model.predict(user_inputs)

# Display prediction results
st.subheader('Forecast Results:')
st.write(f"Predicted Total Energy Generation (MWh): {user_pred[0][0]:.2f}")
st.write(f"Predicted Total Energy Consumption (MWh): {user_pred[0][1]:.2f}")

# Visualize insights: feature importance using coefficients
@st.cache
def plot_feature_importance():
    fig, ax = plt.subplots(figsize=(12, 6))
    feature_importance = multi_model.estimators_[0].coef_
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': feature_importance.flatten()
    }).sort_values(by='Coefficient', ascending=False)
    
    sns.barplot(data=feature_df.head(15), x='Coefficient', y='Feature', palette='crest', ax=ax)
    ax.set_title('Top 15 Features for Energy Generation')
    return fig

st.subheader('Feature Importance Insights:')
fig = plot_feature_importance()
st.pyplot(fig)

# Interactive Visuals
st.subheader('Visual Insights:')
# Example: Show a line plot for the actual vs predicted generation and consumption
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test['Total (MWh)'], label='Actual Total Generation')
ax.plot(y_test.index, y_pred[:, 0], label='Predicted Total Generation')
ax.set_title('Actual vs Predicted Total Energy Generation')
ax.set_xlabel('Time')
ax.set_ylabel('Energy Generation (MWh)')
ax.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test['Consumption (MWh)'], label='Actual Consumption')
ax.plot(y_test.index, y_pred[:, 1], label='Predicted Consumption')
ax.set_title('Actual vs Predicted Total Energy Consumption')
ax.set_xlabel('Time')
ax.set_ylabel('Energy Consumption (MWh)')
ax.legend()
st.pyplot(fig)

