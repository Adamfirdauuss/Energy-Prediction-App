import streamlit as st
import pandas as pd

st.title('Enegy Prediction App')

st.info('This App....')

df = pd.read_csv('https://raw.githubusercontent.com/Adamfirdauuss/Energy-Prediction-App/refs/heads/master/power%20Generation%20and%20consumption.csv')
df
