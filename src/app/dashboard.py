import streamlit as st
import pandas as pd

st.title("Energy Forecast – Proof of Concept")

area = st.selectbox("Vælg område", ["DK1", "DK2"])

df_hist = pd.read_csv(f"data/prices_{area}_hist.csv", parse_dates=["ts"])
df_fc = pd.read_csv(f"data/{area}_price_forecast.csv", parse_dates=["ts"])

st.subheader(f"Historiske priser ({area})")
st.line_chart(df_hist.set_index("ts")["price_eur_mwh"])

st.subheader(f"Day-ahead forecast ({area})")
st.line_chart(df_fc.set_index("ts")["price_eur_mwh"])
