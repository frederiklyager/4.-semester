import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Energy Forecast – POC", layout="wide")
st.title("Energy Forecast – Proof of Concept")

# ------- Helper (cache CSV indlæsning) -------
@st.cache_data
def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=parse_dates)

# ---------- UI ----------
tab_prices, tab_co2 = st.tabs(["📈 Priser", "🌿 CO₂"])

# ============== PRISER (din eksisterende del) ==============
with tab_prices:
    area = st.selectbox("Vælg område", ["DK1", "DK2"], index=0)

    hist_path = f"data/prices_{area}_hist.csv"
    fc_path   = f"data/{area}_price_forecast.csv"

    df_hist = load_csv(hist_path, parse_dates=["ts"])
    df_fc   = load_csv(fc_path,   parse_dates=["ts"])

    st.subheader(f"Historiske priser ({area})")
    if df_hist.empty:
        st.warning(f"Mangler fil: {hist_path}")
    else:
        st.line_chart(df_hist.set_index("ts")["price_eur_mwh"], height=280)

    st.subheader(f"Day-ahead forecast ({area})")
    if df_fc.empty:
        st.warning(f"Mangler fil: {fc_path}")
    else:
        st.line_chart(df_fc.set_index("ts")["price_eur_mwh"], height=280)

# ============== CO2 (NY) ==============
with tab_co2:
    st.caption("Kilde: Energinet Energi Data Service – dataset 'CO2Emis'")
    co2_file = "data/processed/co2_hourly_DK1.csv"   # vi har pt. kun DK1
    df_co2 = load_csv(co2_file, parse_dates=["ts"])

    st.subheader("CO₂-intensitet (DK1) – timegennemsnit")
    if df_co2.empty:
        st.error("Ingen CO₂-data fundet. Kør: `python src/ingest/energinet_co2.py`")
        st.stop()

    fig = px.line(
        df_co2, x="ts", y="co2_g_per_kwh",
        labels={"ts": "Tid", "co2_g_per_kwh": "g CO₂/kWh"},
        title="CO₂-intensitet – seneste 7 dage"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Seneste 24 timer**")
    st.dataframe(df_co2.tail(24), use_container_width=True)
