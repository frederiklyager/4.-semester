# src/app/dashboard.py
"""
Enhanced Energy Forecast Dashboard
Phase 3: Added Forecast visualization tab

Author: Frederik Lyager
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Energy Forecast – POC", layout="wide")
st.title("⚡ Energy Forecast – Proof of Concept")

# ------- Helper Functions -------
@st.cache_data
def load_csv(path: str, parse_dates=None) -> pd.DataFrame:
    """Load CSV with caching."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=parse_dates)


@st.cache_data
def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet with caching."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def plot_actual_vs_forecast(df: pd.DataFrame, title: str = "Actual vs Forecast"):
    """
    Create interactive plot comparing actual and forecast values.
    
    Args:
        df: DataFrame with columns ['ts', 'actual', 'forecast']
        title: Plot title
    """
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Actual</b><br>%{x}<br>%{y:.1f} g CO₂/kWh<extra></extra>'
    ))
    
    # Forecast values
    fig.add_trace(go.Scatter(
        x=df['ts'],
        y=df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>Forecast</b><br>%{x}<br>%{y:.1f} g CO₂/kWh<extra></extra>'
    ))
    
    # Calculate error
    df['error'] = df['actual'] - df['forecast']
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="CO₂ Intensity (g/kWh)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def calculate_quick_metrics(df: pd.DataFrame) -> dict:
    """Calculate quick metrics for display."""
    try:
        from src.eval.metrics import mae, rmse, mape
        
        y_true = df['actual'].values
        y_pred = df['forecast'].values
        
        return {
            'MAE': mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred)
        }
    except Exception as e:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.eval.metrics import mae, rmse, mape
        
        y_true = df['actual'].values
        y_pred = df['forecast'].values
        
        return {
            'MAE': mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred)
        }

# ---------- UI Tabs ----------
tab_prices, tab_co2, tab_forecast = st.tabs(["📈 Priser", "🌿 CO₂", "🎯 Forecast"])

# ============== PRISER TAB (Existing) ==============
with tab_prices:
    area = st.selectbox("Vælg område", ["DK1", "DK2"], index=0, key="price_area")

    hist_path = f"data/prices_{area}_hist.csv"
    fc_path = f"data/{area}_price_forecast.csv"

    df_hist = load_csv(hist_path, parse_dates=["ts"])
    df_fc = load_csv(fc_path, parse_dates=["ts"])

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


# ============== CO₂ TAB (Existing) ==============
with tab_co2:
    st.caption("Kilde: Energinet Energi Data Service – dataset 'CO2Emis'")
    
    co2_area = st.selectbox("Vælg område", ["DK1", "DK2"], index=0, key="co2_area")
    co2_file = f"data/processed/co2_hourly_{co2_area}.csv"
    df_co2 = load_csv(co2_file, parse_dates=["ts"])

    st.subheader(f"CO₂-intensitet ({co2_area}) – timegennemsnit")
    if df_co2.empty:
        st.error(f"Ingen CO₂-data fundet: {co2_file}")
        st.info("💡 Kør: `python src/ingest/energinet_co2.py`")
    else:
        fig = px.line(
            df_co2, x="ts", y="co2_g_per_kwh",
            labels={"ts": "Tid", "co2_g_per_kwh": "g CO₂/kWh"},
            title="CO₂-intensitet – seneste periode"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Seneste 24 timer**")
        st.dataframe(df_co2.tail(24), use_container_width=True)


# ============== FORECAST TAB (NEW) ==============
with tab_forecast:
    st.header("🎯 CO₂ Forecast Evaluation")
    st.caption("Baseline models - Phase 3")
    
    # Zone selector
    forecast_area = st.selectbox(
        "Vælg område", 
        ["DK1", "DK2"], 
        index=0, 
        key="forecast_area"
    )
    
    # Load forecast data
    forecast_path = f"data/forecast/co2_{forecast_area}_baseline.csv"
    df_forecast = load_csv(forecast_path, parse_dates=["ts"])
    
    if df_forecast.empty:
        st.warning(f"⚠️ Ingen forecast data fundet: {forecast_path}")
        st.info("""
        **Sådan genererer du forecast:**
        
        1. Kør feature engineering (Phase 2):
           ```bash
           python src/features/transform.py
           ```
        
        2. Kør baseline models (Phase 3):
           ```bash
           python src/models/baseline.py
           ```
        
        3. Refresh denne side
        """)
        st.stop()
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Data Points", f"{len(df_forecast):,}")
    with col2:
        date_range = f"{df_forecast['ts'].min().date()} til {df_forecast['ts'].max().date()}"
        st.metric("📅 Periode", date_range)
    with col3:
        hours = len(df_forecast)
        st.metric("⏱️ Timer", f"{hours}h ({hours/24:.1f} dage)")
    
    st.markdown("---")
    
    # Quick metrics
    st.subheader("📈 Performance Metrics")
    
    try:
        metrics = calculate_quick_metrics(df_forecast)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error (g CO₂/kWh)")
        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Squared Error (g CO₂/kWh)")
        with col3:
            st.metric("MAPE", f"{metrics['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
        with col4:
            avg_actual = df_forecast['actual'].mean()
            st.metric("Avg CO₂", f"{avg_actual:.1f}", help="Average actual CO₂ intensity")
    
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")
    
    st.markdown("---")
    
    # Time range selector
    st.subheader("🔍 Time Range Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_range = st.selectbox(
            "Hurtig visning",
            ["Hele perioden", "Seneste 7 dage", "Seneste 3 dage", "Seneste 24 timer", "Brugerdefineret"]
        )
    
    # Filter data based on selection
    df_plot = df_forecast.copy()
    
    if time_range == "Seneste 24 timer":
        cutoff = df_forecast['ts'].max() - timedelta(hours=24)
        df_plot = df_forecast[df_forecast['ts'] >= cutoff]
    elif time_range == "Seneste 3 dage":
        cutoff = df_forecast['ts'].max() - timedelta(days=3)
        df_plot = df_forecast[df_forecast['ts'] >= cutoff]
    elif time_range == "Seneste 7 dage":
        cutoff = df_forecast['ts'].max() - timedelta(days=7)
        df_plot = df_forecast[df_forecast['ts'] >= cutoff]
    elif time_range == "Brugerdefineret":
        with col2:
            start_date = st.date_input(
                "Start dato",
                value=df_forecast['ts'].min().date(),
                min_value=df_forecast['ts'].min().date(),
                max_value=df_forecast['ts'].max().date()
            )
            end_date = st.date_input(
                "Slut dato",
                value=df_forecast['ts'].max().date(),
                min_value=df_forecast['ts'].min().date(),
                max_value=df_forecast['ts'].max().date()
            )
        df_plot = df_forecast[
            (df_forecast['ts'].dt.date >= start_date) & 
            (df_forecast['ts'].dt.date <= end_date)
        ]
    
    # Main forecast plot
    st.subheader(f"📊 Actual vs Forecast - {forecast_area}")
    
    if not df_plot.empty:
        fig = plot_actual_vs_forecast(
            df_plot, 
            title=f"CO₂ Intensity Forecast ({time_range})"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        st.subheader("📉 Forecast Error Distribution")
        
        df_plot['error'] = df_plot['actual'] - df_plot['forecast']
        df_plot['abs_error'] = abs(df_plot['error'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error histogram
            fig_hist = px.histogram(
                df_plot, 
                x='error',
                nbins=30,
                title="Error Distribution",
                labels={'error': 'Forecast Error (g CO₂/kWh)'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Error over time
            fig_error = px.line(
                df_plot,
                x='ts',
                y='abs_error',
                title="Absolute Error Over Time",
                labels={'ts': 'Time', 'abs_error': 'Absolute Error (g CO₂/kWh)'}
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        # Error statistics
        st.subheader("📊 Error Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Error", f"{df_plot['error'].mean():.2f}")
        with col2:
            st.metric("Std Dev", f"{df_plot['error'].std():.2f}")
        with col3:
            st.metric("Max Error", f"{df_plot['abs_error'].max():.2f}")
        with col4:
            st.metric("Min Error", f"{df_plot['abs_error'].min():.2f}")
        
        # Data table (expandable)
        with st.expander("📋 Se rådata"):
            display_df = df_plot[['ts', 'actual', 'forecast', 'error', 'abs_error']].copy()
            display_df.columns = ['Tid', 'Faktisk', 'Forecast', 'Fejl', 'Abs Fejl']
            st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = df_plot.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv,
            file_name=f"forecast_{forecast_area}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("Ingen data i valgt tidsperiode")

# ============== FOOTER ==============
st.markdown("---")
st.caption(f"📅 Sidst opdateret: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 🎓 Datamatiker 4. semester | 👨‍💻 Frederik Lyager")