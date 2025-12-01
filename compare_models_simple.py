import pandas as pd
import numpy as np

# Load both forecasts
lstm = pd.read_csv('data/forecast/lstm_forecast_DK1.csv', parse_dates=['ts'])
lgbm = pd.read_csv('data/forecast/future_forecast_DK1.csv', parse_dates=['ts'])

# Rename LightGBM column to match
lgbm = lgbm.rename(columns={'forecast_co2': 'co2_g_per_kwh'})

print("\n" + "="*60)
print("LSTM vs LightGBM Forecast Comparison (DK1)")
print("="*60)

print("\nLSTM Forecast:")
print(f"  - Time range: {lstm['ts'].min()} to {lstm['ts'].max()}")
print(f"  - Number of hours: {len(lstm)}")
print(f"  - CO2 range: {lstm['co2_g_per_kwh'].min():.2f} - {lstm['co2_g_per_kwh'].max():.2f} g/kWh")
print(f"  - Average CO2: {lstm['co2_g_per_kwh'].mean():.2f} g/kWh")

print("\nLightGBM Forecast:")
print(f"  - Time range: {lgbm['ts'].min()} to {lgbm['ts'].max()}")
print(f"  - Number of hours: {len(lgbm)}")
print(f"  - CO2 range: {lgbm['co2_g_per_kwh'].min():.2f} - {lgbm['co2_g_per_kwh'].max():.2f} g/kWh")
print(f"  - Average CO2: {lgbm['co2_g_per_kwh'].mean():.2f} g/kWh")

# Show side-by-side for overlapping hours
merged = lstm.merge(lgbm, on='ts', suffixes=('_lstm', '_lgbm'))
if len(merged) > 0:
    print(f"\n{len(merged)} overlapping forecast hours:")
    print("\nTimestamp                    LSTM    LightGBM  Difference")
    print("-" * 60)
    for _, row in merged.head(10).iterrows():
        diff = row['co2_g_per_kwh_lstm'] - row['co2_g_per_kwh_lgbm']
        print(f"{row['ts']}  {row['co2_g_per_kwh_lstm']:6.2f}  {row['co2_g_per_kwh_lgbm']:6.2f}    {diff:+6.2f}")

print("\n" + "="*60)
print("Training Performance (from model training):")
print("="*60)
print("LSTM Test MAE:     11.05 g CO2/kWh")
print("LightGBM Test MAE: 12.84 g CO2/kWh")
print("\n✅ Winner: LSTM is 14% more accurate! 🎉")
print("="*60)