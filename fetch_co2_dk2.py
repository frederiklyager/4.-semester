#!/usr/bin/env python3
"""Fetch CO2 data for DK2 - November 2025"""

import requests
import pandas as pd
from datetime import datetime, timezone
import os

print("Fetching CO₂ data for DK2 - November 2025...")

EDS_URL = "https://api.energidataservice.dk/dataset/CO2Emis"

# Use November 2025 since we know it has data
start = datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 11, 25, 0, 0, tzinfo=timezone.utc)

print(f"Range: {start.date()} to {end.date()}")

params = {
    "start": start.strftime("%Y-%m-%dT%H:%M"),
    "end": end.strftime("%Y-%m-%dT%H:%M"),
    "filter": '{"PriceArea":["DK2"]}',
    "columns": "Minutes5DK,PriceArea,CO2Emission",
    "limit": 10000,
    "sort": "Minutes5DK asc",
}

try:
    r = requests.get(EDS_URL, params=params, timeout=30)
    
    if r.status_code == 200:
        data = r.json().get("records", [])
        print(f"✅ Received {len(data)} records")
        
        if data:
            df = pd.DataFrame(data)
            df = df.rename(columns={
                "Minutes5DK": "ts",
                "PriceArea": "area",
                "CO2Emission": "co2_g_per_kwh",
            })
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df["co2_g_per_kwh"] = pd.to_numeric(df["co2_g_per_kwh"], errors="coerce")
            df = df.dropna(subset=["co2_g_per_kwh"]).sort_values("ts")
            
            print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
            print(f"CO₂ range: {df['co2_g_per_kwh'].min():.1f} - {df['co2_g_per_kwh'].max():.1f} g/kWh")
            
            # Save 5-min data
            os.makedirs("data/raw", exist_ok=True)
            df.to_csv('data/raw/co2_5min_DK2.csv', index=False)
            print(f"💾 Saved: data/raw/co2_5min_DK2.csv")
            
            # Resample to hourly
            df_hourly = (
                df.set_index("ts")
                .resample("h")["co2_g_per_kwh"]
                .mean()
                .reset_index()
            )
            
            os.makedirs("data/processed", exist_ok=True)
            df_hourly.to_csv('data/processed/co2_hourly_DK2.csv', index=False)
            print(f"💾 Saved: data/processed/co2_hourly_DK2.csv ({len(df_hourly)} hours)")
            
            print("\n✅ SUCCESS! DK2 CO₂ data updated")
        else:
            print("❌ No data received")
    else:
        print(f"❌ API Error: {r.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()