#!/usr/bin/env python3
"""Test CO2 API to find available data in 2025"""

import requests
import pandas as pd
from datetime import datetime, timezone

print("Testing Energinet CO₂ API - 2025...")

EDS_URL = "https://api.energidataservice.dk/dataset/CO2Emis"

# Test multiple date ranges
test_ranges = [
    ("November 2025", datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 25, 0, 0, tzinfo=timezone.utc)),
    ("October 2025", datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)),
    ("September 2025", datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)),
    ("August 2025", datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)),
    ("Last 7 days", datetime(2025, 11, 18, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 25, 0, 0, tzinfo=timezone.utc)),
]

for name, start, end in test_ranges:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Range: {start.date()} to {end.date()}")
    
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": '{"PriceArea":["DK1"]}',
        "columns": "Minutes5DK,PriceArea,CO2Emission",
        "limit": 10000,
        "sort": "Minutes5DK desc",
    }
    
    try:
        r = requests.get(EDS_URL, params=params, timeout=30)
        
        if r.status_code == 200:
            data = r.json().get("records", [])
            print(f"✅ Status 200 - Records: {len(data)}")
            
            if data:
                print(f"   First (newest): {data[0]['Minutes5DK']} - {data[0]['CO2Emission']} g/kWh")
                print(f"   Last (oldest): {data[-1]['Minutes5DK']} - {data[-1]['CO2Emission']} g/kWh")
                
                # Found data - process it!
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    "Minutes5DK": "ts",
                    "PriceArea": "area",
                    "CO2Emission": "co2_g_per_kwh",
                })
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df["co2_g_per_kwh"] = pd.to_numeric(df["co2_g_per_kwh"], errors="coerce")
                df = df.dropna(subset=["co2_g_per_kwh"]).sort_values("ts")
                
                print(f"   Date range: {df['ts'].min()} to {df['ts'].max()}")
                print(f"   CO₂ range: {df['co2_g_per_kwh'].min():.1f} - {df['co2_g_per_kwh'].max():.1f} g/kWh")
                
                # Save 5-min data
                import os
                os.makedirs("data/raw", exist_ok=True)
                os.makedirs("data/processed", exist_ok=True)
                
                df.to_csv('data/raw/co2_5min_DK1.csv', index=False)
                print(f"   💾 Saved 5-min data: data/raw/co2_5min_DK1.csv")
                
                # Resample to hourly
                df_hourly = (
                    df.set_index("ts")
                    .resample("h")["co2_g_per_kwh"]
                    .mean()
                    .reset_index()
                )
                
                df_hourly.to_csv('data/processed/co2_hourly_DK1.csv', index=False)
                print(f"   💾 Saved hourly data: data/processed/co2_hourly_DK1.csv")
                print(f"   📊 Hourly records: {len(df_hourly)}")
                
                print(f"\n   🎉 SUCCESS! Found CO₂ data in '{name}'")
                break
            else:
                print(f"   ⚠️ No records in this range")
        else:
            print(f"   ❌ API Error: {r.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'='*60}")
print("Test complete!")
print("\nNext: Run similar script for DK2 if DK1 succeeded")