#!/usr/bin/env python3
"""Fetch DK2 electricity prices from September 2025"""

import requests
import pandas as pd
from datetime import datetime, timezone
import os

print("Fetching DK2 prices from September 2025...")

# Use September 2025 since that's where data exists
start = datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)
end = datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)

params = {
    'start': start.strftime('%Y-%m-%dT%H:%M'),
    'end': end.strftime('%Y-%m-%dT%H:%M'),
    'filter': '{"PriceArea":["DK2"]}',
    'columns': 'HourUTC,PriceArea,SpotPriceEUR',
    'limit': 2000,
    'sort': 'HourUTC asc'
}

try:
    r = requests.get('https://api.energidataservice.dk/dataset/Elspotprices', params=params, timeout=30)
    
    if r.status_code == 200:
        data = r.json().get('records', [])
        print(f"✅ Received {len(data)} records for DK2")
        
        if data:
            df = pd.DataFrame(data)
            df = df.rename(columns={
                'HourUTC': 'ts',
                'PriceArea': 'area',
                'SpotPriceEUR': 'price_eur_mwh'
            })
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            df['price_eur_mwh'] = pd.to_numeric(df['price_eur_mwh'], errors='coerce')
            df = df.sort_values('ts')
            
            print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
            
            os.makedirs('data', exist_ok=True)
            
            df.to_csv('data/prices_DK2_hist.csv', index=False)
            print(f"💾 Saved to data/prices_DK2_hist.csv")
            
            df_last24 = df.tail(24).copy()
            df_last24.to_csv('data/DK2_price_forecast.csv', index=False)
            print(f"💾 Saved to data/DK2_price_forecast.csv")
            
            print("\n✅ SUCCESS! DK2 files created")
        else:
            print("❌ No data received")
    else:
        print(f"❌ API Error: {r.status_code}")
        
except Exception as e:
    print(f"❌ Error: {e}")