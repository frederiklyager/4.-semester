#!/usr/bin/env python3
"""Simple test to fetch electricity prices from Energinet API"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

print("Testing Energinet API connection...")

now = datetime.now(timezone.utc)
start = now - timedelta(days=7)
end = now

print(f"Fetching prices from {start.date()} to {end.date()}")

params = {
    'start': start.strftime('%Y-%m-%dT%H:%M'),
    'end': end.strftime('%Y-%m-%dT%H:%M'),
    'filter': '{"PriceArea":["DK1"]}',
    'columns': 'HourUTC,PriceArea,SpotPriceEUR',
    'limit': 1000,
    'sort': 'HourUTC asc'
}

try:
    r = requests.get('https://api.energidataservice.dk/dataset/Elspotprices', params=params, timeout=30)
    print(f"API Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json().get('records', [])
        print(f"Records received: {len(data)}")
        
        if data:
            print("\nFirst record:")
            print(data[0])
            
            # Create DataFrame
            df = pd.DataFrame(data)
            df = df.rename(columns={
                'HourUTC': 'ts',
                'PriceArea': 'area',
                'SpotPriceEUR': 'price_eur_mwh'
            })
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            df['price_eur_mwh'] = pd.to_numeric(df['price_eur_mwh'], errors='coerce')
            
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            # Save test file
            import os
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/prices_DK1_hist.csv', index=False)
            print("\n✅ Saved to data/prices_DK1_hist.csv")
            
        else:
            print("❌ No records in response")
    else:
        print(f"❌ API Error: {r.text[:200]}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()