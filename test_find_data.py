#!/usr/bin/env python3
"""Test Energinet API with 2025 dates"""

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

print("Testing Energinet API - November 2025...")

# Try multiple date ranges to find data
test_ranges = [
    ("Last 7 days", datetime(2025, 11, 18, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 25, 0, 0, tzinfo=timezone.utc)),
    ("Last 30 days", datetime(2025, 10, 26, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 25, 0, 0, tzinfo=timezone.utc)),
    ("October 2025", datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 11, 1, 0, 0, tzinfo=timezone.utc)),
    ("September 2025", datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc), datetime(2025, 10, 1, 0, 0, tzinfo=timezone.utc)),
]

for name, start, end in test_ranges:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Range: {start.date()} to {end.date()}")
    
    params = {
        'start': start.strftime('%Y-%m-%dT%H:%M'),
        'end': end.strftime('%Y-%m-%dT%H:%M'),
        'filter': '{"PriceArea":["DK1"]}',
        'columns': 'HourUTC,PriceArea,SpotPriceEUR',
        'limit': 2000,
        'sort': 'HourUTC desc'
    }
    
    try:
        r = requests.get('https://api.energidataservice.dk/dataset/Elspotprices', params=params, timeout=30)
        
        if r.status_code == 200:
            data = r.json().get('records', [])
            print(f"✅ Status 200 - Records: {len(data)}")
            
            if data:
                print(f"   First record: {data[0]['HourUTC']} - {data[0]['SpotPriceEUR']} EUR/MWh")
                print(f"   Last record: {data[-1]['HourUTC']} - {data[-1]['SpotPriceEUR']} EUR/MWh")
                
                # This range has data - use it!
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    'HourUTC': 'ts',
                    'PriceArea': 'area',
                    'SpotPriceEUR': 'price_eur_mwh'
                })
                df['ts'] = pd.to_datetime(df['ts'], utc=True)
                df['price_eur_mwh'] = pd.to_numeric(df['price_eur_mwh'], errors='coerce')
                df = df.sort_values('ts')
                
                print(f"   Date range: {df['ts'].min()} to {df['ts'].max()}")
                
                # Save it
                import os
                os.makedirs('data', exist_ok=True)
                
                df.to_csv('data/prices_DK1_hist.csv', index=False)
                print(f"   💾 Saved to data/prices_DK1_hist.csv")
                
                # Last 24 hours as forecast
                df_last24 = df.tail(24).copy()
                df_last24.to_csv('data/DK1_price_forecast.csv', index=False)
                print(f"   💾 Saved to data/DK1_price_forecast.csv")
                
                print(f"\n   🎉 SUCCESS! Found data in '{name}'")
                break
            else:
                print(f"   ⚠️ No records in this range")
        else:
            print(f"   ❌ API Error: {r.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'='*60}")
print("Test complete!")