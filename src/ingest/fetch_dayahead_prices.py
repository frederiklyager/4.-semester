#!/usr/bin/env python3
"""
Live Day-Ahead Price Fetcher
Fetches the latest published day-ahead electricity prices from Energinet

Prices are published daily at 13:00 CET (12:00 UTC) for the next day
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

EDS_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def fetch_dayahead_prices(area: str = "DK1"):
    """
    Fetch the latest published day-ahead prices
    
    Args:
        area: DK1 or DK2
    
    Returns:
        DataFrame with day-ahead prices
    """
    print(f"\n{'='*70}")
    print(f"📊 FETCHING DAY-AHEAD PRICES - {area}")
    print(f"{'='*70}\n")
    
    now = datetime.now(timezone.utc)
    
    # Get prices from yesterday to tomorrow (includes latest day-ahead)
    start = now - timedelta(days=1)
    end = now + timedelta(days=2)
    
    print(f"📡 Fetching from Energinet API...")
    print(f"   Time range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": f'{{"PriceArea":["{area}"]}}',
        "columns": "HourUTC,PriceArea,SpotPriceEUR",
        "limit": 100000,
        "sort": "HourUTC asc",
    }
    
    try:
        response = requests.get(EDS_URL, params=params, timeout=30)
        response.raise_for_status()
        records = response.json().get("records", [])
        
        if not records:
            raise ValueError(f"No price data received for {area}")
        
        df = pd.DataFrame(records)
        df = df.rename(columns={
            "HourUTC": "ts",
            "PriceArea": "area",
            "SpotPriceEUR": "price_eur_mwh",
        })
        
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
        df = df.dropna(subset=["price_eur_mwh"]).sort_values("ts")
        
        print(f"   ✅ Received {len(df)} hourly prices")
        print(f"   📅 Date range: {df['ts'].min()} to {df['ts'].max()}")
        
        # Split into historical and day-ahead
        cutoff = now.replace(minute=0, second=0, microsecond=0)
        
        df_historical = df[df['ts'] < cutoff].copy()
        df_dayahead = df[df['ts'] >= cutoff].copy()
        
        print(f"\n📊 Price Statistics:")
        print(f"   Historical prices: {len(df_historical)} hours")
        print(f"   Day-ahead prices: {len(df_dayahead)} hours")
        
        if len(df_dayahead) > 0:
            print(f"\n💰 Day-Ahead Price Range:")
            print(f"   Lowest:  {df_dayahead['price_eur_mwh'].min():.2f} EUR/MWh")
            print(f"   Highest: {df_dayahead['price_eur_mwh'].max():.2f} EUR/MWh")
            print(f"   Average: {df_dayahead['price_eur_mwh'].mean():.2f} EUR/MWh")
            
            # Find cheapest hours
            cheapest = df_dayahead.nsmallest(5, 'price_eur_mwh')
            print(f"\n💚 5 Cheapest Hours:")
            for _, row in cheapest.iterrows():
                print(f"      {row['ts'].strftime('%a %H:%M')}: {row['price_eur_mwh']:.2f} EUR/MWh")
        else:
            print(f"\n⚠️  No day-ahead prices available yet")
            print(f"   Prices are published at 13:00 CET (12:00 UTC)")
            print(f"   Current time: {now.strftime('%H:%M UTC')}")
        
        return df_historical, df_dayahead
        
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        raise


def save_prices(df_historical, df_dayahead, area: str = "DK1"):
    """Save historical and day-ahead prices to separate files"""
    
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving price data...")
    
    # Save historical prices
    if not df_historical.empty:
        hist_file = output_dir / f"prices_{area}_hist.csv"
        df_historical.to_csv(hist_file, index=False)
        print(f"   ✅ Historical: {hist_file} ({len(df_historical)} rows)")
    
    # Save day-ahead forecast
    if not df_dayahead.empty:
        da_file = output_dir / f"{area}_price_forecast.csv"
        df_dayahead.to_csv(da_file, index=False)
        print(f"   ✅ Day-ahead: {da_file} ({len(df_dayahead)} rows)")
        
        # Save metadata
        metadata = {
            "area": area,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "forecast_start": str(df_dayahead['ts'].min()),
            "forecast_end": str(df_dayahead['ts'].max()),
            "num_hours": len(df_dayahead),
            "price_min": float(df_dayahead['price_eur_mwh'].min()),
            "price_max": float(df_dayahead['price_eur_mwh'].max()),
            "price_avg": float(df_dayahead['price_eur_mwh'].mean()),
        }
        
        meta_file = output_dir / f"{area}_price_forecast_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Metadata: {meta_file}")
    else:
        print(f"   ⚠️  No day-ahead prices to save")


def main():
    """Fetch day-ahead prices for both zones"""
    
    zones = ['DK1', 'DK2']
    results = {}
    
    for zone in zones:
        try:
            df_hist, df_da = fetch_dayahead_prices(zone)
            save_prices(df_hist, df_da, zone)
            results[zone] = {
                'success': True,
                'historical_count': len(df_hist),
                'dayahead_count': len(df_da)
            }
        except Exception as e:
            print(f"\n❌ Error for {zone}: {e}")
            results[zone] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    
    for zone, result in results.items():
        if result['success']:
            print(f"\n✅ {zone}:")
            print(f"   Historical prices: {result['historical_count']} hours")
            print(f"   Day-ahead forecast: {result['dayahead_count']} hours")
        else:
            print(f"\n❌ {zone}: {result['error']}")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Refresh dashboard to see updated prices")
    print(f"   2. Run this script daily at 13:00 CET for fresh day-ahead prices")
    print(f"   3. Or add auto-refresh button to dashboard")
    print()


if __name__ == "__main__":
    main()