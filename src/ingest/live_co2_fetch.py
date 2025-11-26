#!/usr/bin/env python3
"""
Live CO2 Data Fetcher - Auto-Refresh Integration
Fetches latest CO2 data from Energinet API for real-time dashboard updates
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

def fetch_live_co2_data(zone: str = "DK1", days: int = 7) -> pd.DataFrame:
    """
    Fetch latest CO2 data from Energinet API
    
    Args:
        zone: DK1 or DK2
        days: Number of days to fetch (default 7 for last week)
    
    Returns:
        DataFrame with hourly CO2 data
    """
    
    EDS_URL = "https://api.energidataservice.dk/dataset/CO2Emis"
    
    # Fetch recent data
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": f'{{"PriceArea":["{zone}"]}}',
        "columns": "Minutes5DK,PriceArea,CO2Emission",
        "limit": 10000,
        "sort": "Minutes5DK desc",
    }
    
    try:
        r = requests.get(EDS_URL, params=params, timeout=30)
        
        if r.status_code == 200:
            data = r.json().get("records", [])
            
            if data:
                # Process data
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    "Minutes5DK": "ts",
                    "PriceArea": "area",
                    "CO2Emission": "co2_g_per_kwh",
                })
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                df["co2_g_per_kwh"] = pd.to_numeric(df["co2_g_per_kwh"], errors="coerce")
                df = df.dropna(subset=["co2_g_per_kwh"]).sort_values("ts")
                
                # Resample to hourly
                df_hourly = (
                    df.set_index("ts")
                    .resample("h")["co2_g_per_kwh"]
                    .mean()
                    .reset_index()
                )
                
                return df_hourly
            
    except Exception as e:
        print(f"Error fetching CO2 data for {zone}: {e}")
    
    # Return empty DataFrame if fetch fails
    return pd.DataFrame(columns=["ts", "co2_g_per_kwh"])


def get_co2_data_with_cache(zone: str = "DK1", max_age_minutes: int = 5) -> tuple:
    """
    Get CO2 data with intelligent caching
    
    - Fetches fresh data from API if cache is older than max_age_minutes
    - Uses cached file if fresh enough
    - Returns data + metadata about freshness
    
    Args:
        zone: DK1 or DK2
        max_age_minutes: Maximum age of cache before refresh (default 5)
    
    Returns:
        (DataFrame, dict with metadata)
    """
    
    cache_file = Path(f"data/processed/co2_hourly_{zone}.csv")
    
    # Check cache age
    needs_refresh = True
    cache_age_minutes = None
    
    if cache_file.exists():
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        cache_age = datetime.now() - cache_time
        cache_age_minutes = cache_age.total_seconds() / 60
        
        if cache_age_minutes < max_age_minutes:
            needs_refresh = False
    
    metadata = {
        "zone": zone,
        "cache_exists": cache_file.exists(),
        "cache_age_minutes": cache_age_minutes,
        "needs_refresh": needs_refresh,
        "last_updated": None,
        "data_source": None
    }
    
    # Fetch fresh data if needed
    if needs_refresh:
        print(f"🔄 Fetching fresh CO2 data for {zone} from API...")
        df = fetch_live_co2_data(zone, days=7)
        
        if not df.empty:
            # Save to cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
            
            metadata["data_source"] = "API (fresh)"
            metadata["last_updated"] = datetime.now()
            print(f"✅ Updated cache: {len(df)} hours of data")
        else:
            print(f"⚠️ API fetch failed, using cached data if available")
            if cache_file.exists():
                df = pd.read_csv(cache_file, parse_dates=["ts"])
                metadata["data_source"] = "Cache (API failed)"
            else:
                metadata["data_source"] = "None (no data)"
                return pd.DataFrame(), metadata
    else:
        # Use cached data
        df = pd.read_csv(cache_file, parse_dates=["ts"])
        metadata["data_source"] = "Cache (fresh)"
        metadata["last_updated"] = cache_time
        print(f"📦 Using cached data ({cache_age_minutes:.1f}m old)")
    
    return df, metadata


if __name__ == "__main__":
    # Test the functions
    print("Testing live CO2 data fetch...")
    
    for zone in ["DK1", "DK2"]:
        df, meta = get_co2_data_with_cache(zone, max_age_minutes=5)
        
        if not df.empty:
            print(f"\n✅ {zone}:")
            print(f"   Records: {len(df)}")
            print(f"   Date range: {df['ts'].min()} to {df['ts'].max()}")
            print(f"   Data source: {meta['data_source']}")
            print(f"   Cache age: {meta['cache_age_minutes']:.1f} minutes" if meta['cache_age_minutes'] else "   No cache")
        else:
            print(f"\n❌ {zone}: No data available")