# src/ingest/nord_pool.py
"""
Fetch day-ahead electricity prices from Energinet Energy Data Service
This mirrors official Nord Pool data and is the standard for Danish energy projects
"""
from __future__ import annotations

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.schemas import validate_price

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

EDS_URL = "https://api.energidataservice.dk/dataset/Elspotprices"

def fetch_elspot(start: datetime, end: datetime, area: str) -> pd.DataFrame:
    """Fetch day-ahead electricity prices from Energinet for DK1/DK2"""
    logger.info(f"Fetching Energinet data for {area}: {start.date()} to {end.date()}")
    
    params = {
        "start":   start.strftime("%Y-%m-%dT%H:%M"),
        "end":     end.strftime("%Y-%m-%dT%H:%M"),
        "filter":  f'{{"PriceArea":["{area}"]}}',
        "columns": "HourUTC,PriceArea,SpotPriceEUR",
        "limit":   100000,
        "sort":    "HourUTC asc",
    }
    
    try:
        r = requests.get(EDS_URL, params=params, timeout=30)
        r.raise_for_status()
        recs = r.json().get("records", [])
        
        if not recs:
            logger.warning(f"No records returned for {area}")
            return pd.DataFrame()
        
        df = pd.DataFrame(recs)
        df = df.rename(columns={
            "HourUTC": "ts",
            "PriceArea": "area",
            "SpotPriceEUR": "price_eur_mwh",
        })
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
        df = df.dropna(subset=["price_eur_mwh"]).sort_values("ts")
        
        logger.info(f" Retrieved {len(df)} price records for {area}")
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f" API request failed: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f" Error processing data: {e}")
        return pd.DataFrame()


def main(area: str = "DK1"):
    """
    Fetch electricity prices for specified area
    - Historical: Last 30 days
    - Day-ahead: Next 24-48 hours (if published)
    """
    print(f"\n{'='*60}")
    print(f"Fetching Electricity Prices for {area}")
    print(f"{'='*60}")
    
    now = datetime.now(timezone.utc)
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M UTC')}")

    # 1) Historical prices (last 30 days)
    print("\nFetching historical prices (last 30 days)...")
    start_hist = now - timedelta(days=30)
    end_hist = now
    df_hist = fetch_elspot(start_hist, end_hist, area)

    # 2) Day-ahead prices (published daily at 13:00 CET)
    print("\nFetching day-ahead forecast...")
    start_da = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_da = start_da + timedelta(days=2)  # Today + tomorrow
    df_da = fetch_elspot(start_da, end_da, area)
    
    # Only keep future prices
    if not df_da.empty:
        df_da = df_da[df_da["ts"] >= now]

    # 3) Save files
    os.makedirs("data", exist_ok=True)

    if not df_hist.empty:
        try:
            df_hist = validate_price(df_hist, raise_on_error=True)
            df_hist.to_csv(f"data/prices_{area}_hist.csv", index=False)
            
            latest_hist = df_hist['ts'].max()
            hours_old = (now - latest_hist).total_seconds() / 3600
            
            print(f"✅ Saved historical → data/prices_{area}_hist.csv")
            print(f"   Records: {len(df_hist)}")
            print(f"   Date range: {df_hist['ts'].min().date()} to {df_hist['ts'].max().date()}")
            print(f"   ⏰ Freshness: Latest data is {hours_old:.1f} hours old")
        except Exception as e:
            print(f" Historical validation failed: {e}")
    else:
        print(" No historical data retrieved")

    if not df_da.empty:
        try:
            df_da = validate_price(df_da, raise_on_error=True)
            df_da.to_csv(f"data/{area}_price_forecast.csv", index=False)
            
            latest_da = df_da['ts'].max()
            hours_ahead = (latest_da - now).total_seconds() / 3600
            
            print(f" Saved day-ahead → data/{area}_price_forecast.csv")
            print(f"   Records: {len(df_da)}")
            print(f"   Date range: {df_da['ts'].min()} to {df_da['ts'].max()}")
            print(f"    Coverage: {hours_ahead:.1f} hours ahead")
        except Exception as e:
            print(f" Day-ahead validation failed: {e}")
    else:
        print(" No day-ahead prices available")
        print("    Day-ahead prices are published at 13:00 CET (12:00 UTC)")
        print("    Try running this script after 13:00 CET")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Fetch for both Danish areas
    for area in ("DK1", "DK2"):
        main(area)