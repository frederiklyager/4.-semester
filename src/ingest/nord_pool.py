from __future__ import annotations

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.schemas import validate_price, PRICE_SCHEMA

# Use built-in logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

EDS_URL = "https://api.energidataservice.dk/dataset/Elspotprices"

def fetch_elspot(start: datetime, end: datetime, area: str) -> pd.DataFrame:
    """Hent elspot (day-ahead) priser i EUR/MWh for DK1/DK2 fra Energinet EDS."""
    print(f"Fetching data: {start} to {end}")
    params = {
        "start":   start.strftime("%Y-%m-%dT%H:%M"),
        "end":     end.strftime("%Y-%m-%dT%H:%M"),
        "filter":  f'{{"PriceArea":["{area}"]}}',
        "columns": "HourUTC,PriceArea,SpotPriceEUR",
        "limit":   100000,
        "sort":    "HourUTC asc",
    }
    r = requests.get(EDS_URL, params=params, timeout=30)
    r.raise_for_status()
    recs = r.json().get("records", [])
    
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.rename(columns={
        "HourUTC": "ts",
        "PriceArea": "area",
        "SpotPriceEUR": "price_eur_mwh",
    })
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")
    df = df.dropna(subset=["price_eur_mwh"]).sort_values("ts")
    
    # Validate with schema for security
    try:
        df = validate_price(df, raise_on_error=True)
        logger.info(f"✅ Price data validated: {len(df)} rows for area {area}")
    except Exception as e:
        logger.error(f"❌ Price validation failed for area {area}: {e}")
        raise
    
    return df

def main(area: str = "DK1"):
    print(f"\n=== Henter {area} ===")
    now = datetime.now(timezone.utc)
    print(f"Current time: {now}")

    # 1) Historical (last 30 days)
    start_hist = now - timedelta(days=30)
    end_hist = now
    df_hist = fetch_elspot(start_hist, end_hist, area)

    # 2) Day-ahead (today + next 2 days to get tomorrow's prices)
    start_da = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_da = start_da + timedelta(days=2)
    df_da = fetch_elspot(start_da, end_da, area)
    
    # Only keep future prices
    df_da = df_da[df_da["ts"] >= now]

    # 3) Save files
    os.makedirs("data", exist_ok=True)

    if not df_hist.empty:
        df_hist.to_csv(f"data/prices_{area}_hist.csv", index=False)
        print(f"✅ Gemte historik → data/prices_{area}_hist.csv  ({len(df_hist)} rækker)")
    else:
        print("⚠️ Ingen historik modtaget.")

    if not df_da.empty:
        df_da.to_csv(f"data/{area}_price_forecast.csv", index=False)
        print(f"✅ Gemte day-ahead → data/{area}_price_forecast.csv  ({len(df_da)} rækker)")
    else:
        print("⚠️ Ingen day-ahead publiceret endnu (prøv igen senere).")
