# src/ingest/nordpool_dayahead.py
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

EDS_URL = "https://api.energidataservice.dk/dataset/Elspotprices"

def fetch_elspot(start: datetime, end: datetime, area: str) -> pd.DataFrame:
    """Hent elspot (day-ahead) priser i EUR/MWh for DK1/DK2 fra Energinet EDS."""
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
    return df

def main(area: str = "DK1"):
    now = datetime.now(timezone.utc)

    # 1) Historik (fx 30 dage tilbage)
    start_hist = now - timedelta(days=30)
    end_hist   = now
    df_hist = fetch_elspot(start_hist, end_hist, area)

    # 2) Day-ahead (næste 24 timer hvis publiceret; ellers tom)
    start_da = now.replace(minute=0, second=0, microsecond=0)
    end_da   = start_da + timedelta(days=2)  # hent i morgen også (hvis frigivet)
    df_da = fetch_elspot(start_da, end_da, area)
    df_da = df_da[df_da["ts"] >= start_da]   # kun fra nu og frem

    # 3) Gem i samme filnavne, som dashboardet allerede læser
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

if __name__ == "__main__":
    # Kør for begge områder, så dit eksisterende dashboard virker uændret
    for a in ("DK1", "DK2"):
        print(f"\n=== Henter {a} ===")
        main(a)
