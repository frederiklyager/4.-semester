# src/ingest/energinet_co2.py
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

EDS_URL = "https://api.energidataservice.dk/dataset/CO2Emis"

def fetch_co2(start: datetime, end: datetime, area: str = "DK1") -> pd.DataFrame:
    """Hent 5-min CO2-intensitet for et tidsrum og område (DK1/DK2)."""
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end":   end.strftime("%Y-%m-%dT%H:%M"),
        "filter": f'{{"PriceArea":["{area}"]}}',
        "columns": "Minutes5DK,PriceArea,CO2Emission",
        "limit": 100000,
        "sort": "Minutes5DK asc",
    }
    r = requests.get(EDS_URL, params=params, timeout=30)
    print("GET:", r.url)  # debug så vi kan se det er CO2Emis
    r.raise_for_status()
    data = r.json().get("records", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df.rename(columns={
        "Minutes5DK": "ts",
        "PriceArea": "area",
        "CO2Emission": "co2_g_per_kwh",
    })
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["co2_g_per_kwh"] = pd.to_numeric(df["co2_g_per_kwh"], errors="coerce")
    df = df.dropna(subset=["co2_g_per_kwh"]).sort_values("ts")
    return df

def main(days: int = 7, area: str = "DK1"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    df5 = fetch_co2(start, end, area)
    if df5.empty:
        print("Ingen CO2-data modtaget.")
        return

    # Gem rå 5-min data
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    raw_path = f"data/raw/co2_5min_{area}.csv"
    df5.to_csv(raw_path, index=False)
    print(f"✅ Gemte {len(df5):,} rækker til {raw_path}")

    # Resample til time
    dfh = (
        df5.set_index("ts")
           .resample("h")["co2_g_per_kwh"]  # brug 'h' for at undgå deprecated warning
           .mean()
           .reset_index()
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    proc_path = f"data/processed/co2_hourly_{area}.csv"
    dfh.to_csv(proc_path, index=False)
    print(f"✅ Gemte {len(dfh):,} timer til {proc_path}")

if __name__ == "__main__":
    main()
