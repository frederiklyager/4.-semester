from __future__ import annotations

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.eval.schemas import validate_co2_raw, validate_co2_hourly, CO2_RAW_SCHEMA, CO2_HOURLY_SCHEMA

# Use built-in logging instead
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import os
from dataclasses import dataclass

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


@dataclass
class CO2IngestConfig:
    zone: str = "DK1"   # DK1 | DK2
    hourly_csv_tpl: str = "data/raw/co2_hourly_{zone}.csv"
    min5_csv_tpl: str   = "data/raw/co2_5min_{zone}.csv"


def _find_timestamp_col(df: pd.DataFrame) -> str:
    candidates = ["Minutes5UTC","Minutes5DK","HourUTC","HourDK","timestamp","ts","time"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "minute" in c.lower() or "hour" in c.lower() or "time" in c.lower():
            return c
    raise ValueError(f"Kunne ikke finde timestamp-kolonne i {list(df.columns)}")



def _find_value_col(df: pd.DataFrame) -> str:
    candidates = ["CO2Emission", "CO2_g_per_kWh", "co2_g_per_kwh", "co2", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) == 1:
        return numeric[0]
    raise ValueError("Kunne ikke finde CO₂-værdikolonne.")


def _normalize(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    tcol = _find_timestamp_col(df)
    vcol = _find_value_col(df)
    df = df.rename(columns={tcol: "ts", vcol: "co2_g_per_kwh"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["zone"] = zone
    df = df[["ts", "zone", "co2_g_per_kwh"]].sort_values("ts").reset_index(drop=True)
    
    # Validate with new comprehensive schema
    try:
        df_validate = df.rename(columns={"zone": "area"})
        df_validate = validate_co2_raw(df_validate, raise_on_error=True)
        df = df_validate.rename(columns={"area": "zone"})
        logger.info(f"✅ CO2 data validated: {len(df)} rows for zone {zone}")
    except Exception as e:
        logger.error(f"❌ CO2 validation failed for zone {zone}: {e}")
        raise
    
    return df


def _maybe_resample_to_hourly(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    # Hvis data er 5-min, så aggrégér til time-middel
    dt = (df["ts"].diff().dropna().dt.total_seconds().median() or 3600)
    if dt < 3600:  # 5 min ~ 300s
        # Store the timezone before resampling
        original_tz = df["ts"].dt.tz
        
        df = (
            df.set_index("ts")
              .resample("1h")  # 'H' deprecated → brug '1h'
              .mean(numeric_only=True)
              .dropna()
              .reset_index()
        )
        
        # Restore timezone if it was lost
        if original_tz is not None and df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(original_tz)
        elif original_tz is not None and df["ts"].dt.tz != original_tz:
            df["ts"] = df["ts"].dt.tz_convert(original_tz)
        
        # tilføj konstant zone-kolonne igen
        df["zone"] = zone
        df = df[["ts", "zone", "co2_g_per_kwh"]]
    return df



def ingest_from_csv(csv_path: str, zone: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)
    df = _normalize(df_raw, zone)
    return _maybe_resample_to_hourly(df, zone)


def ingest_from_api(zone: str, start: str, end: str, resolution: str = "hourly") -> pd.DataFrame:
    """
    Henter CO2-intensitet fra Energi Data Service (co2emis).
    Datoformat uden sekunder (yyyy-MM-ddTHH:mm). Ingen 'sort' eller 'columns' i params.
    Vi autodetekterer tidskolonnen (Minutes5* eller Hour*), og resampler til 1H.
    """
    import re
    base_candidates = [
        "https://api.energidataservice.dk/dataset/co2emis",   # korrekt endpoint (lowercase)
        "https://api.energidataservice.dk/dataset/CO2Emis",   # fallback
    ]

    def ensure_iso_minutes(ts: str) -> str:
        if ts and re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts):
            return ts + "T00:00"   # uden sekunder
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})(:\d{2})", ts or "")
        if m:
            return m.group(1)
        return ts

    start_iso = ensure_iso_minutes(start)
    end_iso   = ensure_iso_minutes(end)

    params = {
        "start": start_iso,
        "end": end_iso,
        "filter": f'{{"PriceArea":"{zone}"}}',
        "limit": 100000,
        "offset": 0,
        "timezone": "utc",
        # 🚫 ingen 'columns' her – EDS ændrer feltnavne per dataset/version
    }

    last_err = None
    for base in base_candidates:
        try:
            r = requests.get(base, params=params, timeout=60)
           
            if r.status_code >= 400:
                try:
                    msg = r.json().get("message")
                except Exception:
                    msg = None
                raise requests.HTTPError(f"{r.status_code} from {base} – {msg or r.text}", response=r)

            data = r.json().get("records", [])
            if not data:
                raise ValueError("Ingen data retur (tjek periode/zone).")
            df_raw = pd.DataFrame(data)

            # vælg tidskolonne: forsøger Minutes5*, så Hour*, så generisk
            if   "Minutes5UTC" in df_raw.columns: time_col = "Minutes5UTC"
            elif "Minutes5DK"  in df_raw.columns: time_col = "Minutes5DK"
            elif "HourUTC"     in df_raw.columns: time_col = "HourUTC"
            elif "HourDK"      in df_raw.columns: time_col = "HourDK"
            else:
                # fallback: find en tidskolonne heuristisk
                candidates = [c for c in df_raw.columns if any(k in c.lower() for k in ("minute","hour","time","timestamp"))]
                if not candidates:
                    raise ValueError(f"Kunne ikke finde tidskolonne i {list(df_raw.columns)}")
                time_col = candidates[0]

            # CO2-værdikolonnen (typisk CO2Emission)
            if "CO2Emission" in df_raw.columns:
                val_col = "CO2Emission"
            else:
                co2_candidates = [c for c in df_raw.columns if "co2" in c.lower()]
                if not co2_candidates:
                    raise ValueError(f"Kunne ikke finde CO2-kolonne i {list(df_raw.columns)}")
                val_col = co2_candidates[0]

            # normaliser til (ts, zone, co2_g_per_kwh)
            df = df_raw.rename(columns={time_col: "ts", val_col: "co2_g_per_kwh"})
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["ts"])
            df["zone"] = zone
            df = df[["ts","zone","co2_g_per_kwh"]].sort_values("ts").reset_index(drop=True)

            # hvis 5-min data → resample til time
            df = _maybe_resample_to_hourly(df, zone)
            
            # Ensure timestamps are timezone-aware (might be lost during resampling)
            if df['ts'].dt.tz is None:
                df['ts'] = df['ts'].dt.tz_localize('UTC')
                
            
            # Validate with new comprehensive schema
            try:
                # CO2_HOURLY_SCHEMA only expects 'ts' and 'co2_g_per_kwh'
                # So we validate without the 'zone' column, then add it back
                df_for_validation = df[['ts', 'co2_g_per_kwh']].copy()
                
                # Force timezone awareness (copy() can strip it)
                if df_for_validation['ts'].dt.tz is None:
                    df_for_validation['ts'] = df_for_validation['ts'].dt.tz_localize('UTC')

                df_validated = validate_co2_hourly(df_for_validation, raise_on_error=True)
                
                # Add zone back after validation
                df_validated['zone'] = zone
                df = df_validated[['ts', 'zone', 'co2_g_per_kwh']]
                logger.info(f"✅ CO2 hourly data validated: {len(df)} rows for zone {zone}")
            except Exception as e:
                logger.error(f"❌ CO2 hourly validation failed for zone {zone}: {e}")
                raise
            
            return df

        except Exception as e:
            last_err = e

    raise last_err


def save_processed(df: pd.DataFrame, zone: str) -> str:
    out = DATA_PROCESSED / f"co2_{zone}.parquet"
    df.to_parquet(out, index=False)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", default=os.environ.get("ZONE", "DK1"), choices=["DK1", "DK2"])
    parser.add_argument("--start", default=os.environ.get("START"))  # fx 2024-01-01
    parser.add_argument("--end",   default=os.environ.get("END"))    # fx 2025-01-01
    parser.add_argument("--resolution", default="hourly", choices=["hourly", "5min"])
    parser.add_argument("--source", default=os.environ.get("SOURCE", "csv"), choices=["csv", "api"])
    parser.add_argument("--csv", default=None)  # mulighed for at pege på specifik CSV
    args = parser.parse_args()

    if args.source == "api":
        if not args.start or not args.end:
            raise SystemExit("Angiv --start og --end, fx --start 2024-01-01 --end 2025-01-01")
        df = ingest_from_api(args.zone, args.start, args.end, args.resolution)
    else:
        csv = args.csv or f"data/raw/co2_{args.resolution}_{args.zone}.csv"
        df = ingest_from_csv(csv, args.zone)

    out = save_processed(df, args.zone)
    print(f"✅ Gemte {out}, rækker={len(df)}")
