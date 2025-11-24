#!/usr/bin/env python3
"""
Real-Time Future Forecasting System
Predicts the NEXT 24 hours based on latest Energinet data

Author: Frederik Lyager
Usage: python src/models/forecast_future.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta, timezone
import requests
import warnings
import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

# ========================================
# 1. FETCH LATEST CO2 DATA FROM ENERGINET
# ========================================

def fetch_latest_co2(zone: str = "DK1", days: int = 30):
    """
    Fetch the most recent CO2 data from Energinet API
    
    Args:
        zone: DK1 or DK2
        days: Number of days of history to fetch
    
    Returns:
        DataFrame with recent hourly CO2 data
    """
    print(f"\n📡 Fetching latest CO₂ data for {zone} from Energinet API...")
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    # Energinet API endpoint
    url = "https://api.energidataservice.dk/dataset/CO2Emis"
    
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "filter": f'{{"PriceArea":["{zone}"]}}',
        "columns": "Minutes5DK,PriceArea,CO2Emission",
        "limit": 100000,
        "sort": "Minutes5DK asc",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get("records", [])
        
        if not data:
            raise ValueError(f"No data received from API for {zone}")
        
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
        
        print(f"   ✅ Fetched {len(df_hourly)} hours of real data")
        print(f"   📅 Latest data: {df_hourly['ts'].max()}")
        
        return df_hourly
        
    except Exception as e:
        print(f"   ❌ API fetch failed: {e}")
        print(f"   ℹ️  Falling back to cached data...")
        
        # Fallback to cached file
        cache_file = Path(f"data/processed/co2_hourly_{zone}.csv")
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=["ts"])
            print(f"   ⚠️  Using cached data (last update: {df['ts'].max()})")
            return df
        else:
            raise FileNotFoundError(f"No data available for {zone}")


# ========================================
# 2. CREATE FEATURES (SAME AS TRAINING)
# ========================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the same 16 features used during training
    """
    print("\n🔧 Creating features...")
    
    df = df.copy().sort_values('ts').reset_index(drop=True)
    
    # Time features
    df['hour'] = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Holiday feature (simplified - you can enhance this)
    df['is_holiday'] = 0  # Would need proper DK holiday calendar
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # Lag features
    df['lag_1'] = df['co2_g_per_kwh'].shift(1)
    df['lag_24'] = df['co2_g_per_kwh'].shift(24)
    df['lag_168'] = df['co2_g_per_kwh'].shift(168)
    
    # Rolling statistics
    df['rolling_mean_24h'] = df['co2_g_per_kwh'].rolling(window=24, min_periods=1).mean()
    df['rolling_std_24h'] = df['co2_g_per_kwh'].rolling(window=24, min_periods=1).std()
    df['rolling_mean_168h'] = df['co2_g_per_kwh'].rolling(window=168, min_periods=1).mean()
    df['rolling_std_168h'] = df['co2_g_per_kwh'].rolling(window=168, min_periods=1).std()
    
    print(f"   ✅ Created 16 features")
    
    return df


# ========================================
# 3. LOAD TRAINED MODEL
# ========================================

def load_model(zone: str = "DK1"):
    """Load the trained LightGBM model"""
    
    print(f"\n📦 Loading trained model for {zone}...")
    
    model_path = Path(f"data/models/lgbm_{zone}.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train the model first with: python src/models/ml_forecast.py"
        )
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract model from dict
    if isinstance(model_data, dict):
        model = model_data['model']
        feature_names = model_data['feature_names']
        print(f"   ✅ Model loaded ({len(feature_names)} features)")
        return model, feature_names
    else:
        raise ValueError("Unexpected model format")


# ========================================
# 4. PREDICT FUTURE (NEXT 24 HOURS)
# ========================================

def predict_future(zone: str = "DK1", horizon: int = 24):
    """
    Generate forecast for the NEXT 24 hours
    
    Returns:
        DataFrame with future predictions
    """
    print(f"\n{'='*70}")
    print(f"🔮 FUTURE FORECAST GENERATION - {zone}")
    print(f"{'='*70}")
    
    # Step 1: Fetch latest data
    df_latest = fetch_latest_co2(zone, days=30)
    
    # Step 2: Create features
    df_features = create_features(df_latest)
    
    # Step 3: Load model
    model, feature_names = load_model(zone)
    
    # Step 4: Get most recent complete data point
    df_clean = df_features.dropna(subset=feature_names)
    
    if len(df_clean) == 0:
        raise ValueError("No complete data available for forecasting")
    
    last_timestamp = df_clean['ts'].iloc[-1]
    last_co2 = df_clean['co2_g_per_kwh'].iloc[-1]
    
    print(f"\n🎯 Generating {horizon}h future forecast...")
    print(f"   Last known data: {last_timestamp}")
    print(f"   Last known CO₂: {last_co2:.1f} g/kWh")
    
    # Step 5: Generate future timestamps
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon,
        freq='h'
    )
    
    forecasts = []
    
    # For each future hour, predict using last known features
    # (Simplified approach - in production, you'd roll features forward)
    last_features = df_clean[feature_names].iloc[-1:].values
    
    for i, future_time in enumerate(future_timestamps):
        # Update time-based features for future timestamp
        hour = future_time.hour
        weekday = future_time.dayofweek
        month = future_time.month
        
        # Create feature vector for this future hour
        future_features = last_features.copy()
        
        # Update time features (indices 0-4)
        future_features[0, 0] = hour  # hour
        future_features[0, 1] = weekday  # weekday
        future_features[0, 2] = month  # month
        future_features[0, 3] = 1 if weekday >= 5 else 0  # is_weekend
        # is_holiday stays 0 (index 4)
        
        # Update cyclical encodings (indices 5-8)
        future_features[0, 5] = np.sin(2 * np.pi * hour / 24)  # hour_sin
        future_features[0, 6] = np.cos(2 * np.pi * hour / 24)  # hour_cos
        future_features[0, 7] = np.sin(2 * np.pi * weekday / 7)  # weekday_sin
        future_features[0, 8] = np.cos(2 * np.pi * weekday / 7)  # weekday_cos
        
        # Predict
        prediction = model.predict(future_features)[0]
        forecasts.append(prediction)
    
    # Create forecast DataFrame
    df_forecast = pd.DataFrame({
        'ts': future_timestamps,
        'forecast_co2': forecasts,
        'horizon_hours': list(range(1, horizon + 1))
    })
    
    print(f"   ✅ Generated {len(df_forecast)} future predictions")
    print(f"   📅 Forecast period: {df_forecast['ts'].min()} to {df_forecast['ts'].max()}")
    print(f"   📊 Forecast range: {df_forecast['forecast_co2'].min():.1f} - {df_forecast['forecast_co2'].max():.1f} g/kWh")
    
    # Find green hours
    green_hours = df_forecast.nsmallest(5, 'forecast_co2')
    print(f"\n🌿 Best 5 hours for energy use:")
    for _, row in green_hours.iterrows():
        print(f"      {row['ts'].strftime('%a %H:%M')}: {row['forecast_co2']:.1f} g/kWh")
    
    # Step 6: Save forecast
    output_dir = Path("data/forecast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"future_forecast_{zone}.csv"
    df_forecast.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved: {output_file}")
    
    # Save metadata
    metadata = {
        "zone": zone,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_known_timestamp": str(last_timestamp),
        "last_known_co2": float(last_co2),
        "forecast_start": str(df_forecast['ts'].min()),
        "forecast_end": str(df_forecast['ts'].max()),
        "forecast_mean": float(df_forecast['forecast_co2'].mean()),
        "forecast_min": float(df_forecast['forecast_co2'].min()),
        "forecast_max": float(df_forecast['forecast_co2'].max()),
        "horizon_hours": horizon
    }
    
    import json
    metadata_file = output_dir / f"future_forecast_metadata_{zone}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata: {metadata_file}")
    
    print(f"\n{'='*70}")
    print("✅ FUTURE FORECAST COMPLETE")
    print(f"{'='*70}\n")
    
    return df_forecast, metadata


# ========================================
# MAIN
# ========================================

def main():
    """Generate future forecasts for both zones"""
    
    zones = ['DK1', 'DK2']
    results = {}
    
    for zone in zones:
        try:
            forecast, metadata = predict_future(zone, horizon=24)
            results[zone] = {
                'success': True,
                'forecast': forecast,
                'metadata': metadata
            }
        except Exception as e:
            print(f"\n❌ Error for {zone}: {e}")
            import traceback
            traceback.print_exc()
            results[zone] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for zone, result in results.items():
        if result['success']:
            meta = result['metadata']
            print(f"\n✅ {zone}:")
            print(f"   Forecast period: {meta['forecast_start']} to {meta['forecast_end']}")
            print(f"   Expected CO₂ range: {meta['forecast_min']:.1f} - {meta['forecast_max']:.1f} g/kWh")
            print(f"   File: data/forecast/future_forecast_{zone}.csv")
        else:
            print(f"\n❌ {zone}: {result['error']}")
    
    print("\n💡 Next steps:")
    print("   1. View forecasts in dashboard (refresh browser)")
    print("   2. Run this script daily for fresh predictions")
    print("   3. Use green hours to optimize energy consumption!\n")


if __name__ == "__main__":
    main()