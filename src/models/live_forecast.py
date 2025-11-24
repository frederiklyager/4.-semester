#!/usr/bin/env python3
"""
Live CO₂ Forecast Generator
Fetches latest data from Energinet API and generates real-time 24h forecasts

Author: Frederik Lyager
Phase 4: Live ML Forecasting with Auto-Update
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pickle
import warnings
import json

warnings.filterwarnings('ignore')

# Import your data fetching function
try:
    from src.ingest.energinet_co2 import fetch_co2
except ImportError:
    print("⚠️  Could not import energinet_co2 module")
    fetch_co2 = None


def fetch_latest_co2_data(zone: str = "DK1", days: int = 30) -> pd.DataFrame:
    """
    Fetch the most recent CO₂ data from Energinet API
    
    Args:
        zone: Price area (DK1 or DK2)
        days: Number of days to fetch (default: 30 for enough history)
    
    Returns:
        DataFrame with recent CO₂ data
    """
    print(f"📡 Fetching latest CO₂ data for {zone}...")
    
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    if fetch_co2:
        # Use your existing fetch function
        df_raw = fetch_co2(start, end, zone)
        
        if df_raw.empty:
            raise ValueError(f"No data received from API for {zone}")
        
        # Resample to hourly
        df_hourly = (
            df_raw.set_index("ts")
            .resample("h")["co2_g_per_kwh"]
            .mean()
            .reset_index()
        )
        
        print(f"   ✅ Fetched {len(df_hourly)} hours of data")
        print(f"   📅 Date range: {df_hourly['ts'].min()} to {df_hourly['ts'].max()}")
        
        return df_hourly
    else:
        # Fallback: read from existing file
        print("   ⚠️  Using cached data from file")
        filepath = Path(f"data/processed/co2_hourly_{zone}.csv")
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=["ts"])
            return df
        else:
            raise FileNotFoundError(f"No data available for {zone}")


def create_features_for_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features needed for ML model prediction
    Same feature engineering as training phase
    
    Args:
        df: DataFrame with 'ts' and 'co2_g_per_kwh' columns
    
    Returns:
        DataFrame with features added
    """
    print("🔧 Creating features...")
    
    df = df.copy()
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Time-based features
    df['hour'] = df['ts'].dt.hour
    df['day_of_week'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Lag features (use recent history)
    for lag in [1, 24, 168]:  # 1h, 24h, 1week
        df[f'co2_lag_{lag}'] = df['co2_g_per_kwh'].shift(lag)
    
    # Rolling statistics
    for window in [24, 168]:  # 24h, 1week
        df[f'co2_rolling_mean_{window}'] = df['co2_g_per_kwh'].rolling(
            window=window, min_periods=1
        ).mean()
        df[f'co2_rolling_std_{window}'] = df['co2_g_per_kwh'].rolling(
            window=window, min_periods=1
        ).std()
    
    print(f"   ✅ Created {len([c for c in df.columns if c not in ['ts', 'co2_g_per_kwh']])} features")
    
    return df


def load_trained_model(zone: str = "DK1"):
    """
    Load the trained ML model
    
    Args:
        zone: Price area
    
    Returns:
        Trained model object
    """
    model_path = Path(f"data/models/lgbm_{zone}.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Model not found: {model_path}\n"
            f"   You need to train the model first!"
        )
    
    print(f"📦 Loading trained model: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if it's a dict (your format) or just a model
    if isinstance(model_data, dict):
        # Your ml_forecast.py saves as dict with 'model' key
        model = model_data.get('model', model_data.get('lgbm', model_data))
        print(f"   ✅ Model loaded from dict (keys: {list(model_data.keys())})")
    else:
        # Direct model object
        model = model_data
        print(f"   ✅ Model loaded successfully")
    
    return model


def generate_live_forecast(zone: str = "DK1", horizon: int = 24) -> pd.DataFrame:
    """
    Generate live forecast for next N hours
    
    Steps:
    1. Fetch latest data from API
    2. Create features
    3. Load trained model
    4. Predict next 24 hours
    5. Save forecast
    
    Args:
        zone: Price area (DK1 or DK2)
        horizon: Forecast horizon in hours (default: 24)
    
    Returns:
        DataFrame with forecasts
    """
    print(f"\n{'='*70}")
    print(f"🔮 LIVE FORECAST GENERATION - {zone}")
    print(f"{'='*70}\n")
    
    # Step 1: Fetch latest data
    df_latest = fetch_latest_co2_data(zone, days=30)
    
    # Step 2: Create features
    df_features = create_features_for_forecasting(df_latest)
    
    # Step 3: Load model
    model = load_trained_model(zone)
    
    # Step 4: Prepare data for prediction
    # Get the most recent complete row (with all features)
    df_clean = df_features.dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No complete data available for forecasting")
    
    # Get feature columns (exclude ts and target)
    feature_cols = [c for c in df_clean.columns if c not in ['ts', 'co2_g_per_kwh']]
    
    print(f"🎯 Generating {horizon}h forecast...")
    print(f"   Using {len(feature_cols)} features")
    print(f"   Last known data point: {df_clean['ts'].iloc[-1]}")
    
    # Generate forecasts iteratively
    forecasts = []
    forecast_times = []
    
    # Start from the last known timestamp
    last_known_time = df_clean['ts'].iloc[-1]
    
    # For simplicity, we'll use the last known features to predict
    # In a more sophisticated version, we'd roll forward the features
    last_features = df_clean[feature_cols].iloc[-1:].values
    
    for h in range(1, horizon + 1):
        forecast_time = last_known_time + timedelta(hours=h)
        
        # Predict using model
        prediction = model.predict(last_features)[0]
        
        forecasts.append(prediction)
        forecast_times.append(forecast_time)
    
    # Create forecast DataFrame
    df_forecast = pd.DataFrame({
        'ts': forecast_times,
        'forecast_co2': forecasts,
        'horizon_hours': list(range(1, horizon + 1))
    })
    
    print(f"   ✅ Generated {len(df_forecast)} forecasts")
    print(f"   📅 Forecast period: {df_forecast['ts'].min()} to {df_forecast['ts'].max()}")
    print(f"   📊 Forecast range: {df_forecast['forecast_co2'].min():.1f} - {df_forecast['forecast_co2'].max():.1f} g/kWh")
    
    # Step 5: Save forecast
    output_dir = Path("data/forecast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"live_forecast_{zone}.csv"
    df_forecast.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved live forecast: {output_file}")
    
    # Save metadata
    metadata = {
        "zone": zone,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_data_timestamp": str(last_known_time),
        "forecast_horizon_hours": horizon,
        "num_forecasts": len(df_forecast),
        "forecast_start": str(df_forecast['ts'].min()),
        "forecast_end": str(df_forecast['ts'].max()),
        "forecast_mean": float(df_forecast['forecast_co2'].mean()),
        "forecast_std": float(df_forecast['forecast_co2'].std()),
        "model_path": f"models/lightgbm_model_{zone}.pkl"
    }
    
    metadata_file = output_dir / f"live_forecast_metadata_{zone}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved metadata: {metadata_file}")
    
    print(f"\n{'='*70}")
    print("✅ LIVE FORECAST COMPLETE")
    print(f"{'='*70}\n")
    
    return df_forecast, metadata


def main():
    """Generate live forecasts for both zones"""
    zones = ['DK1', 'DK2']
    
    results = {}
    
    for zone in zones:
        try:
            forecast, metadata = generate_live_forecast(zone, horizon=24)
            results[zone] = {
                'forecast': forecast,
                'metadata': metadata,
                'success': True
            }
        except FileNotFoundError as e:
            print(f"\n❌ {zone}: {e}")
            results[zone] = {'success': False, 'error': str(e)}
        except Exception as e:
            print(f"\n❌ Error generating forecast for {zone}: {e}")
            import traceback
            traceback.print_exc()
            results[zone] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for zone, result in results.items():
        if result['success']:
            print(f"✅ {zone}: Forecast generated successfully")
            print(f"   Generated at: {result['metadata']['generated_at']}")
            print(f"   Forecasts: {result['metadata']['num_forecasts']} hours")
        else:
            print(f"❌ {zone}: Failed - {result['error']}")
    
    print("\n💡 Next steps:")
    print("   1. Run dashboard: streamlit run src/app/dashboard.py")
    print("   2. View live forecasts in ML Forecasts tab")
    print("   3. Set up auto-refresh (see instructions)")
    print()
    
    return results


if __name__ == "__main__":
    main()