# src/features/transform.py
"""
Feature Engineering Module for Energy Forecast Project
Phase 2: Transform CO₂ data into ML-ready features

Author: Frederik Lyager
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Danish holidays library
try:
    import holidays
    DK_HOLIDAYS = holidays.Denmark()
except ImportError:
    print("⚠️ 'holidays' package not installed. Install with: pip install holidays")
    DK_HOLIDAYS = None


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar and time-based features.
    
    Features added:
    - hour (0-23)
    - weekday (0=Monday, 6=Sunday)
    - month (1-12)
    - is_weekend (boolean)
    - is_holiday (boolean, Danish holidays)
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.weekday
    df['month'] = df['ts'].dt.month
    
    # Weekend flag
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Danish holiday flag
    if DK_HOLIDAYS is not None:
        df['is_holiday'] = df['ts'].dt.date.apply(lambda x: x in DK_HOLIDAYS).astype(int)
    else:
        df['is_holiday'] = 0
    
    return df


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic encodings for periodic features.
    
    Transforms hour and weekday into sin/cos pairs to capture cyclical nature:
    - hour_sin, hour_cos (24-hour cycle)
    - weekday_sin, weekday_cos (7-day cycle)
    """
    df = df.copy()
    
    # Hour cyclic encoding (0-23)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Weekday cyclic encoding (0-6)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    return df


def add_lag_features(df: pd.DataFrame, 
                     target_col: str = 'co2_g_per_kwh',
                     lags: list[int] = None) -> pd.DataFrame:
    """
    Add lagged values of the target variable.
    
    Args:
        df: Input DataFrame with datetime index
        target_col: Name of target column to lag
        lags: List of lag periods (default: [1, 24, 168])
              - 1: previous hour
              - 24: same hour yesterday
              - 168: same hour last week
    
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 24, 168]
    
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame,
                        target_col: str = 'co2_g_per_kwh',
                        windows: list[int] = None) -> pd.DataFrame:
    """
    Add rolling statistics (mean and std) over specified windows.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        windows: List of window sizes in hours (default: [24, 168])
                 - 24: last 24 hours (1 day)
                 - 168: last 168 hours (1 week)
    
    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = [24, 168]
    
    df = df.copy()
    
    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}h'] = df[target_col].rolling(
            window=window, 
            min_periods=1
        ).mean()
        
        # Rolling standard deviation
        df[f'rolling_std_{window}h'] = df[target_col].rolling(
            window=window, 
            min_periods=1
        ).std()
    
    return df


def create_features(zone: str = 'DK1', 
                   input_dir: str = 'data/processed',
                   output_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Main orchestrator: Load CO₂ data, engineer features, and save results.
    
    Args:
        zone: Price area ('DK1' or 'DK2')
        input_dir: Directory containing co2_{zone}.parquet
        output_dir: Directory to save features_{zone}.parquet
    
    Returns:
        DataFrame with all engineered features
    """
    print(f"\n{'='*60}")
    print(f"🔧 Feature Engineering for {zone}")
    print(f"{'='*60}")
    
    # 1. Load processed CO₂ data
    input_path = Path(input_dir) / f'co2_{zone}.parquet'

    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    print(f"📂 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Ensure sorted by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    
    # 2. Add calendar features
    print("📅 Adding calendar features...")
    df = add_calendar_features(df)
    
    # 3. Add cyclic encodings
    print("🔄 Adding cyclic encodings...")
    df = add_cyclic_features(df)
    
    # 4. Add lag features
    print("⏱️  Adding lag features (t-1, t-24, t-168)...")
    df = add_lag_features(df, target_col='co2_g_per_kwh')
    
    # 5. Add rolling statistics
    print("📊 Adding rolling features (24h, 168h)...")
    df = add_rolling_features(df, target_col='co2_g_per_kwh')
    
    # 6. Drop rows with NaN (from lag features)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    print(f"🧹 Dropped {dropped_rows} rows with missing values")
    print(f"   Final dataset: {len(df):,} rows")
    
    # 7. Save to parquet
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f'features_{zone}.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved features to: {output_path}")
    
    # 8. Display feature summary
    print(f"\n📋 Feature Summary:")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Feature columns: {list(df.columns)}")
    print(f"\n📊 Dataset info:")
    print(f"   Date range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"   Shape: {df.shape}")
    
    return df


def main():
    """Run feature engineering for both DK1 and DK2."""
    zones = ['DK1', 'DK2']
    
    for zone in zones:
        try:
            df = create_features(zone=zone)
            print(f"\n✅ {zone} feature engineering completed successfully!")
            print(f"   Preview:\n{df.head()}\n")
        except Exception as e:
            print(f"\n❌ Error processing {zone}: {e}\n")
    
    print("="*60)
    print("🎉 Feature engineering pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()