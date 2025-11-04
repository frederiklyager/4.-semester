# src/models/baseline.py
"""
Baseline Forecast Models for Energy Forecast Project
Phase 3: Simple baseline models for CO₂ intensity forecasting

Author: Frederik Lyager

Models:
1. Persistence (Naive) - forecast[t+h] = actual[t]
2. Moving Average - forecast based on rolling mean
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import metrics if available
try:
    from src.eval.metrics import evaluate_forecast, print_metrics, compare_models
    METRICS_AVAILABLE = True
except ImportError as e:
    try:
        # Try alternative import path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.eval.metrics import evaluate_forecast, print_metrics, compare_models
        METRICS_AVAILABLE = True
    except:
        print("⚠️ Metrics module not found. Install or check path.")
        print(f"   Error: {e}")
        METRICS_AVAILABLE = False


class BaselineForecaster:
    """Base class for baseline forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.forecast_df = None
    
    def fit(self, train_data: pd.DataFrame):
        """Fit model on training data (if needed)."""
        pass
    
    def predict(self, test_data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Generate forecasts for test period."""
        raise NotImplementedError
    
    def save_forecast(self, output_path: str):
        """Save forecast to CSV."""
        if self.forecast_df is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self.forecast_df.to_csv(output_path, index=False)
            print(f"✅ Saved forecast to: {output_path}")


class PersistenceModel(BaselineForecaster):
    """
    Persistence (Naive) Forecast Model
    
    Assumes future = current value
    forecast[t+h] = actual[t]
    
    For CO₂ forecasting with 24h horizon:
    - Tomorrow at 14:00 = Today at 14:00
    """
    
    def __init__(self):
        super().__init__(name="Persistence")
    
    def predict(self, test_data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Generate persistence forecasts.
        
        Args:
            test_data: DataFrame with 'ts' and 'co2_g_per_kwh'
            horizon: Forecast horizon in hours (default: 24)
        
        Returns:
            DataFrame with forecasts
        """
        df = test_data.copy()
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Shift by horizon hours to create forecast
        df['forecast'] = df['co2_g_per_kwh'].shift(horizon)
        
        # Remove rows without forecast (first 'horizon' rows)
        df = df.dropna(subset=['forecast'])
        
        self.forecast_df = df[['ts', 'co2_g_per_kwh', 'forecast']].copy()
        self.forecast_df.columns = ['ts', 'actual', 'forecast']
        
        return self.forecast_df


class MovingAverageModel(BaselineForecaster):
    """
    Moving Average Forecast Model
    
    forecast[t+h] = mean(actual[t-window:t])
    
    Common windows:
    - 24h: Daily pattern
    - 168h: Weekly pattern
    """
    
    def __init__(self, window: int = 24):
        """
        Args:
            window: Rolling window size in hours (24, 168, etc.)
        """
        super().__init__(name=f"MovingAverage_{window}h")
        self.window = window
    
    def predict(self, test_data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Generate moving average forecasts.
        
        Args:
            test_data: DataFrame with 'ts' and 'co2_g_per_kwh'
            horizon: Forecast horizon (for alignment)
        
        Returns:
            DataFrame with forecasts
        """
        df = test_data.copy()
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Calculate rolling mean
        df['rolling_mean'] = df['co2_g_per_kwh'].rolling(
            window=self.window, 
            min_periods=1
        ).mean()
        
        # Shift by horizon to simulate forecast
        df['forecast'] = df['rolling_mean'].shift(horizon)
        
        # Remove rows without valid forecast
        df = df.dropna(subset=['forecast'])
        
        self.forecast_df = df[['ts', 'co2_g_per_kwh', 'forecast']].copy()
        self.forecast_df.columns = ['ts', 'actual', 'forecast']
        
        return self.forecast_df


class SeasonalMovingAverageModel(BaselineForecaster):
    """
    Seasonal Moving Average (Same Hour Yesterday/Last Week)
    
    forecast[t+h] = mean of same hour in past N days
    
    Example for 24h horizon:
    - Forecast for tomorrow 14:00 = average of past 7 days at 14:00
    """
    
    def __init__(self, n_days: int = 7):
        """
        Args:
            n_days: Number of past days to average (default: 7)
        """
        super().__init__(name=f"SeasonalMA_{n_days}d")
        self.n_days = n_days
    
    def predict(self, test_data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """
        Generate seasonal moving average forecasts.
        
        Args:
            test_data: DataFrame with 'ts' and 'co2_g_per_kwh'
            horizon: Must be multiple of 24 for daily seasonality
        
        Returns:
            DataFrame with forecasts
        """
        df = test_data.copy()
        df = df.sort_values('ts').reset_index(drop=True)
        df['hour'] = df['ts'].dt.hour
        
        forecasts = []
        
        for idx in range(len(df)):
            current_time = df.loc[idx, 'ts']
            current_hour = df.loc[idx, 'hour']
            
            # Get past N days at same hour
            past_values = []
            for day in range(1, self.n_days + 1):
                past_time = current_time - timedelta(days=day)
                mask = (df['ts'] == past_time) & (df['hour'] == current_hour)
                if mask.any():
                    past_values.append(df.loc[mask, 'co2_g_per_kwh'].values[0])
            
            if past_values:
                forecasts.append(np.mean(past_values))
            else:
                forecasts.append(np.nan)
        
        df['forecast'] = forecasts
        
        # Shift by horizon for proper alignment
        df['forecast'] = df['forecast'].shift(horizon)
        df = df.dropna(subset=['forecast'])
        
        self.forecast_df = df[['ts', 'co2_g_per_kwh', 'forecast']].copy()
        self.forecast_df.columns = ['ts', 'actual', 'forecast']
        
        return self.forecast_df


def train_test_split_time(df: pd.DataFrame, 
                          test_size: float = 0.2,
                          min_train_days: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Args:
        df: DataFrame with 'ts' column
        test_size: Proportion for test set (default: 0.2 = 20%)
        min_train_days: Minimum days in training set
    
    Returns:
        (train_df, test_df)
    """
    df = df.sort_values('ts').reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_size))
    
    # Ensure minimum training data
    min_train_hours = min_train_days * 24
    if split_idx < min_train_hours:
        split_idx = min_train_hours
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def run_baseline_evaluation(zone: str = 'DK1',
                           horizon: int = 24,
                           test_size: float = 0.2) -> pd.DataFrame:
    """
    Complete baseline model evaluation pipeline.
    
    Steps:
    1. Load feature data
    2. Train/test split
    3. Run all baseline models
    4. Evaluate and compare
    5. Save best forecasts
    
    Args:
        zone: Price area ('DK1' or 'DK2')
        horizon: Forecast horizon in hours
        test_size: Test set proportion
    
    Returns:
        Comparison DataFrame with all metrics
    """
    print(f"\n{'='*70}")
    print(f"🎯 Baseline Model Evaluation: {zone}")
    print(f"{'='*70}\n")
    
    # 1. Load data
    input_path = Path(f"data/processed/features_{zone}.parquet")
    
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Features not found: {input_path}\nRun Phase 2 first!")
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df):,} rows\n")
    
    # Select essential columns for baseline (timestamp + target)
    df_base = df[['ts', 'co2_g_per_kwh']].copy()
    
    # 2. Train/test split
    print(f"✂️  Splitting data (test_size={test_size})...")
    train_df, test_df = train_test_split_time(df_base, test_size=test_size)
    print(f"   Train: {len(train_df):,} rows ({train_df['ts'].min()} to {train_df['ts'].max()})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['ts'].min()} to {test_df['ts'].max()})\n")
    
    # 3. Initialize models
    models = [
        PersistenceModel(),
        MovingAverageModel(window=24),
        MovingAverageModel(window=168),
        SeasonalMovingAverageModel(n_days=7),
    ]
    
    # 4. Run forecasts and evaluate
    results = []
    best_mae = float('inf')
    best_model = None
    
    for model in models:
        print(f"🔧 Running {model.name}...")
        
        # Generate forecast
        forecast_df = model.predict(test_df, horizon=horizon)
        
        if len(forecast_df) == 0:
            print(f"   ⚠️  No valid forecasts generated\n")
            continue
        
        y_true = forecast_df['actual'].values
        y_pred = forecast_df['forecast'].values
        
        # Evaluate
        if METRICS_AVAILABLE:
            metrics = evaluate_forecast(y_true, y_pred, model_name=model.name)
            print_metrics(metrics, precision=2)
            results.append(metrics)
            
            # Track best model
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model = model
        else:
            print(f"   Forecast generated: {len(forecast_df)} rows\n")
    
    # 5. Save best model forecast
    if best_model and best_model.forecast_df is not None:
        output_dir = Path(f"data/forecast")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"co2_{zone}_baseline.csv"
        best_model.save_forecast(output_path)
        print(f"\n🏆 Best Model: {best_model.name} (MAE: {best_mae:.2f})")
    
    # 6. Comparison table
    if results and METRICS_AVAILABLE:
        print(f"\n{'='*70}")
        print("📊 Model Comparison")
        print(f"{'='*70}\n")
        comparison_df = compare_models(results)
        print(comparison_df.round(2))
        print(f"\n{'='*70}\n")
        return comparison_df
    
    return pd.DataFrame()


def main():
    """Run baseline evaluation for both zones."""
    zones = ['DK1', 'DK2']
    
    all_results = {}
    
    for zone in zones:
        try:
            results = run_baseline_evaluation(zone=zone, horizon=24, test_size=0.2)
            all_results[zone] = results
        except FileNotFoundError as e:
            print(f"\n❌ {zone}: {e}\n")
        except Exception as e:
            print(f"\n❌ Error processing {zone}: {e}\n")
    
    print("="*70)
    print("🎉 Baseline evaluation complete!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    main()