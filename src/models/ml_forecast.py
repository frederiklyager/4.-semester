# src/models/ml_forecast.py
"""
Machine Learning Forecast Models for Energy Forecast Project
Phase 4: LightGBM and advanced ML forecasting

Author: Frederik Lyager
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import warnings
import pickle

warnings.filterwarnings('ignore')

# ML imports
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")
    print("Install with: pip install lightgbm scikit-learn matplotlib seaborn")
    DEPENDENCIES_OK = False

# Import metrics
try:
    from src.eval.metrics import evaluate_forecast, print_metrics, compare_models
    METRICS_AVAILABLE = True
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.eval.metrics import evaluate_forecast, print_metrics, compare_models
        METRICS_AVAILABLE = True
    except:
        METRICS_AVAILABLE = False


class LightGBMForecaster:
    """
    LightGBM-based forecaster for CO₂ intensity prediction.
    
    Uses gradient boosting with optimized hyperparameters for time series.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM forecaster.
        
        Args:
            params: LightGBM parameters (optional, uses defaults if None)
        """
        self.name = "LightGBM"
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
        # Default hyperparameters optimized for time series
        self.params = params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'co2_g_per_kwh') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML model.
        
        Args:
            df: DataFrame with all features
            target_col: Name of target column
        
        Returns:
            (X, y) - Features and target
        """
        # Drop non-feature columns
        drop_cols = ['ts', 'zone', target_col]
        
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[target_col]
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              num_boost_round: int = 1000,
              early_stopping_rounds: int = 50) -> Dict:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        print(f"🔧 Training {self.name}...")
        print(f"   Training samples: {len(X_train):,}")
        if X_val is not None:
            print(f"   Validation samples: {len(X_val):,}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best score: {self.model.best_score}")
        
        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features DataFrame
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save plot (optional)
        """
        if self.feature_importance is None:
            print("⚠️ No feature importance available. Train model first.")
            return
        
        plt.figure(figsize=(10, 8))
        
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Most Important Features - {self.name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved: {save_path}")
        
        plt.close()
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'name': self.name
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'LightGBMForecaster':
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        forecaster = cls(params=model_data['params'])
        forecaster.model = model_data['model']
        forecaster.feature_names = model_data['feature_names']
        forecaster.feature_importance = model_data['feature_importance']
        forecaster.name = model_data['name']
        
        print(f"✅ Model loaded: {path}")
        return forecaster


def time_series_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train/validation/test sets.
    
    Maintains chronological order (no shuffling).
    
    Args:
        df: DataFrame sorted by time
        val_size: Validation set proportion
        test_size: Test set proportion
    
    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df


def train_and_evaluate_ml(
    zone: str = 'DK1',
    val_size: float = 0.15,
    test_size: float = 0.15,
    save_model_flag: bool = True,
    save_forecast: bool = True
) -> Dict:
    """
    Complete ML training and evaluation pipeline.
    
    Args:
        zone: Price area ('DK1' or 'DK2')
        val_size: Validation set size
        test_size: Test set size
        save_model_flag: Whether to save trained model
        save_forecast: Whether to save forecast CSV
    
    Returns:
        Results dictionary with metrics and model
    """
    print(f"\n{'='*70}")
    print(f"🤖 ML Model Training: {zone}")
    print(f"{'='*70}\n")
    
    # 1. Load features
    features_path = Path(f"data/processed/features_{zone}.parquet")
    
    if not features_path.exists():
        raise FileNotFoundError(f"❌ Features not found: {features_path}\nRun Phase 2 first!")
    
    print(f"📂 Loading: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"   Loaded {len(df):,} rows with {len(df.columns)} columns\n")
    
    # 2. Split data
    print(f"✂️  Splitting data (val={val_size}, test={test_size})...")
    train_df, val_df, test_df = time_series_train_val_test_split(df, val_size, test_size)
    
    print(f"   Train: {len(train_df):,} rows ({train_df['ts'].min()} to {train_df['ts'].max()})")
    print(f"   Val:   {len(val_df):,} rows ({val_df['ts'].min()} to {val_df['ts'].max()})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['ts'].min()} to {test_df['ts'].max()})\n")
    
    # 3. Prepare features
    forecaster = LightGBMForecaster()
    
    X_train, y_train = forecaster.prepare_features(train_df)
    X_val, y_val = forecaster.prepare_features(val_df)
    X_test, y_test = forecaster.prepare_features(test_df)
    
    print(f"📊 Feature preparation complete")
    print(f"   Features: {len(forecaster.feature_names)}")
    print(f"   Top features: {', '.join(forecaster.feature_names[:5])}\n")
    
    # 4. Train model
    train_history = forecaster.train(
        X_train, y_train,
        X_val, y_val,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    # 5. Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    y_pred = forecaster.predict(X_test)
    
    if METRICS_AVAILABLE:
        metrics = evaluate_forecast(y_test, y_pred, model_name=f"LightGBM_{zone}")
        print_metrics(metrics)
    else:
        # Calculate basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"   MAE:  {mae:.2f} g CO₂/kWh")
        print(f"   RMSE: {rmse:.2f} g CO₂/kWh")
        print(f"   MAPE: {mape:.1f}%")
        
        metrics = {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    # 6. Feature importance
    print(f"\n📊 Top 10 Most Important Features:")
    for idx, row in forecaster.feature_importance.head(10).iterrows():
        print(f"   {row['feature']:.<30} {row['importance']:>10.1f}")
    
    # Save feature importance plot
    plot_path = f"data/models/feature_importance_{zone}.png"
    forecaster.plot_feature_importance(top_n=15, save_path=plot_path)
    
    # 7. Save model
    if save_model_flag:
        model_path = f"data/models/lgbm_{zone}.pkl"
        forecaster.save_model(model_path)
    
    # 8. Save forecast
    forecast_df = None
    if save_forecast:
        forecast_df = test_df[['ts']].copy()
        forecast_df['actual'] = y_test.values
        forecast_df['forecast'] = y_pred
        
        output_path = Path(f"data/forecast/co2_{zone}_ml.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(output_path, index=False)
        print(f"\n✅ Forecast saved: {output_path}")
    
    return {
        'zone': zone,
        'model': forecaster,
        'metrics': metrics,
        'train_history': train_history,
        'forecast_df': forecast_df
    }


def compare_baseline_vs_ml(zone: str = 'DK1'):
    """
    Compare baseline and ML forecasts side-by-side.
    
    Args:
        zone: Price area to compare
    """
    print(f"\n{'='*70}")
    print(f"📊 Baseline vs ML Comparison: {zone}")
    print(f"{'='*70}\n")
    
    # Load baseline forecast
    baseline_path = Path(f"data/forecast/co2_{zone}_baseline.csv")
    ml_path = Path(f"data/forecast/co2_{zone}_ml.csv")
    
    if not baseline_path.exists():
        print(f"❌ Baseline forecast not found: {baseline_path}")
        print("   Run: python src/models/baseline.py")
        return
    
    if not ml_path.exists():
        print(f"❌ ML forecast not found: {ml_path}")
        print("   Run ML training first!")
        return
    
    df_baseline = pd.read_csv(baseline_path, parse_dates=['ts'])
    df_ml = pd.read_csv(ml_path, parse_dates=['ts'])
    
    # Find common time range
    common_times = set(df_baseline['ts']).intersection(set(df_ml['ts']))
    
    if not common_times:
        print("⚠️ No overlapping time periods between baseline and ML forecasts")
        return
    
    df_baseline = df_baseline[df_baseline['ts'].isin(common_times)].sort_values('ts')
    df_ml = df_ml[df_ml['ts'].isin(common_times)].sort_values('ts')
    
    print(f"Comparing {len(common_times)} common time points\n")
    
    if METRICS_AVAILABLE:
        # Evaluate both
        baseline_metrics = evaluate_forecast(
            df_baseline['actual'].values,
            df_baseline['forecast'].values,
            model_name="Baseline"
        )
        
        ml_metrics = evaluate_forecast(
            df_ml['actual'].values,
            df_ml['forecast'].values,
            model_name="LightGBM"
        )
        
        # Compare
        comparison = compare_models([baseline_metrics, ml_metrics])
        print("\n" + "="*70)
        print("📊 Model Comparison Results")
        print("="*70 + "\n")
        print(comparison.round(2))
        
        # Calculate improvements
        mae_improvement = ((baseline_metrics['mae'] - ml_metrics['mae']) / baseline_metrics['mae']) * 100
        mape_improvement = ((baseline_metrics['mape'] - ml_metrics['mape']) / baseline_metrics['mape']) * 100
        
        print(f"\n🎯 Improvements:")
        print(f"   MAE:  {mae_improvement:+.1f}%")
        print(f"   MAPE: {mape_improvement:+.1f}%")
        
        if mae_improvement > 0:
            print(f"\n🏆 LightGBM is {mae_improvement:.1f}% better than baseline!")
        else:
            print(f"\n⚠️  Baseline performs better by {abs(mae_improvement):.1f}%")
    
    print("\n" + "="*70 + "\n")


def main():
    """Train ML models for both zones and compare with baseline."""
    
    if not DEPENDENCIES_OK:
        print("❌ Missing required dependencies. Install with:")
        print("   pip install lightgbm scikit-learn matplotlib seaborn")
        return
    
    zones = ['DK1', 'DK2']
    results = {}
    
    for zone in zones:
        try:
            result = train_and_evaluate_ml(
                zone=zone,
                val_size=0.15,
                test_size=0.15,
                save_model_flag=True,
                save_forecast=True
            )
            results[zone] = result
            
            # Compare with baseline
            compare_baseline_vs_ml(zone)
            
        except FileNotFoundError as e:
            print(f"\n❌ {zone}: {e}\n")
        except Exception as e:
            print(f"\n❌ Error processing {zone}: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*70)
    print("🎉 ML training complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()