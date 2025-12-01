"""
Minimal LSTM implementation for CO2 forecasting comparison
Time to implement: ~4 hours
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

def create_sequences(data, seq_length=24, forecast_horizon=24):
    """Create sliding window sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, n_features, forecast_horizon):
    """Build simple LSTM architecture"""
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, 
                         input_shape=(seq_length, n_features)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(forecast_horizon)  # Predict next 24 hours
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_lstm(zone='DK1', epochs=50, seq_length=24):
    """Train LSTM model on CO2 data"""
    
    # Load data
    df = pd.read_csv(f'data/processed/co2_hourly_{zone}.csv', parse_dates=['ts'])
    
    # Use only CO2 values for simplicity
    data = df['co2_g_per_kwh'].values.reshape(-1, 1)
    
    # Normalize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length=seq_length, forecast_horizon=24)
    
    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training LSTM on {len(X_train)} sequences")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    # Build model
    model = build_lstm_model(seq_length, n_features=1, forecast_horizon=24)
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"\nTest MAE: {test_mae:.2f}")
    
    # Save model and scaler
    Path('models').mkdir(exist_ok=True)
    model.save(f'models/lstm_{zone}.keras')
    joblib.dump(scaler, f'models/scaler_{zone}.pkl')
    
    print(f"✅ LSTM model saved to models/lstm_{zone}.keras")
    
    # Compare with actual test performance
    y_pred = model.predict(X_test)
    
    # Reshape predictions and test data properly for inverse transform
    n_samples = y_pred.shape[0]
    n_timesteps = y_pred.shape[1]
    
    # Reshape to 2D for scaler
    y_pred_2d = y_pred.reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)
    
    # Inverse transform
    y_pred_inv = scaler.inverse_transform(y_pred_2d).reshape(n_samples, n_timesteps)
    y_test_inv = scaler.inverse_transform(y_test_2d).reshape(n_samples, n_timesteps)
    
    mae_actual = np.mean(np.abs(y_pred_inv - y_test_inv))
    print(f"Actual MAE (g CO2/kWh): {mae_actual:.2f}")
    return model, scaler, history, mae_actual

def predict_future_lstm(zone='DK1', seq_length=24):
    """Generate 24h forecast using trained LSTM"""
    
    # Load model and scaler
    model = keras.models.load_model(f'models/lstm_{zone}.keras')
    scaler = joblib.load(f'models/scaler_{zone}.pkl')
    
    # Load recent data
    df = pd.read_csv(f'data/processed/co2_hourly_{zone}.csv', parse_dates=['ts'])
    data = df['co2_g_per_kwh'].values[-seq_length:].reshape(-1, 1)
    
    # Normalize
    data_scaled = scaler.transform(data)
    
    # Predict
    X = data_scaled.reshape(1, seq_length, 1)
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)[0]
    
    # Create forecast dataframe
    last_ts = df['ts'].iloc[-1]
    forecast_times = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=24,
        freq='h'
    )
    
    df_forecast = pd.DataFrame({
        'ts': forecast_times,
        'co2_g_per_kwh': y_pred
    })
    
    # Save
    df_forecast.to_csv(f'data/forecast/lstm_forecast_{zone}.csv', index=False)
    print(f"✅ LSTM forecast saved: {len(df_forecast)} hours")
    
    return df_forecast

def compare_models(zone='DK1'):
    """Compare LightGBM vs LSTM performance"""
    
    # Load both forecasts
    lgbm_forecast = pd.read_csv(f'data/forecast/future_forecast_{zone}.csv', parse_dates=['ts'])
    lstm_forecast = pd.read_csv(f'data/forecast/lstm_forecast_{zone}.csv', parse_dates=['ts'])
    
    # Load actual data
    actual = pd.read_csv(f'data/processed/co2_hourly_{zone}.csv', parse_dates=['ts'])
    
    # Merge for comparison
    comparison = lgbm_forecast.merge(lstm_forecast, on='ts', suffixes=('_lgbm', '_lstm'))
    comparison = comparison.merge(actual, on='ts')
    comparison = comparison.rename(columns={'co2_g_per_kwh': 'actual'})
    
    # Calculate MAE for each
    mae_lgbm = np.mean(np.abs(comparison['co2_g_per_kwh_lgbm'] - comparison['actual']))
    mae_lstm = np.mean(np.abs(comparison['co2_g_per_kwh_lstm'] - comparison['actual']))
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"LightGBM MAE: {mae_lgbm:.2f} g CO2/kWh")
    print(f"LSTM MAE:     {mae_lstm:.2f} g CO2/kWh")
    print(f"Winner:       {'LightGBM' if mae_lgbm < mae_lstm else 'LSTM'}")
    print("="*50)
    
    return comparison

def create_ensemble(zone='DK1', lgbm_weight=0.6):
    """Create weighted ensemble of LightGBM and LSTM"""
    
    lgbm = pd.read_csv(f'data/forecast/future_forecast_{zone}.csv', parse_dates=['ts'])
    lstm = pd.read_csv(f'data/forecast/lstm_forecast_{zone}.csv', parse_dates=['ts'])
    
    # Merge
    ensemble = lgbm.merge(lstm, on='ts', suffixes=('_lgbm', '_lstm'))
    
    # Weighted average
    ensemble['co2_g_per_kwh'] = (
        lgbm_weight * ensemble['co2_g_per_kwh_lgbm'] +
        (1 - lgbm_weight) * ensemble['co2_g_per_kwh_lstm']
    )
    
    # Save
    ensemble[['ts', 'co2_g_per_kwh']].to_csv(
        f'data/forecast/ensemble_forecast_{zone}.csv',
        index=False
    )
    
    print(f"✅ Ensemble forecast created (LightGBM: {lgbm_weight*100:.0f}%, LSTM: {(1-lgbm_weight)*100:.0f}%)")
    
    return ensemble

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--zone', default='DK1', choices=['DK1', 'DK2'])
    parser.add_argument('--train', action='store_true', help='Train LSTM model')
    parser.add_argument('--predict', action='store_true', help='Generate forecast')
    parser.add_argument('--compare', action='store_true', help='Compare with LightGBM')
    parser.add_argument('--ensemble', action='store_true', help='Create ensemble')
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    if args.train:
        print(f"Training LSTM for {args.zone}...")
        model, scaler, history, mae = train_lstm(args.zone, epochs=args.epochs)
    
    if args.predict:
        print(f"Generating LSTM forecast for {args.zone}...")
        df = predict_future_lstm(args.zone)
        print(df.head())
    
    if args.compare:
        print(f"Comparing models for {args.zone}...")
        comparison = compare_models(args.zone)
    
    if args.ensemble:
        print(f"Creating ensemble for {args.zone}...")
        ensemble = create_ensemble(args.zone)
