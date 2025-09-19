import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic(area="DK1", days=30):
    """Lav syntetiske elpriser time for time i X dage bagud + day-ahead forecast."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)

    idx = pd.date_range(start=start, end=now, freq="H")
    hours = np.arange(len(idx))
    daily = 10 * (1 + np.sin(2*np.pi*(hours % 24)/24))     # døgnmønster
    weekly = 5 * (1 + np.cos(2*np.pi*(hours % (24*7))/(24*7)))  # ugedagsmønster
    base = 50 if area == "DK1" else 55
    noise = np.random.normal(0, 3, size=len(idx))
    prices = base + daily + weekly + noise

    df_hist = pd.DataFrame({"ts": idx, "area": area, "price_eur_mwh": prices})

    # Day-ahead forecast = brug sidste døgns mønster
    last24 = df_hist.tail(24)["price_eur_mwh"].values
    idx_da = pd.date_range(start=now, periods=24, freq="H")
    df_da = pd.DataFrame({"ts": idx_da, "area": area, "price_eur_mwh": last24})

    return df_hist, df_da


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    for area in ["DK1", "DK2"]:
        hist, forecast = generate_synthetic(area)
        hist_file = f"data/prices_{area}_hist.csv"
        fc_file = f"data/{area}_price_forecast.csv"

        hist.to_csv(hist_file, index=False)
        forecast.to_csv(fc_file, index=False)

        print(f"✅ Gemte {hist_file} og {fc_file}")
