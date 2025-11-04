import pandas as pd
from src.eval.schemas import CO2Schema

def test_co2_schema_ok():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2025-01-01T00:00Z","2025-01-01T01:00Z"]),
        "zone": ["DK1","DK1"],
        "co2_g_per_kwh": [120.0, 115.5],
    })
    CO2Schema.validate(df)  # skal ikke kaste fejl
