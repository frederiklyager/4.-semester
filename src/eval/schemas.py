"""
Data Schema Validation for Energy Forecast Project

This module implements Security-by-Design principles using Pandera
to ensure data integrity and prevent injection attacks.

Author: Frederik Lyager
Course: Datamatiker 4th Semester - Cyber Security Focus
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CO2 EMISSION DATA SCHEMAS
# ============================================

CO2_RAW_SCHEMA = DataFrameSchema(
    {
        "ts": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
            checks=[
                Check(lambda s: s.dt.tz is not None, error="Timestamps must be timezone-aware"),
                Check(lambda s: (s >= pd.Timestamp('2020-01-01', tz='UTC')).all(), 
                      error="Timestamps cannot be before 2020"),
                Check(lambda s: (s <= pd.Timestamp.now(tz='UTC')).all(),
                      error="Timestamps cannot be in the future")
            ],
            title="Timestamp"
        ),
        "area": Column(
            pa.String,
            nullable=False,
            checks=[
                Check.isin(["DK1", "DK2"]),
            ],
            title="Price Area"
        ),
        "co2_g_per_kwh": Column(
            pa.Float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0, error="CO2 cannot be negative"),
                Check.less_than_or_equal_to(1000, error="CO2 value suspiciously high (>1000 g/kWh)"),
                Check(lambda s: ~s.isna().all(), error="All CO2 values are null"),
            ],
            title="CO2 Intensity (g/kWh)"
        ),
    },
    strict=False,  # Allow extra columns
    coerce=True,   # Auto-convert types when possible
    name="CO2_Raw_Data"
)

CO2_HOURLY_SCHEMA = DataFrameSchema(
    {
        "ts": Column(
            "datetime64[ns, UTC]",  # Timezone-aware datetime
            nullable=False,
            checks=[
                Check(lambda s: s.is_monotonic_increasing, error="Timestamps must be sorted"),
            ]
        ),
        "co2_g_per_kwh": Column(
            pa.Float,
            nullable=False,
            checks=[
                Check.in_range(0, 1000, include_min=True, include_max=True),
            ]
        ),
    },
    strict=True,
    coerce=True,
    name="CO2_Hourly_Data"
)

# ============================================
# ELECTRICITY PRICE DATA SCHEMAS
# ============================================

PRICE_SCHEMA = DataFrameSchema(
    {
        "ts": Column(
            "datetime64[ns, UTC]",  # ← Changed to accept timezone
            nullable=False,
            coerce=True,
        ),
        "area": Column(
            pa.String,
            nullable=False,
            checks=[
                Check.isin(["DK1", "DK2"]),
            ]
        ),
        "price_eur_mwh": Column(
            pa.Float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(-500, error="Price suspiciously low"),
                Check.less_than_or_equal_to(5000, error="Price suspiciously high"),
            ],
            title="Spot Price (EUR/MWh)"
        ),
    },
    strict=False,
    coerce=True,
    name="Electricity_Price_Data"
)

# ============================================
# FEATURE ENGINEERING SCHEMA
# ============================================

FEATURES_SCHEMA = DataFrameSchema(
    {
        "ts": Column(pa.DateTime, nullable=False),
        "co2_g_per_kwh": Column(pa.Float, nullable=False),
        "hour": Column(pa.Int, checks=[Check.in_range(0, 23)]),
        "weekday": Column(pa.Int, checks=[Check.in_range(0, 6)]),
        "month": Column(pa.Int, checks=[Check.in_range(1, 12)]),
        "is_weekend": Column(pa.Bool),
        "hour_sin": Column(pa.Float, checks=[Check.in_range(-1, 1)]),
        "hour_cos": Column(pa.Float, checks=[Check.in_range(-1, 1)]),
        "weekday_sin": Column(pa.Float, checks=[Check.in_range(-1, 1)]),
        "weekday_cos": Column(pa.Float, checks=[Check.in_range(-1, 1)]),
        # Lag features
        "lag_1h": Column(pa.Float, nullable=True),
        "lag_24h": Column(pa.Float, nullable=True),
        "lag_168h": Column(pa.Float, nullable=True),
        # Rolling features
        "roll_mean_24h": Column(pa.Float, nullable=True),
        "roll_std_24h": Column(pa.Float, nullable=True),
        "roll_mean_168h": Column(pa.Float, nullable=True),
        "roll_std_168h": Column(pa.Float, nullable=True),
    },
    strict=False,
    coerce=True,
    name="Feature_Data"
)

# ============================================
# VALIDATION FUNCTIONS (with Security Logging)
# ============================================

def validate_co2_raw(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """
    Validate raw CO2 data from API.
    
    Security Features:
    - Schema enforcement prevents injection attacks
    - Range checks detect anomalous data
    - Logging tracks validation failures
    
    Args:
        df: Raw DataFrame from API
        raise_on_error: If True, raise exception on validation failure
        
    Returns:
        Validated DataFrame
        
    Raises:
        pa.errors.SchemaError: If validation fails and raise_on_error=True
    """
    try:
        validated_df = CO2_RAW_SCHEMA.validate(df, lazy=True)
        logger.info(f"✅ CO2 raw data validation passed ({len(df)} rows)")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ CO2 raw data validation FAILED: {e}")
        logger.error(f"Failed checks: {e.failure_cases}")
        if raise_on_error:
            raise
        return df

def validate_co2_hourly(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate processed hourly CO2 data."""
    try:
        validated_df = CO2_HOURLY_SCHEMA.validate(df, lazy=True)
        logger.info(f"✅ CO2 hourly data validation passed ({len(df)} rows)")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ CO2 hourly data validation FAILED: {e}")
        if raise_on_error:
            raise
        return df

def validate_price(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate electricity price data."""
    try:
        validated_df = PRICE_SCHEMA.validate(df, lazy=True)
        logger.info(f"✅ Price data validation passed ({len(df)} rows)")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ Price data validation FAILED: {e}")
        if raise_on_error:
            raise
        return df

def validate_features(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Validate feature-engineered data."""
    try:
        validated_df = FEATURES_SCHEMA.validate(df, lazy=True)
        logger.info(f"✅ Feature data validation passed ({len(df)} rows)")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ Feature data validation FAILED: {e}")
        if raise_on_error:
            raise
        return df

# ============================================
# SECURITY HEALTH CHECK
# ============================================

def security_health_check(df: pd.DataFrame, schema: DataFrameSchema) -> dict:
    """
    Perform comprehensive security health check on data.
    
    Returns dict with:
    - is_valid: bool
    - error_count: int
    - warnings: list
    - security_score: 0-100
    """
    results = {
        "is_valid": False,
        "error_count": 0,
        "warnings": [],
        "security_score": 0,
        "checks_passed": 0,
        "total_checks": 5
    }
    
    try:
        # Check 1: Schema validation
        schema.validate(df, lazy=True)
        results["checks_passed"] += 1
    except pa.errors.SchemaError as e:
        results["error_count"] = len(e.failure_cases)
        results["warnings"].append(f"Schema validation failed: {e}")
    
    # Check 2: No duplicate timestamps
    if not df.duplicated(subset=['ts']).any():
        results["checks_passed"] += 1
    else:
        results["warnings"].append("Duplicate timestamps detected")
    
    # Check 3: No missing critical columns
    required_cols = ['ts', 'co2_g_per_kwh'] if 'co2_g_per_kwh' in df.columns else ['ts', 'price_eur_mwh']
    if all(col in df.columns for col in required_cols):
        results["checks_passed"] += 1
    else:
        results["warnings"].append("Missing critical columns")
    
    # Check 4: Reasonable data range
    if 'co2_g_per_kwh' in df.columns:
        if df['co2_g_per_kwh'].between(0, 1000).all():
            results["checks_passed"] += 1
        else:
            results["warnings"].append("CO2 values outside expected range")
    else:
        results["checks_passed"] += 1
    
    # Check 5: Timezone awareness
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        if df['ts'].dt.tz is not None:
            results["checks_passed"] += 1
        else:
            results["warnings"].append("Timestamps not timezone-aware")
    
    # Calculate security score
    results["security_score"] = int((results["checks_passed"] / results["total_checks"]) * 100)
    results["is_valid"] = results["security_score"] >= 80
    
    logger.info(f"Security Health Check: {results['security_score']}/100 ({results['checks_passed']}/{results['total_checks']} passed)")
    
    return results

def validate_dataframe(df: pd.DataFrame, schema: DataFrameSchema, name: str = "DataFrame") -> pd.DataFrame:
    """
    Generic validation function that works with any schema.
    
    Args:
        df: DataFrame to validate
        schema: Pandera schema to validate against
        name: Name for logging purposes
        
    Returns:
        Validated DataFrame
    """
    try:
        validated_df = schema.validate(df, lazy=True)
        logger.info(f"✅ {name} validation passed ({len(df)} rows)")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ {name} validation FAILED: {e}")
        logger.error(f"Failed checks: {e.failure_cases}")
        raise

if __name__ == "__main__":
    # Example usage / self-test
    print("🔒 Schema Validation Module")
    print("=" * 50)
    print("Available schemas:")
    print("  - CO2_RAW_SCHEMA")
    print("  - CO2_HOURLY_SCHEMA")
    print("  - PRICE_SCHEMA")
    print("  - FEATURES_SCHEMA")
    print("\nValidation functions:")
    print("  - validate_co2_raw()")
    print("  - validate_co2_hourly()")
    print("  - validate_price()")
    print("  - validate_features()")
    print("  - security_health_check()")
