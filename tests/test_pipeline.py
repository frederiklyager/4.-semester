#!/usr/bin/env python3
"""
Energy Forecast Pipeline Test
Tests Phase 1 → Phase 2 → Phase 3 integration

Author: Frederik Lyager
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class PipelineTester:
    """Test suite for Energy Forecast pipeline."""
    
    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def test(self, name: str, condition: bool, message: str = ""):
        """Run a single test."""
        self.total_tests += 1
        status = "✅ PASS" if condition else "❌ FAIL"
        color = GREEN if condition else RED
        
        print(f"{color}{status}{RESET} - {name}")
        if message:
            print(f"       {message}")
        
        if condition:
            self.passed_tests += 1
        
        self.results.append({
            'test': name,
            'passed': condition,
            'message': message
        })
        
        return condition
    
    def section(self, title: str):
        """Print section header."""
        print(f"\n{BLUE}{BOLD}{'='*70}{RESET}")
        print(f"{BLUE}{BOLD}{title}{RESET}")
        print(f"{BLUE}{BOLD}{'='*70}{RESET}\n")
    
    def summary(self):
        """Print test summary."""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}TEST SUMMARY{RESET}")
        print(f"{BOLD}{'='*70}{RESET}")
        
        percentage = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        color = GREEN if percentage == 100 else (YELLOW if percentage >= 70 else RED)
        
        print(f"\nTotal Tests: {self.total_tests}")
        print(f"{color}Passed: {self.passed_tests}{RESET}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"{color}Success Rate: {percentage:.1f}%{RESET}\n")
        
        if percentage == 100:
            print(f"{GREEN}{BOLD}🎉 ALL TESTS PASSED! Pipeline is ready!{RESET}\n")
        elif percentage >= 70:
            print(f"{YELLOW}{BOLD}⚠️  Most tests passed. Review failures before continuing.{RESET}\n")
        else:
            print(f"{RED}{BOLD}❌ Multiple failures. Fix issues before proceeding.{RESET}\n")
        
        return percentage == 100


def test_phase1_data():
    """Test Phase 1: Data ingestion output."""
    tester = PipelineTester()
    tester.section("PHASE 1: Data Ingestion")
    
    zones = ['DK1', 'DK2']
    
    for zone in zones:
        # Check if parquet file exists
        file_path = Path(f"data/processed/co2_{zone}.parquet")
        exists = file_path.exists()
        tester.test(
            f"File exists: co2_{zone}.parquet",
            exists,
            f"Path: {file_path}" if exists else "Run: python src/ingest/energinet_co2.py"
        )
        
        if exists:
            # Load and validate data
            try:
                df = pd.read_parquet(file_path)
                
                # Check columns
                required_cols = ['ts', 'zone', 'co2_g_per_kwh']
                has_cols = all(col in df.columns for col in required_cols)
                tester.test(
                    f"Has required columns ({zone})",
                    has_cols,
                    f"Columns: {list(df.columns)}"
                )
                
                # Check data types
                is_datetime = pd.api.types.is_datetime64_any_dtype(df['ts'])
                tester.test(
                    f"Timestamp is datetime ({zone})",
                    is_datetime,
                    f"Type: {df['ts'].dtype}"
                )
                
                # Check data quantity
                has_data = len(df) > 100
                tester.test(
                    f"Has sufficient data ({zone})",
                    has_data,
                    f"Rows: {len(df):,}"
                )
                
                # Check for nulls
                no_nulls = df['co2_g_per_kwh'].notna().all()
                null_count = df['co2_g_per_kwh'].isna().sum()
                tester.test(
                    f"No null values in CO2 ({zone})",
                    no_nulls,
                    f"Null count: {null_count}"
                )
                
                # Check value range
                min_val = df['co2_g_per_kwh'].min()
                max_val = df['co2_g_per_kwh'].max()
                valid_range = (-50 <= min_val <= 500) and (0 <= max_val <= 500)
                tester.test(
                    f"CO2 values in valid range ({zone})",
                    valid_range,
                    f"Range: {min_val:.1f} to {max_val:.1f} g/kWh"
                )
                
            except Exception as e:
                tester.test(f"Load data ({zone})", False, f"Error: {e}")
    
    return tester


def test_phase2_features():
    """Test Phase 2: Feature engineering output."""
    tester = PipelineTester()
    tester.section("PHASE 2: Feature Engineering")
    
    zones = ['DK1', 'DK2']
    
    for zone in zones:
        # Check if features file exists
        file_path = Path(f"data/processed/features_{zone}.parquet")
        exists = file_path.exists()
        tester.test(
            f"File exists: features_{zone}.parquet",
            exists,
            f"Path: {file_path}" if exists else "Run: python src/features/transform.py"
        )
        
        if exists:
            try:
                df = pd.read_parquet(file_path)
                
                # Check essential features
                essential_features = [
                    'ts', 'co2_g_per_kwh',
                    'hour', 'weekday', 'month',
                    'is_weekend', 'is_holiday',
                    'hour_sin', 'hour_cos',
                    'lag_1', 'lag_24', 'lag_168',
                    'rolling_mean_24h', 'rolling_std_24h'
                ]
                
                missing_features = [f for f in essential_features if f not in df.columns]
                has_features = len(missing_features) == 0
                tester.test(
                    f"Has all essential features ({zone})",
                    has_features,
                    f"Missing: {missing_features}" if not has_features else f"Features: {len(df.columns)}"
                )
                
                # Check data quantity (should have less rows due to lag dropping)
                has_data = len(df) > 50
                tester.test(
                    f"Has sufficient data after feature engineering ({zone})",
                    has_data,
                    f"Rows: {len(df):,}"
                )
                
                # Check for nulls in features
                null_cols = df.columns[df.isna().any()].tolist()
                no_nulls = len(null_cols) == 0
                tester.test(
                    f"No null values in features ({zone})",
                    no_nulls,
                    f"Columns with nulls: {null_cols}" if not no_nulls else "All clean"
                )
                
                # Check cyclic features range
                if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
                    sin_valid = (-1 <= df['hour_sin'].min()) and (df['hour_sin'].max() <= 1)
                    cos_valid = (-1 <= df['hour_cos'].min()) and (df['hour_cos'].max() <= 1)
                    tester.test(
                        f"Cyclic features in valid range ({zone})",
                        sin_valid and cos_valid,
                        "Range: [-1, 1]"
                    )
                
                # Check lag features
                if 'lag_24' in df.columns:
                    lag_corr = df['co2_g_per_kwh'].corr(df['lag_24'])
                    high_corr = lag_corr > 0.5
                    tester.test(
                        f"Lag features correlated with target ({zone})",
                        high_corr,
                        f"Correlation: {lag_corr:.3f}"
                    )
                
            except Exception as e:
                tester.test(f"Load features ({zone})", False, f"Error: {e}")
    
    return tester


def test_phase3_forecasts():
    """Test Phase 3: Baseline forecast output."""
    tester = PipelineTester()
    tester.section("PHASE 3: Baseline Forecasts")
    
    zones = ['DK1', 'DK2']
    
    for zone in zones:
        # Check if forecast file exists
        file_path = Path(f"data/forecast/co2_{zone}_baseline.csv")
        exists = file_path.exists()
        tester.test(
            f"File exists: co2_{zone}_baseline.csv",
            exists,
            f"Path: {file_path}" if exists else "Run: python src/models/baseline.py"
        )
        
        if exists:
            try:
                df = pd.read_csv(file_path, parse_dates=['ts'])
                
                # Check columns
                required_cols = ['ts', 'actual', 'forecast']
                has_cols = all(col in df.columns for col in required_cols)
                tester.test(
                    f"Has required columns ({zone})",
                    has_cols,
                    f"Columns: {list(df.columns)}"
                )
                
                # Check data quantity
                has_data = len(df) > 20
                tester.test(
                    f"Has sufficient forecast data ({zone})",
                    has_data,
                    f"Rows: {len(df):,}"
                )
                
                # Check for nulls
                no_nulls = df[['actual', 'forecast']].notna().all().all()
                tester.test(
                    f"No null values in forecasts ({zone})",
                    no_nulls
                )
                
                # Calculate basic metrics
                if has_cols and has_data and no_nulls:
                    mae = (df['actual'] - df['forecast']).abs().mean()
                    mape = ((df['actual'] - df['forecast']).abs() / df['actual']).mean() * 100
                    
                    # Reasonable MAE (< 50% of average CO2)
                    avg_co2 = df['actual'].mean()
                    reasonable_mae = mae < (avg_co2 * 0.5)
                    tester.test(
                        f"Forecast MAE is reasonable ({zone})",
                        reasonable_mae,
                        f"MAE: {mae:.2f} g/kWh (avg CO2: {avg_co2:.2f})"
                    )
                    
                    # MAPE should be < 50%
                    reasonable_mape = mape < 50
                    tester.test(
                        f"Forecast MAPE is reasonable ({zone})",
                        reasonable_mape,
                        f"MAPE: {mape:.1f}%"
                    )
                
            except Exception as e:
                tester.test(f"Load forecast ({zone})", False, f"Error: {e}")
    
    return tester


def test_code_modules():
    """Test if code modules can be imported."""
    tester = PipelineTester()
    tester.section("CODE MODULES")
    
    # Test metrics module
    try:
        from src.eval.metrics import mae, rmse, mape, evaluate_forecast
        tester.test("Import metrics module", True, "All functions available")
    except ImportError as e:
        tester.test("Import metrics module", False, f"Error: {e}")
    
    # Test transform module
    try:
        from src.features.transform import create_features
        tester.test("Import transform module", True)
    except ImportError as e:
        tester.test("Import transform module", False, f"Error: {e}")
    
    # Test baseline module
    try:
        from src.models.baseline import PersistenceModel, MovingAverageModel
        tester.test("Import baseline module", True)
    except ImportError as e:
        tester.test("Import baseline module", False, f"Error: {e}")
    
    return tester


def test_dependencies():
    """Test required Python packages."""
    tester = PipelineTester()
    tester.section("DEPENDENCIES")
    
    packages = [
        'pandas',
        'numpy',
        'pyarrow',
        'holidays',
        'streamlit',
        'plotly'
    ]
    
    for package in packages:
        try:
            __import__(package)
            tester.test(f"Package installed: {package}", True)
        except ImportError:
            tester.test(f"Package installed: {package}", False, f"Install: pip install {package}")
    
    return tester


def main():
    """Run all tests."""
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}🧪 ENERGY FORECAST PIPELINE TEST{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_testers = []
    
    # Run all test suites
    all_testers.append(test_dependencies())
    all_testers.append(test_code_modules())
    all_testers.append(test_phase1_data())
    all_testers.append(test_phase2_features())
    all_testers.append(test_phase3_forecasts())
    
    # Overall summary
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}OVERALL RESULTS{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")
    
    total_tests = sum(t.total_tests for t in all_testers)
    total_passed = sum(t.passed_tests for t in all_testers)
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_tests - total_passed}")
    print(f"Overall Success Rate: {overall_percentage:.1f}%\n")
    
    if overall_percentage == 100:
        print(f"{GREEN}{BOLD}{'='*70}{RESET}")
        print(f"{GREEN}{BOLD}🎉 ALL SYSTEMS GO! Pipeline is fully operational!{RESET}")
        print(f"{GREEN}{BOLD}{'='*70}{RESET}\n")
        print("✅ You're ready to move on to Phase 4 (ML Models)")
        print("🚀 Run: streamlit run src/app/dashboard.py\n")
        return 0
    elif overall_percentage >= 70:
        print(f"{YELLOW}{BOLD}⚠️  Pipeline mostly working. Review failures above.{RESET}\n")
        return 1
    else:
        print(f"{RED}{BOLD}❌ Pipeline has critical issues. Fix failures before proceeding.{RESET}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
    