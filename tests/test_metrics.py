"""
Evaluation Metrics Tests

Tests for MAE, RMSE, MAPE and other evaluation metrics.

Author: Frederik Lyager
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import mape, rmse


class TestMAPE:
    """Test Mean Absolute Percentage Error."""
    
    def test_mape_perfect_prediction(self):
        """Test MAPE with perfect predictions."""
        y_true = [100, 200, 300]
        y_pred = [100, 200, 300]
        
        result = mape(y_true, y_pred)
        assert result == 0.0
    
    def test_mape_basic_calculation(self):
        """Test MAPE basic calculation."""
        y_true = [100, 200]
        y_pred = [110, 180]
        
        # Expected: (|100-110|/100 + |200-180|/200) / 2 * 100
        # = (10/100 + 20/200) / 2 * 100 = (0.1 + 0.1) / 2 * 100 = 10%
        result = mape(y_true, y_pred)
        assert result == pytest.approx(10.0)
    
    def test_mape_handles_zeros(self):
        """Test that MAPE handles zero values correctly."""
        y_true = [0, 100, 200]
        y_pred = [10, 110, 180]
        
        # Should skip the zero value
        result = mape(y_true, y_pred)
        
        # Only calculate for non-zero values
        assert result > 0
        assert result < 100
    
    def test_mape_numpy_arrays(self):
        """Test MAPE works with numpy arrays."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        result = mape(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0


class TestRMSE:
    """Test Root Mean Squared Error."""
    
    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = [100, 200, 300]
        y_pred = [100, 200, 300]
        
        result = rmse(y_true, y_pred)
        assert result == 0.0
    
    def test_rmse_basic_calculation(self):
        """Test RMSE basic calculation."""
        y_true = [100, 200]
        y_pred = [110, 180]
        
        # Expected: sqrt(((10^2 + 20^2) / 2)) = sqrt(250) ≈ 15.81
        result = rmse(y_true, y_pred)
        assert result == pytest.approx(15.811, abs=0.01)
    
    def test_rmse_penalizes_large_errors(self):
        """Test that RMSE penalizes large errors more."""
        y_true = [100, 100]
        y_pred_small = [101, 101]  # Small errors
        y_pred_large = [100, 102]  # One larger error
        
        rmse_small = rmse(y_true, y_pred_small)
        rmse_large = rmse(y_true, y_pred_large)
        
        # Larger error should have higher RMSE
        assert rmse_large > rmse_small
    
    def test_rmse_numpy_arrays(self):
        """Test RMSE works with numpy arrays."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        
        result = rmse(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
    y_true = []
    y_pred = []
    
    # Empty arrays should return NaN or raise error
    try:
        result = mape(y_true, y_pred)
        assert np.isnan(result) or np.isinf(result)
    except (ValueError, ZeroDivisionError):
        pass  # This is acceptable behavior
    
    def test_mismatched_lengths(self):
        """Test behavior with mismatched array lengths."""
        y_true = [100, 200, 300]
        y_pred = [110, 190]
        
        # Should raise error or handle gracefully
        try:
            result = mape(y_true, y_pred)
            # If it doesn't raise, it should at least work on common length
        except (ValueError, IndexError):
            pass  # Expected behavior
    
    def test_negative_values(self):
        """Test metrics with negative values."""
        y_true = [-100, 200, 300]
        y_pred = [-110, 190, 310]
        
        # Metrics should still work
        result_rmse = rmse(y_true, y_pred)
        assert result_rmse > 0
        
        # MAPE might behave differently with negatives
        result_mape = mape(y_true, y_pred)
        assert isinstance(result_mape, float)
    
    def test_all_zeros(self):
        """Test with all zero actual values."""
        y_true = [0, 0, 0]
        y_pred = [10, 20, 30]
        
        # MAPE undefined (division by zero)
        # Should either raise or return inf/nan
        result = mape(y_true, y_pred)
        assert np.isnan(result) or np.isinf(result) or result == 0


class TestMetricsRealistic:
    """Test metrics with realistic CO2 forecast scenarios."""
    
    def test_good_forecast_scenario(self):
        """Test metrics for a good forecast (< 10% error)."""
        # Realistic CO2 values (g/kWh)
        y_true = [150, 160, 140, 155, 145]
        y_pred = [152, 158, 138, 157, 143]
        
        result_mape = mape(y_true, y_pred)
        result_rmse = rmse(y_true, y_pred)
        
        # Good forecast should have low errors
        assert result_mape < 5.0  # Less than 5% error
        assert result_rmse < 10.0  # Less than 10 g/kWh error
    
    def test_poor_forecast_scenario(self):
        """Test metrics for a poor forecast (> 20% error)."""
        y_true = [150, 160, 140, 155, 145]
        y_pred = [120, 190, 110, 185, 115]  # Poor predictions
        
        result_mape = mape(y_true, y_pred)
        result_rmse = rmse(y_true, y_pred)
        
        # Poor forecast should have high errors
        assert result_mape > 15.0  # More than 15% error
        assert result_rmse > 20.0  # More than 20 g/kWh error
    
    def test_baseline_comparison(self):
        """Test that ML should beat baseline metrics."""
        # Simulate baseline (persistence)
        y_true = [150, 160, 140, 155, 145]
        y_baseline = [150, 150, 160, 140, 155]  # Lag-1 persistence
        
        # Simulate better ML model
        y_ml = [152, 158, 138, 157, 143]
        
        mape_baseline = mape(y_true, y_baseline)
        mape_ml = mape(y_true, y_ml)
        
        # ML should be better (lower error)
        assert mape_ml < mape_baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
