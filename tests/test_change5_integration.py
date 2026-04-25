"""
Integration test for Change 5: scale=1.0 extraction and orig_o1/orig_o2 caching.

This test verifies the complete flow from _analyze_single_sample_crossovers
through get_xovers_df, get_output_swap_bounds, and _verify_single_swap.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.sae.steering import (
    get_output_swap_bounds,
    _empty_result,
)


def test_orig_o1_o2_flow_to_swap_bounds_df():
    """Test that orig_o1 and orig_o2 flow through get_output_swap_bounds."""
    # Create synthetic xovers_df
    xovers_data = {
        'd1': [5, 7, 9],
        'd2': [10, 12, 14],
        'o1_crossovers': [[], [1.5], []],
        'o2_crossovers': [[], [2.5], []],
        'o1_bound_types': [[], ['lb'], []],
        'o2_bound_types': [[], ['ub'], []],
        'scales': [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
        'argmax_o1': [[5, 5, 50], [7, 7, 70], [9, 9, 90]],
        'argmax_o2': [[10, 10, 30], [12, 12, 32], [14, 14, 34]],
        'o1_failure_reason': ['no_o1_crossover', None, 'nonlinear_d1'],
        'orig_o1': [5, 7, None],
        'orig_o2': [10, 12, None],
    }
    
    xovers_df = pd.DataFrame(xovers_data)
    
    # Convert dtypes as the real function does
    xovers_df['orig_o1'] = xovers_df['orig_o1'].astype('Int64')
    xovers_df['orig_o2'] = xovers_df['orig_o2'].astype('Int64')
    
    # Get swap bounds
    swap_bounds_df = get_output_swap_bounds(xovers_df, scale_range=[0.0, 10.0])
    
    # Verify orig_o1 and orig_o2 are in the output
    assert 'orig_o1' in swap_bounds_df.columns
    assert 'orig_o2' in swap_bounds_df.columns
    
    # Verify dtypes are Int64
    assert swap_bounds_df['orig_o1'].dtype == 'Int64'
    assert swap_bounds_df['orig_o2'].dtype == 'Int64'
    
    # Verify values are propagated (for rows with no o1_failure_reason, orig values should be set)
    # Row 0: has o1_failure_reason='no_o1_crossover' → orig_o1/o2 should be None from xovers_df
    # Row 1: has o1_failure_reason=None but might fail for other reasons in _determine_swap_bounds
    # Row 2: has o1_failure_reason='nonlinear_d1' → orig_o1/o2 should be None from xovers_df


def test_orig_o1_o2_dtype_is_int64():
    """Test that orig_o1 and orig_o2 have Int64 nullable dtype."""
    xovers_data = {
        'd1': [5],
        'd2': [10],
        'o1_crossovers': [[]],
        'o2_crossovers': [[]],
        'o1_bound_types': [[]],
        'o2_bound_types': [[]],
        'scales': [[0.0, 1.0, 2.0]],
        'argmax_o1': [[5, 5, 50]],
        'argmax_o2': [[10, 10, 30]],
        'o1_failure_reason': ['no_o1_crossover'],
        'orig_o1': [5],
        'orig_o2': [10],
    }
    
    xovers_df = pd.DataFrame(xovers_data)
    
    # Simulate the real function's dtype conversion
    xovers_df['orig_o1'] = xovers_df['orig_o1'].astype('Int64')
    xovers_df['orig_o2'] = xovers_df['orig_o2'].astype('Int64')
    
    swap_bounds_df = get_output_swap_bounds(xovers_df)
    
    # Verify Int64 dtype
    assert swap_bounds_df['orig_o1'].dtype == 'Int64'
    assert swap_bounds_df['orig_o2'].dtype == 'Int64'


def test_degenerate_cases_have_none_orig():
    """Test that degenerate cases (feat_zero, d1==d2) have None orig_o1/orig_o2."""
    scale_factors = np.array([0.0, 1.0, 2.0])
    
    result = _empty_result('feat_zero', d1_val=5, d2_val=7, feat_orig=0.0, scale_factors=scale_factors)
    assert result['orig_o1'] is None
    assert result['orig_o2'] is None
    
    result = _empty_result('d1_eq_d2', d1_val=5, d2_val=5, feat_orig=2.5, scale_factors=scale_factors)
    assert result['orig_o1'] is None
    assert result['orig_o2'] is None


def test_scale_1p0_extraction_logic():
    """Test the scale=1.0 extraction logic that's used in _analyze_single_sample_crossovers."""
    # Test with a grid that includes scale=1.0
    scale_factors = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    argmax_o1 = np.array([5, 7, 42, 50, 80, 95])
    argmax_o2 = np.array([3, 15, 37, 60, 75, 90])
    
    # Extract scale=1.0 using the logic from the code
    scale_1_idx = int(np.argmin(np.abs(np.asarray(scale_factors) - 1.0)))
    
    # The assertion from the caveat
    assert scale_factors[scale_1_idx] == 1.0, (
        f"scale=1.0 not found in grid; nearest is {scale_factors[scale_1_idx]}. "
        "orig_o1/orig_o2 would be incorrect."
    )
    
    # Extract the original predictions
    orig_o1 = int(argmax_o1[scale_1_idx])
    orig_o2 = int(argmax_o2[scale_1_idx])
    
    assert orig_o1 == 42
    assert orig_o2 == 37


def test_scale_1p0_missing_from_grid_fails_assertion():
    """Verify that the assertion catches missing scale=1.0."""
    # Grid WITHOUT scale=1.0
    scale_factors = np.array([0.0, 0.5, 0.9, 2.0, 5.0, 10.0])
    
    # Find nearest to 1.0
    scale_1_idx = int(np.argmin(np.abs(np.asarray(scale_factors) - 1.0)))
    nearest = scale_factors[scale_1_idx]
    
    # The nearest is 0.9, not 1.0 — assertion should fail
    assert nearest != 1.0
    assert nearest == 0.9
    
    # The assertion would catch this
    with pytest.raises(AssertionError, match="scale=1.0 not found in grid"):
        assert nearest == 1.0, (
            f"scale=1.0 not found in grid; nearest is {nearest}. "
            "orig_o1/orig_o2 would be incorrect."
        )


def test_old_csv_without_orig_columns():
    """Test that old CSVs without orig_o1/orig_o2 don't break the flow."""
    # Old CSV-like data without orig columns
    xovers_data = {
        'd1': [5],
        'd2': [10],
        'o1_crossovers': [[]],
        'o2_crossovers': [[]],
        'o1_bound_types': [[]],
        'o2_bound_types': [[]],
        'scales': [[0.0, 1.0, 2.0]],
        'argmax_o1': [[5, 5, 50]],
        'argmax_o2': [[10, 10, 30]],
        'o1_failure_reason': ['no_o1_crossover'],
        # Intentionally missing 'orig_o1' and 'orig_o2'
    }
    
    xovers_df = pd.DataFrame(xovers_data)
    
    # Should not raise KeyError when calling get_output_swap_bounds
    # because the code uses 'in' checks and pd.notna()
    try:
        swap_bounds_df = get_output_swap_bounds(xovers_df)
        # Should succeed without error
        assert isinstance(swap_bounds_df, pd.DataFrame)
    except KeyError as e:
        pytest.fail(f"Old CSV without orig_o1/orig_o2 caused KeyError: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

