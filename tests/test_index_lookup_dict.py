"""
Test index lookup dict optimization in steering.py

This test verifies that the _build_index_lookup helper function creates correct
{(d1, d2): index} mappings for O(1) lookup, replacing the O(n) _find_input_index calls.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.sae.steering import (
    _find_input_index,
)


@pytest.fixture
def simple_dataset():
    """Create simple test dataset with 10 samples."""
    d1_all = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
    d2_all = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return d1_all, d2_all


@pytest.fixture
def full_enumeration_dataset():
    """Create full-enumeration dataset with all (d1, d2) pairs for 2 digits and list_len=2."""
    # For n_digits=2, list_len=2: we have 2^2 = 4 combinations: (0,0), (0,1), (1,0), (1,1)
    d1_all = torch.tensor([0, 0, 1, 1])
    d2_all = torch.tensor([0, 1, 0, 1])
    return d1_all, d2_all


def test_build_index_lookup_basic(simple_dataset):
    """Test basic functionality of building index lookup dict."""
    d1_all, d2_all = simple_dataset
    
    # Import the function now that we've implemented it
    from src.sae.steering import _build_index_lookup
    
    # Build lookup
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Check it's a dict
    assert isinstance(idx_lookup, dict)
    
    # Check size matches input (all pairs should be unique in full enumeration)
    # In this test dataset, we have duplicates, so it should be 9 unique pairs
    expected_unique = len(set(zip(d1_all.tolist(), d2_all.tolist())))
    assert len(idx_lookup) == expected_unique
    
    # Check specific lookups work
    assert idx_lookup[(0, 1)] == 0  # First (0, 1) pair
    assert idx_lookup[(1, 2)] == 1
    assert idx_lookup[(2, 3)] == 2


def test_build_index_lookup_full_enumeration(full_enumeration_dataset):
    """Test with full-enumeration dataset where all pairs are unique."""
    d1_all, d2_all = full_enumeration_dataset
    
    from src.sae.steering import _build_index_lookup
    
    # Build lookup
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Check all 4 pairs are present
    assert len(idx_lookup) == 4
    assert idx_lookup[(0, 0)] == 0
    assert idx_lookup[(0, 1)] == 1
    assert idx_lookup[(1, 0)] == 2
    assert idx_lookup[(1, 1)] == 3


def test_build_index_lookup_keys_are_tuples(full_enumeration_dataset):
    """Test that dict keys are (int, int) tuples."""
    d1_all, d2_all = full_enumeration_dataset
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # All keys should be tuples of ints
    for key in idx_lookup.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], int)
        assert isinstance(key[1], int)


def test_build_index_lookup_values_are_indices(full_enumeration_dataset):
    """Test that dict values are valid indices."""
    d1_all, d2_all = full_enumeration_dataset
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # All values should be valid indices
    for value in idx_lookup.values():
        assert isinstance(value, int)
        assert 0 <= value < len(d1_all)


def test_lookup_dict_consistency_with_find_input_index(simple_dataset):
    """Test that dict lookup gives same results as _find_input_index."""
    d1_all, d2_all = simple_dataset
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Test several lookups
    test_pairs = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 5)]
    
    for d1_val, d2_val in test_pairs:
        # Get index using both methods
        idx_find = _find_input_index(d1_all, d2_all, d1_val, d2_val)
        idx_lookup_result = idx_lookup[(d1_val, d2_val)]
        
        # They should match
        assert idx_find == idx_lookup_result, f"Mismatch for ({d1_val}, {d2_val}): {idx_find} vs {idx_lookup_result}"


def test_lookup_dict_handles_large_dataset():
    """Test that dict lookup works with larger dataset (e.g., 100 digits, list_len=2)."""
    # Simulate 100^2 = 10000 full-enumeration samples
    n_samples = 10000
    d1_all = torch.arange(n_samples) // 100  # 0-99 each repeated 100 times
    d2_all = torch.arange(n_samples) % 100   # 0-99 cycling
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # All 10000 pairs should be unique in this case
    assert len(idx_lookup) == n_samples
    
    # Test random lookups
    test_indices = [0, 100, 500, 5000, 9999]
    for idx in test_indices:
        d1_val = int(d1_all[idx].item())
        d2_val = int(d2_all[idx].item())
        assert idx_lookup[(d1_val, d2_val)] == idx


def test_lookup_dict_keys_matches_dataset_pairs(full_enumeration_dataset):
    """Test that dict keys exactly match the (d1, d2) pairs in the dataset."""
    d1_all, d2_all = full_enumeration_dataset
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Create set of expected pairs
    expected_pairs = set(zip(d1_all.tolist(), d2_all.tolist()))
    actual_pairs = set(idx_lookup.keys())
    
    assert expected_pairs == actual_pairs


def test_build_index_lookup_empty_inputs():
    """Test handling of empty inputs."""
    d1_all = torch.tensor([], dtype=torch.long)
    d2_all = torch.tensor([], dtype=torch.long)
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Should return empty dict
    assert isinstance(idx_lookup, dict)
    assert len(idx_lookup) == 0


def test_build_index_lookup_single_element():
    """Test with single element."""
    d1_all = torch.tensor([42])
    d2_all = torch.tensor([99])
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Should have one entry
    assert len(idx_lookup) == 1
    assert idx_lookup[(42, 99)] == 0


def test_lookup_dict_with_tensor_dtypes():
    """Test that lookup works correctly regardless of tensor dtype."""
    # Test with different integer dtypes
    d1_all = torch.tensor([0, 1, 2], dtype=torch.int64)
    d2_all = torch.tensor([5, 6, 7], dtype=torch.int32)
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Should work fine - values converted to Python ints
    assert idx_lookup[(0, 5)] == 0
    assert idx_lookup[(1, 6)] == 1
    assert idx_lookup[(2, 7)] == 2


def test_lookup_dict_with_duplicates_maps_to_last_occurrence():
    """Test behavior with duplicate (d1, d2) pairs.
    
    Note: The full-enumeration dataset should not have duplicates, so this is
    documented for clarity. When duplicates exist, dict comprehension maps to
    the last occurrence (overwrites earlier entries).
    """
    # Create dataset with duplicate pairs
    d1_all = torch.tensor([1, 2, 3, 1, 2])  # Duplicate (1, ...) and (2, ...)
    d2_all = torch.tensor([4, 5, 6, 4, 5])  # Matching duplicates
    
    from src.sae.steering import _build_index_lookup
    
    idx_lookup = _build_index_lookup(d1_all, d2_all)
    
    # Duplicates map to last occurrence (dict overwrites)
    assert idx_lookup[(1, 4)] == 3  # Last occurrence at index 3
    assert idx_lookup[(2, 5)] == 4  # Last occurrence at index 4
    assert idx_lookup[(3, 6)] == 2  # Only one occurrence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
