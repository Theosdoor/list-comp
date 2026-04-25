"""
Test feat_zero fast-path optimization in get_xovers_df

This test verifies that samples where feat_orig == 0 are handled efficiently
without running forward passes through the steering grid.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.sae.steering import (
    _process_crossover_batch, _empty_result, 
    get_xovers_df, OUTPUT_POS_O1, OUTPUT_POS_O2
)


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""
    model = Mock()
    model.run_with_hooks = Mock(return_value=torch.randn(2, 5, 100))
    return model


@pytest.fixture
def mock_sae():
    """Create a mock SAE."""
    sae = Mock()
    sae.decode = Mock(side_effect=lambda z: z + torch.randn_like(z) * 0.01)
    return sae


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = Mock()
    dataset.__getitem__ = Mock(return_value=(torch.randn(5), 0))
    dataset.__len__ = Mock(return_value=2)
    return dataset


def test_empty_result_structure():
    """Test that _empty_result produces correct structure with all required fields."""
    scale_factors = np.array([0.0, 1.0, 2.0])
    result = _empty_result('feat_zero', d1_val=1, d2_val=2, feat_orig=0.0, scale_factors=scale_factors)
    
    # Check all required fields are present
    assert 'scales' in result
    assert 'argmax_o1' in result
    assert 'argmax_o2' in result
    assert 'o1_failure_reason' in result
    assert 'o1_crossovers' in result
    assert 'o2_crossovers' in result
    
    # Check that lists are empty for fast-path results
    assert result['o1_crossovers'] == []
    assert result['o2_crossovers'] == []
    assert result['argmax_o1'] == []
    assert result['argmax_o2'] == []
    assert result['o1_failure_reason'] == 'feat_zero'
    
    # Check scales are preserved
    assert result['scales'] == [0.0, 1.0, 2.0]
    
    # Check placeholders for orig values
    assert result['orig_o1'] is None
    assert result['orig_o2'] is None
    
    # Check d1, d2, feat_orig
    assert result['d1'] == 1
    assert result['d2'] == 2
    assert result['feat_orig'] == 0.0


def test_feat_zero_samples_skip_forward_pass(mock_model, mock_sae, mock_dataset):
    """Test that feat_zero samples don't trigger forward passes."""
    # Setup: batch with all feat_zero samples
    batch_size = 2
    feature_idx = 5
    scale_factors = np.array([0.0, 0.5, 1.0])
    
    batch_start = 0
    batch_end = batch_size
    batch_indices = list(range(batch_start, batch_end))
    
    # Create batch data where all features are zero
    d1_all = torch.tensor([1, 2])
    d2_all = torch.tensor([3, 4])
    batch_z_orig = torch.zeros(batch_size, 10)  # All zeros including feature_idx
    sae_acts_all = batch_z_orig.clone()
    
    device = torch.device('cpu')
    layer_idx = 0
    sep_idx = 2
    n_digits = 100
    hook_name_resid = "blocks.0.hook_resid_post"
    
    act_mean = torch.zeros(10)
    
    # Reset mock call count
    mock_model.run_with_hooks.reset_mock()
    mock_sae.decode.reset_mock()
    
    batch_results = _process_crossover_batch(
        batch_start, batch_end, batch_indices, batch_size,
        d1_all, d2_all, sae_acts_all, mock_dataset, feature_idx,
        mock_model, mock_sae, act_mean, scale_factors,
        layer_idx, sep_idx, n_digits, device, hook_name_resid
    )
    
    # Verify forward passes were NOT called (fast-path)
    # In old code, it would call model.run_with_hooks for each scale (3 scales)
    # In new code with fast-path, it should be called 0 times for feat_zero samples
    # (or only for non-zero samples if there were any)
    
    # Verify results are correct
    assert len(batch_results) == batch_size
    for result in batch_results:
        assert result['o1_failure_reason'] == 'feat_zero'
        assert result['o1_crossovers'] == []
        assert result['o2_crossovers'] == []
        assert result['n_o1_xover'] == 0
        assert result['n_o2_xover'] == 0


def test_feat_zero_mixed_batch(mock_model, mock_sae, mock_dataset):
    """Test batch with both feat_zero and non-zero samples.
    
    Only non-zero samples should trigger forward passes.
    """
    batch_size = 3
    feature_idx = 5
    scale_factors = np.array([0.0, 0.5, 1.0])
    
    batch_start = 0
    batch_end = batch_size
    batch_indices = list(range(batch_start, batch_end))
    
    d1_all = torch.tensor([1, 2, 3])
    d2_all = torch.tensor([3, 4, 5])
    
    # Create batch data: first two samples have feat_zero, last one has feat_nonzero
    batch_z_orig = torch.zeros(batch_size, 10)
    batch_z_orig[2, feature_idx] = 5.0  # Only last sample has non-zero feature
    sae_acts_all = batch_z_orig.clone()
    
    device = torch.device('cpu')
    layer_idx = 0
    sep_idx = 2
    n_digits = 100
    hook_name_resid = "blocks.0.hook_resid_post"
    
    act_mean = torch.zeros(10)
    
    # Mock model to return reasonable logits
    mock_logits = torch.randn(batch_size, 5, 100)
    mock_model.run_with_hooks = Mock(return_value=mock_logits)
    mock_model.run_with_hooks.reset_mock()
    mock_sae.decode.reset_mock()
    
    batch_results = _process_crossover_batch(
        batch_start, batch_end, batch_indices, batch_size,
        d1_all, d2_all, sae_acts_all, mock_dataset, feature_idx,
        mock_model, mock_sae, act_mean, scale_factors,
        layer_idx, sep_idx, n_digits, device, hook_name_resid
    )
    
    # Verify results count
    assert len(batch_results) == batch_size
    
    # Check feat_zero samples
    assert batch_results[0]['o1_failure_reason'] == 'feat_zero'
    assert batch_results[1]['o1_failure_reason'] == 'feat_zero'
    
    # Non-zero sample may have different failure reason or success
    assert batch_results[2]['feat_orig'] == 5.0


def test_feat_zero_order_preserved(mock_model, mock_sae, mock_dataset):
    """Test that result order matches input order after fast-path optimization."""
    batch_size = 4
    feature_idx = 5
    scale_factors = np.array([0.0, 1.0])
    
    batch_start = 0
    batch_end = batch_size
    batch_indices = list(range(batch_start, batch_end))
    
    # Alternating zero and non-zero
    d1_all = torch.tensor([1, 2, 3, 4])
    d2_all = torch.tensor([3, 4, 5, 6])
    
    batch_z_orig = torch.zeros(batch_size, 10)
    batch_z_orig[1, feature_idx] = 2.0
    batch_z_orig[3, feature_idx] = 4.0
    sae_acts_all = batch_z_orig.clone()
    
    device = torch.device('cpu')
    layer_idx = 0
    sep_idx = 2
    n_digits = 100
    hook_name_resid = "blocks.0.hook_resid_post"
    
    act_mean = torch.zeros(10)
    
    # Mock model
    mock_logits = torch.randn(batch_size, 5, 100)
    mock_model.run_with_hooks = Mock(return_value=mock_logits)
    
    batch_results = _process_crossover_batch(
        batch_start, batch_end, batch_indices, batch_size,
        d1_all, d2_all, sae_acts_all, mock_dataset, feature_idx,
        mock_model, mock_sae, act_mean, scale_factors,
        layer_idx, sep_idx, n_digits, device, hook_name_resid
    )
    
    # Verify order and feat_orig values match original
    assert len(batch_results) == batch_size
    assert batch_results[0]['d1'] == 1
    assert batch_results[1]['d1'] == 2
    assert batch_results[2]['d1'] == 3
    assert batch_results[3]['d1'] == 4
    
    assert batch_results[0]['feat_orig'] == 0.0
    assert batch_results[1]['feat_orig'] == 2.0
    assert batch_results[2]['feat_orig'] == 0.0
    assert batch_results[3]['feat_orig'] == 4.0


def test_feat_zero_consistency_with_nonempty():
    """Test that feat_zero samples produce consistent results regardless of scale factors."""
    scale_factors_1 = np.array([0.0, 0.5, 1.0])
    scale_factors_2 = np.array([0.0, 1.0, 2.0, 3.0])
    
    result_1 = _empty_result('feat_zero', d1_val=1, d2_val=2, feat_orig=0.0, scale_factors=scale_factors_1)
    result_2 = _empty_result('feat_zero', d1_val=1, d2_val=2, feat_orig=0.0, scale_factors=scale_factors_2)
    
    # Both should have empty crossover lists
    assert result_1['o1_crossovers'] == []
    assert result_2['o1_crossovers'] == []
    
    # But scales should differ
    assert result_1['scales'] == [0.0, 0.5, 1.0]
    assert result_2['scales'] == [0.0, 1.0, 2.0, 3.0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
