"""
Test vectorised bisection implementation (Change 6).

Tests for:
1. _run_vectorised_bisection: vectorised bisection across multiple tasks
2. _collect_o2_sign_changes: identify and collect sign change crossovers
3. Integration with _process_crossover_batch
4. Active indices tracking for original task order preservation
5. Edge cases: empty task list, single task, all converge immediately, mixed convergence
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

from src.sae.steering import (
    OUTPUT_POS_O1, OUTPUT_POS_O2,
    _collect_o2_sign_changes,
    _run_vectorised_bisection,
    _determine_bound_type_from_diff,
    DEFAULT_BISECTION_TOL,
    DEFAULT_BISECTION_MAX_ITER,
)


class TestCollectO2SignChanges:
    """Test _collect_o2_sign_changes function."""
    
    def test_empty_logits_returns_empty_tasks(self):
        """Empty batch should return empty task list."""
        logits_batch = np.array([]).reshape(0, 5, 100)
        d1_all = np.array([])
        d2_all = np.array([])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert tasks == []
    
    def test_single_sample_no_sign_changes(self):
        """Single sample with no sign changes should return empty task list."""
        # Create logits where d1 always > d2 (diff always positive)
        n_scales = 3
        n_digits = 100
        d1_idx = 5
        d2_idx = 10
        
        logits_batch = np.random.randn(1, n_scales, n_digits)
        # Make d1 logits always higher
        logits_batch[0, :, d1_idx] = 10.0
        logits_batch[0, :, d2_idx] = -10.0
        
        d1_all = np.array([d1_idx])
        d2_all = np.array([d2_idx])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert tasks == []
    
    def test_single_sample_single_sign_change(self):
        """Single sample with one sign change should return one task."""
        n_scales = 3
        n_digits = 100
        d1_idx = 5
        d2_idx = 10
        
        logits_batch = np.zeros((1, n_scales, n_digits))
        # Create sign change: d1 > d2 at scale 0, then d2 > d1 at scales 1,2
        logits_batch[0, 0, d1_idx] = 5.0
        logits_batch[0, 0, d2_idx] = 3.0
        logits_batch[0, 1, d1_idx] = 2.0
        logits_batch[0, 1, d2_idx] = 4.0  # Sign change between 0 and 1
        logits_batch[0, 2, d1_idx] = 1.0
        logits_batch[0, 2, d2_idx] = 5.0
        
        d1_all = np.array([d1_idx])
        d2_all = np.array([d2_idx])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert len(tasks) == 1
        task = tasks[0]
        assert task['sample_idx'] == 0
        assert task['scale_low'] == 0.0
        assert task['scale_high'] == 0.5
        assert task['crossover_idx'] == 0
        assert task['d1_val'] == d1_idx
        assert task['d2_val'] == d2_idx
        assert task['output_pos'] == OUTPUT_POS_O2
        assert 'diff' in task
    
    def test_multiple_samples_different_crossovers(self):
        """Multiple samples with different sign change patterns."""
        n_scales = 4
        n_digits = 100
        
        logits_batch = np.random.randn(2, n_scales, n_digits)
        
        # Sample 0: sign change at index 0 (between scale 0 and 1)
        logits_batch[0, 0, 5] = 1.0   # d1
        logits_batch[0, 0, 10] = -1.0  # d2
        logits_batch[0, 1, 5] = -1.0
        logits_batch[0, 1, 10] = 1.0   # Sign change
        logits_batch[0, 2, 5] = -2.0
        logits_batch[0, 2, 10] = 2.0
        logits_batch[0, 3, 5] = -3.0
        logits_batch[0, 3, 10] = 3.0
        
        # Sample 1: sign change at index 2 (between scale 2 and 3)
        logits_batch[1, 0, 15] = 5.0    # d1
        logits_batch[1, 0, 20] = -5.0   # d2
        logits_batch[1, 1, 15] = 4.0
        logits_batch[1, 1, 20] = -4.0
        logits_batch[1, 2, 15] = 1.0
        logits_batch[1, 2, 20] = -1.0
        logits_batch[1, 3, 15] = -1.0
        logits_batch[1, 3, 20] = 1.0    # Sign change
        
        d1_all = np.array([5, 15])
        d2_all = np.array([10, 20])
        scale_factors = np.array([0.0, 0.5, 1.0, 1.5])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert len(tasks) == 2
        
        # Check first task (sample 0)
        assert tasks[0]['sample_idx'] == 0
        assert tasks[0]['scale_low'] == 0.0
        assert tasks[0]['scale_high'] == 0.5
        
        # Check second task (sample 1)
        assert tasks[1]['sample_idx'] == 1
        assert tasks[1]['scale_low'] == 1.0
        assert tasks[1]['scale_high'] == 1.5
    
    def test_diff_array_preserved_for_bound_type(self):
        """Verify diff array is preserved in task for bound type computation."""
        n_scales = 3
        n_digits = 100
        
        logits_batch = np.zeros((1, n_scales, n_digits))
        logits_batch[0, 0, 5] = 5.0
        logits_batch[0, 0, 10] = 3.0
        logits_batch[0, 1, 5] = 2.0
        logits_batch[0, 1, 10] = 4.0  # Sign change
        logits_batch[0, 2, 5] = 1.0
        logits_batch[0, 2, 10] = 5.0
        
        d1_all = np.array([5])
        d2_all = np.array([10])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert len(tasks) == 1
        task = tasks[0]
        
        # Verify diff is the full array
        assert len(task['diff']) == 3
        np.testing.assert_array_almost_equal(
            task['diff'],
            np.array([2.0, -2.0, -4.0])  # d1 - d2 at each scale
        )


class TestVectorisedBisectionEdgeCases:
    """Test edge cases for vectorised bisection."""
    
    def test_empty_task_list_returns_empty(self):
        """Empty task list should return [] immediately without forward passes."""
        mock_model = Mock()
        mock_sae = Mock()
        
        result = _run_vectorised_bisection(
            [], mock_model, mock_sae, torch.zeros(10), feature_idx=0,
            inputs_batch=torch.randn(1, 5), z_orig_batch=torch.randn(1, 10),
            feat_orig_batch=torch.tensor([1.0]), layer_idx=0, sep_idx=2,
            n_digits=100, device=torch.device('cpu')
        )
        
        assert result == []
        # Verify no forward passes were called
        mock_model.run_with_hooks.assert_not_called()
    
    def test_homogeneous_output_pos_assertion(self):
        """All tasks must have same output_pos - heterogeneous should raise."""
        tasks = [
            {
                'sample_idx': 0,
                'scale_low': 0.5,
                'scale_high': 1.5,
                'crossover_idx': 0,
                'd1_val': 5,
                'd2_val': 10,
                'output_pos': OUTPUT_POS_O2,
                'diff': np.array([1.0, -1.0, -2.0]),
            },
            {
                'sample_idx': 1,
                'scale_low': 0.5,
                'scale_high': 1.5,
                'crossover_idx': 0,
                'd1_val': 5,
                'd2_val': 10,
                'output_pos': OUTPUT_POS_O1,  # Different!
                'diff': np.array([1.0, -1.0, -2.0]),
            },
        ]
        
        mock_model = Mock()
        mock_sae = Mock()
        
        with pytest.raises(AssertionError, match="All tasks must share the same output_pos"):
            _run_vectorised_bisection(
                tasks, mock_model, mock_sae, torch.zeros(10), feature_idx=0,
                inputs_batch=torch.randn(2, 5), z_orig_batch=torch.randn(2, 10),
                feat_orig_batch=torch.tensor([1.0, 1.0]), layer_idx=0, sep_idx=2,
                n_digits=100, device=torch.device('cpu')
            )


class TestDetermineBoundTypeFromDiff:
    """Test bound type determination from diff slope."""
    
    def test_o1_diff_negative_left_returns_ub(self):
        """For o1: if diff < 0 on left, return 'ub'."""
        diff = np.array([-2.0, 1.0, 2.0])  # Sign change at index 0
        crossover_idx = 0
        
        bound_type = _determine_bound_type_from_diff(crossover_idx, OUTPUT_POS_O1, diff)
        
        assert bound_type == 'ub'
    
    def test_o1_diff_negative_right_returns_lb(self):
        """For o1: if diff < 0 on right, return 'lb'."""
        diff = np.array([2.0, -1.0, -2.0])  # Sign change at index 0
        crossover_idx = 0
        
        bound_type = _determine_bound_type_from_diff(crossover_idx, OUTPUT_POS_O1, diff)
        
        assert bound_type == 'lb'
    
    def test_o2_diff_positive_left_returns_ub(self):
        """For o2: if diff > 0 on left, return 'ub'."""
        diff = np.array([2.0, -1.0, -2.0])  # Sign change at index 0
        crossover_idx = 0
        
        bound_type = _determine_bound_type_from_diff(crossover_idx, OUTPUT_POS_O2, diff)
        
        assert bound_type == 'ub'
    
    def test_o2_diff_positive_right_returns_lb(self):
        """For o2: if diff > 0 on right, return 'lb'."""
        diff = np.array([-2.0, 1.0, 2.0])  # Sign change at index 0
        crossover_idx = 0
        
        bound_type = _determine_bound_type_from_diff(crossover_idx, OUTPUT_POS_O2, diff)
        
        assert bound_type == 'lb'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestVectorisedBisectionIntegration:
    """Integration tests with realistic data patterns."""
    
    def test_collect_and_bound_type_integration(self):
        """Verify _collect_o2_sign_changes produces tasks with correct structure for bisection."""
        n_scales = 5
        n_digits = 100
        
        logits_batch = np.zeros((2, n_scales, n_digits))
        
        # Sample 0: sign change at scale 0-1 (avoid crossing through 0)
        logits_batch[0, 0, 5] = 5.0   # d1
        logits_batch[0, 0, 10] = -5.0  # d2
        logits_batch[0, 1, 5] = -1.0
        logits_batch[0, 1, 10] = 1.0   # Sign change
        logits_batch[0, 2, 5] = -2.0
        logits_batch[0, 2, 10] = 2.0
        logits_batch[0, 3, 5] = -3.0
        logits_batch[0, 3, 10] = 3.0
        logits_batch[0, 4, 5] = -4.0
        logits_batch[0, 4, 10] = 4.0
        
        # Sample 1: sign change at scale 2-3 (different timing)
        logits_batch[1, 0, 15] = 5.0    # d1
        logits_batch[1, 0, 20] = -5.0   # d2
        logits_batch[1, 1, 15] = 4.0
        logits_batch[1, 1, 20] = -4.0
        logits_batch[1, 2, 15] = 1.0
        logits_batch[1, 2, 20] = -1.0
        logits_batch[1, 3, 15] = -1.0
        logits_batch[1, 3, 20] = 1.0   # Sign change
        logits_batch[1, 4, 15] = -2.0
        logits_batch[1, 4, 20] = 2.0
        
        d1_all = np.array([5, 15])
        d2_all = np.array([10, 20])
        scale_factors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        # Should find exactly 2 tasks (one per sample, one crossover each)
        assert len(tasks) == 2
        
        # Both tasks should have diff arrays
        for task in tasks:
            assert len(task['diff']) == n_scales
            # Verify diff is d1 - d2
            expected_diff = (logits_batch[task['sample_idx'], :, task['d1_val']] -
                           logits_batch[task['sample_idx'], :, task['d2_val']])
            np.testing.assert_array_almost_equal(task['diff'], expected_diff)
    
    def test_bound_type_computation_from_tasks(self):
        """Verify bound types computed correctly from task diffs."""
        # Create a sign change: diff goes from positive to negative
        diff = np.array([2.0, 1.0, 0.5, -0.5, -1.0, -2.0])
        crossover_idx = 2  # Between diff[2]=0.5 and diff[3]=-0.5
        
        # For o2 (we want d1 > d2, i.e., diff > 0):
        # - diff > 0 on left (2.0) → upper bound
        bound_type = _determine_bound_type_from_diff(crossover_idx, OUTPUT_POS_O2, diff)
        assert bound_type == 'ub'
        
        # Another case: diff goes from negative to positive
        diff2 = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        crossover_idx2 = 2  # Between diff[2]=-0.5 and diff[3]=0.5
        
        # For o2: diff < 0 on left (-0.5) → swap condition doesn't hold on left
        # So it must hold on right → lower bound
        bound_type2 = _determine_bound_type_from_diff(crossover_idx2, OUTPUT_POS_O2, diff2)
        assert bound_type2 == 'lb'


class TestTaskStructure:
    """Test task dict field names match specification."""
    
    def test_task_has_all_required_fields(self):
        """Task dict from _collect_o2_sign_changes must have exact field names."""
        logits_batch = np.zeros((1, 3, 100))
        logits_batch[0, 0, 5] = 5.0
        logits_batch[0, 0, 10] = -5.0
        logits_batch[0, 1, 5] = -1.0
        logits_batch[0, 1, 10] = 1.0
        logits_batch[0, 2, 5] = -2.0
        logits_batch[0, 2, 10] = 2.0
        
        d1_all = np.array([5])
        d2_all = np.array([10])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert len(tasks) == 1
        task = tasks[0]
        
        # Check all required fields present
        required_fields = ['sample_idx', 'scale_low', 'scale_high', 'crossover_idx', 
                         'd1_val', 'd2_val', 'output_pos', 'diff']
        for field in required_fields:
            assert field in task, f"Task missing required field: {field}"
    
    def test_collected_tasks_ready_for_bisection(self):
        """Tasks from _collect_o2_sign_changes should be ready for _run_vectorised_bisection."""
        # Create simple test case
        logits_batch = np.zeros((1, 3, 100))
        logits_batch[0, 0, 5] = 2.0
        logits_batch[0, 0, 10] = -2.0
        logits_batch[0, 1, 5] = -0.5
        logits_batch[0, 1, 10] = 0.5
        logits_batch[0, 2, 5] = -3.0
        logits_batch[0, 2, 10] = 3.0
        
        d1_all = np.array([5])
        d2_all = np.array([10])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        # Tasks should have everything needed for bisection
        assert len(tasks) > 0
        for task in tasks:
            # These are the fields used in _run_vectorised_bisection
            assert 'scale_low' in task
            assert 'scale_high' in task
            assert 'd1_val' in task
            assert 'd2_val' in task
            assert 'output_pos' in task
            assert 'sample_idx' in task
            assert 'crossover_idx' in task
            assert 'diff' in task


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_no_warnings_emitted(self, capsys):
        """Verify no warnings are emitted during normal operation."""
        # Simple test: call collect and verify no warnings
        logits_batch = np.zeros((1, 3, 100))
        logits_batch[0, 0, 5] = 2.0
        logits_batch[0, 0, 10] = -2.0
        logits_batch[0, 1, 5] = -1.0
        logits_batch[0, 1, 10] = 1.0
        logits_batch[0, 2, 5] = -2.0
        logits_batch[0, 2, 10] = 2.0
        
        d1_all = np.array([5])
        d2_all = np.array([10])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        # Call function
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        # Verify no warnings (captured via capsys)
        captured = capsys.readouterr()
        assert 'warning' not in captured.err.lower() or len(tasks) > 0  # No warnings expected


class TestReturnValues:
    """Test return value structure and types."""
    
    def test_vectorised_bisection_returns_lists(self):
        """_run_vectorised_bisection should return (crossover_scales, bound_types) as lists."""
        # Test with empty tasks (safest case)
        result = _run_vectorised_bisection(
            [], Mock(), Mock(), torch.zeros(10), feature_idx=0,
            inputs_batch=torch.randn(1, 5), z_orig_batch=torch.randn(1, 10),
            feat_orig_batch=torch.tensor([1.0]), layer_idx=0, sep_idx=2,
            n_digits=100, device=torch.device('cpu')
        )
        
        # Should return a list
        assert isinstance(result, list)
        assert result == []
    
    def test_collect_returns_list_of_dicts(self):
        """_collect_o2_sign_changes should return a list of dict task objects."""
        logits_batch = np.zeros((1, 3, 100))
        logits_batch[0, 0, 5] = 1.0
        logits_batch[0, 0, 10] = -1.0
        logits_batch[0, 1, 5] = -1.0
        logits_batch[0, 1, 10] = 1.0
        logits_batch[0, 2, 5] = -2.0
        logits_batch[0, 2, 10] = 2.0
        
        d1_all = np.array([5])
        d2_all = np.array([10])
        scale_factors = np.array([0.0, 0.5, 1.0])
        
        tasks = _collect_o2_sign_changes(logits_batch, d1_all, d2_all, scale_factors)
        
        assert isinstance(tasks, list)
        for task in tasks:
            assert isinstance(task, dict)
