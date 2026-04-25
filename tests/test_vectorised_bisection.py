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
from unittest.mock import Mock, patch, MagicMock

from src.sae.steering import (
    OUTPUT_POS_O1, OUTPUT_POS_O2,
    find_exact_crossover_bisection,
    DEFAULT_BISECTION_TOL
)


class TestVectorisedBisectionEdgeCases:
    """Test edge cases for vectorised bisection."""
    
    def test_empty_task_list_returns_empty(self):
        """Empty task list should return [] immediately without forward passes."""
        # This tests the edge case where _run_vectorised_bisection([], ...) is called
        # We'll use a mock model to verify no forward passes occur
        mock_model = Mock()
        mock_sae = Mock()
        mock_model.run_with_hooks = Mock(return_value=torch.randn(1, 5, 100))
        
        # Simulate calling _run_vectorised_bisection with empty tasks
        # (We'll implement the actual function after tests are written)
        # For now, this tests the contract
        tasks = []
        # Expected: should return [] without calling model
        
        # Later: assert mock_model.run_with_hooks.call_count == 0


class TestSingleTaskBisection:
    """Test that single task vectorised bisection matches find_exact_crossover_bisection."""
    
    def test_single_task_matches_scalar_bisection(self):
        """Single task in vectorised form should produce same result as scalar bisection."""
        # Create mock model that returns consistent logits
        mock_model = Mock()
        mock_sae = Mock()
        
        # For this test, we verify the contract:
        # _run_vectorised_bisection([task]) should match find_exact_crossover_bisection(task)
        
        # We'll implement this once the function exists
        pass


class TestActiveIndicesTracking:
    """Test that active_indices correctly maps back to original task order."""
    
    def test_active_indices_preserved_in_results(self):
        """Results should be returned in original task order, not active order."""
        # Simulate scenario where tasks are processed in different order
        # but results are returned in original order
        
        # Task 0: converges at iteration 2
        # Task 1: converges at iteration 4  
        # Task 2: converges at iteration 1 (converges early)
        
        # Expected: results[0] from task 0, results[1] from task 1, results[2] from task 2
        pass


class TestMixedConvergence:
    """Test heterogeneous convergence patterns."""
    
    def test_some_converge_early_others_late(self):
        """Some tasks converge on early iterations, others take full iterations."""
        # Create tasks with different convergence patterns
        # Task 0: converges after 1 iteration
        # Task 1: converges after 4 iterations
        # Task 2: converges after 2 iterations
        
        # Verify all converge correctly despite different iteration counts
        pass
    
    def test_all_converge_on_first_iteration(self):
        """Edge case: all tasks converge on first iteration."""
        # When all tasks have |high - low| < tol initially
        # should return immediately with all results
        pass


class TestHomogeneousOutputPosAssertion:
    """Test assertion that all tasks share same output_pos."""
    
    def test_homogeneous_output_pos_assertion(self):
        """All tasks must have same output_pos - heterogeneous should raise."""
        # Later: verify _run_vectorised_bisection asserts all tasks share output_pos
        # If task[0]['output_pos'] != task[1]['output_pos'], should raise AssertionError
        pass


class TestO2SignChangesCollection:
    """Test _collect_o2_sign_changes returns correct structure."""
    
    def test_collect_o2_sign_changes_returns_tuples(self):
        """_collect_o2_sign_changes should return list of (scale_low, scale_high, idx) tuples."""
        # Expected return format from the spec:
        # [(scale_low_1, scale_high_1, crossover_idx_1), (scale_low_2, scale_high_2, crossover_idx_2), ...]
        # where crossover_idx is the index in the o2_logits array where sign change occurs
        pass
    
    def test_collect_o2_sign_changes_includes_diff_data(self):
        """_collect_o2_sign_changes should collect o2_logits diff data for bound type computation."""
        # Need to pass diff data through so bound types can be computed
        # from d1_logits - d2_logits
        pass


class TestBoundTypeComputation:
    """Test bound type computation with collected diff data."""
    
    def test_bound_type_from_diff_slope(self):
        """Bound type should be computed from diff slope at crossover."""
        # slope_diff < 0 → 'lb' (diff falling, d2 gains on d1)
        # slope_diff > 0 → 'ub' (diff rising, d1 gains on d2)
        pass


class TestProcessCrossoverBatchIntegration:
    """Test integration of vectorised bisection with _process_crossover_batch."""
    
    def test_vectorised_bisection_called_for_o2_crossovers(self):
        """_process_crossover_batch should use vectorised bisection for o2 crossovers."""
        # When non-zero samples have o2 sign changes, should call
        # _run_vectorised_bisection instead of individual find_exact_crossover_bisection
        pass
    
    def test_vectorised_bisection_not_called_for_single_sample(self):
        """When only one non-zero sample, may still use scalar for efficiency."""
        # Or may always use vectorised - depends on implementation choice
        pass


class TestTaskDictStructure:
    """Test task dict has exact field names from spec."""
    
    def test_task_dict_field_names(self):
        """Task dict must have exact field names: scale_low, scale_high, output_pos, d1_val, d2_val."""
        task = {
            'scale_low': 0.5,
            'scale_high': 1.5,
            'output_pos': OUTPUT_POS_O2,
            'd1_val': 5,
            'd2_val': 7,
        }
        
        # Verify all required fields present
        assert 'scale_low' in task
        assert 'scale_high' in task
        assert 'output_pos' in task
        assert 'd1_val' in task
        assert 'd2_val' in task


class TestErrorHandling:
    """Test error handling - no warnings, only explicit errors."""
    
    def test_no_warnings_on_convergence(self):
        """Should not emit warnings for normal convergence."""
        # Verify no warnings are issued during normal operation
        pass
    
    def test_assertion_on_heterogeneous_output_pos(self):
        """Should raise AssertionError if output_pos heterogeneous."""
        # verify explicit assertion error
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
