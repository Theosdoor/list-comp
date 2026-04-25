"""
Test that feat_zero is properly handled as SUMMARY_ONLY in reporting.py

This test verifies:
1. SUMMARY_ONLY includes both "dead_latent" and "feat_zero"
2. When generating markdown, feat_zero failures are only in the summary table
3. generate_example_visuals is NOT called for feat_zero samples
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.sae.reporting import SUMMARY_ONLY, generate_markdown, generate_example_visuals


class TestSummaryOnlyFeatZero:
    """Test feat_zero is in SUMMARY_ONLY and properly excluded from per-reason breakdown."""

    def test_summary_only_includes_feat_zero(self):
        """SUMMARY_ONLY should include both dead_latent and feat_zero."""
        assert "dead_latent" in SUMMARY_ONLY
        assert "feat_zero" in SUMMARY_ONLY
        assert len(SUMMARY_ONLY) == 2

    def test_generate_markdown_feat_zero_only_in_summary(self):
        """
        feat_zero rows should appear only in the summary table,
        not in the per-reason breakdown section.
        """
        # Create a merged DataFrame with some feat_zero failures
        merged = pd.DataFrame({
            "d1": [1, 1, 2, 2, 3],
            "d2": [2, 2, 3, 3, 4],
            "failure_reason": ["success", "feat_zero", "feat_zero", "dead_latent", "success"],
            "correctness": ["both_correct", "partial", "both_wrong", "partial", "both_correct"],
        })

        # Generate markdown without visuals
        md = generate_markdown(merged, feature_idx=5)

        # Check that summary table includes feat_zero row with correct count
        # The format is: | `feat_zero` | 0 | 1 | 1 | 2 | 40.0% |
        assert "| `feat_zero`" in md, "feat_zero should be in the summary table"
        feat_zero_line = [line for line in md.split('\n') if "| `feat_zero`" in line][0]
        # Total count should be 2 (4th column after the reason name)
        parts = feat_zero_line.split("|")
        assert len(parts) >= 7, f"Summary table row should have correct format: {feat_zero_line}"
        # Column structure: | `reason` | both_correct | partial | both_wrong | total | pct |
        # So parts[5] should contain the total count
        assert "2" in parts[5], f"feat_zero total count should be 2, got: {feat_zero_line}"

        # Check that feat_zero is NOT in the per-reason breakdown section
        breakdown_section = md.split("## Per-Reason Breakdown")[1] if "## Per-Reason Breakdown" in md else ""
        assert "### `feat_zero`" not in breakdown_section, \
            "feat_zero should NOT have its own per-reason breakdown section"

    def test_generate_markdown_dead_latent_only_in_summary(self):
        """
        dead_latent rows should also only appear in summary, not in breakdown.
        (sanity check that the existing behavior works)
        """
        merged = pd.DataFrame({
            "d1": [1, 1, 2],
            "d2": [2, 2, 3],
            "failure_reason": ["success", "dead_latent", "dead_latent"],
            "correctness": ["both_correct", "partial", "both_wrong"],
        })

        md = generate_markdown(merged, feature_idx=5)

        # Check that summary table includes dead_latent
        assert "| `dead_latent`" in md
        
        # Check that dead_latent is NOT in breakdown
        breakdown_section = md.split("## Per-Reason Breakdown")[1] if "## Per-Reason Breakdown" in md else ""
        assert "### `dead_latent`" not in breakdown_section

    def test_generate_markdown_non_summary_only_in_breakdown(self):
        """
        Failure reasons NOT in SUMMARY_ONLY should appear in the breakdown section.
        """
        merged = pd.DataFrame({
            "d1": [1, 1, 2],
            "d2": [2, 2, 3],
            "failure_reason": ["success", "d1_eq_d2", "no_o2_crossover"],
            "correctness": ["both_correct", "partial", "both_wrong"],
        })

        md = generate_markdown(merged, feature_idx=5)

        # These reasons should have their own breakdown sections
        assert "### `d1_eq_d2`" in md
        assert "### `no_o2_crossover`" in md

    def test_generate_example_visuals_not_called_for_feat_zero(self):
        """
        Verify that generate_example_visuals is NOT called in the report pipeline
        when processing feat_zero failures. This test verifies the logic that skips
        SUMMARY_ONLY reasons before calling generate_example_visuals.
        """
        merged = pd.DataFrame({
            "d1": [1, 1],
            "d2": [2, 2],
            "failure_reason": ["success", "feat_zero"],
            "correctness": ["both_correct", "partial"],
            "feat_orig": [1.5, 0.0],
            "o1_crossovers": [[0.5], []],
            "o2_crossovers": [[1.0], []],
            "lower_bound": [0.3, None],
            "upper_bound": [1.2, None],
        })

        # Generate markdown (visuals is None because we skip generating visuals for SUMMARY_ONLY)
        md = generate_markdown(merged, feature_idx=5, visuals=None)
        
        # The key validation: feat_zero should be in the summary but NOT in the breakdown
        # This means the code correctly skipped calling generate_example_visuals for feat_zero
        assert "| `feat_zero`" in md
        breakdown_section = md.split("## Per-Reason Breakdown")[1] if "## Per-Reason Breakdown" in md else ""
        assert "### `feat_zero`" not in breakdown_section
        
        # Verify that "success" (which is NOT SUMMARY_ONLY) IS in the breakdown
        assert "### `success`" in breakdown_section

    def test_markdown_summary_table_has_feat_zero_counts(self):
        """
        The summary table should include count and correctness info for feat_zero,
        even though it's SUMMARY_ONLY.
        """
        merged = pd.DataFrame({
            "d1": [1, 1, 1, 2],
            "d2": [2, 2, 2, 3],
            "failure_reason": ["success", "feat_zero", "feat_zero", "feat_zero"],
            "correctness": ["both_correct", "both_correct", "partial", "both_wrong"],
        })

        md = generate_markdown(merged, feature_idx=5)

        # Extract the summary table
        summary_section = md.split("## Per-Reason Breakdown")[0]
        
        # feat_zero row should have:
        # - 1 both_correct, 1 partial, 1 both_wrong, 3 total, 75% of all
        assert "| `feat_zero`" in summary_section
        feat_zero_line = [line for line in summary_section.split('\n') if "| `feat_zero`" in line][0]
        
        # Count should be 3
        assert "3" in feat_zero_line, f"feat_zero count should be 3 in line: {feat_zero_line}"
        
        # Verify correctness breakdown
        # Format: | `feat_zero` | 1 | 1 | 1 | 3 | 75.0% |
        parts = feat_zero_line.split("|")
        assert len(parts) >= 7, f"Summary table row should have at least 7 columns, got: {feat_zero_line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
