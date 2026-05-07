import pandas as pd

from src.sae.reporting import generate_markdown


def test_failure_report_shows_four_examples_per_reason():
    merged = pd.DataFrame(
        [
            {
                "d1": i,
                "d2": i + 10,
                "feat_orig": 0.25 + i,
                "o1_crossovers": [],
                "o2_crossovers": [],
                "lower_bound": 1.0,
                "upper_bound": 2.0,
                "correctness": "both_correct",
                "failure_reason": "success",
            }
            for i in range(6)
        ]
    )

    markdown = generate_markdown(merged, feature_idx=30)

    assert "**Examples** (up to 4 of 6):" in markdown
    assert "| 0 | 10 |" in markdown
    assert "| 3 | 13 |" in markdown
    assert "| 4 | 14 |" not in markdown
