"""Test the updated get_output_swap_bounds function on the 49,19 input."""
import pandas as pd
# Import directly from module to avoid visualization dependencies
from src.sae.steering import get_output_swap_bounds

# Load the xovers data
xovers_df = pd.read_csv('results/xover/xovers_feat30.csv')

# Filter to just the 49,19 case
row_4919 = xovers_df[(xovers_df['d1'] == 49) & (xovers_df['d2'] == 19)]
print('Input row for 49,19:')
print(f"  o1_crossovers: {row_4919['o1_crossovers'].values[0]}")
print(f"  o2_crossovers: {row_4919['o2_crossovers'].values[0]}")
print(f"  Old bounds from crossovers: [2.159, 4.259]")

# Get new swap bounds
result = get_output_swap_bounds(row_4919)
print()
print('New swap bounds (with argmax dominance):')
print(result[['d1', 'd2', 'lower_bound', 'upper_bound', 'midpoint', 'failure_reason']].to_string(index=False))

if result['failure_reason'].isna().values[0]:
    print(f"\n✓ Valid swap zone found: [{result['lower_bound'].values[0]:.3f}, {result['upper_bound'].values[0]:.3f}]")
    print(f"  Midpoint: {result['midpoint'].values[0]:.3f}")
else:
    print(f"\n✗ Failed: {result['failure_reason'].values[0]}")
