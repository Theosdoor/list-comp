"""
SAE Feature Steering and Crossover Analysis

Functions for analyzing how scaling SAE features affects model outputs,
including crossover detection and output swap verification.

CONSTANTS:
    OUTPUT_POS_O1: Position index for first output token
    OUTPUT_POS_O2: Position index for second output token
    DEFAULT_BISECTION_TOL: Tolerance for bisection convergence
    DEFAULT_BISECTION_MAX_ITER: Maximum iterations for bisection
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .hooks import make_sae_patch_hook, make_batched_sae_patch_hook


# Module-level constants
OUTPUT_POS_O1 = -2
OUTPUT_POS_O2 = -1
DEFAULT_BISECTION_TOL = 0.0005
DEFAULT_BISECTION_MAX_ITER = 50
DEFAULT_SEP_IDX = 2
DEFAULT_N_DIGITS = 100

# Linear fit constants for o1 crossover detection
# R² threshold: require near-perfect fit since o1 logits are empirically linear
LINEAR_FIT_R2_THRESHOLD = 0.999
# Slope threshold for declaring lines parallel (unresponsive):
# 1e-5 logit-diff per scale unit → <0.001 total shift across [0,10], functionally zero
LINEAR_FIT_SLOPE_EPS = 1e-5
# Soft upper cap for extrapolated intersections: 2× the default grid ceiling (10.0).
# Beyond scale 20 the linear model is extrapolating into completely untested territory
# (20× the original activation), so we flag these rather than accept them silently.
LINEAR_FIT_SCALE_CAP = 20.0


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_inputs(model, sae, d1_all, d2_all, sae_acts_all, dataset, feature_idx):
    """Validate common inputs across functions."""
    if len(d1_all) != len(d2_all):
        raise ValueError(f"d1_all and d2_all must have same length, got {len(d1_all)} vs {len(d2_all)}")
    
    if len(d1_all) != len(sae_acts_all):
        raise ValueError(f"d1_all length {len(d1_all)} doesn't match sae_acts_all length {len(sae_acts_all)}")
    
    if len(dataset) != len(d1_all):
        raise ValueError(f"dataset length {len(dataset)} doesn't match d1_all length {len(d1_all)}")
    
    if feature_idx < 0 or feature_idx >= sae_acts_all.shape[1]:
        raise ValueError(f"feature_idx {feature_idx} out of range [0, {sae_acts_all.shape[1]})")


def _get_device(model, device=None):
    """Get device from model or use provided device."""
    if device is None:
        return next(model.parameters()).device
    return device


def _find_input_index(d1_all, d2_all, d1_val, d2_val):
    """Find index of input pair in dataset."""
    mask = (d1_all == d1_val) & (d2_all == d2_val)
    if mask.sum() == 0:
        raise ValueError(f"Input pair ({d1_val}, {d2_val}) not found in dataset")
    return torch.where(mask)[0][0].item()


def _run_model_with_scaled_feature(
    model, sae, act_mean, inputs, z_orig, feature_idx, 
    feat_orig, scale, layer_idx, sep_idx, hook_name_resid
):
    """
    Run model with a single scaled feature activation.
    
    Returns:
        patched_logits: Model output logits [batch_size, seq_len, vocab_size]
    """
    z_scaled = z_orig.clone()
    z_scaled[feature_idx] = feat_orig * scale
    recon = sae.decode(z_scaled.unsqueeze(0))
    
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            inputs,
            fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
        )
    
    return patched_logits


def _extract_logits_at_positions(patched_logits, n_digits):
    """
    Extract logits at both output positions.
    
    Returns:
        logits_o1: Logits at first output position [n_digits]
        logits_o2: Logits at second output position [n_digits]
    """
    logits_o1 = patched_logits[0, OUTPUT_POS_O1, :n_digits].cpu().numpy()
    logits_o2 = patched_logits[0, OUTPUT_POS_O2, :n_digits].cpu().numpy()
    return logits_o1, logits_o2


def _parse_list_field(field):
    """Safely parse a field that might be string or list."""
    if isinstance(field, str):
        import ast
        return ast.literal_eval(field)
    return field


# ============================================================================
# Core Analysis Functions
# ============================================================================

def inspect_steered_output(
    model, sae, act_mean, feature_idx, scale,
    inputs_i, z_orig, feat_orig,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS, device=None
):
    """
    Inspect model output when a specific feature is scaled to a custom factor.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the feature being steered
        scale: Scale factor for the feature (e.g., 2.0 = double activation, 0.0 = ablate)
        inputs_i: Input tensor [1, seq_len]
        z_orig: Original SAE activations [d_sae]
        feat_orig: Original feature activation value
        layer_idx: Layer to patch (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        device: Device to use (default: auto-detect)
    
    Returns:
        dict with keys:
            - 'scale': Scale factor used
            - 'logits_o1': Logits at output position 1 [n_digits]
            - 'logits_o2': Logits at output position 2 [n_digits]
            - 'pred_o1': Predicted digit at o1
            - 'pred_o2': Predicted digit at o2
    """
    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    patched_logits = _run_model_with_scaled_feature(
        model, sae, act_mean, inputs_i, z_orig, feature_idx,
        feat_orig, scale, layer_idx, sep_idx, hook_name_resid
    )
    
    logits_o1, logits_o2 = _extract_logits_at_positions(patched_logits, n_digits)
    
    return {
        'scale': scale,
        'logits_o1': logits_o1,
        'logits_o2': logits_o2,
        'pred_o1': int(logits_o1.argmax()),
        'pred_o2': int(logits_o2.argmax()),
    }


def inspect_steered_outputs_batch(
    model, sae, act_mean, feature_idx, scales,
    inputs_i, z_orig, feat_orig, d1_val=None, d2_val=None,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS, device=None
):
    """
    Inspect model outputs at multiple scale factors.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the feature being steered
        scales: List of scale factors to test (e.g., [0.0, 0.5, 1.0, 2.0, 5.0])
        inputs_i: Input tensor [1, seq_len]
        z_orig: Original SAE activations [d_sae]
        feat_orig: Original feature activation value
        d1_val, d2_val: (Optional) Input digits for labeling output
        layer_idx: Layer to patch (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        device: Device to use (default: auto-detect)
    
    Returns:
        results: List of dicts with inspection results for each scale
        df: Summary dataframe
    """
    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    results = []
    
    for scale in scales:
        patched_logits = _run_model_with_scaled_feature(
            model, sae, act_mean, inputs_i, z_orig, feature_idx,
            feat_orig, scale, layer_idx, sep_idx, hook_name_resid
        )
        
        logits_o1, logits_o2 = _extract_logits_at_positions(patched_logits, n_digits)
        
        row = {
            'scale': scale,
            'pred_o1': int(logits_o1.argmax()),
            'pred_o2': int(logits_o2.argmax()),
            'logits_o1': logits_o1,
            'logits_o2': logits_o2,
        }
        
        # Add specific digit logits if provided
        if d1_val is not None and d2_val is not None:
            row['d1_logit_o1'] = logits_o1[d1_val]
            row['d2_logit_o1'] = logits_o1[d2_val]
            row['d1_logit_o2'] = logits_o2[d1_val]
            row['d2_logit_o2'] = logits_o2[d2_val]
        
        results.append(row)
    
    # Create summary dataframe
    summary_cols = ['scale', 'pred_o1', 'pred_o2']
    if d1_val is not None and d2_val is not None:
        summary_cols.extend(['d1_logit_o1', 'd2_logit_o1', 'd1_logit_o2', 'd2_logit_o2'])
    
    df = pd.DataFrame(results)[summary_cols]
    
    return results, df


def find_exact_crossover_bisection(
    model, sae, act_mean, feature_idx,
    inputs_i, z_orig, feat_orig, d1_val, d2_val,
    scale_low, scale_high, output_pos,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS,
    tol=DEFAULT_BISECTION_TOL, max_iter=DEFAULT_BISECTION_MAX_ITER, device=None
):
    """
    Use bisection to find exact scale where d1 and d2 logits cross.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the feature being steered
        inputs_i: Input tensor [1, seq_len]
        z_orig: Original SAE activations [d_sae]
        feat_orig: Original feature activation value
        d1_val: d1 digit value
        d2_val: d2 digit value
        scale_low: Lower bound of scale range
        scale_high: Upper bound of scale range
        output_pos: Output position (OUTPUT_POS_O1 or OUTPUT_POS_O2)
        layer_idx: Layer to patch (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digits (default: 100)
        tol: Tolerance for convergence (default: 0.0005 for 3dp accuracy)
        max_iter: Maximum iterations (default: 20)
        device: Device to use (default: auto-detect)
    
    Returns:
        exact_scale: Scale value where crossover occurs (to 3 decimal places)
    """
    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    def get_logit_diff_at_scale(scale):
        """Helper: Run model at given scale and return d1_logit - d2_logit"""
        patched_logits = _run_model_with_scaled_feature(
            model, sae, act_mean, inputs_i, z_orig, feature_idx,
            feat_orig, scale, layer_idx, sep_idx, hook_name_resid
        )
        
        logits = patched_logits[0, output_pos, :n_digits].cpu().numpy()
        return logits[d1_val] - logits[d2_val]
    
    # Bisection loop
    for iteration in range(max_iter):
        if scale_high - scale_low < tol:
            break
        
        scale_mid = (scale_low + scale_high) / 2
        diff_mid = get_logit_diff_at_scale(scale_mid)
        diff_low = get_logit_diff_at_scale(scale_low)
        
        # Check which half contains the root
        # If signs are the same, root is in upper half; otherwise lower half
        if diff_mid * diff_low > 0:
            scale_low = scale_mid
        else:
            scale_high = scale_mid
    
    return (scale_low + scale_high) / 2


# ============================================================================
# Feature Steering Experiment
# ============================================================================

def feature_steering_experiment(
    model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS,
    scale_factors=None, scale_range=[-1.0, 4.0], sample_step_size=0.05, n_test_cases=5, seed=42,
    test_pairs=None, device=None, plot=True, save_dir=None
):
    """
    Perform feature steering experiment by scaling a specific SAE feature's activation.
    
    Tests how scaling a feature's activation affects model outputs across different inputs.
    Samples test cases from inputs where the feature actually fires, or uses specified inputs.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature to steer
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to patch activations at (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        scale_factors: Array of scale factors to test (overrides scale_range if provided)
        scale_range: Tuple (min, max) for scale factors (default: [-1.0, 4.0])
        n_test_cases: Number of test cases to sample (ignored if test_pairs is provided)
        seed: Random seed for sampling (ignored if test_pairs is provided)
        test_pairs: Optional list of (d1, d2) tuples to plot specific inputs.
                   If provided, overrides n_test_cases and seed.
                   Example: [(1, 83), (14, 67), (0, 10)]
        device: Device to use (default: auto-detect from model)
        plot: Whether to create visualization (default: True)
        save_dir: Directory to save plot (if None and plot=True, shows plot)
    
    Returns:
        all_results: List of dicts with keys:
            - 'd1', 'd2': Input pair
            - 'scales': Scale factors used
            - 'all_logits_o1': All logits at output position 1 [n_scales, n_digits]
            - 'all_logits_o2': All logits at output position 2 [n_scales, n_digits]
            - 'logit_d1_o1', 'logit_d2_o1': Logits at output position 1
            - 'logit_d1_o2', 'logit_d2_o2': Logits at output position 2
            - 'output_o1', 'output_o2': Predicted outputs
            - 'order_feat_orig': Original feature activation
    """
    _validate_inputs(model, sae, d1_all, d2_all, sae_acts_all, dataset, feature_idx)
    
    if scale_factors is None:
        if sample_step_size <= 0:
            raise ValueError(f"sample_step_size must be > 0, got {sample_step_size}")
        if scale_range is None or len(scale_range) != 2:
            raise ValueError(
                f"scale_range must be a length-2 sequence [min, max], got {scale_range}"
            )

        scale_min, scale_max = scale_range[0], scale_range[1]
        if scale_max < scale_min:
            raise ValueError(
                f"scale_range max must be >= min, got min={scale_min}, max={scale_max}"
            )

        num_steps = int(round((scale_max - scale_min) / sample_step_size)) + 1
        scale_factors = np.linspace(scale_min, scale_max, num_steps)
    
    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Determine test pairs
    test_pairs = _determine_test_pairs(
        test_pairs, sae_acts_all, feature_idx, d1_all, d2_all, 
        n_test_cases, seed
    )
    
    # Run steering experiment
    all_results = _run_steering_for_test_pairs(
        test_pairs, d1_all, d2_all, dataset, sae_acts_all,
        model, sae, act_mean, feature_idx, scale_factors,
        layer_idx, sep_idx, n_digits, device, hook_name_resid
    )
    
    # Create visualization if requested
    if plot and len(all_results) > 0:
        _plot_steering_results(all_results, feature_idx, n_digits, save_dir)
    
    return all_results


def _determine_test_pairs(test_pairs, sae_acts_all, feature_idx, d1_all, d2_all, n_test_cases, seed):
    """Determine which input pairs to test."""
    if test_pairs is not None:
        print(f"Using {len(test_pairs)} specified input pairs")
        return test_pairs
    
    # Select test cases where the feature actually fires
    active_indices = torch.where(sae_acts_all[:, feature_idx] > 0)[0]
    print(f"Feature {feature_idx} fires on {len(active_indices)} / {len(d1_all)} inputs")
    
    if len(active_indices) == 0:
        raise ValueError(f"Feature {feature_idx} never fires in the dataset")
    
    # Sample from active inputs only
    np.random.seed(seed)
    n_samples = min(n_test_cases, len(active_indices))
    test_indices = np.random.choice(active_indices.numpy(), size=n_samples, replace=False)
    
    return [(d1_all[i].item(), d2_all[i].item()) for i in test_indices]


def _run_steering_for_test_pairs(
    test_pairs, d1_all, d2_all, dataset, sae_acts_all,
    model, sae, act_mean, feature_idx, scale_factors,
    layer_idx, sep_idx, n_digits, device, hook_name_resid
):
    """Run steering experiment for all test pairs."""
    all_results = []
    
    for d1_val, d2_val in test_pairs:
        try:
            idx = _find_input_index(d1_all, d2_all, d1_val, d2_val)
        except ValueError:
            print(f"Warning: Skipping pair ({d1_val}, {d2_val}) - not found in dataset")
            continue
        
        inputs_i = dataset[idx][0].unsqueeze(0).to(device)
        z_orig = sae_acts_all[idx].clone().to(device)
        feat_orig = z_orig[feature_idx].item()
        
        # Storage for all logits at o1 and o2 (for ALL digits)
        all_logits_o1 = []
        all_logits_o2 = []
        
        for scale in scale_factors:
            patched_logits = _run_model_with_scaled_feature(
                model, sae, act_mean, inputs_i, z_orig, feature_idx,
                feat_orig, scale, layer_idx, sep_idx, hook_name_resid
            )
            
            logits_o1, logits_o2 = _extract_logits_at_positions(patched_logits, n_digits)
            all_logits_o1.append(logits_o1)
            all_logits_o2.append(logits_o2)
        
        all_logits_o1 = np.array(all_logits_o1)  # [n_scales, n_digits]
        all_logits_o2 = np.array(all_logits_o2)  # [n_scales, n_digits]
        
        all_results.append({
            'd1': d1_val, 
            'd2': d2_val,
            'scales': scale_factors,
            'all_logits_o1': all_logits_o1,
            'all_logits_o2': all_logits_o2,
            'logit_d1_o1': all_logits_o1[:, d1_val],
            'logit_d2_o1': all_logits_o1[:, d2_val],
            'logit_d1_o2': all_logits_o2[:, d1_val],
            'logit_d2_o2': all_logits_o2[:, d2_val],
            'output_o1': all_logits_o1.argmax(axis=1),
            'output_o2': all_logits_o2.argmax(axis=1),
            'order_feat_orig': feat_orig,
        })
    
    return all_results


def _plot_steering_results(all_results, feature_idx, n_digits, save_dir):
    """Create visualization of steering results."""
    fig, axes = plt.subplots(2, len(all_results), figsize=(4*len(all_results), 10), squeeze=False)
    
    for col, result in enumerate(all_results):
        d1, d2 = result['d1'], result['d2']
        scales = result['scales']
        
        # Top row: Logits at o1 position
        _plot_output_position(
            axes[0, col], scales, result['all_logits_o1'],
            result['logit_d1_o1'], result['logit_d2_o1'],
            d1, d2, feature_idx, result['order_feat_orig'],
            n_digits, output_name="Output Position 1"
        )
        
        # Bottom row: Logits at o2 position
        _plot_output_position(
            axes[1, col], scales, result['all_logits_o2'],
            result['logit_d1_o2'], result['logit_d2_o2'],
            d1, d2, feature_idx, result['order_feat_orig'],
            n_digits, output_name="Output Position 2"
        )
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f'feature_{feature_idx}_logit_steering.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def _plot_output_position(ax, scales, all_logits, logit_d1, logit_d2, 
                         d1, d2, feature_idx, feat_orig, n_digits, output_name):
    """Plot logits for a single output position."""
    # Plot all other logits in grey
    for digit in range(n_digits):
        if digit != d1 and digit != d2:
            ax.plot(scales, all_logits[:, digit], 'grey', alpha=0.2, linewidth=0.5)
    
    # Plot d1 and d2 logits on top
    ax.plot(scales, logit_d1, 'b-', linewidth=2, label=f'd1={d1} logit')
    ax.plot(scales, logit_d2, 'r-', linewidth=2, label=f'd2={d2} logit')
    
    # Mark reference points
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Original')
    ax.axvline(x=0.0, color='black', linestyle=':', alpha=0.5, label='Ablated')
    
    ax.set_xlabel(f'Feature {feature_idx} Scale Factor')
    ax.set_ylabel(f'Logit at {output_name.split()[-1]}')
    ax.set_title(f'All Logits at {output_name}\nInput: ({d1}, {d2}), Original f{feature_idx}={feat_orig:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)


# ============================================================================
# Crossover Detection and Analysis
# ============================================================================

def get_xovers_df(
    model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS,
    scale_range=[0.0, 10.0], sample_step_size = 0.05,
    batch_size=64, device=None
):
    """
    Find all crossover points where d1 and d2 logits intersect across all inputs.
    
    Uses a coarse grid search followed by bisection to find exact crossover points.
    Also determines whether each crossover is an upper or lower bound for output swapping.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature to analyze
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to patch (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        scale_range: Tuple (min, max) for scale range (default: [0.0, 10.0])
        sample_step_size: Step size for coarse grid search (default: 0.1)
        batch_size: Batch size for processing (default: 64)
        device: Device to use (default: auto-detect)
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - feat_orig: Original feature activation
            - o1_crossovers: List of scale values where o1 logits cross (analytically via linear fit)
            - o2_crossovers: List of scale values where o2 logits cross (grid + bisection)
            - o1_bound_types: List of bound types ('ub', 'lb') for o1 crossovers
            - o2_bound_types: List of bound types ('ub', 'lb', 'unknown') for o2 crossovers
            - n_o1_xover: Number of o1 crossovers
            - n_o2_xover: Number of o2 crossovers
            - scales: List of scale factors used in coarse search
            - argmax_o1: List of argmax predictions at o1 for each scale
            - argmax_o2: List of argmax predictions at o2 for each scale
            - o1_failure_reason: None if successful, else one of:
                'feat_zero'           - feature has zero activation (no steering possible)
                'd1_eq_d2'            - d1 == d2, degenerate input
                'nonlinear_d1'        - d1 logit at o1 fails R²≥0.999 linearity check
                'nonlinear_d2'        - d2 logit at o1 fails R²≥0.999 linearity check
                'unresponsive'        - diff slope < 1e-5, feature has no effect on o1
                'o1_negative_scale'   - intersection is at scale < 0 (suppression, not amplification, swaps outputs)
                'o1_extrapolated'     - intersection is beyond scale cap (20.0), implausible extrapolation
    """
    _validate_inputs(model, sae, d1_all, d2_all, sae_acts_all, dataset, feature_idx)

    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"

    # Create scale factors for grid search
    # Note: the grid is used for o2 crossover detection and argmax dominance checks.
    # o1 crossovers are found analytically from linear fits to the grid logits.
    n_scale_points = int(round((scale_range[1] - scale_range[0]) / sample_step_size)) + 1
    scale_factors = np.linspace(scale_range[0], scale_range[1], n_scale_points)
    
    # Process all data samples in batches
    n_data_samples = len(d1_all)
    all_results = []
    n_batches = (n_data_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Finding crossovers (batched)", leave=True):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_data_samples)
        batch_indices = range(batch_start, batch_end)
        current_batch_size = batch_end - batch_start
        
        # Get batch data
        batch_results = _process_crossover_batch(
            batch_start, batch_end, batch_indices, current_batch_size,
            d1_all, d2_all, sae_acts_all, dataset, feature_idx,
            model, sae, act_mean, scale_factors,
            layer_idx, sep_idx, n_digits, device, hook_name_resid
        )
        
        all_results.extend(batch_results)
    
    return pd.DataFrame(all_results)


def _process_crossover_batch(
    batch_start, batch_end, batch_indices, current_batch_size,
    d1_all, d2_all, sae_acts_all, dataset, feature_idx,
    model, sae, act_mean, scale_factors,
    layer_idx, sep_idx, n_digits, device, hook_name_resid
):
    """Process a single batch for crossover detection."""
    batch_d1 = d1_all[batch_start:batch_end]
    batch_d2 = d2_all[batch_start:batch_end]
    batch_z_orig = sae_acts_all[batch_start:batch_end].to(device)
    batch_feat_orig = batch_z_orig[:, feature_idx].clone()
    
    batch_inputs = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
    
    # Run model for all scales
    batch_logits_o1, batch_logits_o2 = _run_batched_steering(
        model, sae, act_mean, batch_inputs, batch_z_orig, 
        feature_idx, batch_feat_orig, scale_factors,
        layer_idx, sep_idx, n_digits, hook_name_resid
    )
    
    # Process each sample in batch
    batch_results = []
    for i in range(current_batch_size):
        global_idx = batch_start + i
        result = _analyze_single_sample_crossovers(
            i, global_idx, batch_d1, batch_d2, batch_feat_orig,
            batch_logits_o1, batch_logits_o2, batch_z_orig,
            dataset, model, sae, act_mean, feature_idx, scale_factors,
            layer_idx, sep_idx, n_digits, device
        )
        batch_results.append(result)
    
    return batch_results


def _run_batched_steering(
    model, sae, act_mean, batch_inputs, batch_z_orig,
    feature_idx, batch_feat_orig, scale_factors,
    layer_idx, sep_idx, n_digits, hook_name_resid
):
    """Run steering for all scales in batch."""
    batch_logits_o1 = []
    batch_logits_o2 = []
    
    for scale in scale_factors:
        batch_z_scaled = batch_z_orig.clone()
        batch_z_scaled[:, feature_idx] = batch_feat_orig * scale
        
        batch_recon = sae.decode(batch_z_scaled)
        
        with torch.no_grad():
            batch_patched_logits = model.run_with_hooks(
                batch_inputs,
                fwd_hooks=[(hook_name_resid, make_batched_sae_patch_hook(batch_recon, act_mean, sep_idx))]
            )
        
        logits_o1 = batch_patched_logits[:, OUTPUT_POS_O1, :n_digits].cpu().numpy()
        logits_o2 = batch_patched_logits[:, OUTPUT_POS_O2, :n_digits].cpu().numpy()
        batch_logits_o1.append(logits_o1)
        batch_logits_o2.append(logits_o2)
    
    # Transpose to [batch_size, n_scales, n_digits]
    batch_logits_o1 = np.array(batch_logits_o1).transpose(1, 0, 2)
    batch_logits_o2 = np.array(batch_logits_o2).transpose(1, 0, 2)
    
    return batch_logits_o1, batch_logits_o2


def _analyze_single_sample_crossovers(
    i, global_idx, batch_d1, batch_d2, batch_feat_orig,
    batch_logits_o1, batch_logits_o2, batch_z_orig,
    dataset, model, sae, act_mean, feature_idx, scale_factors,
    layer_idx, sep_idx, n_digits, device
):
    """Analyze crossovers for a single sample."""
    d1_val = batch_d1[i].item()
    d2_val = batch_d2[i].item()
    feat_orig = batch_feat_orig[i].item()

    logits_o1 = batch_logits_o1[i]
    logits_o2 = batch_logits_o2[i]

    argmax_o1 = logits_o1.argmax(axis=1)
    argmax_o2 = logits_o2.argmax(axis=1)

    def _empty_result(failure_reason):
        return {
            'd1': d1_val, 'd2': d2_val, 'feat_orig': feat_orig,
            'o1_crossovers': [], 'o2_crossovers': [],
            'o1_bound_types': [], 'o2_bound_types': [],
            'n_o1_xover': 0, 'n_o2_xover': 0,
            'scales': scale_factors.tolist(),
            'argmax_o1': argmax_o1.tolist(),
            'argmax_o2': argmax_o2.tolist(),
            'o1_failure_reason': failure_reason,
        }

    # Degenerate: feature not active
    if feat_orig == 0:
        return _empty_result('feat_zero')

    # Degenerate: both digits are the same — logit diff is always ~0
    if d1_val == d2_val:
        return _empty_result('d1_eq_d2')

    # --- o1: analytical linear fit ---
    o1_crossovers, o1_bound_types, o1_failure_reason = _find_o1_crossover_linear(
        logits_o1, d1_val, d2_val, scale_factors
    )

    # --- o2: grid + bisection (logits are nonlinear) ---
    inputs_i = dataset[global_idx][0].unsqueeze(0).to(device)
    z_orig = batch_z_orig[i]

    o2_crossovers, o2_bound_types = _find_crossovers_for_position(
        logits_o2, d1_val, d2_val, scale_factors, argmax_o2,
        model, sae, act_mean, feature_idx, inputs_i, z_orig, feat_orig,
        OUTPUT_POS_O2, layer_idx, sep_idx, n_digits, device
    )

    return {
        'd1': d1_val,
        'd2': d2_val,
        'feat_orig': feat_orig,
        'o1_crossovers': o1_crossovers,
        'o2_crossovers': o2_crossovers,
        'o1_bound_types': o1_bound_types,
        'o2_bound_types': o2_bound_types,
        'n_o1_xover': len(o1_crossovers),
        'n_o2_xover': len(o2_crossovers),
        'scales': scale_factors.tolist(),
        'argmax_o1': argmax_o1.tolist(),
        'argmax_o2': argmax_o2.tolist(),
        'o1_failure_reason': o1_failure_reason,
    }


def _r_squared(y, y_fit):
    """Compute R² between observed values and a linear fit."""
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        # Constant signal — perfect fit if residuals also zero, else undefined
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _find_o1_crossover_linear(logits_o1, d1_val, d2_val, scale_factors):
    """
    Find the o1 crossover analytically by fitting lines to d1 and d2 logits.

    o1 logits are empirically linear in the scale factor, so we:
      1. Fit a line to each of d1_logit(scale) and d2_logit(scale)
      2. Verify R² ≥ LINEAR_FIT_R2_THRESHOLD for both — fail with 'nonlinear_d1/d2' otherwise
      3. Fit a line to the diff (d1-d2) and check |slope| < LINEAR_FIT_SLOPE_EPS → 'unresponsive'
      4. Compute the analytical intersection scale = -intercept_diff / slope_diff
      5. If intersection < 0 → 'o1_negative_scale'; if > LINEAR_FIT_SCALE_CAP → 'o1_extrapolated'
      6. Determine bound type from diff slope:
           slope_diff < 0  →  diff is falling  →  d2 overtakes d1 going right  →  'lb'
           slope_diff > 0  →  diff is rising   →  d2 is beaten back going right →  'ub'

    Returns:
        crossovers: list with 0 or 1 scale values
        bound_types: list with 0 or 1 bound type strings
        failure_reason: None if successful, else a string
    """
    d1_logits = logits_o1[:, d1_val].astype(float)
    d2_logits = logits_o1[:, d2_val].astype(float)
    x = scale_factors.astype(float)

    # Fit lines
    slope_d1, intercept_d1 = np.polyfit(x, d1_logits, 1)
    slope_d2, intercept_d2 = np.polyfit(x, d2_logits, 1)

    d1_fit = slope_d1 * x + intercept_d1
    d2_fit = slope_d2 * x + intercept_d2

    r2_d1 = _r_squared(d1_logits, d1_fit)
    r2_d2 = _r_squared(d2_logits, d2_fit)

    if r2_d1 < LINEAR_FIT_R2_THRESHOLD:
        return [], [], 'nonlinear_d1'
    if r2_d2 < LINEAR_FIT_R2_THRESHOLD:
        return [], [], 'nonlinear_d2'

    # Diff line: d1 - d2
    slope_diff = slope_d1 - slope_d2
    intercept_diff = intercept_d1 - intercept_d2

    if abs(slope_diff) < LINEAR_FIT_SLOPE_EPS:
        return [], [], 'unresponsive'

    # Analytical intersection: slope_diff * scale + intercept_diff = 0
    xover_scale = -intercept_diff / slope_diff

    if xover_scale < 0:
        # Intersection at negative scale means the swap already occurred before
        # scale=0 (i.e., suppressing the feature, not amplifying it, would swap outputs).
        # Flag separately from the too-large extrapolation case.
        return [], [], 'o1_negative_scale'
    if xover_scale > LINEAR_FIT_SCALE_CAP:
        return [], [], 'o1_extrapolated'

    xover_scale = round(float(xover_scale), 4)

    # Bound type from diff slope:
    #   slope_diff < 0: diff decreasing → d2 gains on d1 going right → lb (swap starts here)
    #   slope_diff > 0: diff increasing → d1 gains on d2 going right → ub (swap ends here)
    bound_type = 'lb' if slope_diff < 0 else 'ub'

    return [xover_scale], [bound_type], None


def _find_crossovers_for_position(
    logits, d1_val, d2_val, scale_factors, argmax,
    model, sae, act_mean, feature_idx, inputs_i, z_orig, feat_orig,
    output_pos, layer_idx, sep_idx, n_digits, device
):
    """
    Find all crossovers for a specific output position (grid + bisection).

    Used for o2, whose logits are nonlinear in scale.

    For o1: use _find_o1_crossover_linear instead.
    For o2: We want d1 > d2 (so output predicts d1)
    This creates the swap pattern (d2, d1)

    Args:
        argmax: Array of argmax predictions at each scale [n_scales]
    """
    d1_logits = logits[:, d1_val]
    d2_logits = logits[:, d2_val]
    diff = d1_logits - d2_logits  # d1 - d2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    crossovers = []
    bound_types = []
    
    for idx in sign_changes:
        # Find exact crossover using bisection
        exact_scale = find_exact_crossover_bisection(
            model, sae, act_mean, feature_idx,
            inputs_i, z_orig, feat_orig, d1_val, d2_val,
            scale_factors[idx], scale_factors[idx + 1], output_pos,
            layer_idx, sep_idx, n_digits, device=device
        )
        exact_scale = round(exact_scale, 3)
        crossovers.append(exact_scale)
        
        # Determine bound type using logit difference at grid points
        bound_type = _determine_bound_type_from_diff(
            idx, output_pos, diff
        )
        bound_types.append(bound_type)
    
    return crossovers, bound_types


def _determine_bound_type_from_diff(crossover_idx, output_pos, diff):
    """
    Determine if crossover is an upper bound (ub) or lower bound (lb) using logit differences.
    
    The crossover occurs between diff[crossover_idx] and diff[crossover_idx+1]
    where diff = d1_logits - d2_logits.
    
    For o1: We want d2 > d1, i.e., diff < 0
        - If diff < 0 on left side: swap condition holds → upper bound
        - If diff < 0 on right side: swap condition holds → lower bound
    
    For o2: We want d1 > d2, i.e., diff > 0
        - If diff > 0 on left side: swap condition holds → upper bound
        - If diff > 0 on right side: swap condition holds → lower bound
    
    Args:
        crossover_idx: Index where sign change detected (crossover between idx and idx+1)
        output_pos: OUTPUT_POS_O1 or OUTPUT_POS_O2
        diff: Array of d1_logits - d2_logits at each scale [n_scales]
    
    Returns:
        'lb' or 'ub' (never 'unknown' since we know there's a sign change)
    """
    # At crossover_idx and crossover_idx+1, diff has opposite signs
    diff_left = diff[crossover_idx]
    diff_right = diff[crossover_idx + 1]
    
    if output_pos == OUTPUT_POS_O1:
        # For o1: want d2 > d1, i.e., diff < 0
        if diff_left < 0:
            return 'ub'  # Swap condition holds on left
        else:  # diff_right < 0
            return 'lb'  # Swap condition holds on right
    else:  # OUTPUT_POS_O2
        # For o2: want d1 > d2, i.e., diff > 0
        if diff_left > 0:
            return 'ub'  # Swap condition holds on left
        else:  # diff_right > 0
            return 'lb'  # Swap condition holds on right


def get_output_swap_bounds(xovers_df, scale_range=[0.0, 10.0]):
    """
    Identify scale ranges where outputs should swap from (d1, d2) to (d2, d1).
    
    For outputs to swap:
    - o1 must predict d2 (argmax at o1 == d2)
    - o2 must predict d1 (argmax at o2 == d1)
    
    Uses both crossover information AND argmax dominance to determine valid swap zones.
    A third digit becoming dominant (e.g., 70) will constrain the bounds.
    
    Args:
        xovers_df: DataFrame from get_xovers_df
        scale_range: Tuple (min, max) for initial bounds (default: [0.0, 10.0])
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - lower_bound: Lower bound of swap zone
            - upper_bound: Upper bound of swap zone
            - midpoint: Midpoint of swap zone
            - swap_zone_width: Width of swap zone
            - failure_reason: None if successful, otherwise reason for failure
    """
    results = []
    
    for _, row in xovers_df.iterrows():
        result = _determine_swap_bounds_for_sample(row, scale_range)
        results.append(result)
    
    return pd.DataFrame(results)


def _find_argmax_dominance_ranges(scales, argmax_seq, target_val):
    """
    Find scale ranges where argmax equals target_val.
    
    Returns list of (lower, upper) tuples representing contiguous ranges
    where argmax == target_val.
    """
    ranges = []
    in_range = False
    range_start = None
    
    for i, (scale, argmax) in enumerate(zip(scales, argmax_seq)):
        if argmax == target_val:
            if not in_range:
                in_range = True
                range_start = scale
        else:
            if in_range:
                # Range ended at previous scale
                ranges.append((range_start, scales[i-1]))
                in_range = False
    
    # Handle range extending to end
    if in_range:
        ranges.append((range_start, scales[-1]))
    
    return ranges


def _intersect_ranges(ranges1, ranges2):
    """
    Compute intersection of two lists of (lower, upper) ranges.
    Returns list of (lower, upper) tuples.
    """
    result = []
    for r1_lo, r1_hi in ranges1:
        for r2_lo, r2_hi in ranges2:
            lo = max(r1_lo, r2_lo)
            hi = min(r1_hi, r2_hi)
            if lo <= hi:
                result.append((lo, hi))
    return result


def _determine_swap_bounds_for_sample(row, scale_range):
    """Determine swap bounds for a single sample."""
    d1_val = row['d1']
    d2_val = row['d2']

    # Propagate o1 failure reasons from get_xovers_df directly.
    # Use pd.notna() rather than `is not None`: pandas stores Python None as float NaN
    # in object columns, so `is not None` would fire on success rows loaded from CSV.
    o1_failure = row.get('o1_failure_reason', None)
    if pd.notna(o1_failure):
        return {
            'd1': d1_val, 'd2': d2_val,
            'lower_bound': None, 'upper_bound': None,
            'midpoint': None, 'swap_zone_width': None,
            'failure_reason': o1_failure,
        }

    # Parse crossover lists and bound types
    o1_xovers = _parse_list_field(row['o1_crossovers'])
    o2_xovers = _parse_list_field(row['o2_crossovers'])
    o1_bound_types = _parse_list_field(row['o1_bound_types'])
    o2_bound_types = _parse_list_field(row['o2_bound_types'])

    # Parse scales and argmax sequences for dominance check
    scales = _parse_list_field(row['scales'])
    argmax_o1 = _parse_list_field(row['argmax_o1'])
    argmax_o2 = _parse_list_field(row['argmax_o2'])

    # Initialize bounds — o1 provides at most one crossover (linear fit)
    # so lower/upper start unconstrained and are tightened below.
    lower_bound = scale_range[0]
    upper_bound = scale_range[1]
    failure_reason = None

    # Process o1 crossover (0 or 1 entry from linear fit)
    # An empty list here means o1_failure_reason was None but no intersection existed
    # within [0, scale_cap] — this path should not normally be reached because
    # o1_failure_reason would have been set; guard anyway.
    if len(o1_xovers) == 0:
        failure_reason = "no_o1_crossover"
    else:
        for xover, bound_type in zip(o1_xovers, o1_bound_types):
            if bound_type == 'lb':
                lower_bound = max(lower_bound, xover)
            elif bound_type == 'ub':
                upper_bound = min(upper_bound, xover)
            # 'unknown' bound types are ignored
    
    # Process o2 crossovers using pre-computed bound types
    if failure_reason is None:
        if len(o2_xovers) == 0:
            # No o2 logit crossovers at all — valid if argmax_o2 equals d1
            # somewhere in [lower_bound, upper_bound].
            in_window = any(
                argmax_o2[i] == d1_val
                for i, s in enumerate(scales)
                if lower_bound <= s <= upper_bound
            )
            if not in_window:
                failure_reason = "no_o2_crossover"
        else:
            # Filter o2 crossovers within current bounds along with their types
            valid_o2 = [(x, bt) for x, bt in zip(o2_xovers, o2_bound_types) 
                        if lower_bound <= x <= upper_bound]
            
            if len(valid_o2) == 0:
                # No o2 logit crossover (d1 vs d2) within bounds. This can happen when d1
                # was already beating d2 at o2 from the start, with only an exit crossover
                # beyond upper_bound. In that case, argmax_o2 may still equal d1 somewhere
                # within [lower_bound, upper_bound] — the downstream dominance check will
                # nail down the exact range. Only fail if o2 never predicts d1 in the window.
                in_window = any(
                    argmax_o2[i] == d1_val
                    for i, s in enumerate(scales)
                    if lower_bound <= s <= upper_bound
                )
                if not in_window:
                    failure_reason = "no_o2_crossover_in_bounds"
            else:
                for xover, bound_type in valid_o2:
                    if bound_type == 'lb':
                        lower_bound = max(lower_bound, xover)
                    elif bound_type == 'ub':
                        upper_bound = min(upper_bound, xover)
    
    # Check for invalid bounds from crossovers
    if failure_reason is None and lower_bound > upper_bound:
        failure_reason = "invalid_bounds"
    
    # Now constrain by argmax dominance (d1/d2 must actually be the top prediction)
    if failure_reason is None:
        # Find ranges where o1 predicts d2 AND o2 predicts d1
        o1_d2_ranges = _find_argmax_dominance_ranges(scales, argmax_o1, d2_val)
        o2_d1_ranges = _find_argmax_dominance_ranges(scales, argmax_o2, d1_val)
        
        if len(o1_d2_ranges) == 0:
            failure_reason = "o1_never_predicts_d2"
        elif len(o2_d1_ranges) == 0:
            failure_reason = "o2_never_predicts_d1"
        else:
            # Find intersection of o1 and o2 dominance ranges
            swap_ranges = _intersect_ranges(o1_d2_ranges, o2_d1_ranges)
            
            if len(swap_ranges) == 0:
                failure_reason = "no_overlapping_dominance"
            else:
                # Filter swap_ranges to those overlapping with crossover bounds
                crossover_bounds = [(lower_bound, upper_bound)]
                valid_swap_ranges = _intersect_ranges(swap_ranges, crossover_bounds)
                
                if len(valid_swap_ranges) == 0:
                    failure_reason = "dominance_outside_crossover_bounds"
                else:
                    # Use the first valid range (usually there's only one)
                    # Could also use the widest one if there are multiple
                    best_range = max(valid_swap_ranges, key=lambda r: r[1] - r[0])
                    lower_bound, upper_bound = best_range
    
    # Calculate midpoint and width
    if failure_reason is None:
        midpoint = (lower_bound + upper_bound) / 2
        swap_zone_width = upper_bound - lower_bound
    else:
        midpoint = None
        swap_zone_width = None
        lower_bound = None
        upper_bound = None
    
    return {
        'd1': d1_val,
        'd2': d2_val,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'midpoint': midpoint,
        'swap_zone_width': swap_zone_width,
        'failure_reason': failure_reason,
    }


def swap_outputs(
    model, sae, act_mean, feature_idx,
    swap_bounds_df,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS,
    device=None
):
    """
    Verify actual model outputs when feature is scaled to the identified midpoints.
    
    Takes the swap bounds from get_output_swap_bounds and verifies that the model
    actually produces swapped outputs (d2, d1) at the predicted scale values.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature to steer
        swap_bounds_df: DataFrame from get_output_swap_bounds
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to patch activations at (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        device: Device to use (default: auto-detect from model)
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - scale: Scale value (midpoint)
            - orig_o1, orig_o2: Original model outputs
            - patched_o1, patched_o2: Patched model outputs
            - swapped: Boolean indicating if outputs were swapped
    """
    _validate_inputs(model, sae, d1_all, d2_all, sae_acts_all, dataset, feature_idx)
    
    device = _get_device(model, device)
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Filter to only successful swap bounds (failure_reason must be null AND midpoint must be non-null)
    valid_df = swap_bounds_df[
        swap_bounds_df['failure_reason'].isna() & swap_bounds_df['midpoint'].notna()
    ].copy()
    
    if len(valid_df) == 0:
        print("Warning: No valid swap bounds found")
        return pd.DataFrame(columns=['d1', 'd2', 'scale', 'orig_o1', 'orig_o2', 
                                    'patched_o1', 'patched_o2', 'swapped'])
    
    results = []
    
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Verifying swaps", leave=True):
        result = _verify_single_swap(
            row, d1_all, d2_all, dataset, sae_acts_all,
            model, sae, act_mean, feature_idx,
            layer_idx, sep_idx, n_digits, device, hook_name_resid
        )
        results.append(result)
    
    return pd.DataFrame(results)


def _verify_single_swap(
    row, d1_all, d2_all, dataset, sae_acts_all,
    model, sae, act_mean, feature_idx,
    layer_idx, sep_idx, n_digits, device, hook_name_resid
):
    """Verify swap for a single input pair."""
    d1_val = int(row['d1'])
    d2_val = int(row['d2'])
    scale = row['midpoint']

    if scale is None or (isinstance(scale, float) and pd.isna(scale)):
        raise ValueError(f"midpoint is None/NaN for ({d1_val}, {d2_val}) — this row should have been filtered out")
    
    # Find this input in the dataset
    idx = _find_input_index(d1_all, d2_all, d1_val, d2_val)
    
    inputs_i = dataset[idx][0].unsqueeze(0).to(device)
    z_orig = sae_acts_all[idx].clone().to(device)
    feat_orig = z_orig[feature_idx].item()
    
    # Get original output (scale = 1.0)
    orig_logits = _run_model_with_scaled_feature(
        model, sae, act_mean, inputs_i, z_orig, feature_idx,
        feat_orig, 1.0, layer_idx, sep_idx, hook_name_resid
    )
    orig_o1 = orig_logits[0, OUTPUT_POS_O1, :n_digits].argmax().item()
    orig_o2 = orig_logits[0, OUTPUT_POS_O2, :n_digits].argmax().item()
    
    # Get patched output at midpoint
    patched_logits = _run_model_with_scaled_feature(
        model, sae, act_mean, inputs_i, z_orig, feature_idx,
        feat_orig, scale, layer_idx, sep_idx, hook_name_resid
    )
    patched_o1 = patched_logits[0, OUTPUT_POS_O1, :n_digits].argmax().item()
    patched_o2 = patched_logits[0, OUTPUT_POS_O2, :n_digits].argmax().item()
    
    # Check if swapped
    swapped = (patched_o1 == d2_val and patched_o2 == d1_val)
    
    return {
        'd1': d1_val,
        'd2': d2_val,
        'scale': scale,
        'orig_o1': orig_o1,
        'orig_o2': orig_o2,
        'patched_o1': patched_o1,
        'patched_o2': patched_o2,
        'swapped': swapped,
    }


def analyze_feature_crossovers(
    results, model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=DEFAULT_SEP_IDX, n_digits=DEFAULT_N_DIGITS,
    device=None, verbose=True
):
    """
    Analyze crossover points from feature_steering_experiment results using bisection.
    
    Finds exact scale values (to 3 decimal places) where d1 and d2 logits cross,
    and verifies whether the model output is swapped at those crossovers.
    
    Args:
        results: List of result dicts returned from feature_steering_experiment().
                Each dict should have: d1, d2, scales, all_logits_o1, all_logits_o2, 
                output_o1, output_o2
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature being steered
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to analyze (default: 0)
        sep_idx: SEP token position (default: 2)
        n_digits: Number of possible digit values (default: 100)
        device: Device to use (default: auto-detect from model)
        verbose: If True, prints detailed crossover analysis (default: True)
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - feat_orig: Original feature activation
            - o1_crossovers: List of scale values where o1 logits cross
            - o2_crossovers: List of scale values where o2 logits cross
            - o1_swapped: List of boolean for whether output swapped at each o1 crossover
            - o2_swapped: List of boolean for whether output swapped at each o2 crossover
    """
    device = _get_device(model, device)
    
    crossover_data = []
    
    if verbose:
        print("\n" + "="*60)
        print("CROSSOVER ANALYSIS (exact to 3dp)")
        print("="*60)
    
    for i, result in enumerate(results):
        analysis = _analyze_single_result_crossovers(
            i, result, d1_all, d2_all, sae_acts_all, dataset,
            model, sae, act_mean, feature_idx,
            layer_idx, sep_idx, n_digits, device, verbose
        )
        crossover_data.append(analysis)
    
    return pd.DataFrame(crossover_data)


def _analyze_single_result_crossovers(
    i, result, d1_all, d2_all, sae_acts_all, dataset,
    model, sae, act_mean, feature_idx,
    layer_idx, sep_idx, n_digits, device, verbose
):
    """Analyze crossovers for a single result from steering experiment."""
    d1_val = result['d1']
    d2_val = result['d2']
    scale_factors = result['scales']
    all_logits_o1 = result['all_logits_o1']
    all_logits_o2 = result['all_logits_o2']
    
    # Get data needed for bisection
    idx = _find_input_index(d1_all, d2_all, d1_val, d2_val)
    inputs_i = dataset[idx][0].unsqueeze(0).to(device)
    z_orig = sae_acts_all[idx].clone().to(device)
    feat_orig = z_orig[feature_idx].item()
    
    # Find original output (at scale = 1.0)
    original_scale_idx = np.argmin(np.abs(scale_factors - 1.0))
    original_output_o1 = result['output_o1'][original_scale_idx]
    original_output_o2 = result['output_o2'][original_scale_idx]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: Input ({d1_val}, {d2_val})")
        print(f"Original feature {feature_idx} activation: {feat_orig:.4f}")
        print(f"Original model output: ({original_output_o1}, {original_output_o2})")
        print(f"{'='*60}")
    
    # Analyze o1 crossovers
    o1_crossovers, o1_swapped = _find_and_analyze_crossovers(
        all_logits_o1, d1_val, d2_val, scale_factors, result['output_o1'], result['output_o2'],
        model, sae, act_mean, feature_idx, inputs_i, z_orig, feat_orig,
        OUTPUT_POS_O1, layer_idx, sep_idx, n_digits, device, verbose, "O1"
    )
    
    # Analyze o2 crossovers
    o2_crossovers, o2_swapped = _find_and_analyze_crossovers(
        all_logits_o2, d1_val, d2_val, scale_factors, result['output_o1'], result['output_o2'],
        model, sae, act_mean, feature_idx, inputs_i, z_orig, feat_orig,
        OUTPUT_POS_O2, layer_idx, sep_idx, n_digits, device, verbose, "O2"
    )
    
    return {
        'd1': d1_val,
        'd2': d2_val,
        'feat_orig': feat_orig,
        'o1_crossovers': o1_crossovers,
        'o2_crossovers': o2_crossovers,
        'o1_swapped': o1_swapped,
        'o2_swapped': o2_swapped,
    }


def _find_and_analyze_crossovers(
    logits, d1_val, d2_val, scale_factors, output_o1, output_o2,
    model, sae, act_mean, feature_idx, inputs_i, z_orig, feat_orig,
    output_pos, layer_idx, sep_idx, n_digits, device, verbose, position_name
):
    """Find and analyze crossovers for a specific output position."""
    d1_logits = logits[:, d1_val]
    d2_logits = logits[:, d2_val]
    diff = d1_logits - d2_logits
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    crossovers = []
    swapped_list = []
    
    if len(sign_changes) > 0:
        if verbose:
            print(f"\n📍 {position_name}: Found {len(sign_changes)} crossover point(s)")
        
        for j, crossover_idx in enumerate(sign_changes, 1):
            exact_scale = find_exact_crossover_bisection(
                model, sae, act_mean, feature_idx,
                inputs_i, z_orig, feat_orig, d1_val, d2_val,
                scale_factors[crossover_idx], scale_factors[crossover_idx + 1],
                output_pos, layer_idx, sep_idx, n_digits, device=device
            )
            exact_scale = round(exact_scale, 3)
            crossovers.append(exact_scale)
            
            pred_o1 = output_o1[crossover_idx]
            pred_o2 = output_o2[crossover_idx]
            is_swapped = (pred_o1 == d2_val and pred_o2 == d1_val)
            swapped_list.append(is_swapped)
            
            if verbose:
                swap_indicator = " SWAPPED!" if is_swapped else ""
                print(f"   Crossover #{j} at scale = {exact_scale:.3f} (3dp)")
                print(f"      d1 logit = {d1_logits[crossover_idx]:.3f}, d2 logit = {d2_logits[crossover_idx]:.3f}")
                print(f"      → Model output: ({pred_o1}, {pred_o2}){swap_indicator}")
    else:
        if verbose:
            print(f"\n❌ {position_name}: No crossover detected in range [{scale_factors[0]:.1f}, {scale_factors[-1]:.1f}]")
            if d1_logits[0] > d2_logits[0]:
                print("   d1 logit remains higher throughout")
            else:
                print("   d2 logit remains higher throughout")
    
    return crossovers, swapped_list