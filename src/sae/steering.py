"""
SAE Feature Steering and Crossover Analysis

Functions for analyzing how scaling SAE features affects model outputs,
including crossover detection and output swap verification.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .hooks import make_sae_patch_hook, make_batched_sae_patch_hook


def find_exact_crossover_bisection(
    model, sae, act_mean, feature_idx,
    inputs_i, z_orig, feat_orig, d1_val, d2_val,
    scale_low, scale_high, output_pos,
    layer_idx=0, sep_idx=2, n_digits=100,
    tol=0.0005, max_iter=20, device=None
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
        output_pos: Output position (-2 for o1, -1 for o2)
        layer_idx: Layer to patch
        sep_idx: SEP token position
        n_digits: Number of possible digits
        tol: Tolerance for convergence (default: 0.0005 for 3dp accuracy)
        max_iter: Maximum iterations
        device: Device to use
    
    Returns:
        exact_scale: Scale value where crossover occurs (to 3 decimal places)
    """
    if device is None:
        device = next(model.parameters()).device
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    def get_logit_diff_at_scale(scale):
        """Helper: Run model at given scale and return d1_logit - d2_logit"""
        z_scaled = z_orig.clone()
        z_scaled[feature_idx] = feat_orig * scale
        recon = sae.decode(z_scaled.unsqueeze(0))
        
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                inputs_i,
                fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
            )
        
        logits = patched_logits[0, output_pos, :n_digits].cpu().numpy()
        return logits[d1_val] - logits[d2_val]
    
    # Bisection loop
    for _ in range(max_iter):
        if scale_high - scale_low < tol:
            break
        
        scale_mid = (scale_low + scale_high) / 2
        diff_mid = get_logit_diff_at_scale(scale_mid)
        diff_low = get_logit_diff_at_scale(scale_low)
        
        # Check which half contains the root
        if diff_mid * diff_low > 0:
            # Same sign, root is in upper half
            scale_low = scale_mid
        else:
            # Different sign, root is in lower half
            scale_high = scale_mid
    
    return (scale_low + scale_high) / 2


def feature_steering_experiment(
    model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=2, n_digits=100,
    scale_factors=None, scale_range=[0.0, 10.0], n_test_cases=5, seed=42,
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
        layer_idx: Layer to patch activations at
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        scale_factors: Array of scale factors to test (overrides scale_range if provided)
        scale_range: Tuple (min, max) for scale factors (default: [0.0, 10.0])
        n_test_cases: Number of test cases to sample (ignored if test_pairs is provided)
        seed: Random seed for sampling (ignored if test_pairs is provided)
        test_pairs: Optional list of (d1, d2) tuples to plot specific inputs.
                   If provided, overrides n_test_cases and seed.
                   Example: [(1, 83), (14, 67), (0, 10)]
        device: Device to use (default: auto-detect from model)
        plot: Whether to create visualization
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
    if scale_factors is None:
        scale_factors = np.linspace(scale_range[0], scale_range[1], 100)
    
    # Auto-detect device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Determine test pairs
    if test_pairs is not None:
        # Use specified test pairs
        print(f"Using {len(test_pairs)} specified input pairs")
    else:
        # Select test cases where the feature actually fires
        active_indices = torch.where(sae_acts_all[:, feature_idx] > 0)[0]
        print(f"Feature {feature_idx} fires on {len(active_indices)} / {len(d1_all)} inputs")
        
        # Sample from active inputs only
        np.random.seed(seed)
        n_samples = min(n_test_cases, len(active_indices))
        test_indices = np.random.choice(active_indices.numpy(), size=n_samples, replace=False)
        test_pairs = [(d1_all[i].item(), d2_all[i].item()) for i in test_indices]
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Storage for results
    all_results = []
    
    for d1_val, d2_val in test_pairs:
        # Find this pair in dataset
        mask = (d1_all == d1_val) & (d2_all == d2_val)
        if mask.sum() == 0:
            continue
        idx = torch.where(mask)[0][0].item()
        
        # Get inputs from dataset
        inputs_i = dataset[idx][0].unsqueeze(0).to(device)
        z_orig = sae_acts_all[idx].clone().to(device)
        feat_orig = z_orig[feature_idx].item()
        
        # Storage for all logits at o1 and o2 (for ALL digits)
        all_logits_o1 = []  # Will be shape [n_scales, n_digits]
        all_logits_o2 = []  # Will be shape [n_scales, n_digits]
        
        for scale in scale_factors:
            z_scaled = z_orig.clone()
            z_scaled[feature_idx] = feat_orig * scale
            
            recon = sae.decode(z_scaled.unsqueeze(0))
            
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    inputs_i,
                    fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
                )
            
            # Get logits at o1 (position -2) and o2 (position -1) for ALL digits
            logits_o1 = patched_logits[0, -2, :n_digits].cpu().numpy()
            logits_o2 = patched_logits[0, -1, :n_digits].cpu().numpy()
            all_logits_o1.append(logits_o1)
            all_logits_o2.append(logits_o2)
        
        all_logits_o1 = np.array(all_logits_o1)  # [n_scales, n_digits]
        all_logits_o2 = np.array(all_logits_o2)  # [n_scales, n_digits]
        
        all_results.append({
            'd1': d1_val, 'd2': d2_val,
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
    
    # Create visualization if requested
    if plot and len(all_results) > 0:
        fig, axes = plt.subplots(2, len(all_results), figsize=(4*len(all_results), 10), squeeze=False)
        
        for col, result in enumerate(all_results):
            d1, d2 = result['d1'], result['d2']
            scales = result['scales']
            
            # Top row: Logits at o1 position
            ax1 = axes[0, col]
            
            # Plot all other logits in grey
            for digit in range(n_digits):
                if digit != d1 and digit != d2:
                    ax1.plot(scales, result['all_logits_o1'][:, digit], 'grey', alpha=0.2, linewidth=0.5)
            
            # Plot d1 and d2 logits on top
            ax1.plot(scales, result['logit_d1_o1'], 'b-', linewidth=2, label=f'd1={d1} logit')
            ax1.plot(scales, result['logit_d2_o1'], 'r-', linewidth=2, label=f'd2={d2} logit')
            
            # Mark original activation point
            ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Original')
            ax1.axvline(x=0.0, color='black', linestyle=':', alpha=0.5, label='Ablated')
            
            ax1.set_xlabel(f'Feature {feature_idx} Scale Factor')
            ax1.set_ylabel('Logit at o1')
            ax1.set_title(f'All Logits at Output Position 1\\nInput: ({d1}, {d2}), Original f{feature_idx}={result["order_feat_orig"]:.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom row: Logits at o2 position
            ax2 = axes[1, col]
            
            # Plot all other logits in grey
            for digit in range(n_digits):
                if digit != d1 and digit != d2:
                    ax2.plot(scales, result['all_logits_o2'][:, digit], 'grey', alpha=0.2, linewidth=0.5)
            
            # Plot d1 and d2 logits on top
            ax2.plot(scales, result['logit_d1_o2'], 'b-', linewidth=2, label=f'd1={d1} logit')
            ax2.plot(scales, result['logit_d2_o2'], 'r-', linewidth=2, label=f'd2={d2} logit')
            
            # Mark original activation point
            ax2.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Original')
            ax2.axvline(x=0.0, color='black', linestyle=':', alpha=0.5, label='Ablated')
            
            ax2.set_xlabel(f'Feature {feature_idx} Scale Factor')
            ax2.set_ylabel('Logit at o2')
            ax2.set_title(f'All Logits at Output Position 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'feature_{feature_idx}_logit_steering.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    return all_results


def get_xovers_df(
    model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=2, n_digits=100,
    scale_range=[0.0, 10.0], n_coarse_samples=20,
    batch_size=64, device=None
):
    """
    Find all crossover points for all inputs in the dataset (BATCHED VERSION).
    
    Uses batched coarse sampling followed by bisection to efficiently detect where
    d1 and d2 logits intersect at output positions o1 and o2.
    
    Args:
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature to steer
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to patch activations at
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        scale_range: Tuple (min, max) for scale factors
        n_coarse_samples: Number of coarse samples for initial detection
        batch_size: Number of inputs to process in parallel (default: 64)
        device: Device to use (default: auto-detect from model)
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - feat_orig: Original feature activation
            - o1_crossovers: List of crossover scales at output position 1
            - o2_crossovers: List of crossover scales at output position 2
            - n_o1_xover: Number of o1 crossovers
            - n_o2_xover: Number of o2 crossovers
    """
    if device is None:
        device = next(model.parameters()).device
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    scale_factors = np.linspace(scale_range[0], scale_range[1], n_coarse_samples)
    n_samples = len(d1_all)
    
    # Storage for all logits across batches
    all_results = []
    
    # Process in batches for coarse sampling
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Finding crossovers (batched)", leave=True):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        batch_indices = range(batch_start, batch_end)
        current_batch_size = batch_end - batch_start
        
        # Collect batch data
        batch_d1 = d1_all[batch_start:batch_end]
        batch_d2 = d2_all[batch_start:batch_end]
        batch_z_orig = sae_acts_all[batch_start:batch_end].to(device)
        batch_feat_orig = batch_z_orig[:, feature_idx].clone()
        
        # Collect inputs for this batch
        batch_inputs = torch.stack([dataset[i][0] for i in batch_indices]).to(device)
        
        # Storage for this batch's logits [batch_size, n_scales, n_digits]
        batch_logits_o1 = []
        batch_logits_o2 = []
        
        # Process all scales for entire batch
        for scale in scale_factors:
            # Scale the feature for entire batch
            batch_z_scaled = batch_z_orig.clone()
            batch_z_scaled[:, feature_idx] = batch_feat_orig * scale
            
            # Decode entire batch
            batch_recon = sae.decode(batch_z_scaled)  # [batch_size, d_model]
            
            # Run model on entire batch
            with torch.no_grad():
                batch_patched_logits = model.run_with_hooks(
                    batch_inputs,
                    fwd_hooks=[(hook_name_resid, make_batched_sae_patch_hook(batch_recon, act_mean, sep_idx))]
                )
            
            # Extract logits [batch_size, n_digits]
            logits_o1 = batch_patched_logits[:, -2, :n_digits].cpu().numpy()
            logits_o2 = batch_patched_logits[:, -1, :n_digits].cpu().numpy()
            batch_logits_o1.append(logits_o1)
            batch_logits_o2.append(logits_o2)
        
        # Reshape: [n_scales, batch_size, n_digits] -> [batch_size, n_scales, n_digits]
        batch_logits_o1 = np.array(batch_logits_o1).transpose(1, 0, 2)
        batch_logits_o2 = np.array(batch_logits_o2).transpose(1, 0, 2)
        
        # Process each input in batch to find crossovers
        for i in range(current_batch_size):
            global_idx = batch_start + i
            d1_val = batch_d1[i].item()
            d2_val = batch_d2[i].item()
            feat_orig = batch_feat_orig[i].item()
            
            # Skip if feature doesn't fire
            if feat_orig == 0:
                all_results.append({
                    'd1': d1_val,
                    'd2': d2_val,
                    'feat_orig': feat_orig,
                    'o1_crossovers': [],
                    'o2_crossovers': [],
                    'n_o1_xover': 0,
                    'n_o2_xover': 0,
                })
                continue
            
            # Get logits for this input [n_scales, n_digits]
            logits_o1 = batch_logits_o1[i]
            logits_o2 = batch_logits_o2[i]
            
            # Find crossovers for o1
            d1_logits_o1 = logits_o1[:, d1_val]
            d2_logits_o1 = logits_o1[:, d2_val]
            diff_o1 = d1_logits_o1 - d2_logits_o1
            sign_changes_o1 = np.where(np.diff(np.sign(diff_o1)))[0]
            
            # Use bisection for exact crossovers
            inputs_i = dataset[global_idx][0].unsqueeze(0).to(device)
            z_orig = batch_z_orig[i]
            
            o1_crossovers = []
            for idx in sign_changes_o1:
                exact_scale = find_exact_crossover_bisection(
                    model, sae, act_mean, feature_idx,
                    inputs_i, z_orig, feat_orig, d1_val, d2_val,
                    scale_factors[idx], scale_factors[idx + 1], -2,
                    layer_idx, sep_idx, n_digits, device=device
                )
                o1_crossovers.append(round(exact_scale, 3))
            
            # Find crossovers for o2
            d1_logits_o2 = logits_o2[:, d1_val]
            d2_logits_o2 = logits_o2[:, d2_val]
            diff_o2 = d1_logits_o2 - d2_logits_o2
            sign_changes_o2 = np.where(np.diff(np.sign(diff_o2)))[0]
            
            o2_crossovers = []
            for idx in sign_changes_o2:
                exact_scale = find_exact_crossover_bisection(
                    model, sae, act_mean, feature_idx,
                    inputs_i, z_orig, feat_orig, d1_val, d2_val,
                    scale_factors[idx], scale_factors[idx + 1], -1,
                    layer_idx, sep_idx, n_digits, device=device
                )
                o2_crossovers.append(round(exact_scale, 3))
            
            all_results.append({
                'd1': d1_val,
                'd2': d2_val,
                'feat_orig': feat_orig,
                'o1_crossovers': o1_crossovers,
                'o2_crossovers': o2_crossovers,
                'n_o1_xover': len(o1_crossovers),
                'n_o2_xover': len(o2_crossovers),
            })
    
    return pd.DataFrame(all_results)


def get_output_swap_bounds(xovers_df, scale_range=[0.0, 10.0]):
    """
    Identify scale ranges where outputs should swap from (d1, d2) to (d2, d1).
    
    For outputs to swap:
    - o1 must predict d2 (d2_logit > d1_logit at o1)
    - o2 must predict d1 (d1_logit > d2_logit at o2)
    
    Args:
        xovers_df: DataFrame from get_xovers_df
        scale_range: Tuple (min, max) for initial bounds
    
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
        d1_val = row['d1']
        d2_val = row['d2']
        o1_xovers = row['o1_crossovers']
        o2_xovers = row['o2_crossovers']
        
        # Initialize bounds
        lower_bound = scale_range[0]
        upper_bound = scale_range[1]
        failure_reason = None
        
        # Check o1 crossover
        if len(o1_xovers) == 0:
            failure_reason = "no_o1_crossover"
        else:
            # For simplicity, use first o1 crossover
            # TODO: Could handle multiple o1 crossovers more sophisticatedly
            o1_xover = o1_xovers[0]
            
            # Need to determine which side of crossover has d2 > d1
            # We'll check a point slightly left and right of crossover
            # For now, assume standard behavior: d1 > d2 on left, d2 > d1 on right
            # This means we want scale >= o1_xover for o1 to predict d2
            lower_bound = max(lower_bound, o1_xover)
        
        # Check o2 crossover(s)
        if failure_reason is None:
            if len(o2_xovers) == 0:
                failure_reason = "no_o2_crossover"
            else:
                # Filter o2 crossovers within current bounds
                valid_o2_xovers = [x for x in o2_xovers if lower_bound <= x <= upper_bound]
                
                if len(valid_o2_xovers) == 0:
                    failure_reason = "no_o2_crossover"
                else:
                    # For o2, we need d1 > d2, which typically happens on right side of crossover
                    # If there are 2 crossovers, the zone is typically between them
                    if len(valid_o2_xovers) == 1:
                        # Single crossover: assume d1 > d2 on right
                        lower_bound = max(lower_bound, valid_o2_xovers[0])
                    else:
                        # Multiple crossovers: assume swap zone is between first and last
                        lower_bound = max(lower_bound, valid_o2_xovers[0])
                        upper_bound = min(upper_bound, valid_o2_xovers[-1])
        
        # Check for invalid bounds
        if failure_reason is None and lower_bound > upper_bound:
            failure_reason = "invalid_bounds"
        
        # Calculate midpoint and width
        if failure_reason is None:
            midpoint = (lower_bound + upper_bound) / 2
            swap_zone_width = upper_bound - lower_bound
        else:
            midpoint = None
            swap_zone_width = None
            lower_bound = None
            upper_bound = None
        
        results.append({
            'd1': d1_val,
            'd2': d2_val,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'midpoint': midpoint,
            'swap_zone_width': swap_zone_width,
            'failure_reason': failure_reason,
        })
    
    return pd.DataFrame(results)


def swap_outputs(
    model, sae, act_mean, feature_idx,
    swap_bounds_df,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=2, n_digits=100,
    device=None
):
    """
    Verify actual model outputs when feature is scaled to the identified midpoints.
    
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
        layer_idx: Layer to patch activations at
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        device: Device to use (default: auto-detect from model)
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - scale: Scale value (midpoint)
            - orig_o1, orig_o2: Original model outputs
            - patched_o1, patched_o2: Patched model outputs
            - swapped: Boolean indicating if outputs were swapped
    """
    if device is None:
        device = next(model.parameters()).device
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Filter to only successful swap bounds
    valid_df = swap_bounds_df[swap_bounds_df['failure_reason'].isna()].copy()
    
    results = []
    
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Verifying swaps", leave=True):
        d1_val = int(row['d1'])
        d2_val = int(row['d2'])
        scale = row['midpoint']
        
        # Find this input in the dataset
        mask = (d1_all == d1_val) & (d2_all == d2_val)
        idx = torch.where(mask)[0][0].item()
        
        inputs_i = dataset[idx][0].unsqueeze(0).to(device)
        z_orig = sae_acts_all[idx].clone().to(device)
        feat_orig = z_orig[feature_idx].item()
        
        # Get original output (scale = 1.0)
        z_orig_scaled = z_orig.clone()
        z_orig_scaled[feature_idx] = feat_orig * 1.0
        recon_orig = sae.decode(z_orig_scaled.unsqueeze(0))
        
        with torch.no_grad():
            orig_logits = model.run_with_hooks(
                inputs_i,
                fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon_orig, act_mean, sep_idx))]
            )
        orig_o1 = orig_logits[0, -2, :n_digits].argmax().item()
        orig_o2 = orig_logits[0, -1, :n_digits].argmax().item()
        
        # Get patched output at midpoint
        z_patched = z_orig.clone()
        z_patched[feature_idx] = feat_orig * scale
        recon_patched = sae.decode(z_patched.unsqueeze(0))
        
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                inputs_i,
                fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon_patched, act_mean, sep_idx))]
            )
        patched_o1 = patched_logits[0, -2, :n_digits].argmax().item()
        patched_o2 = patched_logits[0, -1, :n_digits].argmax().item()
        
        # Check if swapped
        swapped = (patched_o1 == d2_val and patched_o2 == d1_val)
        
        results.append({
            'd1': d1_val,
            'd2': d2_val,
            'scale': scale,
            'orig_o1': orig_o1,
            'orig_o2': orig_o2,
            'patched_o1': patched_o1,
            'patched_o2': patched_o2,
            'swapped': swapped,
        })
    
    return pd.DataFrame(results)
