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


def inspect_steered_output(
    model, sae, act_mean, feature_idx, scale,
    inputs_i, z_orig, feat_orig,
    layer_idx=0, sep_idx=2, n_digits=100, device=None
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
        layer_idx: Layer to patch
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        device: Device to use
    
    Returns:
        dict with keys:
            - 'scale': Scale factor used
            - 'logits_o1': Logits at output position 1 [n_digits]
            - 'logits_o2': Logits at output position 2 [n_digits]
            - 'pred_o1': Predicted digit at o1
            - 'pred_o2': Predicted digit at o2
            - 'd1_logit_o1', 'd2_logit_o1': Specific logits if needed
            - 'd1_logit_o2', 'd2_logit_o2': Specific logits if needed
    """
    if device is None:
        device = next(model.parameters()).device
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    
    # Scale the feature
    z_scaled = z_orig.clone()
    z_scaled[feature_idx] = feat_orig * scale
    
    # Decode and get reconstructed activations
    recon = sae.decode(z_scaled.unsqueeze(0))
    
    # Run model with patch
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            inputs_i,
            fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
        )
    
    # Extract logits at o1 and o2
    logits_o1 = patched_logits[0, -2, :n_digits].cpu().numpy()
    logits_o2 = patched_logits[0, -1, :n_digits].cpu().numpy()
    
    pred_o1 = int(logits_o1.argmax())
    pred_o2 = int(logits_o2.argmax())
    
    return {
        'scale': scale,
        'logits_o1': logits_o1,
        'logits_o2': logits_o2,
        'pred_o1': pred_o1,
        'pred_o2': pred_o2,
    }


def inspect_steered_outputs_batch(
    model, sae, act_mean, feature_idx, scales,
    inputs_i, z_orig, feat_orig, d1_val=None, d2_val=None,
    layer_idx=0, sep_idx=2, n_digits=100, device=None
):
    """
    Inspect model outputs at multiple scale factors (efficient batch version).
    
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
        layer_idx: Layer to patch
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        device: Device to use
    
    Returns:
        list of dicts with inspection results for each scale, plus a summary dataframe
    """
    if device is None:
        device = next(model.parameters()).device
    
    hook_name_resid = f"blocks.{layer_idx}.hook_resid_post"
    results = []
    
    for scale in scales:
        # Scale the feature
        z_scaled = z_orig.clone()
        z_scaled[feature_idx] = feat_orig * scale
        
        # Decode
        recon = sae.decode(z_scaled.unsqueeze(0))
        
        # Run model with patch
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                inputs_i,
                fwd_hooks=[(hook_name_resid, make_sae_patch_hook(recon, act_mean, sep_idx))]
            )
        
        # Extract logits
        logits_o1 = patched_logits[0, -2, :n_digits].cpu().numpy()
        logits_o2 = patched_logits[0, -1, :n_digits].cpu().numpy()
        
        pred_o1 = int(logits_o1.argmax())
        pred_o2 = int(logits_o2.argmax())
        
        row = {
            'scale': scale,
            'pred_o1': pred_o1,
            'pred_o2': pred_o2,
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
    scale_factors=None, scale_range=[-1.0, 4.0], n_test_cases=5, seed=42,
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
            ax1.set_title(f'All Logits at Output Position 1\nInput: ({d1}, {d2}), Original f{feature_idx}={result["order_feat_orig"]:.3f}')
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
            - scales: Array of scale factors sampled [n_coarse_samples]
            - argmax_o1: Array of argmax predictions at o1 [n_coarse_samples]
            - argmax_o2: Array of argmax predictions at o2 [n_coarse_samples]
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
            
            # Get logits for this input [n_scales, n_digits]
            logits_o1 = batch_logits_o1[i]
            logits_o2 = batch_logits_o2[i]
            
            # Compute argmax at each scale
            argmax_o1 = logits_o1.argmax(axis=1)  # [n_scales]
            argmax_o2 = logits_o2.argmax(axis=1)  # [n_scales]
            
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
                    'scales': scale_factors.tolist(),
                    'argmax_o1': argmax_o1.tolist(),
                    'argmax_o2': argmax_o2.tolist(),
                })
                continue
            
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
                'scales': scale_factors.tolist(),
                'argmax_o1': argmax_o1.tolist(),
                'argmax_o2': argmax_o2.tolist(),
            })
    
    return pd.DataFrame(all_results)

def get_output_swap_bounds(xovers_df):
    """
    Identify scale ranges where outputs should swap from (d1, d2) to (d2, d1).
    
    For outputs to swap:
    - o1 must predict d2 (argmax at o1 == d2)
    - o2 must predict d1 (argmax at o2 == d1)
    
    Uses cached argmax predictions from get_xovers_df.
    
    Args:
        xovers_df: DataFrame from get_xovers_df (must contain scales, argmax_o1, argmax_o2)
    
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
        scales = np.array(row['scales'])
        argmax_o1 = np.array(row['argmax_o1'])
        argmax_o2 = np.array(row['argmax_o2'])
        
        # Find where d2 is predicted at o1 AND d1 is predicted at o2
        d2_wins_o1 = (argmax_o1 == d2_val)
        d1_wins_o2 = (argmax_o2 == d1_val)
        both_true = d2_wins_o1 & d1_wins_o2
        
        if not both_true.any():
            failure_reason = "no_swap_zone"
            lower_bound = None
            upper_bound = None
            midpoint = None
            swap_zone_width = None
        else:
            # Find contiguous regions where both are true
            true_indices = np.where(both_true)[0]
            
            # Find gaps in the indices
            gaps = np.diff(true_indices) > 1
            if gaps.any():
                # Multiple regions - use the first one
                first_gap = np.where(gaps)[0][0]
                true_indices = true_indices[:first_gap + 1]
            
            lower_bound = float(scales[true_indices[0]])
            upper_bound = float(scales[true_indices[-1]])
            midpoint = (lower_bound + upper_bound) / 2
            swap_zone_width = upper_bound - lower_bound
            failure_reason = None
        
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


def analyze_feature_crossovers(
    results, model, sae, act_mean, feature_idx,
    d1_all, d2_all, sae_acts_all, dataset,
    layer_idx=0, sep_idx=2, n_digits=100,
    device=None, verbose=True
):
    """
    Analyze crossover points from feature_steering_experiment results using bisection for exact values.
    
    Finds exact scale values (to 3 decimal places) where d1 and d2 logits cross at,
    and verifies whether the model output is swapped at those crossovers.
    
    Args:
        results: List of result dicts returned from feature_steering_experiment().
                Each dict should have: d1, d2, scales, all_logits_o1, all_logits_o2, output_o1, output_o2
        model: Base transformer model
        sae: Trained SAE
        act_mean: Activation mean for centering
        feature_idx: Index of the SAE feature being steered
        d1_all: Tensor of d1 values [n_samples]
        d2_all: Tensor of d2 values [n_samples]
        sae_acts_all: SAE activations [n_samples, d_sae]
        dataset: PyTorch dataset for getting inputs
        layer_idx: Layer to analyze
        sep_idx: SEP token position
        n_digits: Number of possible digit values
        device: Device to use (default: auto-detect from model)
        verbose: If True, prints detailed crossover analysis; if False, only returns data
    
    Returns:
        pd.DataFrame with columns:
            - d1, d2: Input digit pair
            - feat_orig: Original feature activation
            - o1_crossovers: List of scale values where o1 logits cross
            - o2_crossovers: List of scale values where o2 logits cross
            - o1_swapped: List of boolean for whether output swapped at each o1 crossover
            - o2_swapped: List of boolean for whether output swapped at each o2 crossover
    """
    if device is None:
        device = next(model.parameters()).device
    
    crossover_data = []
    
    if verbose:
        print("\n" + "="*60)
        print("CROSSOVER ANALYSIS (exact to 3dp)")
        print("="*60)
    
    for i, result in enumerate(results):
        d1_val = result['d1']
        d2_val = result['d2']
        scale_factors = result['scales']
        all_logits_o1 = result['all_logits_o1']
        all_logits_o2 = result['all_logits_o2']
        
        # Get data needed for bisection
        mask = (d1_all == d1_val) & (d2_all == d2_val)
        idx = torch.where(mask)[0][0].item()
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
        
        # Extract d1 and d2 logits
        d1_logits_o1 = all_logits_o1[:, d1_val]
        d2_logits_o1 = all_logits_o1[:, d2_val]
        diff_o1 = d1_logits_o1 - d2_logits_o1
        
        d1_logits_o2 = all_logits_o2[:, d1_val]
        d2_logits_o2 = all_logits_o2[:, d2_val]
        diff_o2 = d1_logits_o2 - d2_logits_o2
        
        # Find where sign changes (crossover) for o1
        o1_crossovers = []
        o1_swapped = []
        sign_changes_o1 = np.where(np.diff(np.sign(diff_o1)))[0]
        
        if len(sign_changes_o1) > 0:
            if verbose:
                print(f"\n📍 O1: Found {len(sign_changes_o1)} crossover point(s)")
            for j, crossover_idx in enumerate(sign_changes_o1, 1):
                exact_scale = find_exact_crossover_bisection(
                    model=model, sae=sae, act_mean=act_mean,
                    feature_idx=feature_idx,
                    inputs_i=inputs_i, z_orig=z_orig, feat_orig=feat_orig,
                    d1_val=d1_val, d2_val=d2_val,
                    scale_low=scale_factors[crossover_idx],
                    scale_high=scale_factors[crossover_idx + 1],
                    output_pos=-2,  # o1 position
                    layer_idx=layer_idx, sep_idx=sep_idx, n_digits=n_digits,
                    device=device
                )
                o1_crossovers.append(round(exact_scale, 3))
                
                pred_o1 = result['output_o1'][crossover_idx]
                pred_o2 = result['output_o2'][crossover_idx]
                is_swapped = (pred_o1 == d2_val and pred_o2 == d1_val)
                o1_swapped.append(is_swapped)
                
                swap_indicator = " SWAPPED!" if is_swapped else ""
                if verbose:
                    print(f"   Crossover #{j} at scale = {exact_scale:.3f} (3dp)")
                    print(f"      d1 logit = {d1_logits_o1[crossover_idx]:.3f}, d2 logit = {d2_logits_o1[crossover_idx]:.3f}")
                    print(f"      → Model output: ({pred_o1}, {pred_o2}){swap_indicator}")
        else:
            if verbose:
                print(f"\n❌ O1: No crossover detected in range [{scale_factors[0]:.1f}, {scale_factors[-1]:.1f}]")
                if d1_logits_o1[0] > d2_logits_o1[0]:
                    print("   d1 logit remains higher throughout")
                else:
                    print("   d2 logit remains higher throughout")
        
        # Find where sign changes (crossover) for o2
        o2_crossovers = []
        o2_swapped = []
        sign_changes_o2 = np.where(np.diff(np.sign(diff_o2)))[0]
        
        if len(sign_changes_o2) > 0:
            if verbose:
                print(f"\n📍 O2: Found {len(sign_changes_o2)} crossover point(s)")
            for j, crossover_idx in enumerate(sign_changes_o2, 1):
                exact_scale = find_exact_crossover_bisection(
                    model=model, sae=sae, act_mean=act_mean,
                    feature_idx=feature_idx,
                    inputs_i=inputs_i, z_orig=z_orig, feat_orig=feat_orig,
                    d1_val=d1_val, d2_val=d2_val,
                    scale_low=scale_factors[crossover_idx],
                    scale_high=scale_factors[crossover_idx + 1],
                    output_pos=-1,  # o2 position
                    layer_idx=layer_idx, sep_idx=sep_idx, n_digits=n_digits,
                    device=device
                )
                o2_crossovers.append(round(exact_scale, 3))
                
                pred_o1 = result['output_o1'][crossover_idx]
                pred_o2 = result['output_o2'][crossover_idx]
                is_swapped = (pred_o1 == d2_val and pred_o2 == d1_val)
                o2_swapped.append(is_swapped)
                
                swap_indicator = " SWAPPED!" if is_swapped else ""
                if verbose:
                    print(f"   Crossover #{j} at scale = {exact_scale:.3f} (3dp)")
                    print(f"      d1 logit = {d1_logits_o2[crossover_idx]:.3f}, d2 logit = {d2_logits_o2[crossover_idx]:.3f}")
                    print(f"      → Model output: ({pred_o1}, {pred_o2}){swap_indicator}")
        else:
            if verbose:
                print(f"\n❌ O2: No crossover detected in range [{scale_factors[0]:.1f}, {scale_factors[-1]:.1f}]")
                if d1_logits_o2[0] > d2_logits_o2[0]:
                    print("   d1 logit remains higher throughout")
                else:
                    print("   d2 logit remains higher throughout")
        
        crossover_data.append({
            'd1': d1_val,
            'd2': d2_val,
            'feat_orig': feat_orig,
            'o1_crossovers': o1_crossovers,
            'o2_crossovers': o2_crossovers,
            'o1_swapped': o1_swapped,
            'o2_swapped': o2_swapped,
        })
    
    return pd.DataFrame(crossover_data)
