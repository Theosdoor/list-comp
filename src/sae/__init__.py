"""Sparse Autoencoder (SAE) analysis and training utilities."""

# Hook utilities
from .hooks import (
    make_sae_patch_hook,
    make_dynamic_sae_patch_hook,
    make_batched_sae_patch_hook,
)

# Activation collection
from .activation_collection import (
    collect_sae_activations,
    collect_attention_patterns,
    collect_attention_weights,  # Backward compatibility
    identify_special_features,
)

# Visualization
from .visualization import (
    create_feature_heatmaps,
    create_firing_rate_histogram,
)

# Metrics
from .metrics import (
    compute_reconstruction_metrics,
    compute_sae_patched_accuracy,
)

# SAE loading
from .loading import (
    load_sae_from_local,
    load_sae_from_wandb_run,
    compare_sweep_runs,
)

# Feature steering and crossover analysis
from .steering import (
    find_exact_crossover_bisection,
    feature_steering_experiment,
    analyze_feature_crossovers,
    get_xovers_df,
    get_output_swap_bounds,
    swap_outputs,
)

__all__ = [
    # Hook utilities
    "make_sae_patch_hook",
    "make_dynamic_sae_patch_hook",
    "make_batched_sae_patch_hook",
    # Activation collection
    "collect_sae_activations",
    "collect_attention_patterns",
    "collect_attention_weights",
    "identify_special_features",
    # Visualization
    "create_feature_heatmaps",
    "create_firing_rate_histogram",
    # Metrics
    "compute_reconstruction_metrics",
    "compute_sae_patched_accuracy",
    # SAE loading
    "load_sae_from_local",
    "load_sae_from_wandb_run",
    "compare_sweep_runs",
    # Feature steering
    "find_exact_crossover_bisection",
    "feature_steering_experiment",
    "analyze_feature_crossovers",
    "get_xovers_df",
    "get_output_swap_bounds",
    "swap_outputs",
]
