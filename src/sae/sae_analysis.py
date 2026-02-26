"""
SAE Analysis Utilities (DEPRECATED)

This module has been split into focused submodules for better organization:
- hooks.py: Hook creation utilities
- activation_collection.py: Data collection functions
- visualization.py: Plotting and visualization
- metrics.py: Reconstruction metrics
- steering.py: Feature steering experiments
- loading.py: SAE loading utilities

Import from individual modules or from src.sae for better organization:
    from src.sae import collect_sae_activations, feature_steering_experiment
"""

import warnings

warnings.warn(
    "Importing from sae_analysis.py directly is deprecated. "
    "Use 'from src.sae import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from .hooks import *  # noqa: F401, F403
from .activation_collection import *  # noqa: F401, F403
from .visualization import *  # noqa: F401, F403
from .metrics import *  # noqa: F401, F403
from .steering import *  # noqa: F401, F403
from .loading import *  # noqa: F401, F403
