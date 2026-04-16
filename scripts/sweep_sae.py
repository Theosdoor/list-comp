"""
W&B sweep entry point — thin wrapper around train_sae.py.

Kept for compatibility with running sweeps (e.g. btksae sweep launched before
train_sae.py existed). New sweeps should point directly to scripts/train_sae.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_sae import train_sae_sweep

if __name__ == "__main__":
    train_sae_sweep()
