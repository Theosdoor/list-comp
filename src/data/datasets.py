import torch
from torch.utils.data import TensorDataset
import itertools

# When n_digits^list_len exceeds this, enumerate all combinations becomes infeasible:
# 100^4 = 100M rows ≈ 14 GB RAM, and 50k training steps covers less than one epoch.
# Above this threshold we randomly sample MAX_DATASET_SIZE unique sequences instead,
# keeping multiple epochs of training within a fixed step budget.
MAX_DATASET_SIZE = 1_000_000


def _shuffle_pair(inputs: torch.Tensor, targets: torch.Tensor, seed=None):
    """Shuffle aligned input/target tensors with the same permutation."""
    if len(inputs) != len(targets):
        raise ValueError(f"inputs and targets must have same length, got {len(inputs)} vs {len(targets)}")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(seed)

    perm = torch.randperm(len(inputs), generator=generator, device=inputs.device)
    return inputs[perm], targets[perm]


def _split_pair(inputs: torch.Tensor, targets: torch.Tensor, train_split: float):
    """Split aligned input/target tensors into train and validation portions."""
    split = int(train_split * len(inputs))
    return inputs[:split], targets[:split], inputs[split:], targets[split:]


def get_dataset(
    list_len=2, # [d1, d2]
    n_digits=100,
    train_split=0.8, # 80% train, 20% test
    train_dupes_only=False, # whether to remove duplicates (where d1 == d2) from the validation set
    no_dupes=False, # whether to use only non-duplicates (i.e. all d1 != d2)
    mask_tok=None, # special masking token for o1 and o2
    sep_tok=None, # special seperator token for the model to think about the input
    seed=0, # seed for reproducible shuffle
    max_dataset_size=MAX_DATASET_SIZE, # cap on total sequences; None = always enumerate all
):
    """
    Generate train/validation datasets for list comparison tasks.
    
    Returns:
        (train_ds, val_ds): Tuple of TensorDataset objects
    
    Important for evaluation:
        To get validation data matching training config, use defaults:
            _, val_ds = get_dataset(list_len=2, n_digits=100)
        
        DO NOT use train_split=1.0 for evaluation - this includes training data
        and inflates accuracy by ~4%. The default train_split=0.8 matches how
        models were trained.
    """
    # Set seed for reproducible dataset generation
    torch.manual_seed(seed)
    
    seq_len = list_len * 2 + 1 # [d1, d2, SEP, o1, o2]
    if mask_tok is None:
        mask_tok = n_digits 
    if sep_tok is None:
        sep_tok = n_digits + 1 

    # Create all possible combinations of digits, or sample if the full set is too large.
    # Full enumeration: n_digits^list_len rows. For list_len=4 that's 100M rows (≈14 GB),
    # and 50k training steps would cover less than one epoch — the model can't converge.
    # When capped, we sample max_dataset_size rows uniformly at random (with replacement
    # to keep it simple; collisions are negligible when sampling << population size).
    full_size = n_digits ** list_len
    sampled = max_dataset_size is not None and full_size > max_dataset_size
    if sampled and (no_dupes or train_dupes_only):
        raise ValueError(
            "no_dupes/train_dupes_only require full enumeration but dataset is too large "
            f"(n_digits^list_len={full_size:,} > max_dataset_size={max_dataset_size:,}). "
            "Pass max_dataset_size=None to force enumeration, or don't use these flags."
        )
    if max_dataset_size is not None and full_size > max_dataset_size:
        rng = torch.Generator()
        rng.manual_seed(seed)
        all_data = torch.randint(0, n_digits, (max_dataset_size, list_len),
                                 dtype=torch.int64, generator=rng)
    else:
        digits = list(range(n_digits))
        all_data = list(itertools.product(digits, repeat=list_len))
        all_data = torch.tensor(all_data, dtype=torch.int64)

    # Split into dupes (all elements equal) and non-dupes
    # For list_len=2: [d1,d2] is dupe if d1==d2
    # For list_len=3: [d1,d2,d3] is dupe if d1==d2==d3
    dupes_mask = (all_data == all_data[:, 0:1]).all(dim=1)
    dupes = all_data[dupes_mask]
    non_dupes = all_data[~dupes_mask]

    def build_inputs_targets(data_tensor: torch.Tensor):
        n = len(data_tensor)
        targets = torch.full((n, seq_len), sep_tok, dtype=torch.int64)
        targets[:, :list_len] = data_tensor
        targets[:, list_len + 1 :] = data_tensor
        inputs = targets.clone()
        inputs[:, list_len + 1 :] = mask_tok
        return inputs, targets

    if no_dupes:
        # Use only non-duplicates for both train and val
        all_inputs, all_targets = build_inputs_targets(non_dupes)
        all_inputs, all_targets = _shuffle_pair(all_inputs, all_targets)
        train_inputs, train_targets, val_inputs, val_targets = _split_pair(
            all_inputs, all_targets, train_split
        )

    else: # if allowed dupes
        if train_dupes_only:
            # Split non-dupes into train/val, then add dupes only to train and reshuffle
            nd_inputs, nd_targets = build_inputs_targets(non_dupes)
            nd_inputs, nd_targets = _shuffle_pair(nd_inputs, nd_targets)
            train_inputs, train_targets, val_inputs, val_targets = _split_pair(
                nd_inputs, nd_targets, train_split
            )

            # Build dupes and append to train only
            d_inputs, d_targets = build_inputs_targets(dupes)
            train_inputs = torch.cat([train_inputs, d_inputs], dim=0)
            train_targets = torch.cat([train_targets, d_targets], dim=0)

            # Shuffle the augmented training set
            train_inputs, train_targets = _shuffle_pair(train_inputs, train_targets)
        else:
            # Use all data (dupes + non-dupes) for both train and val
            all_inputs, all_targets = build_inputs_targets(all_data)
            all_inputs, all_targets = _shuffle_pair(all_inputs, all_targets)
            train_inputs, train_targets, val_inputs, val_targets = _split_pair(
                all_inputs, all_targets, train_split
            )

    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds = TensorDataset(val_inputs, val_targets)

    return train_ds, val_ds
