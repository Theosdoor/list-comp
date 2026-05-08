import torch

from src.data.datasets import _shuffle_pair, _split_pair, get_dataset


def test_shuffle_pair_applies_same_permutation_to_inputs_and_targets():
    inputs = torch.arange(12).reshape(4, 3)
    targets = inputs + 100

    shuffled_inputs, shuffled_targets = _shuffle_pair(inputs, targets, seed=123)

    assert shuffled_inputs.shape == inputs.shape
    assert torch.equal(shuffled_targets, shuffled_inputs + 100)
    assert not torch.equal(shuffled_inputs, inputs)


def test_split_pair_preserves_alignment_and_uses_floor_split():
    inputs = torch.arange(15).reshape(5, 3)
    targets = inputs + 100

    train_inputs, train_targets, val_inputs, val_targets = _split_pair(
        inputs, targets, train_split=0.6
    )

    assert len(train_inputs) == 3
    assert len(val_inputs) == 2
    assert torch.equal(train_targets, train_inputs + 100)
    assert torch.equal(val_targets, val_inputs + 100)


def test_get_dataset_keeps_copy_task_target_format_after_refactor():
    train_ds, val_ds = get_dataset(list_len=2, n_digits=4, train_split=0.5, seed=0)

    inputs, targets = train_ds[0]
    assert inputs.tolist()[2] == 5
    assert inputs.tolist()[3:] == [4, 4]
    assert targets.tolist()[:2] == targets.tolist()[3:]
    assert len(train_ds) == len(val_ds) == 8
