# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import einops
import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import utils
from src.utils.runtime import configure_runtime
from src.models.transformer import build_attention_mask
from src.models.utils import load_model, accuracy, infer_model_config
from src.data.datasets import get_dataset

# Configure plotly to use static rendering if widgets fail
import plotly.io as pio
pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# %% [markdown]
# # Data and Model setup

# %%
# ---------- parameters ----------
MODEL_NAME = '2layer_100dig_64d'
# Construct path relative to project root (parent of model_scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", MODEL_NAME + ".pt")

# Check model exists before proceeding
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist. Please train the model first.")

# Infer model config from checkpoint
inferred = infer_model_config(MODEL_PATH)
D_MODEL = inferred['d_model']
N_LAYER = inferred['n_layers']
N_HEAD = inferred['n_heads']
VOCAB = inferred['d_vocab']
USE_LN = inferred['use_ln']
USE_BIAS = inferred['use_bias']
USE_WV = inferred['use_wv']
USE_WO = inferred['use_wo']

# Derive data parameters from inferred vocab size and model name
from src.models.transformer import parse_model_name_safe
parsed = parse_model_name_safe(MODEL_NAME)
N_DIGITS = VOCAB - 2  # vocab = digits + MASK + SEP
LIST_LEN = parsed.list_len  # parsed from model name, defaults to 2
SEQ_LEN = LIST_LEN * 2 + 1  # [d1, ..., dn, SEP, o1, ..., on]

DIGITS = list(range(N_DIGITS))  # 0 .. N_DIGITS-1
MASK = N_DIGITS  # special masking token for o1 and o2
SEP = N_DIGITS + 1  # special separator token

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
SEED = 0

# Provide runtime config so we don't need to thread constants everywhere
configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=SEED)


# %%
# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

mask_bias, _mask_bias_l0 = build_attention_mask()

# %%
# ---------- dataset ----------
train_ds, val_ds = get_dataset(
    list_len=LIST_LEN, 
    n_digits=N_DIGITS, 
    train_split=0.8,
    mask_tok=MASK, # use MASK as mask token
    sep_tok=SEP, # use SEP as separator token
    )

train_batch_size = min(128, len(train_ds))
val_batch_size = min(256, len(val_ds))
train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

print("Input:", train_ds[0][0])
print("Target:", train_ds[0][1])
print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")

# %%
# LOAD existing model (config already inferred at top of file)
model = load_model(
    MODEL_PATH,
    n_layers=N_LAYER,
    n_heads=N_HEAD,
    d_model=D_MODEL,
    ln=USE_LN,
    use_bias=USE_BIAS,
    use_wv=USE_WV,
    use_wo=USE_WO,
    device=DEV,
)

accuracy(model, val_dl)


# %%
# --- Model Parameters Overview ---
m_for_overview = globals().get('model', None)
if m_for_overview is not None:
    print("--- Overview of Model Parameters ---")   
    total_params = 0
    trainable_params = 0

    # Use a formatted string for better alignment
    print(f"{'Parameter Name':<40} | {'Shape':<20} | {'Trainable':<10}")
    print("-" * 80)

    for name, param in m_for_overview.named_parameters():
        shape_str = str(tuple(param.shape))
        is_trainable = "Yes" if param.requires_grad else "No"
        total_params += param.numel()

        if not param.requires_grad:
            continue
        # Print only trainable parameters
        print(f"{name:<40} | {shape_str:<20} | {is_trainable:<10}")
        trainable_params += param.numel()

    print("-" * 80)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("-" * 80)



# %% [markdown]
# # Results

# %%
# --- Setup ---
head_index_to_ablate = 0 # fixed

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Check loss on validation set
val_inputs = val_ds.tensors[0].to(DEV)
val_targets = val_ds.tensors[1].to(DEV)
sample_idx = 0  # Use the xth sample in the validation set for comparing predictions
sample_list = val_inputs[sample_idx].cpu().numpy()

# --- Calculate Original Loss on last 2 digits ---
with torch.no_grad():
    original_logits, cache = model.run_with_cache(val_inputs, return_type="logits")
    output_logits = original_logits[:, LIST_LEN+1:] # Slice to get logits for the last two positions
    output_targets = val_targets[:, LIST_LEN+1:] # Slice to get the target tokens
    
    original_loss = loss_fn(output_logits.reshape(-1, VOCAB), output_targets.reshape(-1)) # Calculate the loss
    # Calculate accuracy
    original_predictions = original_logits.argmax(dim=-1) 
    original_output_predictions = original_predictions[:, LIST_LEN+1:]
    original_accuracy = (original_output_predictions == output_targets).float().mean()

print(f"Original loss: {original_loss.item()}")
print(f"Original accuracy: {original_accuracy.item()}")
print(f"Sample sequence {sample_idx}: {sample_list}")

# %% [markdown]
# ## Attention Analysis and Ablations


# %% [markdown]
# ### Mean Attention Patterns

# %%
# --- Mean Attention Patterns ---

all_pats = [[] for _ in range(model.cfg.n_layers)]
for inputs, _ in val_dl:
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs.to(DEV))
    for l in range(model.cfg.n_layers):
        pat = cache["pattern", l][:, 0]  # (batch, Q, K)
        all_pats[l].append(pat)
all_pats = [torch.cat(pats, dim=0) for pats in all_pats]

for l, pats in enumerate(all_pats):
    identical = torch.allclose(pats, pats[0].expand_as(pats))
    print(f"Layer {l}: all attention patterns identical? {'✅' if identical else '❌'}")

with torch.no_grad():
    avg_pats = [
        torch.zeros(SEQ_LEN, SEQ_LEN, device=DEV) for _ in range(model.cfg.n_layers)
    ]
    n = 0
    for inputs, _ in val_dl:
        _, cache = model.run_with_cache(inputs.to(DEV))
        for l in range(model.cfg.n_layers):
            avg_pats[l] += cache["pattern", l][:, 0].sum(0)
        n += inputs.shape[0]
    avg_pats = [p / n for p in avg_pats]


# Create a deep copy of the model to avoid modifying the original
model_with_avg_attn = copy.deepcopy(model)

def mk_hook(avg):
    logits = (avg + 1e-12).log()  # log-prob so softmax≈avg, ε avoids -∞

    def f(scores, hook):
        return logits.unsqueeze(0).unsqueeze(0).expand_as(scores)

    return f

for l in range(model_with_avg_attn.cfg.n_layers):
    model_with_avg_attn.blocks[l].attn.hook_attn_scores.add_hook(
        mk_hook(avg_pats[l]), dir="fwd"
    )

print("Accuracy with avg-attn:", accuracy(model_with_avg_attn, val_dl))


# %% [markdown]
# Using the mean attention pattern destroys performance.

# %% [markdown]
# Earlier research found that RNNs use a fixed attention pattern where the embeddings are projected into an "onion ring" pattern. However, in our model, the attention pattern to each of the input tokens is roughly normally distributed, for each of the positions (though always summing to 1).
# 

# %% [markdown]
# ### Ablation of specific attn edges

# %%
# ---- Ablation of Specific Attention Edges ----

renorm_rows = True # whether to renormalize rows after ablation
# ^ False gets graph from paper, but True is arguably more correct (Results don't signigicantly change)
ablate_in_l0 = [
                (4,3),
                (0,0),
                (1,0)
                ]
ablate_in_l1 = [
                (0,0),
                (1,0),
                (2,0),
                (2,1),
                (4,3)
                ]

ablate_in_l2 = [(0,0),(1,0),(2,0), (2,1), (3,0),  (4,0), (4,1), (4,2), (4,3)]

# Try ablating multiple layer attention patterns at same time
def build_qk_mask(positions=None, queries=None, keys=None, seq_len=SEQ_LEN):
    """
    Create a boolean mask of shape (seq_len, seq_len) where True means "ablate this (q,k)".
    You can pass:
      - positions: list of (q, k) tuples
      - or queries: iterable of q, and keys: iterable of k (outer-product mask)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if positions is not None:
        for q, k in positions:
            if 0 <= q < seq_len and 0 <= k < seq_len:
                mask[q, k] = True
    else:
        if queries is None:
            queries = range(seq_len)
        if keys is None:
            keys = range(seq_len)
        for q in queries:
            mask[q, keys] = True
    return mask

def make_pattern_hook(mask_2d: torch.Tensor, head_index=None, set_to=0.0, renorm=True, eps=1e-12):
    """
    Returns a fwd hook for the 'pattern' activation that:
      - sets masked entries to set_to (default 0.0)
      - optionally renormalizes rows so they sum to 1 again (per head, per batch, per query row)
    Args:
      mask_2d: Bool tensor [Q, K]
      head_index: int to affect a single head, or None to affect all heads
      set_to: value to write into masked entries (usually 0.0)
      renorm: whether to renormalize rows after masking
    """
    mask_2d = mask_2d.detach()

    def hook(pattern, hook):
        # pattern: [batch, n_heads, Q, K]
        B, H, Q, K = pattern.shape
        m4_all = mask_2d.to(pattern.device).view(1, 1, Q, K)  # broadcastable
        # Keep a copy for safe fallback in renorm
        pre = pattern.clone()
        print(f"\nLayer {hook.layer()} Ablation")
        print(f'BEFORE Ablation:\n{pattern[sample_idx, head_index, :, :].cpu().numpy()}')
        # print(f'Mask:\n{m4_all[0, 0, :, :].cpu().numpy()}')
        

        if head_index is None:
            pattern = torch.where(m4_all, torch.as_tensor(set_to, device=pattern.device), pattern)
        else:
            m3 = m4_all.squeeze(1)  # [1, Q, K]
            ph = pattern[:, head_index]  # [B, Q, K]
            ph = torch.where(m3, torch.as_tensor(set_to, device=pattern.device), ph)
            pattern[:, head_index] = ph

        if renorm:
            # Renormalize only rows whose query index has any masked key
            rows_to_fix = mask_2d.any(dim=-1)  # [Q]
            if rows_to_fix.any():
                rows_idx = rows_to_fix.nonzero(as_tuple=False).squeeze(-1)  # [Nr]
                heads = range(H) if head_index is None else [head_index]
                for h in heads:
                    # p: [B, Nr, K]
                    p = pattern[:, h, rows_idx, :]
                    s = p.sum(dim=-1, keepdim=True).clamp_min(eps)   # [B, Nr, 1]
                    pattern[:, h, rows_idx, :] = p / s

        print(f'AFTER Ablation:\n{pattern[sample_idx, head_index, :, :].cpu().numpy()}')
        return pattern

    return hook

# Example usage:
# Define what to ablate per layer:
# - As explicit (q,k) pairs
# - Or as queries/keys sets (outer-product)
_layers_to_ablate_raw = {
    0: ablate_in_l0,
    1: ablate_in_l1,
    # 2: ablate_in_l2,
}
layers_to_ablate = {
    layer_idx: build_qk_mask(positions=positions, seq_len=SEQ_LEN)
    for layer_idx, positions in _layers_to_ablate_raw.items()
    if layer_idx < model.cfg.n_layers
}

# Apply to a single head or all heads
head = None  # Set to None to affect all heads, or specify a head index (e.g., 0)

# Build hooks
fwd_hooks = []
for layer_idx, mask in layers_to_ablate.items():
    hook_name = utils.get_act_name("pattern", layer_idx)
    fwd_hooks.append((hook_name, make_pattern_hook(mask, head_index=head, set_to=0.0, renorm=renorm_rows)))

# Run with hooks and evaluate on last two positions
with torch.no_grad():
    logits_multi = model.run_with_hooks(val_inputs, return_type="logits", fwd_hooks=fwd_hooks)

output_logits_multi = logits_multi[:, LIST_LEN+1:]
ablated_output_predictions = output_logits_multi.argmax(dim=-1)
output_targets = val_targets[:, LIST_LEN+1:]

ablated_loss = loss_fn(output_logits_multi.reshape(-1, VOCAB), val_targets[:, LIST_LEN+1:].reshape(-1))
ablated_acc = (ablated_output_predictions == val_targets[:, LIST_LEN+1:]).float().mean()

print("\n--- Performance Metrics ---")
print(f"Multi-layer attention ablation -> Loss: {ablated_loss.item():.3f}, Acc: {ablated_acc.item():.3f}")

# Optional: inspect a sample
idx = sample_idx
print("Sample sequence:", val_inputs[idx].cpu().numpy())
print("Original:", original_predictions[idx].cpu().numpy())
print("Ablated: ", logits_multi.argmax(dim=-1)[idx].cpu().numpy())

# %%
# Analyse errors from the ablated model wrt o_2

def analyze_o2_errors(preds, targets, inputs=None, top_k=10):
    """
    preds, targets: [B, 2] (o1, o2)
    inputs: [B, 5] full token seq [d1, d2, SEP, o1, o2] (optional)
    """
    assert preds.shape == targets.shape and preds.ndim == 2 and preds.shape[1] == 2
    B = preds.shape[0]
    o1_pred, o2_pred = preds[:, 0], preds[:, 1]
    o1_true, o2_true = targets[:, 0], targets[:, 1]

    o2_wrong_mask = (o2_pred != o2_true)
    o1_wrong_mask = (o1_pred != o1_true)
    num_o2_wrong = int(o2_wrong_mask.sum().item())
    num_o1_wrong = int(o1_wrong_mask.sum().item())
    if num_o2_wrong == 0:
        print("No o2 errors.")
        return

    # Relations to o1
    dupes_rate = (o2_pred[o2_wrong_mask] == o1_pred[o2_wrong_mask]).float().mean().item()
    equals_o1_true_rate = (o2_pred[o2_wrong_mask] == o1_true[o2_wrong_mask]).float().mean().item()
    o1_correct_given_o2_wrong = (o1_pred[o2_wrong_mask] == o1_true[o2_wrong_mask]).float().mean().item()

    # Relations to inputs (if provided)
    if inputs is not None:
        d1, d2 = inputs[:, 0], inputs[:, 1]
        eq_d1 = int((o2_pred[o2_wrong_mask] == d1[o2_wrong_mask]).sum().item())
        eq_d2 = int((o2_pred[o2_wrong_mask] == d2[o2_wrong_mask]).sum().item())
    else:
        eq_d1 = eq_d2 = None

    # Frequency of o2 predictions when wrong
    vals = o2_pred[o2_wrong_mask]
    counts = torch.bincount(vals, minlength=VOCAB).cpu() 
    top_idx = counts.argsort(descending=True)[:top_k]
    

    def tok_label(t):
        t = int(t)
        if t < N_DIGITS: return f"{t}"
        if t == MASK: return "MASK"
        if t == SEP: return "SEP"
        return f"tok{t}"

    print(f"o1 wrong: {num_o1_wrong}/{B} ({num_o1_wrong/B:.2%})")
    print(f"o2 wrong: {num_o2_wrong}/{B} ({num_o2_wrong/B:.2%})")
    print(f"P(o2_pred == o1_pred | o2 wrong): {dupes_rate:.2%}")
    print(f"P(o2_pred == o1_true | o2 wrong): {equals_o1_true_rate:.2%}")
    print(f"P(o1 correct | o2 wrong): {o1_correct_given_o2_wrong:.2%}")
    if eq_d1 is not None:
        print(f"P(o2_pred == d1 | o2 wrong): {eq_d1/num_o2_wrong:.2%} ({eq_d1})")
        print(f"P(o2_pred == d2 | o2 wrong): {eq_d2/num_o2_wrong:.2%} ({eq_d2})")

    print("\nTop o2 predictions when wrong:")
    for t in top_idx.tolist():
        c = int(counts[t].item())
        if c == 0: continue
        print(f"  {tok_label(t):>4}: {c} ({c/num_o2_wrong:.2%})")

    # Show a few concrete examples
    show = min(5, num_o2_wrong)
    idxs = torch.nonzero(o2_wrong_mask).squeeze(-1)[:show].cpu().tolist()
    print("\nExamples (d1,d2) -> (o1_true,o2_true) | (o1_pred,o2_pred):")
    for i in idxs:
        if inputs is not None:
            d1i, d2i = int(inputs[i, 0]), int(inputs[i, 1])
            left = f"({d1i},{d2i}) -> ({int(o1_true[i])},{int(o2_true[i])})"
        else:
            left = f"-> ({int(o1_true[i])},{int(o2_true[i])})"
        print(f"  {left} | ({int(o1_pred[i])},{int(o2_pred[i])})")

analyze_o2_errors(ablated_output_predictions, output_targets, inputs=val_inputs)

# %%
#  ------ Fig 1 ------

from src.interpretability.interp_utils import gen_attn_flow, find_critical_attention_edges

# Use a single validation example
example = val_inputs[sample_idx].unsqueeze(0).to(DEV)

# Run ablation to find critical edges
ablation_results = find_critical_attention_edges(
    model=model,
    inputs=val_inputs,
    targets=val_targets,
    list_len=LIST_LEN,
    accuracy_tolerance=0.01,
    head_index=0,
    verbose=True,
    renorm=renorm_rows,
)

# Generate attention flow diagram with critical edges and delta labels
gen_attn_flow(
    model=model,
    example_input=example,
    list_len=LIST_LEN,
    ablation_results=ablation_results,
    attention_threshold=0.04,
    show_delta_labels=True,
    figsize=(8, 5),
    dpi=300,
    show_plot=True,
    return_fig=False,
)


# %% [markdown]
# ## Circuit Analysis

# %%
# ---- constants for equation ----
# Get attention patterns for both layers on the validation set

with torch.no_grad():
    logits, cache = model.run_with_cache(val_inputs)
    dig_logits = logits[:,:,:-2]  # exclude SEP and mask token logits
    b_size = dig_logits.shape[0]  # batch size

# get required attention values
alpha = cache["pattern", 0][:, head_index_to_ablate]  # Layer 0
beta = cache["pattern", 1][:, head_index_to_ablate]   # Layer 1
alpha_sep_d1 = alpha[:,2, 0].unsqueeze(-1)  # SEP -> d1
alpha_sep_d2 = alpha[:,2, 1].unsqueeze(-1)  # SEP -> d2
beta_o2_o1 = beta[:,-1, -2].unsqueeze(-1)
beta_o2_sep = beta[:,-1, 2].unsqueeze(-1) # beta_o2_o1 + beta_o2_sep = 1.0

# Weights and embeddings
W_E = model.W_E.detach()  # (vocab, d_model)
W_pos = model.W_pos.detach()  # (seq_len, d_model)
W_U = model.unembed.W_U.detach()  # (d_model, vocab)

# Input tokens for d1, d2
d1_tok = val_inputs[:, 0]
d2_tok = val_inputs[:, 1]

# get embeds
big_d1 = W_E[d1_tok] + W_pos[0,:]  # d1 embedding (d_model)
big_d2 = W_E[d2_tok] + W_pos[1,:] # d2 embedding (d_model)
pos_o1 = W_pos[-2,:]  # o1 position (d_model)
pos_o2 = W_pos[-1,:]  # o2 position (d_model)
mask_embed = W_E[MASK]  # (d_model)
sep_embed = W_E[SEP] + W_pos[2,:]  # SEP token embedding (d_model)

# get shapes right
pos_o1 = pos_o1.expand(b_size, -1)
pos_o2 = pos_o2.expand(b_size, -1)
mask_embed = mask_embed.expand(b_size, -1)
sep_embed = sep_embed.expand(b_size, -1)

# print("mask_embed shape:", mask_embed.shape)
# print("sep_embed shape:", sep_embed.shape)
# print("pos_o1 shape:", pos_o1.shape)
# print("pos_o2 shape:", pos_o2.shape)
# print("big_d1 shape:", big_d1.shape)
# print("big_d2 shape:", big_d2.shape)
# print("alpha_sep_d1 shape:", alpha_sep_d1.shape)
# print("alpha_sep_d2 shape:", alpha_sep_d2.shape)
# print("beta_o2_o1 shape:", beta_o2_o1.shape)
# print("beta_o2_sep shape:", beta_o2_sep.shape)

# verify by reconstructing logits

l_o1 = (mask_embed + sep_embed + pos_o1 + alpha_sep_d1*big_d1+ alpha_sep_d2*big_d2) @ W_U  # logits for o1 (d_model)
l_o1_digits = l_o1[:, :N_DIGITS]
patched_o1_logits = l_o1_digits.argmax(dim=-1)
acc_patched_o1 = (patched_o1_logits == val_targets[:, -2]).float().mean().item()

l_o2 = ((1+beta_o2_o1)*mask_embed + beta_o2_o1*pos_o1 + pos_o2 + beta_o2_sep*(alpha_sep_d1*big_d1 + alpha_sep_d2*big_d2 + sep_embed)) @ W_U  # logits for o2 (d_model)
l_o2_digits = l_o2[:, :N_DIGITS]
patched_o2_logits = l_o2_digits.argmax(dim=-1)
acc_patched_o2 = (patched_o2_logits == val_targets[:, -1]).float().mean().item()


# Compare reconstructed logits to model logits for o2
with torch.no_grad():
    model_o2_logits = logits[:, -1, :N_DIGITS]  # [B, N_DIGITS]
    l2_diff = ((l_o2_digits - model_o2_logits).norm(dim=-1).mean().item())
    print(f"Mean L2 diff between reconstructed and model o2 logits: {l2_diff:.4f}")

print(f'\nReconstructed accuracy: {(acc_patched_o1 + acc_patched_o2) / 2.0:.3f}')

# %% [markdown]
# Epic! This suggests we have successfully reconstructed the logits as the accuracy is the same using the reconstructed logits to make predictions.

# %%
# --- Logit Difference Calculation ---
# scale constants
scaled_pos_o1 = -beta_o2_sep * pos_o1
scaled_big_d1 = -beta_o2_o1 * alpha_sep_d1 * big_d1
scaled_big_d2 = -beta_o2_o1 * alpha_sep_d2 * big_d2
scaled_sep_embed = -beta_o2_o1 * sep_embed
scaled_mask_embed = beta_o2_o1 * mask_embed

# logit_o2 - logit_o1
logit_diff = pos_o2 + scaled_pos_o1 + scaled_big_d1 + scaled_big_d2 + scaled_sep_embed + scaled_mask_embed
logit_diff = (logit_diff @ W_U )[:,:-2]  # exclude sep and mask token logits


# %%
# ----- Fig 4 - Logit Difference Contributions -----

# get embeds in d_model space
term_pos_o2 = W_pos[-1,:].expand(b_size, -1)
term_pos_o1 = -beta_o2_sep * W_pos[-2,:].expand(b_size, -1)
term_big_d1 = -beta_o2_o1 * alpha_sep_d1 * (W_E[d1_tok] + W_pos[0,:])
term_big_d2 = -beta_o2_o1 * alpha_sep_d2 * (W_E[d2_tok] + W_pos[1,:])
term_sep = -beta_o2_o1 * (W_E[SEP] + W_pos[2,:]).expand(b_size, -1)
term_mask = beta_o2_o1 * W_E[MASK].expand(b_size, -1)


# Define number of digits from logit shape
N_DIGITS = dig_logits.shape[-1]

# Helper to project d_model vectors to logit space for N_DIGITS
def unembed_digits(x):
    return (x @ W_U)[:, :N_DIGITS]

# Unembed all our terms into logit-space contributions
base_o1_digits = logits[:, -2, :N_DIGITS]
contrib_pos_o2 = unembed_digits(term_pos_o2)
contrib_pos_o1 = unembed_digits(term_pos_o1)
contrib_big_d1 = unembed_digits(term_big_d1)
contrib_big_d2 = unembed_digits(term_big_d2)
contrib_sep = unembed_digits(term_sep)
contrib_mask = unembed_digits(term_mask)

# Helper to gather the specific logits for d1 and d2 from a logit matrix
def gather_pair_cols(logits, d1, d2):
    # logits: [B, N_DIGITS], d1: [B], d2: [B]
    d1_logits = logits.gather(-1, d1.unsqueeze(-1))
    d2_logits = logits.gather(-1, d2.unsqueeze(-1))
    return torch.cat([d1_logits, d2_logits], dim=-1) # -> [B, 2]

# Define the mask for the specific cases we're analyzing (where target_o2 is d2)
# This assumes `val_targets` (shape [batch, seq_len]) is defined.
tgt_o2 = val_targets[:, -1]
mask_unique = (d1_tok != d2_tok) & ((tgt_o2 == d1_tok) | (tgt_o2 == d2_tok))
m = mask_unique & (tgt_o2 == d2_tok)

# Calculate the final scalar values required for the plot
base_diff_d2_d1 = (
    gather_pair_cols(base_o1_digits[m], d1_tok[m], d2_tok[m])[:, 1] - 
    gather_pair_cols(base_o1_digits[m], d1_tok[m], d2_tok[m])[:, 0]
).mean().cpu()

contribs_list = [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2, contrib_sep, contrib_mask]
diff_contributions = []
for c in contribs_list:
    c_d2 = gather_pair_cols(c[m], d1_tok[m], d2_tok[m])[:, 1]
    c_d1 = gather_pair_cols(c[m], d1_tok[m], d2_tok[m])[:, 0]
    diff_contributions.append((c_d2 - c_d1).mean().item())

final_calc_diff = base_diff_d2_d1 + sum(diff_contributions)



fig, ax = plt.subplots(figsize=(13, 7))

# Labels for the plot, matching the paper's notation
base_name = r'$\ell_{o_1}$ (base)'
contrib_names = [
    r'$P(o_2)$', r'$-\beta_{o_2 \to s} P(o_1)$', r'$-\beta_{o_2 \to o_1}\alpha_{s \to d_1}D_1$',
    r'$-\beta_{o_2 \to o_1}\alpha_{s \to d_2}D_2$', r'$-\beta_{o_2 \to o_1}S$', r'$+\beta_{o_2 \to o_1}E(m)$'
]

# Combine base and contribution data for plotting
all_names = [base_name] + contrib_names
all_values = [base_diff_d2_d1.item()] + diff_contributions
step_plot_values = np.insert(np.cumsum(all_values), 0, base_diff_d2_d1.item())
colors = ['grey'] + ['#377eb8' if v > 0 else '#e41a1c' for v in diff_contributions]

# Plot the main horizontal bars
bars = ax.barh(all_names, all_values, color=colors, alpha=0.9, zorder=2)
ax.invert_yaxis()

# Add text annotations for the value of each bar
for bar, value in zip(bars, all_values):
    ha = 'left' if value >= 0 else 'right'
    offset = 0.5
    ax.text(bar.get_width() + (offset if value >= 0 else -offset), bar.get_y() + bar.get_height()/2,
            f'{value:.2f}', va='center', ha=ha, fontsize=10)

# Create and configure the top axis for the cumulative sum step plot
ax_top = ax.twiny()
y_steps = np.arange(len(all_names) + 1) - 0.5
step_line, = ax_top.step(step_plot_values, y_steps, color='dimgray', where='post', 
                         linestyle='--', linewidth=1.5, label=r'Cumulative $\Delta\ell$')
# ax_top.set_xlabel(r"Cumulative $\Delta\ell$", fontsize=11, color='dimgray')
ax_top.tick_params(axis='x', colors='dimgray')

# Manually draw the final sum line for robustness
y_limits = ax.get_ylim()
final_line, = ax.plot([final_calc_diff, final_calc_diff], y_limits, 
                       color='purple', linestyle='--', linewidth=2, 
                       label=r'Final $\Delta\ell$', zorder=10)
ax.text(final_calc_diff, y_limits[1] - 0.2, f' {final_calc_diff:.2f}', color='purple', 
        ha='left', va='bottom', fontsize=10)

# Create a combined legend for elements from both axes
handles = [step_line, final_line]
ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=10)

# --- Final Aesthetics and Layout ---
ax.set_xlabel(r"Contribution to $\Delta\ell$ (Negative favors $d_1$ $\leftarrow$ | $\rightarrow$ Positive favors $d_2$)", fontsize=12)
ax.grid(axis='x', linestyle=':', alpha=0.6, zorder=1)
ax.axvline(0, color='dimgray', linestyle='-', linewidth=1.2, zorder=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 5))
ax.spines['left'].set_position(('outward', 5))
ax_top.spines['top'].set_position(('outward', 5))
ax_top.spines['right'].set_visible(False)
ax_top.tick_params(axis='y', right=False, labelright=False)
plt.yticks(fontsize=12)

# Set axis limits and apply tight layout
ax.set_xlim(left=-15, right=30)
ax_top.set_xlim(ax.get_xlim()) 
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# ## Linearly Separable Unembedding Matrix
# 

# %% [markdown]
# The projections of the last and second-to-last token vectors onto the positional dimensions are linearly separable, revealing that this composition of positional encodings is key to separating the tokens back out.

# %%
# Ensure we have a cache to use in the analysis
with torch.no_grad():
    logits, cache = model.run_with_cache(val_inputs, return_type="logits")

# Residuals: [batch, seq, d_model] or [seq, d_model]
resid = cache["resid_post", 1].detach().cpu().numpy()
if resid.ndim == 2:  # no batch
    resid = resid[None, ...]
if resid.shape[1] < 2:
    raise ValueError("Need sequence length >= 2 to compare -1 and -2.")

# Last and second-to-last token vectors
vecs_last = resid[:, -1, :]  # [B, d_model] -> o2 (last)
vecs_prev = resid[:, -2, :]  # [B, d_model] -> o1 (second-last)

# Positional matrix and projections
W_pos = model.W_pos.detach().cpu().numpy()
projs_last = (vecs_last @ W_pos.T)[:]
projs_prev = (vecs_prev @ W_pos.T)[:]

with torch.no_grad():
    # Least-squares projection onto span(W_pos) before unembedding
    # projs_last/prev are (B, n_pos) = vec @ W_pos.T
    W_pos_t = model.W_pos.cpu()                         # [n_pos, d_model]
    G = W_pos_t @ W_pos_t.T                             # [n_pos, n_pos]
    G_inv = torch.linalg.pinv(G)                        # robust pseudo-inverse

    last_coeffs = torch.from_numpy(projs_last).to(W_pos_t.dtype) @ G_inv  # (B, n_pos)
    prev_coeffs = torch.from_numpy(projs_prev).to(W_pos_t.dtype) @ G_inv  # (B, n_pos)

    last_proj = last_coeffs @ W_pos_t                   # (B, d_model)
    prev_proj = prev_coeffs @ W_pos_t                   # (B, d_model)

    last_pos_projs_unembed = last_proj @ model.W_U.cpu()  # (B, vocab)
    prev_pos_projs_unembed = prev_proj @ model.W_U.cpu()  # (B, vocab)


# %%
# --- SVM analysis ---
from sklearn.svm import SVC

# X: points, y: labels
X = np.vstack([projs_last, projs_prev])
y = np.hstack([np.ones(len(projs_last)), -np.ones(len(projs_prev))])

# hard-margin SVM via large C
clf = SVC(kernel="linear", C=1e6).fit(X, y)

separable = clf.score(X, y) > 0.99
print(f"Linearly separable: {separable} (accuracy: {clf.score(X, y)})")

if separable:
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 1.0 / np.linalg.norm(w)
    print("Margin:", margin)

# %% [markdown]
# ### Fig 5 - UMAP visualization of separable groups

# %%
# Project the positional projections onto 2D with UMAP and color by class (o2 last vs o1 prev)
if 'X' not in locals() or 'y' not in locals():
    # Fallback: rebuild from previously computed projections
    X = np.vstack([projs_last, projs_prev])
    y = np.hstack([np.ones(len(projs_last)), -np.ones(len(projs_prev))])

from sklearn.preprocessing import StandardScaler
# Robust import for UMAP across umap-learn versions
try:
    from umap import UMAP  # preferred if exposed at top-level
except Exception:  # pragma: no cover - fallback path
    from umap.umap_ import UMAP

# Standardize features for a more stable embedding
X_std = StandardScaler().fit_transform(X)
umap_model = UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                  metric="euclidean", random_state=42)
Z = umap_model.fit_transform(X_std)
# Ensure coordinates are a NumPy array for safe indexing
Z_np = np.asarray(Z)

# Plot
plt.figure(figsize=(6.8, 6.0), dpi=300)
mask_last = y == 1
mask_prev = y == -1
plt.scatter(Z_np[mask_last, 0], Z_np[mask_last, 1], s=16, c="#1f77b4", alpha=0.6, label="o2")
plt.scatter(Z_np[mask_prev, 0], Z_np[mask_prev, 1], s=16, c="#DC2626", alpha=0.6, label="o1")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(frameon=True, loc="best", title="Token")
plt.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Additional Plots

# %% 
# ----- Fig 2 - Attn histogram -----

# Extract attention weights (reuse existing cache)
attn_sep_d1 = cache["pattern", 0].squeeze()[:, 2, 0].cpu().numpy()
attn_sep_d2 = cache["pattern", 0].squeeze()[:, 2, 1].cpu().numpy()
attn_o2_sep = cache["pattern", 1].squeeze()[:, -1, 2].cpu().numpy()
attn_o2_o1  = cache["pattern", 1].squeeze()[:, -1, 3].cpu().numpy()

# Compute global x-range across all histograms
all_vals = np.concatenate([attn_sep_d1, attn_sep_d2, attn_o2_sep, attn_o2_o1])
x_min, x_max = float(all_vals.min()), float(all_vals.max())

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

# Row 1
ax = axes[0, 0]
ax.hist(attn_sep_d1, bins=30, color="skyblue", edgecolor="black")
ax.set_xlabel("Attention to d1 (pos 1)")
ax.set_ylabel("Count")
ax.set_title("SEP → d1 (Layer 1)")

ax = axes[0, 1]
ax.hist(attn_sep_d2, bins=30, color="salmon", edgecolor="black")
ax.set_xlabel("Attention to d2 (pos 2)")
ax.set_ylabel("")
ax.set_title("SEP → d2 (Layer 1)")

# Row 2
ax = axes[1, 0]
ax.hist(attn_o2_sep, bins=30, color="skyblue", edgecolor="black")
ax.set_xlabel("Attention to SEP (pos 3)")
ax.set_ylabel("Count")
ax.set_title("o2 → SEP (Layer 2)")

ax = axes[1, 1]
ax.hist(attn_o2_o1, bins=30, color="salmon", edgecolor="black")
ax.set_xlabel("Attention to o1 (pos 4)")
ax.set_ylabel("")
ax.set_title("o2 → o1 (Layer 2)")

# Apply the shared x-limits to all axes
for ax in axes.flat:
    ax.set_xlim(x_min, x_max)

# Make y-axes the same within each row
ymax_row0 = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
ymax_row1 = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
axes[0, 0].set_ylim(0, ymax_row0)
axes[0, 1].set_ylim(0, ymax_row0)
# axes[0,1].set_yticks([])
axes[1, 0].set_ylim(0, ymax_row1)
axes[1, 1].set_ylim(0, ymax_row1)
# axes[1,1].set_yticks([])

plt.tight_layout()
plt.show()

# %%
# ------ Fig 3 - SEP attn vs accuracy ------
from src.interpretability.interp_utils import plot_sep_attention_vs_accuracy


plot_sep_attention_vs_accuracy(
    model=model,
    val_inputs=val_inputs,
    val_targets=val_targets,
    list_len=LIST_LEN,
    layer=0,
    dpi=300,
)

# %%
# ---- Fig - scatter plot ----
# Indices of incorrect samples
incorrect_idx = np.where(~all_correct)[0]

plt.figure(figsize=(6, 6))
plt.scatter(
    all_inputs[incorrect_idx, 0].detach().cpu().numpy(),
    all_inputs[incorrect_idx, 1].detach().cpu().numpy(),
    c="blue",
    alpha=0.5,
    label="Incorrect",
)
plt.xlabel("d1 value")
plt.ylabel("d2 value")
# plt.title("Scatter plot of incorrect predictions by d1 and d2 values")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# This scatter plot shows the model's incorrect predictions ($o_i$) by input values ($d_i$). 
# There is no clear relationship between the input value and whether the model is more likely to fail.
# Whilst there is a clear relationship between the attention scores of the $d_i$ positions and misclassifications in Figure 3, this is not due to the input tokens, as shown here.

# %%
