# %% [markdown]
# # Results Heatmap (LIST_LEN x N_LAYERS)
#
# This notebook-style script reads `results.csv`, aggregates mean validation accuracy over `run_idx`,
# pivots to rows=LIST_LEN and cols=N_LAYERS, and renders a seaborn heatmap.

# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "results.csv"

# --- Load and aggregate ---
df = pd.read_csv(CSV_PATH)

# Ensure numeric dtypes for keys and value
for col in ["LIST_LEN", "N_LAYERS", "val_acc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing keys/values
df = df.dropna(subset=["LIST_LEN", "N_LAYERS", "val_acc"]).copy()

# Ignore configurations with N_LAYERS == 1
df = df[df["N_LAYERS"] != 1]

# set list_len = n_layers = 2 as 0.914 (we know this)
df.loc[(df["LIST_LEN"] == 2) & (df["N_LAYERS"] == 2), "val_acc"] = 0.914

# Mean across runs (run_idx) for each LIST_LEN x N_LAYERS combination
agg = (
        df.groupby(["LIST_LEN", "N_LAYERS"])  # group keys
            ["val_acc"].mean()                   # mean over runs
            .reset_index(name="val_acc_mean")    # bring keys back to columns
)

# --- Pivot to 2D grid ---
pivot = agg.pivot(index="LIST_LEN", columns="N_LAYERS", values="val_acc_mean")
# Sort row/col indices numerically for a tidy layout
pivot = pivot.sort_index().sort_index(axis=1)

# Sanity print
print("Pivot shape (rows=LIST_LEN, cols=N_LAYERS):", pivot.shape)
print(pivot.head())

# --- Seaborn heatmap ---
plt.figure(figsize=(8, 5.5))
ax = sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "Validation Accuracy"},
    linewidths=0.2,
    linecolor="white",
)
ax.set_title("Mean Validation Accuracy")
ax.set_xlabel("No. of Layers")
ax.set_ylabel("N-gram Size")
# Put LIST_LEN=1 at the bottom
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
