# Further Experiments for Order by Scale

Further experiments based on the results from the paper:

> Farrell, Theo, Patrick Leask, and Noura Al Moubayed. "Order by Scale: Relative‑Magnitude Relational Composition in Attention‑Only Transformers." Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025.
Link: https://openreview.net/forum?id=vWRVzNtk7W

## Overview

tba

## Model Architecture

- **Attention-only transformer** (no MLPs)
- **2-3 layers** with single attention head per layer
- **Constrained weights**: Identity value and matrices ($W_V = W_O = I$)
- **Custom attention mask** to enforce causal structure and token-specific attention patterns

## Repository Structure

tba

## Installation

tba

## Usage

tba

## Dependencies

See `pyproject.toml`.
