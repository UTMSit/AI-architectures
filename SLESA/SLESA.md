# UIT:SLESA — Special Low-End Sensitive Agent

**Version:** 1.0
**License:** UOPLicense
**Author:** UTMS Innovative Technologies

## Overview

SLESA is a neural architecture designed for **sensitive low-level code generation** (OS development, kernels, assembly, firmware) on **low-end hardware** (RX 570 8GB, ARM CPUs). It combines:

- **RWKV8-style infinite context** — exact global state without quadratic complexity.
- **Large precise local window** — full attention over 8k-16k tokens for syntax and structure.
- **Prefrontal Lobe** — RNN with noise and iterative deliberation for creative reasoning.
- **Symbolic Store** — exact memory for addresses, registers, and constants.
- **Precision Gate** — adaptive switching between precise and creative modes.
- **Fill-in-the-Middle** — native support for code completion.

## Architecture

### Cortex (Main Processor)

The Cortex handles input processing and maintains two types of context:

**Global State (RWKV8-style):**

S_t = decay_t ⊙ S_{t-1} + v_t ⊗ k_t

Where:
- S_t ∈ R^{d×d} — state matrix (or factorized as a_t, b_t).
- k_t, v_t ∈ R^d — key and value projections of input x_t.
- decay_t = exp(−exp(W_decay · x_t)) — data-dependent decay ∈ (0,1)^d.

Readout: y_glob = S_t^T · r_t, where r_t is a query projection.

**Local Window (Precise Short-term Memory):**

Ring buffer of W tokens (W = 8192–16384) with full multi-head attention and Rotary Position Embeddings (RoPE).

y_loc = Attention(Q_t, K_win, V_win)

**Gate Mixing:**

g = σ(W_g · [x_t, y_glob, y_loc])
y_cortex = g ⊙ y_loc + (1 − g) ⊙ y_glob

**Mixture of Experts FFN:**

y_cortex = MoE(y_cortex) — router selects top-k experts from N total.

### Prefrontal Lobe (Creative Reasoning)

A small RNN with internal noise and iterative deliberation.

**State Update:**

h_t = tanh(W_h · h_{t-1} + W_in · [y_cortex, diag(S_t)] + σ ⊙ ε)

Where:
- h_t ∈ R^{d_pref} — prefrontal state.
- σ ⊙ ε — learned noise (σ trained, ε ~ N(0,1)).

**Iterative Deliberation (K = 3–6 steps):**

At each step i:
1. Generate M hypotheses with different noise:
   hyp_m = W_hyp · (h_t + σ_m ⊙ ε_m)
2. Evaluate each via Value Head:
   val_m = W_val · hyp_m
3. Select best:
   h_best = hyp_argmax(val_m)
4. Update state:
   h_t ← h_t + α · h_best

**Output:**

y_pref = W_out · h_t
β = σ(W_β · [y_cortex, h_t])
y_combined = y_cortex + β ⊙ y_pref

### Precision Gate

Controlled by the Prefrontal Lobe:

p = σ(W_p · h_t)

If p > 0.5: precise mode — Symbolic Store active, low softmax temperature.
If p ≤ 0.5: creative mode — standard softmax sampling.

### Symbolic Store

Exact associative memory for numerical values and addresses.

- Key: 32-bit hash computed from numerical literals (e.g., token `0x3F8` → hash `0x3F8`).
- Operations: WRITE(hash, value), READ(hash) → exact value.
- Used instead of softmax for numerical tokens in precise mode.

### Fill-in-the-Middle (FIM)

Special tokens: `<|prefix|>`, `<|suffix|>`, `<|middle|>`.

Training: sequences rearranged as prefix + suffix + middle.
Inference: model receives prefix + suffix, generates middle.

### Output

y_ffn = MoE(y_combined)
logits = W_vocab · y_ffn
output = READ(hash) if p > 0.5 else softmax(logits / τ)

## Key Properties

| Property | Value |
|----------|-------|
| Context length | ∞ (RWKV8 state), 8k-16k precise (local window) |
| Complexity per token | O(d²) for state, O(W·d) for local attention |
| Memory for 5B model (6-bit) | ~3.75 GB weights + 32 MB local window + Symbolic Store |
| Target hardware | RX 570 8GB, ARM CPUs |
| Precise mode | Deterministic, uses Symbolic Store |
| Creative mode | Stochastic, governed by Prefrontal Lobe noise |

## License

UOPLicense — see LICENSE file.
