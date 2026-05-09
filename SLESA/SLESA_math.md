# UIT:SLESA — Mathematical Specification

## Forward Pass (per token t)

### Embedding
x_t = Embed(token_t) + RoPE(pos_t)

### Cortex: Global State (RWKV8)
k_t, v_t, r_t = W_k·x_t, W_v·x_t, W_r·x_t
decay_t = exp(−exp(W_d·x_t))
S_t = decay_t ⊙ S_{t-1} + v_t ⊗ k_t
y_glob = S_t^T · r_t

### Cortex: Local Window
K_win, V_win ← append(K_t, V_t), drop oldest
y_loc = softmax(Q_t·K_win^T / √d) · V_win

### Cortex: Gate
g = σ(W_g·[x_t, y_glob, y_loc])
y_cortex = g ⊙ y_loc + (1−g) ⊙ y_glob

### Prefrontal Lobe
h_t = tanh(W_h·h_{t-1} + W_in·[y_cortex, diag(S_t)] + σ ⊙ ε)

for i = 1..K:
    {hyp_m} = {W_hyp·(h_t + σ_m ⊙ ε_m), m=1..M}
    m* = argmax_m (W_val·hyp_m)
    h_t ← h_t + α·hyp_m*

y_pref = W_out·h_t
β = σ(W_β·[y_cortex, h_t])
y_combined = y_cortex + β ⊙ y_pref

### Precision Gate
p = σ(W_p·h_t)

### Output
y_ffn = MoE(y_combined)
logits = W_vocab · y_ffn

if p > 0.5:
    out_t = SymbolicRead(hash(token_t))
else:
    out_t ∼ softmax(logits / τ)

## Notation

- x_t ∈ R^d — input embedding
- S_t ∈ R^{d×d} — global state matrix
- K_win, V_win ∈ R^{W×d_head} — local window buffer
- h_t ∈ R^{d_pref} — prefrontal state
- σ, σ_m — learned noise parameters
- ε, ε_m ~ N(0,1) — Gaussian noise
- σ(·) — sigmoid function
- ⊙ — element-wise multiplication
- ⊗ — outer product
- MoE(·) — Mixture of Experts feed-forward
- SymbolicRead — exact lookup in Symbolic Store
