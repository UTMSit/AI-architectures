x_t = embed(token_t) + rope(pos)

S_t = decay_t ⊙ S_{t-1} + v_t ⊗ k_t
y_glob = S_t^T · r_t

y_loc = attn(Q_t, K_win, V_win)

g = σ(W_g·[x, y_glob, y_loc])
y_cortex = g⊙y_loc + (1-g)⊙y_glob

h_t = tanh(W_h·h_{t-1} + W_in·[y_cortex, diag(S_t)] + σ⊙ε)
for i in 1..K:
    hyp_m = W_hyp·(h_t + σ_m⊙ε_m)
    h_t += α·hyp_argmax(W_val·hyp_m)

y_pref = W_out·h_t
β = σ(W_β·[y_cortex, h_t])
p = σ(W_p·h_t)

y = MoE(y_cortex + β⊙y_pref)
out = READ(hash) if p>0.5 else softmax(W_vocab·y)

S_t = RWKV8-style состояние (∞ контекст)
attn = точное окно 8k-16k токенов
h_t = лобная доля (RNN+шум+обдумывание)
p = precision gate
READ = Symbolic Store (точные значения)
