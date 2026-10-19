"""Parameter, FLOP and KV-cache calculator from a config dict (Chapters 9, 10, 11).
Config keys: vocab, d, layers, n_heads, n_kv_heads, head_dim, d_ff, mlp ('gelu' | 'swiglu'),
norm ('layernorm' | 'rmsnorm'), qk_norm (bool), tied (bool), positions ('learned' | 'rope'), n_ctx,
bias (bool: GPT-2 style biases on every linear layer)."""

def params(c):
    d, L, H, Hkv, hd, F, V = c["d"], c["layers"], c["n_heads"], c["n_kv_heads"], c["head_dim"], c["d_ff"], c["vocab"]
    b = c.get("bias", False)
    norm = 2 * d if c["norm"] == "layernorm" else d                 # LayerNorm has a scale and a shift
    out = {}
    out["embed"] = V * d
    out["pos"] = c["n_ctx"] * d if c["positions"] == "learned" else 0
    q, kv, o = d * H * hd, d * Hkv * hd, H * hd * d
    attn = q + 2 * kv + o
    if b: attn += H * hd + 2 * Hkv * hd + d
    if c.get("qk_norm"): attn += 2 * hd
    mlp = (3 if c["mlp"] == "swiglu" else 2) * d * F
    if b: mlp += F + d
    out["per_layer_attn"], out["per_layer_mlp"], out["per_layer_norm"] = attn, mlp, 2 * norm
    out["per_layer"] = attn + mlp + 2 * norm
    out["layers"] = L * out["per_layer"]
    out["final_norm"] = norm
    out["unembed"] = 0 if c["tied"] else V * d
    out["total"] = out["embed"] + out["pos"] + out["layers"] + out["final_norm"] + out["unembed"]
    out["non_embedding"] = out["total"] - out["embed"] - out["pos"] - out["unembed"]
    return out

def flops_per_token(c, train=False):
    """Kaplan et al. (2020): about 2N per token forward, 6N for training (forward + backward),
    with N the non-embedding parameters; plus attention-score work 2 * layers * ctx * H * hd."""
    N = params(c)["non_embedding"]
    fwd = 2 * N + 2 * c["layers"] * c["n_ctx"] * c["n_heads"] * c["head_dim"]
    return 3 * fwd if train else fwd

def kv_bytes(c, tokens, bytes_per=2, batch=1):
    """2 (K and V) x layers x kv_heads x head_dim x tokens x bytes."""
    return 2 * c["layers"] * c["n_kv_heads"] * c["head_dim"] * tokens * bytes_per * batch

GPT2_SMALL = dict(vocab=50257, d=768, layers=12, n_heads=12, n_kv_heads=12, head_dim=64, d_ff=3072,
                  mlp="gelu", norm="layernorm", qk_norm=False, tied=True, positions="learned", n_ctx=1024, bias=True)
# Qwen3-0.6B as listed in the Qwen3 technical report (2025); re-check against config.json when downloadable.
QWEN3_06B = dict(vocab=151936, d=1024, layers=28, n_heads=16, n_kv_heads=8, head_dim=128, d_ff=3072,
                 mlp="swiglu", norm="rmsnorm", qk_norm=True, tied=True, positions="rope", n_ctx=40960, bias=False)
TOY = dict(vocab=12, d=4, layers=1, n_heads=1, n_kv_heads=1, head_dim=4, d_ff=16,
           mlp="gelu", norm="layernorm", qk_norm=False, tied=True, positions="rope", n_ctx=8, bias=False)
