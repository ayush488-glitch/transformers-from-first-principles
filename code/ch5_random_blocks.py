"""Random (untrained) transformer blocks for the depth experiments of Chapters 5 and 6.
Forward pass only, fixed seeds, d = 64, 4 heads, d_ff = 256, ReLU MLP, causal attention.
Weights ~ N(0, 1/fan_in), the usual starting scale; embeddings ~ N(0, 1), so each starts with RMS about 1."""
import numpy as np

D, H, DFF, V, T, NSEQ = 64, 4, 256, 100, 16, 32

def rms(x): return x / np.sqrt((x ** 2).mean(-1, keepdims=True))
def ln(x):
    mu = x.mean(-1, keepdims=True); return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True))
NORM = {"none": lambda x: x, "rms": rms, "layer": ln}

def make_blocks(L, seed=0):
    g = np.random.default_rng(seed); dk = D // H; bl = []
    for _ in range(L):
        bl.append(dict(WQ=g.normal(0, D ** -0.5, (H, D, dk)), WK=g.normal(0, D ** -0.5, (H, D, dk)),
                       WV=g.normal(0, D ** -0.5, (H, D, dk)), WO=g.normal(0, D ** -0.5, (D, D)),
                       W1=g.normal(0, D ** -0.5, (D, DFF)), W2=g.normal(0, DFF ** -0.5, (DFF, D))))
    return bl

def attn(x, b):
    Tn = x.shape[1]; mask = np.triu(np.ones((Tn, Tn)), 1).astype(bool); outs = []
    for h in range(H):
        q, k, v = x @ b["WQ"][h], x @ b["WK"][h], x @ b["WV"][h]
        s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]); s[:, mask] = -np.inf
        a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True); outs.append(a @ v)
    return np.concatenate(outs, -1) @ b["WO"]

def mlp(x, b): return np.maximum(x @ b["W1"], 0) @ b["W2"]

def run(L, mode="pre", norm="rms", residual=True, seed=0):
    """mode 'pre': x + F(norm(x)); 'post': norm(x + F(x)); 'none': x + F(x) (or F(x) without residual).
    Returns token ids, embedding table and the stream after every layer: list of (NSEQ, T, D) arrays."""
    g = np.random.default_rng(seed + 1000)
    Emb = g.normal(0, 1, (V, D)); ids = g.integers(0, V, (NSEQ, T)); x = Emb[ids]
    n = NORM[norm]; stream = [x]
    for b in make_blocks(L, seed):
        for F in (attn, mlp):
            if mode == "pre": u = F(n(x), b)
            else: u = F(x, b)
            x = (x + u) if residual else u
            if mode == "post": x = n(x)
        stream.append(x)
    return ids, Emb, stream

def identity_accuracy(ids, Emb, x):
    """Fraction of positions whose nearest embedding (by angle) is the token that started there."""
    xs = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12); es = Emb / np.linalg.norm(Emb, axis=1, keepdims=True)
    return float(((xs @ es.T).argmax(-1) == ids).mean())

def stream_rms(x): return float(np.sqrt((x ** 2).mean()))

def cos_to_input(x0, x):
    a = (x0 * x).sum(-1) / (np.linalg.norm(x0, axis=-1) * np.linalg.norm(x, axis=-1) + 1e-12); return float(a.mean())
