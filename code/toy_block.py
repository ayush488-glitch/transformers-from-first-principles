"""The toy transformer block of Chapters 5-7, built on code/vocab.py (d = 4, 12-token vocabulary).

  two heads, d_k = 2 each        head 1 looks for a judgement word, head 2 looks for a function word
  W_O = identity                 so you can see each head's output land in its own two coordinates
  MLP, d_ff = 16, ReLU           only neurons 1-3 are non-zero, so you can follow them by hand
  residual additions             h1 = x + attn(norm(x)),  h2 = h1 + mlp(norm(h1))
  norm                           None (Chapter 5), "layer" or "rms" (Chapter 6), pre-norm by default
  final norm + tied unembedding  logits = norm(h2) E^T (Chapter 7)

Every number is pencil-friendly. The function block_reference() is an independent, loop-based
implementation used to check block() to 1e-12.
"""
import math
import numpy as np
from vocab import E, VOCAB, ID

# ---------------- attention: two heads of width 2 ----------------
WQ1 = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)   # head 1 asks with coords 3,4
WK1 = np.array([[1, 1], [0, 0], [1, 1], [0, 0]], dtype=float)   # waste's key lines up with it
WV1 = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=float)   # hands over coords 1,3
WQ2 = np.array([[0, 0], [0, 0], [1, 0], [0, 0]], dtype=float)   # head 2 asks with coord 3 only
WK2 = np.array([[0, 0], [2, 0], [0, 0], [0, 0]], dtype=float)   # only "of" answers
WV2 = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=float)   # hands over coords 2,4
HEADS = [(WQ1, WK1, WV1), (WQ2, WK2, WV2)]
WO = np.eye(4)

# ---------------- MLP: 16 neurons, 3 in use ----------------
D_FF = 16
W1 = np.zeros((4, D_FF)); b1 = np.zeros(D_FF); W2 = np.zeros((D_FF, 4))
KEYS = [((1, 0, -1, 0), 0.0, (-4, -1, -2, -1)),    # neuron 1: fires on "waste", writes towards "of"
        ((0, 1, 0, 0), -1.0, (-1, -4, 0, 2)),       # neuron 2: fires on "of", writes towards "time"
        ((-1, 0, 1, 1), -0.5, (0, 0, -4, -1))]      # neuron 3: fires on "time", writes towards "Sorry"
for j, (k, b, v) in enumerate(KEYS):
    W1[:, j] = k; b1[j] = b; W2[j, :] = v

def layernorm(x):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var)

def rmsnorm(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True))

NORMS = {None: lambda x: x, "layer": layernorm, "rms": rmsnorm}

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def head(X, WQ, WK, WV):
    T = len(X)
    S = (X @ WQ) @ (X @ WK).T / math.sqrt(WQ.shape[1])
    S = np.where(np.triu(np.ones((T, T)), 1).astype(bool), -np.inf, S)
    A = softmax(S)
    return A, A @ (X @ WV)

def mha(X):
    outs, As = [], []
    for WQ, WK, WV in HEADS:
        A, O = head(X, WQ, WK, WV); As.append(A); outs.append(O)
    return np.concatenate(outs, 1) @ WO, As

def mlp(X):
    pre = X @ W1 + b1
    act = np.maximum(pre, 0)
    return act @ W2, act

def block(X, norm="rms", residual=True, prenorm=True):
    """One block. Returns a dict with every intermediate result."""
    n = NORMS[norm]
    a_in = n(X) if prenorm else X
    o, As = mha(a_in)
    h1 = X + o if residual else o
    if not prenorm: h1 = n(h1)
    m_in = n(h1) if prenorm else h1
    m, act = mlp(m_in)
    h2 = h1 + m if residual else m
    if not prenorm: h2 = n(h2)
    return dict(attn_in=a_in, A=As, o=o, h1=h1, mlp_in=m_in, act=act, m=m, h2=h2)

def logits(h, norm="rms"):
    return NORMS[norm](h) @ E.T

def forward(words, norm="rms"):
    X = E[[ID[w] for w in words]]
    r = block(X, norm=norm)
    r["logits"] = logits(r["h2"], norm=norm)
    r["probs"] = softmax(r["logits"])
    return r

# ---------------- independent check ----------------
def block_reference(words, norm="rms"):
    """Plain loops, no matrix library calls; must agree with block()."""
    X = [list(E[ID[w]]) for w in words]
    def nrm(v):
        if norm is None: return v[:]
        if norm == "rms":
            s = math.sqrt(sum(t * t for t in v) / len(v)); return [t / s for t in v]
        mu = sum(v) / len(v); s = math.sqrt(sum((t - mu) ** 2 for t in v) / len(v)); return [(t - mu) / s for t in v]
    def vm(v, M): return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(len(M[0]))]
    xin = [nrm(x) for x in X]
    h1 = []
    for t in range(len(X)):
        cat = []
        for WQ, WK, WV in HEADS:
            q = vm(xin[t], WQ.tolist())
            sc = [sum(a * b for a, b in zip(q, vm(xin[j], WK.tolist()))) / math.sqrt(2) for j in range(t + 1)]
            mx = max(sc); ex = [math.exp(s - mx) for s in sc]; z = sum(ex)
            vals = [vm(xin[j], WV.tolist()) for j in range(t + 1)]
            cat += [sum(ex[j] / z * vals[j][c] for j in range(t + 1)) for c in range(2)]
        o = vm(cat, WO.tolist())
        h1.append([X[t][c] + o[c] for c in range(4)])
    h2 = []
    for t in range(len(X)):
        u = nrm(h1[t])
        act = [max(sum(u[i] * W1[i][j] for i in range(4)) + b1[j], 0.0) for j in range(D_FF)]
        m = [sum(act[j] * W2[j][c] for j in range(D_FF)) for c in range(4)]
        h2.append([h1[t][c] + m[c] for c in range(4)])
    lg = [[sum(a * b for a, b in zip(nrm(h), E[k])) for k in range(len(VOCAB))] for h in h2]
    return np.array(h1), np.array(h2), np.array(lg)

def check():
    for norm in (None, "layer", "rms"):
        for words in (["waste", "of", "time"], ["not", "very", "useful"]):
            r = forward(words, norm=norm)
            h1, h2, lg = block_reference(words, norm=norm)
            assert np.allclose(r["h1"], h1, atol=1e-12) and np.allclose(r["h2"], h2, atol=1e-12)
            assert np.allclose(r["logits"], lg, atol=1e-12)
    try:
        import torch
        X = torch.tensor(E[[0, 1, 2]])
        def tnorm(x): return x / torch.sqrt((x ** 2).mean(-1, keepdim=True))
        xin = tnorm(X); outs = []
        for WQ, WK, WV in HEADS:
            q, k, v = xin @ torch.tensor(WQ), xin @ torch.tensor(WK), xin @ torch.tensor(WV)
            outs.append(torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0])
        h1 = X + torch.cat(outs, 1) @ torch.tensor(WO)
        act = torch.relu(tnorm(h1) @ torch.tensor(W1) + torch.tensor(b1))
        h2 = h1 + act @ torch.tensor(W2)
        assert np.allclose(h2.numpy(), forward(["waste", "of", "time"])["h2"], atol=1e-10)
        return "numpy == loops == torch"
    except ImportError:
        return "numpy == loops (torch not installed)"

if __name__ == "__main__":
    print(check())
    for norm in (None, "rms"):
        r = forward(["waste", "of", "time"], norm=norm)
        print(norm, [VOCAB[i] for i in r["logits"].argmax(1)], np.round(r["probs"].max(1), 3))
