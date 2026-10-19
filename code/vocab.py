"""The toy vocabulary and embedding table shared by every chapter (d = 4).
Rows 0-2 are the running example X of Chapter 3; rows 3-5 are the second sentence of Section 3.9.
Do not change these numbers: every chapter's worked examples depend on them."""
import numpy as np
VOCAB = ["waste", "of", "time", "not", "very", "useful", "Sorry", ",", "this", "has", "been", "."]
E = np.array([
    [1,0,1,0],  # waste
    [0,1,0,0],  # of
    [0,0,1,1],  # time
    [0,1,0,1],  # not
    [1,0,0,0],  # very
    [0,0,1,0],  # useful
    [1,1,0,0],  # Sorry
    [0,0,0,1],  # ,
    [1,0,0,1],  # this
    [0,1,1,0],  # has
    [1,1,1,0],  # been
    [0,1,1,1],  # .
], dtype=float)
ID = {w: i for i, w in enumerate(VOCAB)}
assert len({tuple(r) for r in E}) == len(VOCAB), "embeddings must be distinct"
# the toy attention machines of Chapter 3 (one head, d_k = 4)
WQ = np.eye(4)
WK = np.array([[0,0,1,1],[1,0,0,0],[0,0,1,1],[0,1,0,0]], dtype=float)
WV = np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], dtype=float)
REPLY = ["Sorry", ",", "this", "has", "not", "been", "useful"]   # the target reply (Chapter 1)

def attention(X, WQ=WQ, WK=WK, WV=WV, causal=True):
    """Single-head causal attention, exactly as in Chapter 3. Returns (A, O)."""
    Q, K, V = X @ WQ, X @ WK, X @ WV
    S = Q @ K.T / np.sqrt(K.shape[1])
    if causal:
        S = np.where(np.triu(np.ones_like(S), 1).astype(bool), -np.inf, S)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return A, A @ V

if __name__ == "__main__":
    X = E[[ID["waste"], ID["of"], ID["time"]]]
    A, O = attention(X)
    print(np.round(A, 3)); print(np.round(O, 3))
