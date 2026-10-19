"""Chapter 8 toy: the skeleton model (embeddings -> one attention head -> tied unembedding),
trained on one exercise: after 'waste of time', predict 'Sorry'. Writes tex/num_ch8_toy part via ch8_numbers."""
import sys, math, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vocab import VOCAB, E, WQ, WK, WV, ID, attention
torch.manual_seed(0); torch.set_num_threads(1)
X = E[[ID["waste"], ID["of"], ID["time"]]]
TARGET = ID["Sorry"]

def forward_np(WQ_, WK_, WV_):
    A, O = attention(X, WQ_, WK_, WV_)
    logits = O[-1] @ E.T
    p = np.exp(logits - logits.max()); p /= p.sum()
    return A, O, logits, p

def run(lr, steps, opt="sgd"):
    Et = torch.tensor(E); Xt = torch.tensor(X)
    P = [torch.tensor(W.copy(), requires_grad=True) for W in (WQ, WK, WV)]
    o = torch.optim.SGD(P, lr=lr) if opt == "sgd" else torch.optim.Adam(P, lr=lr)
    mask = torch.triu(torch.ones(3, 3), 1).bool()
    hist = []
    for s in range(steps + 1):
        Q, K, V = Xt @ P[0], Xt @ P[1], Xt @ P[2]
        S = (Q @ K.T / 2.0).masked_fill(mask, -float("inf"))
        A = torch.softmax(S, -1); Ov = A @ V
        logits = Ov[-1] @ Et.T
        loss = -torch.log_softmax(logits, -1)[TARGET]
        q, k = Q[2].detach().numpy(), K[0].detach().numpy()
        ang = math.degrees(math.acos(np.clip(q @ k / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-12), -1, 1)))
        hist.append((s, loss.item(), A[2, 0].item(), torch.softmax(logits, -1)[TARGET].item(),
                     ang, float(np.linalg.norm(q))))
        if s < steps:
            o.zero_grad(); loss.backward(); o.step()
    return hist, [p.detach().numpy() for p in P]

if __name__ == "__main__":
    A, O, logits, p = forward_np(WQ, WK, WV)
    print("logits", np.round(logits, 3)); print("p", np.round(p, 3)); print("loss", -np.log(p[TARGET]))
    for lr in (0.01, 0.5, 5.0, 30.0):
        h, _ = run(lr, 40)
        print(lr, [tuple(round(v, 3) for v in h[i]) for i in (0, 10, 40)])
