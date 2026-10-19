"""Numbers for Chapter 5 (many heads, the MLP, the residual stream). Writes tex/num_ch5.tex."""
import math, pathlib, numpy as np
import toy_block as tb, ch5_random_blocks as rb
from vocab import E, VOCAB, ID, WK as WK3, WV as WV3, attention
OUT = pathlib.Path(__file__).resolve().parent.parent / "tex"
M = []
def m(k, v): M.append(f"\\newcommand{{\\{k}}}{{{v}}}")
def f(v, n=3): return f"{v:.{n}f}"
def vec(v, n=3): return "(" + ", ".join(f(x, n) for x in v) + ")"
def mat(A, n=0):
    rows = [" & ".join((str(int(round(x))) if n == 0 else f(x, n)) for x in r) for r in A]
    return "\\begin{bmatrix}" + " \\\\ ".join(rows) + "\\end{bmatrix}"

assert tb.check().startswith("numpy == loops")
X = E[[ID["waste"], ID["of"], ID["time"]]]
r = tb.block(X, norm=None)                         # Chapter 5 works without the norm (Chapter 6 adds it)
A1, A2 = r["A"]
# heads
for name, W in [("WQone", tb.WQ1), ("WKone", tb.WK1), ("WVone", tb.WV1), ("WQtwo", tb.WQ2), ("WKtwo", tb.WK2), ("WVtwo", tb.WV2)]:
    m("nFive" + name, mat(W))
for j, t in enumerate(["Waste", "Of", "Time"]):
    m(f"nFiveAone{t}", f(A1[2, j])); m(f"nFiveAtwo{t}", f(A2[2, j]))
    m(f"nFivePone{t}", f(100 * A1[2, j], 1)); m(f"nFivePtwo{t}", f(100 * A2[2, j], 1))
q1 = X[2] @ tb.WQ1; K1 = X @ tb.WK1; q2 = X[2] @ tb.WQ2; K2 = X @ tb.WK2
m("nFiveQone", vec(q1, 0)); m("nFiveKone", mat(K1)); m("nFiveQtwo", vec(q2, 0)); m("nFiveKtwo", mat(K2))
m("nFiveSone", vec(K1 @ q1, 0)); m("nFiveStwo", vec(K2 @ q2, 0))
m("nFiveSoneScaled", vec(K1 @ q1 / math.sqrt(2))); m("nFiveStwoScaled", vec(K2 @ q2 / math.sqrt(2)))
O1 = A1 @ (X @ tb.WV1); O2 = A2 @ (X @ tb.WV2)
m("nFiveVone", mat(X @ tb.WV1)); m("nFiveVtwo", mat(X @ tb.WV2))
m("nFiveOone", vec(O1[2])); m("nFiveOtwo", vec(O2[2])); m("nFiveO", vec(r["o"][2]))
m("nFiveHone", mat(r["h1"], 3)); m("nFiveHoneTime", vec(r["h1"][2]))
# one head cannot keep two answers apart: a 50/50 blend of two value arrows
A3, O3 = attention(X)
m("nFiveChThreeWaste", f(100 * A3[2, 0], 1)); m("nFiveChThreeOf", f(100 * A3[2, 1], 1))
blend = 0.5 * (X[0] @ WV3) + 0.5 * (X[1] @ WV3)
m("nFiveBlend", vec(blend, 1)); m("nFiveVwasteThree", vec(X[0] @ WV3, 0)); m("nFiveVofThree", vec(X[1] @ WV3, 0))
# a different pair of values with the same average: the blend cannot tell them apart
alt = (X[0] @ WV3 + X[1] @ WV3) - np.array([1, 0, 0, 0]), np.array([1, 0, 0, 0])
assert np.allclose(0.5 * alt[0] + 0.5 * alt[1], blend)
m("nFiveAltA", vec(alt[0], 0)); m("nFiveAltB", vec(alt[1], 0))
# MLP
for j, (k, b, v) in enumerate(tb.KEYS, 1):
    w = ["One", "Two", "Three"][j - 1]
    m(f"nFiveKey{w}", vec(k, 0)); m(f"nFiveBias{w}", f(b, 1)); m(f"nFiveVal{w}", vec(v, 0))
    pre = r["h1"] @ np.array(k, float) + b
    m(f"nFivePre{w}", vec(pre)); m(f"nFiveAct{w}", vec(np.maximum(pre, 0)))
m("nFiveNThreeTime", f(r["act"][2, 2])); m("nFiveNTwoOf", f(r["act"][1, 1], 1))
m("nFiveMtime", vec(r["m"][2])); m("nFiveHtwo", mat(r["h2"], 3)); m("nFiveHtwoTime", vec(r["h2"][2]))
m("nFiveDff", str(tb.D_FF))
lg = r["h2"] @ E.T; P = tb.softmax(lg)
preds = [VOCAB[i] for i in lg.argmax(1)]
assert preds == ["of", "time", "Sorry"], preds
m("nFivePredWaste", preds[0]); m("nFivePredOf", preds[1]); m("nFivePredTime", preds[2])
m("nFiveLogitSorry", f(lg[2, ID["Sorry"]], 2)); m("nFiveLogitOf", f(lg[2, ID["of"]], 2))
m("nFiveTopTwoTime", VOCAB[np.argsort(lg[2])[-2]]); m("nFiveTopTwoTimeLogit", f(np.sort(lg[2])[-2], 2))
# without the MLP, what does time predict?
lg0 = r["h1"] @ E.T; m("nFiveNoMlpPredTime", VOCAB[lg0[2].argmax()])
# without the residual connection, the toy loses track of which token is which
r_nores = tb.block(X, norm=None, residual=False)
m("nFiveNoResHtime", vec(r_nores["h2"][2]))
lg_nr = r_nores["h2"] @ E.T
m("nFiveNoResPreds", ", ".join("\\tok{" + VOCAB[i] + "}" for i in lg_nr.argmax(1)))
# parameter count of the toy block
n_attn = sum(W.size for h in tb.HEADS for W in h) + tb.WO.size
n_mlp = tb.W1.size + tb.b1.size + tb.W2.size
m("nFiveParamsAttn", str(n_attn)); m("nFiveParamsMlp", str(n_mlp))
# random deep stacks: does the token still know who it is?
Ls = list(range(0, 13))
rows = []
for res in (True, False):
    ids, Emb, st = rb.run(12, mode="pre", norm="rms", residual=res, seed=5)
    rows.append([rb.identity_accuracy(ids, Emb, st[L]) for L in Ls])
with open(OUT / "data_ch5" / "identity.dat", "w") as fh:
    fh.write("L res nores\n")
    for L, a, b in zip(Ls, rows[0], rows[1]): fh.write(f"{L} {a:.4f} {b:.4f}\n")
for L in (1, 2, 4, 12):
    w = {1: "One", 2: "Two", 4: "Four", 12: "Twelve"}[L]
    m(f"nFiveIdRes{w}", f(100 * rows[0][L], 1)); m(f"nFiveIdNoRes{w}", f(100 * rows[1][L], 1))
m("nFiveChance", f(100 / rb.V, 1)); m("nFiveRbD", str(rb.D)); m("nFiveRbV", str(rb.V)); m("nFiveRbH", str(rb.H))
m("nFiveRbDff", str(rb.DFF)); m("nFiveRbSeqs", str(rb.NSEQ)); m("nFiveRbT", str(rb.T))
# exercise numbers
m("nFiveAtwoOfRow", vec(A2[1, :2])); m("nFiveOtwoOfRow", vec(O2[1])); m("nFiveQtwoOf", vec(X[1] @ tb.WQ2, 0))
d = 768
m("nFiveGptAttn", f"{4*d*d:,}".replace(",", "{,}")); m("nFiveGptMlp", f"{8*d*d:,}".replace(",", "{,}"))
m("nFiveGptBlock", f"{12*d*d:,}".replace(",", "{,}")); m("nFiveGptD", str(d)); m("nFiveGptDff", str(4 * d))
m("nFiveHeadsDk", str(d // 12))
(OUT / "num_ch5.tex").write_text("% AUTO-GENERATED by code/ch5_numbers.py\n" + "\n".join(M) + "\n")
print("ch5 ok", preds, [round(x, 3) for x in rows[0][:5]], [round(x, 3) for x in rows[1][:5]])
