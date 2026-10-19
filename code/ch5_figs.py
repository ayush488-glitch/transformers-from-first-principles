"""Chapter 5 pictures: one head's blend, each head's products, stitching the heads, the MLP, the residual
additions. Writes tex/fig_ch5_*.tex. Every number comes from code/toy_block.py (no norm, as in Chapter 5)."""
import math, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import figlib, toy_block as tb
from vocab import E, ID, WV as WV3
from ch4_figs import eq, mm, G

def write(name, *pics):
    figlib.write(ROOT / "tex" / f"fig_ch5_{name}.tex", *pics)

toks = ["waste", "of", "time"]
X = E[[ID[w] for w in toks]]
r = tb.block(X, norm=None)
A1, A2 = r["A"]

# A. one head, one blend: two different pairs of reports give the same average
va, vo = X[0] @ WV3, X[1] @ WV3
blend = 0.5 * va + 0.5 * vo
altA = va + vo - np.array([1, 0, 0, 0.]); altB = np.array([1, 0, 0, 0.])
assert np.allclose(0.5 * altA + 0.5 * altB, blend)
write("blend",
      eq(["$0.5\\ \\times$", G([va], 0, title="$\\bm v_{\\text{waste}}$"), "$+\\ 0.5\\ \\times$",
          G([vo], 0, title="$\\bm v_{\\text{of}}$"), "$=$", G([blend], 1, title="what \\texttt{time} receives", hl_cells=((0, 0), (0, 1), (0, 2), (0, 3))),]),
      eq(["$0.5\\ \\times$", G([altA], 0, title="another pair"), "$+\\ 0.5\\ \\times$",
          G([altB], 0, title="\\phantom{p}"), "$=$", G([0.5 * altA + 0.5 * altB], 1, title="the same blend", hl_cells=((0, 0), (0, 1), (0, 2), (0, 3)))]))

# B. head 1, fully worked: the query, then the scores against every key
q1 = X[2] @ tb.WQ1; K1 = X @ tb.WK1
write("headone",
      mm([X[2]], tb.WQ1, 0, 0, "\\bm x_{\\text{time}}", "W_Q^{(1)}", "\\bm q^{(1)}_{\\text{time}}",
         rowsA=["time"], lab_r="\\texttt{time}", lab_c="1 of $W_Q^{(1)}$"),
      mm(X, tb.WK1, 0, 0, "X", "W_K^{(1)}", "K^{(1)}", rowsA=toks, lab_c="1 of $W_K^{(1)}$"),
      mm([q1], K1.T, 0, 0, "\\bm q^{(1)}_{\\text{time}}", "K^{(1)\\top}", "\\text{raw scores}",
         rowsA=["time"], colsB=toks, lab_r="\\texttt{time}"))

# C. head 2, the reader's turn: the machines are shown, the answers are left blank
q2 = X[2] @ tb.WQ2; K2 = X @ tb.WK2
write("headtwo",
      eq([G([X[2]], 0, rows=["time"], title="$\\bm x_{\\text{time}}$"), "$\\times$",
          G(tb.WQ2, 0, title="$W_Q^{(2)}$"), "$=$", G([["?", "?"]], 0, title="$\\bm q^{(2)}_{\\text{time}}$")]),
      eq([G(X, 0, rows=toks, title="$X$"), "$\\times$", G(tb.WK2, 0, title="$W_K^{(2)}$"), "$=$",
          G([["?", "?"]] * 3, 0, rows=toks, title="$K^{(2)}$")]))
write("headtwoanswer",
      mm([q2], K2.T, 0, 1, "\\bm q^{(2)}_{\\text{time}}", "K^{(2)\\top}", "\\text{raw scores}",
         rowsA=["time"], colsB=toks, lab_r="\\texttt{time}"))

# D. stitching the heads: side by side, then the editor W_O
o1, o2 = A1[2] @ (X @ tb.WV1), A2[2] @ (X @ tb.WV2)
cat = np.concatenate([o1, o2])
assert np.allclose(cat @ tb.WO, r["o"][2])
write("concat",
      eq([G([o1], 3, title="$\\bm o^{(1)}_{\\text{time}}$", below="head 1's note"), "$\\big|$",
          G([o2], 3, title="$\\bm o^{(2)}_{\\text{time}}$", below="head 2's note"), "$\\longrightarrow$",
          G([cat], 3, title="placed side by side", hl_cells=((0, 0), (0, 1)), below="width $2+2=4$")]),
      mm([cat], tb.WO, 0, 2, "[\\bm o^{(1)}\\ \\bm o^{(2)}]", "W_O", "\\bm o_{\\text{time}}",
         nd=(3, 0, 3), rowsA=["time"], lab_r="\\texttt{time}", lab_c="3 of $W_O$"))

# E. the MLP: keys, thresholds, ReLU, values (only the three neurons that are switched on)
H1 = r["h1"]; W1s = tb.W1[:, :3]; b1s = tb.b1[:3]; W2s = tb.W2[:3, :]
pre = H1 @ W1s + b1s; act = np.maximum(pre, 0)
assert np.allclose(act, r["act"][:, :3]) and np.allclose(r["act"][:, 3:], 0)
neur = ["n1", "n2", "n3"]
write("mlpone",
      mm(H1, W1s, 2, 2, "H_1", "W_1\\ (\\text{neurons }1\\text{--}3)", "H_1W_1", nd=(3, 0, 3),
         rowsA=toks, colsB=neur, lab_c="\\texttt{n3}"))
write("mlptwo",
      eq([G(H1 @ W1s, 3, rows=toks, cols=neur, title="$H_1W_1$"), "$+$",
          G([b1s], 1, cols=neur, title="thresholds $\\bm b_1$"), "$=$",
          G(pre, 3, rows=toks, cols=neur, title="before ReLU")]),
      eq([G(pre, 3, rows=toks, cols=neur, title="before ReLU"), "$\\xrightarrow{\\ \\max(0,\\,\\cdot)\\ }$",
          G(act, 3, rows=toks, cols=neur, title="after ReLU: negatives become 0", hl_cells=((0, 0), (1, 1), (2, 2)))]))
write("mlpthree",
      mm(act, W2s, 2, 2, "\\text{after ReLU}", "W_2\\ (\\text{rows }1\\text{--}3)", "\\text{MLP output}",
         nd=(3, 0, 3), rowsA=toks, rowsB=neur))

# F. the residual stream: vector + vector = vector, twice, for time
x = X[2]; o = r["o"][2]; h1 = r["h1"][2]; mo = r["m"][2]; h2 = r["h2"][2]
assert np.allclose(x + o, h1) and np.allclose(h1 + mo, h2)
write("residual",
      eq([G([x], 0, title="$\\bm x_{\\text{time}}$", below="the token itself"), "$+$",
          G([o], 3, title="attention's note"), "$=$", G([h1], 3, title="$\\bm h_{1,\\text{time}}$", hl_cells=((0, 0), (0, 1), (0, 2), (0, 3)))]),
      eq([G([h1], 3, title="$\\bm h_{1,\\text{time}}$"), "$+$",
          G([mo], 3, title="the MLP's note"), "$=$", G([h2], 3, title="$\\bm h_{2,\\text{time}}$", hl_cells=((0, 0), (0, 1), (0, 2), (0, 3)))]))

# G. the batch: the same machines applied to every sentence at once (both customer replies)
X2 = E[[ID[w] for w in ["not", "very", "useful"]]]
H1b = [r["h1"], tb.block(X2, norm=None)["h1"]]
write("batch", figlib.stack_fig([np.round(h, 1) for h in H1b], titles=["\\texttt{waste of time}", "\\texttt{not very useful}"],
                                nd=1, rows=toks, name="H_1"))
print("ch5 figs ok")
