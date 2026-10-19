"""Pictures of every vector and matrix product in Chapter 3 (numbers from code/vocab.py)."""
import sys, pathlib, math
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vocab import E, ID, WQ, WK, WV, attention
from figlib import matmul_fig, matrix_fig, vec_fig, stack_fig, write
T = pathlib.Path(__file__).resolve().parent.parent / "tex"
toks = ["waste", "of", "time"]; toks2 = ["not", "very", "useful"]
X = E[[ID[t] for t in toks]]; X2 = E[[ID[t] for t in toks2]]
Q, K, V = X @ WQ, X @ WK, X @ WV
S = Q @ K.T
A, O = attention(X)
dims = ["1", "2", "3", "4"]
def stretch(pic, f):
    """Widen cells (x unit) without changing text size, so decimals fit."""
    return pic.replace("\\begin{tikzpicture}", f"\\begin{{tikzpicture}}[x={f}cm]", 1)

# the running sentence as a matrix
write(T / "fig_ch3_X.tex", matrix_fig(X, title="$X$", rows=toks, cols=dims, shape="$T\\times d=3\\times4$"))
# churn score: clues . importances
clues = np.array([[4, 2, 1.0]]); w = np.array([[0.5], [1.0], [-2.0]])
assert abs((clues @ w).item() - 2.0) < 1e-12
write(T / "fig_ch3_churn.tex", stretch(matmul_fig(clues, w, 0, 0, "\\text{clues}", "\\text{importance}", "\\text{score}",
      nd=(0, 1, 0), colsA=["days", "tickets", "buys"], rowsA=["customer"]), 1.45))
# 2-D dot product
write(T / "fig_ch3_dot2d.tex", matmul_fig(np.array([[3, 1.0]]), np.array([[1.0], [2.0]]), 0, 0,
      "\\bm{a}", "\\bm{b}^{\\top}", "\\bm{a}\\cdot\\bm{b}", rowsA=["a"], colsB=["b"]))
# one arrow through the key machine
xw = X[[0]]
write(T / "fig_ch3_xwk.tex", matmul_fig(xw, WK, 0, 2, "\\bm{x}_{\\text{waste}}", "W_K", "\\bm{k}_{\\text{waste}}",
      rowsA=["waste"], colsB=dims))
# Q, K, V for the whole sentence
write(T / "fig_ch3_qkv.tex",
      matmul_fig(X, WQ, 2, 3, "X", "W_Q", "Q", rowsA=toks),
      matmul_fig(X, WK, 0, 3, "X", "W_K", "K", rowsA=toks),
      matmul_fig(X, WV, 0, 1, "X", "W_V", "V", rowsA=toks))
# Step 1: the table of all meetings, future cells greyed
write(T / "fig_ch3_qkt.tex", stretch(matmul_fig(Q, K.T, 2, 0, "Q", "K^{\\top}", "S", rowsA=toks, colsB=toks, grey_upper=True), 1.25))
# Step 3: scaled and masked
Ssc = S / math.sqrt(4)
Sm = np.where(np.triu(np.ones((3, 3)), 1).astype(bool), -np.inf, Ssc)
grey = tuple((r, c) for r in range(3) for c in range(3) if c > r)
write(T / "fig_ch3_mask.tex", stretch(matrix_fig(Sm, title="$S/\\sqrt{d_k}$, masked", rows=toks, cols=toks,
      shape="$T\\times T$", grey_cells=grey, hl_rows=(2,)), 1.3))
# Step 4: softmax of the row of time, as three rows
z = Ssc[2]; e = np.exp(z); a = e / e.sum()
assert np.allclose(a, A[2])
rows = ["score", "$e^{\\text{score}}$", "share"]
tab = np.array([[f"${v:g}$" for v in z], [f"${v:.3f}$" for v in e], [f"$\\mathbf{{{v:.3f}}}$" for v in a]], dtype=object)
write(T / "fig_ch3_softmax.tex", stretch(matrix_fig(tab, title="row \\texttt{time}", rows=rows, cols=toks,
      shape=f"total of $e^{{\\text{{score}}}}$: ${e.sum():.3f}$; shares add to 1"), 1.6))
# Step 5: collect values
write(T / "fig_ch3_av.tex", stretch(matmul_fig(A, V, 2, 1, "A", "V", "O", nd=(3, 0, 3), rowsA=toks), 1.4))
# batch of two sentences
Q2, K2 = X2 @ WQ, X2 @ WK
S2 = Q2 @ K2.T
A2, O2 = attention(X2)
def stack_with_label(mats, name, shape, rows):
    pic = stack_fig(mats, titles=["sentence 1", "sentence 2"], rows=rows, name=name, batch_label=False)
    return pic.replace("\\end{tikzpicture}",
        f"\\node[anchor=north, font=\\small] at (1.2,-2.25) {{${name}$: ${shape}$}};\n\\end{{tikzpicture}}")
write(T / "fig_ch3_batch.tex",
      stack_with_label([X, X2], "X", "B\\times T\\times d = 2\\times3\\times4", toks),
      stack_with_label([S, S2], "S", "B\\times T\\times T = 2\\times3\\times3", toks),
      sep="\\qquad")
write(T / "fig_ch3_batch2.tex", stretch(matmul_fig(Q2, K2.T, 2, 1, "Q^{(2)}", "K^{(2)\\top}", "S^{(2)}", rowsA=toks2,
      colsB=toks2, grey_upper=True), 1.5))
write(T / "fig_ch3_atwo.tex", stretch(matrix_fig(A2, title="$A^{(2)}$: after scaling, masking and softmax", nd=3, rows=toks2, cols=toks2, shape="$T\\times T$; every row adds to 1", grey_cells=tuple((r, c) for r in range(3) for c in range(3) if c > r)), 1.5))
assert np.allclose(attention(X)[1], O)
print("ch3 figures written")
