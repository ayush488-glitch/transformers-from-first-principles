"""Pictures for Appendix A: the gradient at the end (p - y) and the gradient of V (A^T G) for the toy's
real loss after 'waste of time' -> 'Sorry', with a finite-difference check. Writes tex/fig_appA_*.tex."""
import sys, math, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figlib
from ch8_figs import set_cw, save, expansion
from vocab import VOCAB, E, WQ, WK, WV, ID, attention

def main():
    T = ROOT / "tex"
    X = E[[ID["waste"], ID["of"], ID["time"]]]
    s = ID["Sorry"]
    def loss_from(Wv):
        A, O = attention(X, WQ, WK, Wv)
        z = O[-1] @ E.T; return -(z[s] - z.max() - math.log(np.exp(z - z.max()).sum()))
    A, O = attention(X)
    z = O[-1] @ E.T; p = np.exp(z - z.max()); p /= p.sum()
    y = np.zeros(len(VOCAB)); y[s] = 1
    g = (p - y) @ E                       # dL/do_time, 1 x 4
    G = np.zeros_like(O); G[-1] = g       # dL/dO: only the row of 'time' feeds the loss
    dV = A.T @ G                          # dL/dV, 3 x 4
    dWv = X.T @ dV
    # finite-difference check of dL/dW_V for this real loss
    num = np.zeros_like(WV); h = 1e-6
    for i in range(4):
        for j in range(4):
            Wp = WV.copy(); Wp[i, j] += h; Wm = WV.copy(); Wm[i, j] -= h
            num[i, j] = (loss_from(Wp) - loss_from(Wm)) / (2 * h)
    assert np.abs(num - dWv).max() < 1e-7
    assert np.argmax(np.abs(dV).sum(1)) == 0   # caption: waste's row is the biggest
    labs = [w if w != "," else "{,}" for w in VOCAB]
    # 1. p, y and p - y
    rows = np.array([[f"${v:.3f}$" for v in p], [f"${int(v)}$" for v in y], [f"${v:.3f}$" for v in p - y]], dtype=object)
    set_cw(1.18)
    f1 = figlib.matrix_fig(rows, rows=["$\\bm{p}$", "$\\bm{y}$", "$\\bm{p}-\\bm{y}$"], cols=labs, hl_cols=(s,),
                           shape="three vectors of length 12 (one number per word); the highlighted column is \\texttt{Sorry}")
    figlib.write(T / "fig_appA_py.tex", f1)
    # 2. from the scores back to o_time: E^T (4 x 12) times (p - y)^T (12 x 1) = g^T (4 x 1)
    set_cw(1.15)
    jj = 0
    f2 = figlib.matmul_fig(E.T, (p - y)[:, None], jj, 0, "E^\\top", "(\\bm{p}-\\bm{y})^\\top",
                           "\\bm{g}^\\top", nd=(0, 3, 3), expand=False)
    figlib.write(T / "fig_appA_g.tex", f2)
    save("fig_appA_g_exp.tex", expansion(E.T[jj], p - y, 0, 3, 3, "row 1 of $E^\\top$", "$\\bm{p}-\\bm{y}$"))
    # 3. dV = A^T G
    set_cw(1.22)
    toks = ["waste", "of", "time"]
    i, j = 0, int(np.argmax(np.abs(dV[0])))
    f3 = figlib.matmul_fig(A.T, G, i, j, "A^\\top", "G", "\\partial L/\\partial V", nd=(3, 3, 3),
                           rowsA=toks, expand=False)
    figlib.write(T / "fig_appA_dv.tex", f3)
    save("fig_appA_dv_exp.tex", expansion(A.T[i], G[:, j], 3, 3, 3, "\\texttt{waste}'s column of $A$", f"column {j+1} of $G$"))
    print("appA figures written; max |dW_V - finite difference| =", float(np.abs(num - dWv).max()))

if __name__ == "__main__":
    main()
