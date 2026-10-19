"""Pictures for Chapter 9: the KV cache growing row by row, the new query meeting the cached keys,
and the memory formula as a labelled product. Writes tex/fig_ch9_*.tex."""
import sys, math, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figlib
from ch8_figs import set_cw, save, expansion
from vocab import E, ID, WQ, WK, WV, REPLY, attention
from calc import kv_bytes, QWEN3_06B, GPT2_SMALL

def main():
    T = ROOT / "tex"
    seq = ["waste", "of", "time"] + REPLY
    X = E[[ID[w] for w in seq]]
    K, V = X @ WK, X @ WV
    lab = [w if w != "," else "{,}" for w in seq]
    # 1. the cache after the prompt (3 rows), after Sorry (4), after the comma (5): new row highlighted
    set_cw(0.78)
    pics = []
    for t in (3, 4, 5):
        hl = (t - 1,) if t > 3 else ()
        pics.append(figlib.matrix_fig(K[:t], title=f"$K_{{\\text{{cache}}}}$, $t={t}$", rows=lab[:t],
                                      hl_rows=hl, shape=f"${t}\\times4$"))
    figlib.write(T / "fig_ch9_growK.tex", *pics, sep="\\qquad")
    pics = []
    for t in (3, 4, 5):
        hl = (t - 1,) if t > 3 else ()
        pics.append(figlib.matrix_fig(V[:t], title=f"$V_{{\\text{{cache}}}}$, $t={t}$", rows=lab[:t],
                                      hl_rows=hl, shape=f"${t}\\times4$"))
    figlib.write(T / "fig_ch9_growV.tex", *pics, sep="\\qquad")
    # 2. the Sorry turn: q (1x4) times K_cache^T (4x4) = scores (1x4); then shares (1x4) times V_cache (4x4)
    t0 = 4
    q = (E[ID["Sorry"]] @ WQ)[None, :]
    Kc, Vc = K[:t0], V[:t0]
    sc = q @ Kc.T
    j = 0  # the key of 'waste'
    set_cw(0.98)
    f1 = figlib.matmul_fig(q, Kc.T, 0, j, "\\bm{q}_{\\text{Sorry}}", "K_{\\text{cache}}^\\top", "\\text{scores}",
                           nd=(0, 0, 0), rowsA=["Sorry"], colsB=lab[:t0], expand=False)
    figlib.write(T / "fig_ch9_qk.tex", f1)
    save("fig_ch9_qk_exp.tex", expansion(q[0], Kc.T[:, j], 0, 0, 0, "\\texttt{Sorry}'s query", "\\texttt{waste}'s key"))
    a = np.exp(sc / 2 - (sc / 2).max()); a /= a.sum()
    o = a @ Vc
    # the cached output equals the full recompute (as asserted in ch9_numbers.py)
    _, Ofull = attention(X[:t0]); assert np.allclose(o[0], Ofull[-1])
    jv = int(np.argmax(o[0]))
    set_cw(1.1)
    f2 = figlib.matmul_fig(a, Vc, 0, jv, "\\text{shares}", "V_{\\text{cache}}", "\\bm{o}_{\\text{Sorry}}",
                           nd=(3, 0, 3), rowsA=["Sorry"], colsA=lab[:t0], expand=False)
    figlib.write(T / "fig_ch9_av.tex", f2)
    save("fig_ch9_av_exp.tex", expansion(a[0], Vc[:, jv], 3, 0, 3, "the shares", f"column {jv+1} of $V_{{\\text{{cache}}}}$"))
    # 3. the memory formula as a labelled product, for Qwen3-0.6B
    c = QWEN3_06B
    L, h, dh = c["layers"], c["n_kv_heads"], c["head_dim"]
    per = kv_bytes(c, 1)
    assert per == 2 * L * h * dh * 2
    facs = [("2", "K and V"), (str(L), "layers"), (str(h), "key-value heads"), (str(dh), "numbers per head"),
            ("2", "bytes per number (fp16)")]
    parts = ["\\begin{tikzpicture}[font=\\normalsize]"]
    x = 0.0
    for i, (v, l) in enumerate(facs):
        w = 1.25 if len(v) < 3 else 1.45
        parts.append(f"\\node[draw, line width=0.9pt, rounded corners=2pt, minimum width={w}cm, minimum height=0.9cm, fill=black!7] (f{i}) at ({x + w/2:.2f},0) {{${v}$}};")
        parts.append(f"\\node[font=\\footnotesize, text width=1.9cm, align=center, anchor=north] at ({x + w/2:.2f},-0.6) {{{l}}};")
        x += w
        if i < len(facs) - 1:
            parts.append(f"\\node at ({x + 0.35:.2f},0) {{$\\times$}};"); x += 0.7
    parts.append(f"\\node at ({x + 0.35:.2f},0) {{$=$}};"); x += 0.7
    parts.append(f"\\node[draw, line width=1.6pt, rounded corners=2pt, minimum height=0.9cm, fill=black!25, anchor=west] at ({x:.2f},0) {{${per:,}$ bytes per token}};".replace(f"{per:,}", f"{per:,}".replace(",", "{,}")))
    parts.append("\\end{tikzpicture}")
    save("fig_ch9_formula.tex", "\n".join(parts))
    print("ch9 figures written")

if __name__ == "__main__":
    main()
