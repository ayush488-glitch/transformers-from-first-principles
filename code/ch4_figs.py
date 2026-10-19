"""Chapter 4 pictures: positions as matrices, the rotation acting on a pair, q and k before/after RoPE,
and the score tables for both word orders. Writes tex/fig_ch4_*.tex. Numbers come from code/ch4_rope.py.

Also defines eq() and mm(), small layout helpers on top of figlib._grid, reused by code/ch5_figs.py:
  eq(items)  lays out matrices and operator symbols in one row, vertically centred.
  mm(A, B, i, j, ...)  A x B = C with row i, column j and cell (i, j) highlighted and the dot product
                       written out; like figlib.matmul_fig but with cell widths that fit decimals.
"""
import math, pathlib, sys, io, contextlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import figlib
from figlib import _grid, _fmt

def cw_for(M, nd):
    M = np.atleast_2d(np.array(M, dtype=object))
    longest = max(len(_fmt(v, nd).strip("$")) for v in M.ravel())
    return max(0.7, 0.17 * longest + 0.28)

def G(M, nd=0, **kw):
    """A grid item for eq()."""
    return dict(M=M, nd=nd, **kw)

def eq(items, gap=0.42, below_text=None):
    parts, x = [], 0.0
    placed = []
    for it in items:
        if isinstance(it, str):
            w = 0.5 if len(it) < 3 else 0.17 * len(it) + 0.3
            placed.append(("op", it, x, w)); x += w + gap
            continue
        M = np.atleast_2d(np.array(it["M"], dtype=object))
        cw = it.get("cw") or cw_for(M, it["nd"])
        if it.get("cols"): cw = max(cw, 0.16 * max(len(c) for c in it["cols"]) + 0.22)
        lab = 0.15 + 0.165 * max((len(r) for r in it.get("rows") or [""]), default=0) if it.get("rows") else 0.0
        x += lab
        h = M.shape[0] * CH
        placed.append(("grid", it, x, cw, h, M))
        x += M.shape[1] * cw + gap
    hmax = max(p[4] for p in placed if p[0] == "grid")
    for p in placed:
        if p[0] == "op":
            _, s, x0, w = p
            parts.append(f"\\node[font=\\Large] at ({x0 + w/2:.3f},0) {{{s if s.startswith('$') else s}}};")
        else:
            _, it, x0, cw, h, M = p
            kw = {k: v for k, v in it.items() if k not in ("M", "nd", "cw")}
            t, w, hh = _grid_ch(M, x0, h / 2, it["nd"], cw, **kw)
            parts.append(t)
    if below_text:
        parts.append(f"\\node[anchor=north west, font=\\small, align=left] at (0,{-hmax/2 - 0.75:.3f}) {{{below_text}}};")
    return "\\begin{tikzpicture}\n" + "\n".join(parts) + "\n\\end{tikzpicture}"

CH = 0.62   # fixed cell height (cm), whatever the cell width

def _grid_ch(M, x0, y0, nd, cw, rows=None, cols=None, hl_rows=(), hl_cols=(), hl_cells=(), grey_cells=(),
             title=None, below=None):
    """Same drawing rules as figlib._grid, but the cell height stays fixed when cells are widened for decimals."""
    M = np.atleast_2d(np.array(M, dtype=object)); R, C = M.shape; ch = CH; out = []
    for r in range(R):
        for c in range(C):
            x, y = x0 + c * cw, y0 - r * ch
            fill = "white"
            if r in hl_rows or c in hl_cols: fill = "black!12"
            if (r, c) in hl_cells: fill = "black!30"
            if (r, c) in grey_cells: fill = "black!6"
            v = M[r, c]
            if not isinstance(v, str) and abs(float(v)) < 5e-4: v = 0.0
            txt = _fmt(v, nd); col = "black!60" if (r, c) in grey_cells else "black"
            out.append(f"\\draw[black!45, fill={fill}] ({x:.3f},{y:.3f}) rectangle ({x+cw:.3f},{y-ch:.3f});")
            out.append(f"\\node[font=\\small, text={col}] at ({x+cw/2:.3f},{y-ch/2:.3f}) {{{txt}}};")
    for r in hl_rows:
        out.append(f"\\draw[line width=1.4pt] ({x0:.3f},{y0-r*ch:.3f}) rectangle ({x0+C*cw:.3f},{y0-(r+1)*ch:.3f});")
    for c in hl_cols:
        out.append(f"\\draw[line width=1.4pt] ({x0+c*cw:.3f},{y0:.3f}) rectangle ({x0+(c+1)*cw:.3f},{y0-R*ch:.3f});")
    for (r, c) in hl_cells:
        out.append(f"\\draw[line width=1.6pt] ({x0+c*cw:.3f},{y0-r*ch:.3f}) rectangle ({x0+(c+1)*cw:.3f},{y0-(r+1)*ch:.3f});")
    out.append(f"\\draw[line width=0.9pt] ({x0:.3f},{y0:.3f}) rectangle ({x0+C*cw:.3f},{y0-R*ch:.3f});")
    for r, lab in enumerate(rows or []):
        out.append(f"\\node[anchor=east, font=\\footnotesize\\ttfamily] at ({x0-0.08:.3f},{y0-r*ch-ch/2:.3f}) {{{lab}}};")
    for c, lab in enumerate(cols or []):
        out.append(f"\\node[anchor=south, font=\\footnotesize\\ttfamily] at ({x0+c*cw+cw/2:.3f},{y0+0.02:.3f}) {{{lab}}};")
    top = y0 + (0.42 if cols else 0.05)
    if title:
        out.append(f"\\node[anchor=south, font=\\normalsize] at ({x0+C*cw/2:.3f},{top:.3f}) {{{title}}};")
    if below:
        out.append(f"\\node[anchor=north, font=\\footnotesize, text=black!75] at ({x0+C*cw/2:.3f},{y0-R*ch-0.08:.3f}) {{{below}}};")
    return "\n".join(out), C * cw, R * ch

def mm(A, B, i, j, nameA, nameB, nameC, nd=(0, 0, 0), rowsA=None, colsB=None, rowsB=None, colsA=None,
       C=None, grey_upper=False, shapes=True, lab_r=None, lab_c=None):
    A = np.array(A, dtype=float); B = np.array(B, dtype=float)
    C = A @ B if C is None else np.array(C, dtype=float)
    ndA, ndB, ndC = nd
    def sh(M): return f"${M.shape[0]}\\times{M.shape[1]}$" if shapes else None
    gA = G(A, ndA, rows=rowsA, cols=colsA, title=f"${nameA}$", below=sh(A), hl_rows=(i,))
    gB = G(B, ndB, rows=rowsB, cols=colsB, title=f"${nameB}$", below=sh(B), hl_cols=(j,))
    gC = G(C, ndC, rows=rowsA, cols=colsB, title=f"${nameC}$", below=sh(C), hl_cells=((i, j),))
    if grey_upper:
        gC["grey_cells"] = tuple((r, c) for r in range(C.shape[0]) for c in range(C.shape[1]) if c > r)
    def f(v, n): return _fmt(float(v), n).strip("$")
    terms = " + ".join(f"({f(a, ndA)})({f(b, ndB)})" for a, b in zip(A[i], B[:, j]))
    lr = lab_r or (f"\\texttt{{{rowsA[i]}}}" if rowsA else f"{i+1}")
    lc = lab_c or (f"\\texttt{{{colsB[j]}}}" if colsB else f"{j+1}")
    text = f"row {lr} $\\cdot$ column {lc} $= {terms} = \\mathbf{{{f(A[i] @ B[:, j], ndC)}}}$"
    return eq([gA, "$\\times$", gB, "$=$", gC], below_text=text)

def write(name, *pics):
    figlib.write(ROOT / "tex" / f"fig_ch4_{name}.tex", *pics)

if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        import ch4_rope as R          # recomputes and rewrites tex/num_ch4.tex (deterministic)
    toks1, toks2 = ["waste", "of", "time"], ["of", "waste", "time"]
    pos = ["pos 0", "pos 1", "pos 2"]
    # A. sinusoid badges added to the arrows
    write("pe", eq([G(R.X1, 0, rows=toks1, title="$X$", below="token arrows"), "$+$",
                    G(R.P, 2, rows=pos, cols=["sin", "cos", "sin", "cos"], title="$\\mathrm{PE}$", below="position badges"),
                    "$=$", G(R.X1 + R.P, 2, rows=toks1, title="$X+\\mathrm{PE}$", below="what attention now sees")]))
    # B. the rotation acting on time's second pair at position 2
    th2 = R.thetas(4)[1]; ang = 2 * th2
    Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pair = R.Q1[2][2:4].reshape(2, 1)
    out = Rm @ pair
    assert np.allclose(out.ravel(), R.Qr1[2][2:4])
    write("rot", mm(Rm, pair, 0, 0, f"R(2\\theta_2)", "\\text{pair 2 of }\\bm q_{\\text{time}}", "\\text{turned pair}",
                    nd=(3, 0, 3), rowsB=["$x_3$", "$x_4$"], lab_r="1 of $R$", lab_c="$(x_3, x_4)$"))
    # C. queries and keys before and after turning
    rowsp = [f"{t} (p={p})" for p, t in enumerate(toks1)]
    write("qk",
          eq([G(R.Q1, 0, rows=rowsp, title="$Q$ before"), "$\\longrightarrow$",
              G(R.Qr1, 2, title="$Q$ after: row $p$ turned by $p\\theta$")]),
          eq([G(R.K1, 0, rows=rowsp, title="$K$ before"), "$\\longrightarrow$",
              G(R.Kr1, 2, title="$K$ after: row $p$ turned by $p\\theta$")]))
    # D. score tables (q . k before scaling, masked cells grey), both orders, without and with RoPE
    def scores(X, rope):
        Q, K = X @ R.WQ, X @ R.WK
        if rope:
            Q = np.array([R.rope(q, t) for t, q in enumerate(Q)]); K = np.array([R.rope(k, t) for t, k in enumerate(K)])
        return Q @ K.T
    grey = tuple((r, c) for r in range(3) for c in range(3) if c > r)
    pics = []
    for X, toks, lab in ((R.X1, toks1, "\\texttt{waste of time}"), (R.X2, toks2, "\\texttt{of waste time}")):
        S0, S1 = scores(X, False), scores(X, True)
        pics.append(eq([G(S0, 0, rows=toks, cols=toks, title=f"{lab}: no positions", grey_cells=grey, hl_rows=(2,)),
                        "\\quad", G(S1, 2, rows=toks, cols=toks, title=f"{lab}: with RoPE", grey_cells=grey, hl_rows=(2,))]))
    # the time rows without positions hold the same numbers in both orders (re-ordered); with RoPE they do not
    s_a, s_b = scores(R.X1, False)[2], scores(R.X2, False)[2]
    assert sorted(s_a) == sorted(s_b)
    assert not np.allclose(sorted(scores(R.X1, True)[2]), sorted(scores(R.X2, True)[2]))
    write("scores", *pics)
    print("ch4 figs ok")
