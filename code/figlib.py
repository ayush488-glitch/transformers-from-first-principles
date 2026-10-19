"""Draw matrices, vectors and their multiplication as TikZ, with real numbers.

Every worked example in the notes uses these so the reader SEES the shapes:
  matmul_fig(A, B, i, j, ...)   A x B = C, row i of A and column j of B highlighted, cell (i, j) of C
                                highlighted, and the dot product written out underneath.
  matrix_fig(M, ...)            one labelled matrix (or vector), optional highlighted rows/cols/cells.
  stack_fig([M1, M2, ...], ...) a batch: matrices stacked one behind the other, all numbers shown.
  vec_fig(v, ...)               a row vector as a strip of cells.
Each returns a complete tikzpicture string. write(path, *pictures) saves them for \\input.
"""
import numpy as np

CELL = 0.78          # cm per cell (width); height uses CELL*0.82
FONT = r"\small"

def _fmt(v, nd):
    if isinstance(v, str): return v
    if np.isneginf(v): return r"$-\infty$"
    if np.isposinf(v): return r"$\infty$"
    if nd == 0 and abs(v - round(v)) < 1e-12: return f"${int(round(v))}$"
    s = f"{v:.{nd}f}"
    return f"${s}$"

def _grid(M, x0, y0, name, nd=0, rows=None, cols=None, hl_rows=(), hl_cols=(), hl_cells=(),
          grey_cells=(), cw=CELL, title=None, below=None):
    """Draw matrix M with top-left corner at (x0, y0). Returns (tikz, width, height)."""
    M = np.atleast_2d(np.array(M, dtype=object))
    R, C = M.shape
    ch = cw * 0.82
    out = []
    for r in range(R):
        for c in range(C):
            x, y = x0 + c * cw, y0 - r * ch
            fill = "white"
            if r in hl_rows or c in hl_cols: fill = "black!12"
            if (r, c) in hl_cells: fill = "black!30"
            if (r, c) in grey_cells: fill = "black!6"
            txt = _fmt(M[r, c], nd)
            col = "black!60" if (r, c) in grey_cells else "black"
            out.append(f"\\draw[black!45, fill={fill}] ({x:.3f},{y:.3f}) rectangle ({x+cw:.3f},{y-ch:.3f});")
            out.append(f"\\node[font={FONT}, text={col}] at ({x+cw/2:.3f},{y-ch/2:.3f}) {{{txt}}};")
    # thick outline of highlighted row / column
    for r in hl_rows:
        out.append(f"\\draw[line width=1.4pt] ({x0:.3f},{y0-r*ch:.3f}) rectangle ({x0+C*cw:.3f},{y0-(r+1)*ch:.3f});")
    for c in hl_cols:
        out.append(f"\\draw[line width=1.4pt] ({x0+c*cw:.3f},{y0:.3f}) rectangle ({x0+(c+1)*cw:.3f},{y0-R*ch:.3f});")
    for (r, c) in hl_cells:
        out.append(f"\\draw[line width=1.6pt] ({x0+c*cw:.3f},{y0-r*ch:.3f}) rectangle ({x0+(c+1)*cw:.3f},{y0-(r+1)*ch:.3f});")
    out.append(f"\\draw[line width=0.9pt] ({x0:.3f},{y0:.3f}) rectangle ({x0+C*cw:.3f},{y0-R*ch:.3f});")
    if rows:
        for r, lab in enumerate(rows):
            out.append(f"\\node[anchor=east, font=\\footnotesize\\ttfamily] at ({x0-0.08:.3f},{y0-r*ch-ch/2:.3f}) {{{lab}}};")
    if cols:
        for c, lab in enumerate(cols):
            out.append(f"\\node[anchor=south, font=\\footnotesize\\ttfamily] at ({x0+c*cw+cw/2:.3f},{y0+0.02:.3f}) {{{lab}}};")
    top = y0 + (0.42 if cols else 0.05)
    if title:
        out.append(f"\\node[anchor=south, font=\\normalsize] at ({x0+C*cw/2:.3f},{top:.3f}) {{{title}}};")
    if below:
        out.append(f"\\node[anchor=north, font=\\footnotesize, text=black!75] at ({x0+C*cw/2:.3f},{y0-R*ch-0.08:.3f}) {{{below}}};")
    return "\n".join(out), C * cw, R * ch

def _wrap(body):
    return "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}"

def matrix_fig(M, title=None, nd=0, rows=None, cols=None, shape=None, **kw):
    t, w, h = _grid(M, 0, 0, "M", nd=nd, rows=rows, cols=cols, title=title, below=shape, **kw)
    return _wrap(t)

def vec_fig(v, title=None, nd=0, labels=None, shape=None):
    return matrix_fig(np.atleast_2d(v), title=title, nd=nd, cols=labels, shape=shape)

def matmul_fig(A, B, i=None, j=None, nameA="A", nameB="B", nameC=None, nd=(0, 0, 0),
               rowsA=None, colsB=None, rowsB=None, colsA=None, C=None, expand=True, grey_upper=False,
               shapes=True):
    """A x B = C with row i of A, column j of B and cell (i, j) of C highlighted."""
    A = np.array(A, dtype=float); B = np.array(B, dtype=float)
    if C is None: C = A @ B
    C = np.array(C, dtype=float)
    ndA, ndB, ndC = nd if isinstance(nd, tuple) else (nd, nd, nd)
    gap = 0.75
    parts = []
    hlA = dict(hl_rows=(i,) if i is not None else ())
    hlB = dict(hl_cols=(j,) if j is not None else ())
    hlC = dict(hl_cells=((i, j),) if (i is not None and j is not None) else ())
    if grey_upper:
        hlC["grey_cells"] = tuple((r, c) for r in range(C.shape[0]) for c in range(C.shape[1]) if c > r)
    sA = f"${A.shape[0]}\\times{A.shape[1]}$" if shapes else None
    sB = f"${B.shape[0]}\\times{B.shape[1]}$" if shapes else None
    sC = f"${C.shape[0]}\\times{C.shape[1]}$" if shapes else None
    ta, wa, ha = _grid(A, 0, 0, "A", nd=ndA, rows=rowsA, cols=colsA, title=f"${nameA}$", below=sA, **hlA)
    xb = wa + gap
    tb, wb, hb = _grid(B, xb, 0, "B", nd=ndB, rows=rowsB, cols=colsB, title=f"${nameB}$", below=sB, **hlB)
    xc = xb + wb + gap + (1.05 if rowsA else 0)
    tc, wc, hc = _grid(C, xc, 0, "C", nd=ndC, rows=rowsA, cols=colsB,
                       title=f"${nameC or nameA + nameB}$", below=sC, **hlC)
    mid = -max(ha, hb, hc) / 2
    parts += [ta, tb, tc,
              f"\\node[font=\\Large] at ({wa + gap/2:.3f},{mid:.3f}) {{$\\times$}};",
              f"\\node[font=\\Large] at ({xb + wb + gap/2:.3f},{mid:.3f}) {{$=$}};"]
    if expand and i is not None and j is not None:
        a, b = A[i], B[:, j]
        terms = " + ".join(f"({_fmt(x, ndA).strip('$')})({_fmt(y, ndB).strip('$')})" for x, y in zip(a, b))
        res = _fmt(float(a @ b), ndC).strip("$")
        lab_r = rowsA[i] if rowsA else f"row {i+1}"
        lab_c = colsB[j] if colsB else f"column {j+1}"
        ybot = -max(ha, hb, hc) - 0.75
        parts.append(f"\\node[anchor=north west, font=\\small, align=left] at (0,{ybot:.3f}) "
                     f"{{highlighted row (\\texttt{{{lab_r}}}) $\\cdot$ highlighted column (\\texttt{{{lab_c}}}) $= {terms} = \\mathbf{{{res}}}$}};")
    return _wrap("\n".join(parts))

def ch_(M): return CELL*0.82

def stack_fig(mats, titles=None, nd=0, rows=None, cols=None, name="X", dx=None, dy=None, batch_label=True):
    """A batch: matrices drawn one behind the other (front = first). All numbers visible on the front;
    the ones behind show their top row and right column so the reader sees there are B of them."""
    parts = []
    B = len(mats)
    dx = CELL * 1.0 if dx is None else dx
    dy = CELL * 0.82 if dy is None else dy
    for b in reversed(range(B)):
        M = np.atleast_2d(np.array(mats[b], dtype=float))
        x0, y0 = b * dx, b * dy
        t, w, h = _grid(M, x0, y0, name, nd=nd, rows=rows if b == 0 else None,
                        cols=cols if b == B - 1 else None,
                        title=None)
        # cover with white first so back sheets are partly hidden by front ones
        parts.append(f"\\fill[white] ({x0:.3f},{y0:.3f}) rectangle ({x0+w:.3f},{y0-h:.3f});")
        parts.append(t)
        if titles:
            parts.append(f"\\node[anchor=west, font=\\footnotesize] at ({x0+w+0.12:.3f},{(y0-h+ch_(M)/2) if b == 0 else (y0-ch_(M)/2):.3f}) {{$\\leftarrow$ {titles[b]}}};")
    if batch_label:
        M0 = np.atleast_2d(mats[0]); R, C = M0.shape
        w, h = C * CELL, R * CELL * 0.82
        parts.append(f"\\node[anchor=north, font=\\small] at ({w/2:.3f},{-h-0.1:.3f}) "
                     f"{{${name}$: $B\\times T\\times d = {B}\\times{R}\\times{C}$}};")
    return _wrap("\n".join(parts))

def write(path, *pics, sep="\\\\[10pt]"):
    with open(path, "w") as f:
        f.write("% AUTO-GENERATED by code/figlib.py helpers -- do not edit\n")
        f.write(("\n" + sep + "\n").join(pics) + "\n")

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from vocab import E, ID, WK, WQ
    X = E[[ID["waste"], ID["of"], ID["time"]]]
    toks = ["waste", "of", "time"]
    out = pathlib.Path(__file__).resolve().parent.parent / "tex" / "_figlib_demo.tex"
    Q, K = X @ WQ, X @ WK
    write(out,
          matmul_fig(X, WK, 0, 2, "X", "W_K", "K", rowsA=toks),
          matmul_fig(Q, K.T, 2, 0, "Q", "K^\\top", "S", rowsA=toks, colsB=toks, grey_upper=True),
          stack_fig([X, E[[3, 4, 5]]], titles=["sentence 1", "sentence 2"], rows=toks))
    print("wrote", out)
