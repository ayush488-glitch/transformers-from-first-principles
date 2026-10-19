"""Chapter 10 figures: weight matrices drawn to scale (shapes read from the real model classes),
grouped-query sharing (Qwen3 layout + a toy with numbers), SwiGLU on the toy, and the tensor shapes of a
batch. Writes tex/fig_ch10_*.tex. Shapes come from models built on the meta device (no weights)."""
import sys, pathlib, math, re, warnings
warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import logging; logging.set_verbosity_error()
import figlib
from figlib import _grid, matmul_fig, vec_fig, stack_fig, write, _fmt
from vocab import E, ID, WQ, WK, WV, attention
TEX = HERE.parent / "tex"
_write = write
def write(path, *pics):
    _write(path, *["\\nTenFit{" + p + "}" for p in pics])
SCALE = 1024.0          # 1 cm per 1,024 numbers along each side

def comma(n): return f"{int(n):,}".replace(",", "{,}")
def short(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.0f}k" if n >= 1e3 else str(int(n)))

# ---------- real shapes, read from the reference implementations ----------
with torch.device("meta"):
    g = GPT2LMHeadModel(GPT2Config())
    q = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
        num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        tie_word_embeddings=True, rope_theta=1_000_000, max_position_embeddings=40960))
gl, ql = g.transformer.h[0], q.model.layers[0]
# GPT-2 uses Conv1D, whose weight is already (inputs x outputs)
GPT = [("W_{QKV}", tuple(gl.attn.c_attn.weight.shape)), ("W_O", tuple(gl.attn.c_proj.weight.shape)),
       ("W_{\\text{in}}", tuple(gl.mlp.c_fc.weight.shape)), ("W_{\\text{out}}", tuple(gl.mlp.c_proj.weight.shape))]
# nn.Linear stores (outputs x inputs); transpose to our x W convention (inputs x outputs)
def io(lin): o, i = lin.weight.shape; return (i, o)
QW = [("W_Q", io(ql.self_attn.q_proj)), ("W_K", io(ql.self_attn.k_proj)), ("W_V", io(ql.self_attn.v_proj)),
      ("W_O", io(ql.self_attn.o_proj)), ("W_{\\text{gate}}", io(ql.mlp.gate_proj)),
      ("W_{\\text{up}}", io(ql.mlp.up_proj)), ("W_{\\text{down}}", io(ql.mlp.down_proj))]
assert GPT == [("W_{QKV}", (768, 2304)), ("W_O", (768, 768)), ("W_{\\text{in}}", (768, 3072)), ("W_{\\text{out}}", (3072, 768))]
assert [s for _, s in QW] == [(1024, 2048), (1024, 1024), (1024, 1024), (2048, 1024), (1024, 3072), (1024, 3072), (3072, 1024)]

def scaled_row(mats, x0, y0, gap=0.5, fill="black!10"):
    """Draw matrices as rectangles to scale, top-aligned at y0. Returns (tikz, right edge)."""
    out, x = [], x0
    for name, (r, c) in mats:
        w, h = c / SCALE, r / SCALE
        out.append(f"\\draw[line width=0.8pt, fill={fill}] ({x:.3f},{y0:.3f}) rectangle ({x+w:.3f},{y0-h:.3f});")
        out.append(f"\\node[anchor=south, font=\\small] at ({x+w/2:.3f},{y0+0.02:.3f}) {{${name}$}};")
        out.append(f"\\node[anchor=north, font=\\footnotesize, align=center] at ({x+w/2:.3f},{y0-h-0.04:.3f}) "
                   f"{{{comma(r)}\\\\$\\times$\\,{comma(c)}\\\\[1pt]\\textbf{{{short(r*c)}}}}};")
        x += w + gap
    return "\n".join(out), x

# ---------- figure 1: one block's weights, to scale ----------
parts = []
parts.append("\\node[anchor=west, font=\\normalsize\\bfseries] at (0,0.75) {GPT-2 small, one block};")
t, xr = scaled_row(GPT, 0.0, 0.0, gap=0.55); parts.append(t)
# toy zoom on the right of the GPT-2 row
tz, wz, hz = _grid(WQ, xr + 1.6, 0.0, "WQ", cw=0.6, title=None)
parts.append(tz)
parts.append(f"\\node[anchor=south, font=\\small, align=center] at ({xr+1.6+wz/2:.3f},0.05) {{toy $W_Q$, $4\\times4$}};")
parts.append(f"\\node[anchor=north, font=\\footnotesize, align=center, text width=4.6cm] at ({xr+1.6+wz/2:.3f},{-hz-0.08:.3f}) "
             f"{{at the same scale it would be {4/SCALE*10:.2f}\\,mm wide; drawn here {0.6*SCALE/4:.0f} times larger}};")
y2 = -5.5
parts.append(f"\\node[anchor=west, font=\\normalsize\\bfseries] at (0,{y2+0.75:.2f}) {{Qwen3-0.6B, one block}};")
t, xr2 = scaled_row(QW, 0.0, y2, gap=0.55); parts.append(t)
# scale bar
parts.append(f"\\draw[line width=1pt] (0,{y2-4.75:.2f}) -- (1,{y2-4.75:.2f}); "
             f"\\draw (0,{y2-4.65:.2f}) -- (0,{y2-4.85:.2f}) (1,{y2-4.65:.2f}) -- (1,{y2-4.85:.2f}); "
             f"\\node[anchor=west, font=\\footnotesize] at (1.15,{y2-4.75:.2f}) {{this bar is 1{{,}}024 rows or columns long}};")
write(TEX / "fig_ch10_scale.tex", "\\begin{tikzpicture}\n" + "\n".join(parts) + "\n\\end{tikzpicture}")

# ---------- figure 2: grouped-query sharing ----------
Hq, Hkv, hd = q.config.num_attention_heads, q.config.num_key_value_heads, q.config.head_dim
assert (Hq, Hkv, hd) == (16, 8, 128)
sw = 0.95
p = []
p.append(f"\\node[anchor=south west, font=\\small] at (0,0.05) {{queries of one token: {Hq} heads $\\times$ {hd} numbers $=$ {comma(Hq*hd)}}};")
for i in range(Hq):
    p.append(f"\\draw[line width=0.7pt, fill=black!{8 if i % 2 == 0 else 20}] ({i*sw:.3f},0) rectangle ({(i+1)*sw:.3f},-0.7);")
    p.append(f"\\node[font=\\small] at ({(i+0.5)*sw:.3f},-0.35) {{$q_{{{i+1}}}$}};")
ky = -2.1
for k in range(Hkv):
    x0 = k * 2 * sw
    p.append(f"\\draw[line width=0.7pt, fill=black!14] ({x0:.3f},{ky:.2f}) rectangle ({x0+2*sw:.3f},{ky-0.7:.2f});")
    p.append(f"\\node[font=\\small] at ({x0+sw:.3f},{ky-0.35:.2f}) {{$k_{{{k+1}}}$}};")
    for j in (2 * k, 2 * k + 1):
        p.append(f"\\draw[-{{Stealth}}, line width=0.8pt] ({(j+0.5)*sw:.3f},-0.72) -- ({x0+sw+(-0.3 if j%2==0 else 0.3):.3f},{ky+0.02:.2f});")
p.append(f"\\node[anchor=north west, font=\\small] at (0,{ky-0.75:.2f}) {{keys of one token: {Hkv} heads $\\times$ {hd} numbers $=$ {comma(Hkv*hd)} (values are shared the same way)}};")
gqa_layout = "\\begin{tikzpicture}\n" + "\n".join(p) + "\n\\end{tikzpicture}"
# toy with numbers: 2 query heads of width 2 share one key head of width 2
toks = ["waste", "of", "time"]
X = E[[ID[t] for t in toks]]
Q = X @ WQ
Q1, Q2 = Q[:, :2], Q[:, 2:]
Kshared = (X @ WK)[:, 2:]          # one key head: the last two columns of the toy's keys
S1, S2 = Q1 @ Kshared.T, Q2 @ Kshared.T
assert S1[2, 0] == 0 and S2[2, 0] == 4
toy1 = matmul_fig(Q1, Kshared.T, 2, 0, "Q^{(1)}", "K_{\\text{shared}}^\\top", "S^{(1)}", rowsA=toks, colsB=toks, grey_upper=True)
toy2 = matmul_fig(Q2, Kshared.T, 2, 0, "Q^{(2)}", "K_{\\text{shared}}^\\top", "S^{(2)}", rowsA=toks, colsB=toks, grey_upper=True)
write(TEX / "fig_ch10_gqa.tex", gqa_layout)
write(TEX / "fig_ch10_gqa_toy.tex", toy1, toy2)


def matmul_w(A, B, i, j, nameA, nameB, nameC, nd=(3, 0, 3), rowsA=None, cwA=1.22, cwB=0.78, cwC=1.22):
    """Like figlib.matmul_fig, with wider cells for decimals (figlib.py is shared; not edited)."""
    A = np.atleast_2d(np.array(A, float)); B = np.array(B, float); C = A @ B
    gap = 0.75; parts = []
    ta, wa, ha = _grid(A, 0, 0, "A", nd=nd[0], rows=rowsA, title=f"${nameA}$", below=f"${A.shape[0]}\\times{A.shape[1]}$", hl_rows=(i,), cw=cwA)
    xb = wa + gap
    tb, wb, hb = _grid(B, xb, 0, "B", nd=nd[1], title=f"${nameB}$", below=f"${B.shape[0]}\\times{B.shape[1]}$", hl_cols=(j,), cw=cwB)
    xc = xb + wb + gap + (1.05 if rowsA else 0)
    tc, wc, hc = _grid(C, xc, 0, "C", nd=nd[2], rows=rowsA, title=f"${nameC}$", below=f"${C.shape[0]}\\times{C.shape[1]}$", hl_cells=((i, j),), cw=cwC)
    mid = -max(ha, hb, hc) / 2
    parts += [ta, tb, tc, f"\\node[font=\\Large] at ({wa+gap/2:.3f},{mid:.3f}) {{$\\times$}};",
              f"\\node[font=\\Large] at ({xb+wb+gap/2:.3f},{mid:.3f}) {{$=$}};"]
    a, b = A[i], B[:, j]
    terms = " + ".join(f"({_fmt(x, nd[0]).strip('$')})({_fmt(y, nd[1]).strip('$')})" for x, y in zip(a, b))
    parts.append(f"\\node[anchor=north west, font=\\small] at (0,{-max(ha,hb,hc)-0.75:.3f}) "
                 f"{{highlighted row $\\cdot$ highlighted column $= {terms} = \\mathbf{{{_fmt(float(a@b), nd[2]).strip('$')}}}$}};")
    return "\\begin{tikzpicture}\n" + "\n".join(parts) + "\n\\end{tikzpicture}"

# ---------- figure 3: SwiGLU on the toy ----------
A, O = attention(X)
x = np.round(O[2], 3)                       # time's attention output from Chapter 3, rounded as printed there
Wg = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [1, 0, 0, 0], [0, -1, 0, 1]], float)
Wu = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]], float)
Wd = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 1]], float)
g_ = x @ Wg; u_ = x @ Wu
silu = g_ / (1 + np.exp(-g_))
h = silu * u_; y = h @ Wd
tx = torch.tensor(x); ref = (torch.nn.functional.silu(tx @ torch.tensor(Wg)) * (tx @ torch.tensor(Wu))) @ torch.tensor(Wd)
assert np.allclose(ref.numpy(), y, atol=1e-12)
f1 = matmul_w(x, Wg, 0, 2, "\\bm{x}", "W_{\\text{gate}}", "\\bm{g}", rowsA=["time"])
f2 = matmul_w(x, Wu, 0, 2, "\\bm{x}", "W_{\\text{up}}", "\\bm{u}", rowsA=["time"])
# elementwise row: silu(g) (.) u = h
cw = 1.22
parts = []
xs = 0.0
for k, (vec, nm) in enumerate([(silu, "\\operatorname{SiLU}(\\bm{g})"), (u_, "\\bm{u}"), (h, "\\bm{h}")]):
    tq, w, hh = _grid(vec[None, :], xs, 0, nm, nd=3, cw=cw, title=f"${nm}$", hl_cols=(2,))
    parts.append(tq)
    if k < 2:
        parts.append(f"\\node[font=\\Large] at ({xs+w+0.4:.3f},{-hh/2:.3f}) {{${'\\odot' if k == 0 else '='}$}};")
    xs += w + 0.8
parts.append(f"\\node[anchor=north west, font=\\small] at (0,-1.25) {{entry by entry, e.g.\\ third entry: "
             f"$\\operatorname{{SiLU}}({_fmt(g_[2],3).strip('$')}) \\times {_fmt(u_[2],3).strip('$')} = "
             f"{_fmt(silu[2],3).strip('$')}\\times{_fmt(u_[2],3).strip('$')} = \\mathbf{{{_fmt(h[2],3).strip('$')}}}$; "
             f"a negative gate shrinks its entry}};")
f3 = "\\begin{tikzpicture}\n" + "\n".join(parts) + "\n\\end{tikzpicture}"
f4 = matmul_w(h, Wd, 0, 1, "\\bm{h}", "W_{\\text{down}}", "\\text{MLP output}", rowsA=["time"])
write(TEX / "fig_ch10_swiglu.tex", f1, f2, f3, f4)

# ---------- figure 4: tensor shapes of a batch through Qwen3 ----------
num = (TEX / "num_ch10.tex").read_text()
def shape(name):
    s = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*)\}\s*$", num, re.M).group(1)
    return [int(v.replace("{,}", "")) for v in s.split("\\times")]
stages = [("embeddings", "nTenShapeEmbed"), ("queries", "nTenShapeQ"), ("keys", "nTenShapeK"),
          ("after $W_O$", "nTenShapeAttnOut"), ("MLP gate", "nTenShapeGate"), ("after one block", "nTenShapeBlock"),
          ("scores for every word", "nTenShapeLogits")]
p = []
def cuboid(x0, y0, B, T, D, label):
    w = min(1.6 * D / SCALE, 5.0); hT = 0.42 * T; dx, dy = 0.3, 0.27
    out = []
    for b in reversed(range(B)):
        ox, oy = x0 + b * dx, y0 + b * dy
        out.append(f"\\draw[line width=0.7pt, fill={'black!8' if b else 'black!16'}] ({ox:.3f},{oy:.3f}) rectangle ({ox+w:.3f},{oy-hT:.3f});")
        for t in range(1, T):
            out.append(f"\\draw[black!55] ({ox:.3f},{oy-0.42*t:.3f}) -- ({ox+w:.3f},{oy-0.42*t:.3f});")
    if 1.6 * D / SCALE > 5.0:   # broken: too long to draw
        out.append(f"\\draw[white, line width=3pt] ({x0+w-0.55:.3f},{y0+0.35:.3f}) -- ({x0+w-0.75:.3f},{y0-hT-0.05:.3f});")
        out.append(f"\\draw[line width=0.7pt] ({x0+w-0.48:.3f},{y0+0.35:.3f}) -- ({x0+w-0.68:.3f},{y0-hT-0.05:.3f}) "
                   f"({x0+w-0.62:.3f},{y0+0.35:.3f}) -- ({x0+w-0.82:.3f},{y0-hT-0.05:.3f});")
    out.append(f"\\node[anchor=south west, font=\\normalsize] at ({x0:.3f},{y0+(B-1)*dy+0.08:.3f}) {{{label}}};")
    out.append(f"\\node[anchor=north west, font=\\small] at ({x0:.3f},{y0-hT-0.08:.3f}) "
               f"{{${B}\\times{T}\\times{comma(D)}$}};")
    return "\n".join(out), w
xpos, ypos = 0.0, 0.0
for i, (lab, key) in enumerate(stages):
    B, T, D = shape(key)
    if i == 4: xpos, ypos = 0.0, -2.9
    t, w = cuboid(xpos, ypos, B, T, D, lab)
    p.append(t); xpos += max(w, 2.0) + 1.0
shapes_fig = "\\begin{tikzpicture}\n" + "\n".join(p) + "\n\\end{tikzpicture}"
X2 = E[[ID[t] for t in ["not", "very", "useful"]]]
toy_batch = stack_fig([X, X2], titles=["\\texttt{waste of time}", "\\texttt{not very useful}"], rows=toks, name="X")
write(TEX / "fig_ch10_tensors.tex", shapes_fig)
write(TEX / "fig_ch10_toybatch.tex", toy_batch)

# a few macros for the text
M = ["% AUTO-GENERATED by code/ch10_figs.py"]
def m(k, v): M.append(f"\\newcommand{{\\{k}}}{{{v}}}")
m("nTenFigSOneTW", f"{int(S1[2,0])}"); m("nTenFigSTwoTW", f"{int(S2[2,0])}")
m("nTenFigX", "(" + ", ".join(f"{v:.3f}" for v in x) + ")")
m("nTenFigY", "(" + ", ".join(f"{v:.3f}" for v in y) + ")")
m("nTenFigGateClosed", f"{np.argmin(silu)+1}")
(TEX / "num_ch10figs.tex").write_text("\n".join(M) + "\n")
print("ok", S1[2], S2[2], np.round(g_, 3), np.round(silu, 3), np.round(u_, 3), np.round(h, 3), np.round(y, 3))
