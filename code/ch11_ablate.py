"""Chapter 11 ablations: tiny decoder-only transformers trained on the author's transcripts.
Character level. Train on ai_week1-8, validate on ai_week9. Every run uses the same number of steps,
batches and seeds, so the only difference is the architectural change being tested.
Results are cached in tex/data_ch11/ablations.json; rerun with --retrain (about 5-6 CPU minutes)."""
import json, math, pathlib, re, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tex" / "data_ch11" / "ablations.json"

def clean(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    s = re.sub(r"\n[0-9:]+\n", " ", s).lower()
    s = re.sub(r"[^a-z0-9 .,?'\n]", " ", s)
    return re.sub(r"\s+", " ", s)
train_txt = " ".join(clean(ROOT / "data" / f"ai_week{i}.txt") for i in range(1, 9))
val_txt = clean(ROOT / "data" / "ai_week9.txt")
chars = sorted(set(train_txt) | set(val_txt)); stoi = {c: i for i, c in enumerate(chars)}
enc = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
TR, VA = enc(train_txt), enc(val_txt)
V, T, D, L, B, STEPS, LR = len(chars), 64, 64, 2, 32, 1500, 3e-3

class Block(nn.Module):
    def __init__(s, heads, kv_heads, resid, norm):
        super().__init__()
        s.h, s.kv, s.hd, s.resid = heads, kv_heads, D // heads, resid
        s.q = nn.Linear(D, D, bias=False); s.k = nn.Linear(D, kv_heads * s.hd, bias=False)
        s.v = nn.Linear(D, kv_heads * s.hd, bias=False); s.o = nn.Linear(D, D, bias=False)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        s.n1 = nn.LayerNorm(D) if norm else nn.Identity(); s.n2 = nn.LayerNorm(D) if norm else nn.Identity()
    def attn(s, x):
        b, t, _ = x.shape
        q = s.q(x).view(b, t, s.h, s.hd).transpose(1, 2)
        k = s.k(x).view(b, t, s.kv, s.hd).transpose(1, 2).repeat_interleave(s.h // s.kv, 1)
        v = s.v(x).view(b, t, s.kv, s.hd).transpose(1, 2).repeat_interleave(s.h // s.kv, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(b, t, D))
    def forward(s, x):
        if s.resid:
            x = x + s.attn(s.n1(x)); return x + s.mlp(s.n2(x))
        x = s.attn(s.n1(x)); return s.mlp(s.n2(x))

class Model(nn.Module):
    def __init__(s, heads=2, kv_heads=None, pos=True, resid=True, norm=True):
        super().__init__()
        s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(T, D) if pos else None
        s.blocks = nn.ModuleList(Block(heads, kv_heads or heads, resid, norm) for _ in range(L))
        s.nf = nn.LayerNorm(D) if norm else nn.Identity(); s.head = nn.Linear(D, V, bias=False)
    def forward(s, idx):
        x = s.emb(idx)
        if s.pos is not None: x = x + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.head(s.nf(x))

def batch(data, g):
    i = torch.randint(0, len(data) - T - 1, (B,), generator=g)
    return torch.stack([data[j:j + T] for j in i]), torch.stack([data[j + 1:j + T + 1] for j in i])

def run(name, seed=0, **kw):
    torch.manual_seed(seed); m = Model(**kw); opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1000 + seed); t0 = time.time(); curve = []
    for step in range(STEPS):
        x, y = batch(TR, g); loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == STEPS - 1: curve.append((step, round(loss.item(), 4)))
    gv = torch.Generator().manual_seed(7); m.eval(); vl = []
    with torch.no_grad():
        for _ in range(40):
            x, y = batch(VA, gv); vl.append(F.cross_entropy(m(x).view(-1, V), y.view(-1)).item())
    r = dict(name=name, seed=seed, params=sum(p.numel() for p in m.parameters()), val=sum(vl) / len(vl),
             train_last=curve[-1][1], seconds=round(time.time() - t0, 1), curve=curve)
    print(r["name"], r["seed"], r["params"], round(r["val"], 4), r["seconds"], "s", flush=True); return r

RUNS = [("baseline", 0, {}), ("baseline", 1, {}), ("no position", 0, dict(pos=False)),
        ("no residual", 0, dict(resid=False)), ("no normalisation", 0, dict(norm=False)),
        ("1 head", 0, dict(heads=1)), ("4 heads", 0, dict(heads=4)), ("4 heads, 2 KV (GQA)", 0, dict(heads=4, kv_heads=2))]

if __name__ == "__main__":
    if "--retrain" in sys.argv or not OUT.exists():
        res = dict(vocab=V, T=T, D=D, L=L, B=B, steps=STEPS, lr=LR, train_chars=len(TR), val_chars=len(VA),
                   uniform_loss=math.log(V), torch=torch.__version__, runs=[run(n, s, **k) for n, s, k in RUNS])
        OUT.write_text(json.dumps(res, indent=1))
    print("cached results in", OUT)
