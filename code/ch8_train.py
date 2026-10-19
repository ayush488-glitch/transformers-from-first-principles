"""Chapter 8: train a tiny decoder-only transformer on the lecture transcripts (character level).
Train on sessions 1-8, validate on session 9. Fixed seed, one CPU thread.
Writes tex/data_ch8/curve.dat, tex/data_ch8/overfit.dat and tex/data_ch8/train.json."""
import re, json, math, time, pathlib, sys
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tex" / "data_ch8"; OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)

def clean(path):
    lines = []
    for l in open(path, encoding="utf-8", errors="ignore"):
        l = l.strip()
        if re.fullmatch(r"[0-9:/ ]+", l) or l in ("Transcript", "Collapse", "Search transcript"): continue
        lines.append(l)
    t = " ".join(lines).lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9 ,.?']", " ", t)
    return re.sub(r"\s+", " ", t)

train_txt = " ".join(clean(ROOT / "data" / f"ai_week{i}.txt") for i in range(1, 9))
val_txt = clean(ROOT / "data" / "ai_week9.txt")
chars = sorted(set(train_txt) | set(val_txt)); stoi = {c: i for i, c in enumerate(chars)}
enc = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
tr, va = enc(train_txt), enc(val_txt)
V, D, L, H, T = len(chars), 64, 2, 2, 64

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, T_, _ = x.shape
        q, k, v = s.qkv(s.n1(x)).split(D, -1)
        sh = lambda z: z.view(B, T_, H, D // H).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        S = q @ k.transpose(-1, -2) / math.sqrt(D // H)
        S = S.masked_fill(torch.triu(torch.ones(T_, T_, dtype=torch.bool), 1), float("-inf"))
        o = (S.softmax(-1) @ v).transpose(1, 2).reshape(B, T_, D)
        x = x + s.proj(o)
        return x + s.mlp(s.n2(x))

class TinyGPT(nn.Module):
    def __init__(s, small_init=True):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block() for _ in range(L)]); s.nf = nn.LayerNorm(D)
        if small_init:
            for p in (s.tok.weight, s.pos.weight): nn.init.normal_(p, std=0.02)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
        return s.nf(s.blocks(x)) @ s.tok.weight.T          # tied unembedding, as in the toy

def batch(data, B, g):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])

@torch.no_grad()
def evaluate(m, data, n=20):
    g = torch.Generator().manual_seed(123); m.eval(); tot = 0.0
    for _ in range(n):
        x, y = batch(data, 32, g); tot += F.cross_entropy(m(x).view(-1, V), y.view(-1)).item()
    m.train(); return tot / n

@torch.no_grad()
def sample(m, prompt="waste of time", n=80):
    g = torch.Generator().manual_seed(7); idx = enc(prompt)[None]
    for _ in range(n):
        p = m(idx[:, -T:])[0, -1].softmax(-1)
        idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)[None]], 1)
    return "".join(chars[i] for i in idx[0].tolist())

def train(data, steps, every, tag, lr=3e-3):
    torch.manual_seed(229); m = TinyGPT(); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(1); rows, samples = [], {}
    for s in range(steps + 1):
        if s % every == 0:
            rows.append((s, evaluate(m, data), evaluate(m, va)))
            if s in (0, every * round(steps / 2 / every), steps): samples[s] = sample(m)
        if s == steps: break
        x, y = batch(data, 32, g)
        loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    (OUT / f"{tag}.dat").write_text("step train val\n" + "\n".join(f"{a} {b:.4f} {c:.4f}" for a, b, c in rows) + "\n")
    return m, rows, samples

if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(229); bad_init_loss = evaluate(TinyGPT(small_init=False), tr)   # PyTorch's default init
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    m, rows, samples = train(tr, steps, 100, "curve")
    t1 = time.time()
    small = tr[:4000]                       # a tiny slice: the model can memorise it
    _, orows, _ = train(small, 1200, 100, "overfit")
    info = dict(bad_init_loss=bad_init_loss, V=V, params=sum(p.numel() for p in m.parameters()), train_chars=len(tr), val_chars=len(va),
                small_chars=len(small), steps=steps, rows=rows, orows=orows, samples=samples,
                secs_main=round(t1 - t0), secs_total=round(time.time() - t0), chars="".join(chars))
    (OUT / "train.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("rows", "orows")}, indent=1)); print(rows[-1], orows[-1])
