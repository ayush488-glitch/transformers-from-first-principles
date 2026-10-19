"""Chapter 11 budget calculator and design scenarios. Writes tex/num_ch11.tex.
Independent of code/calc.py on purpose: at the end both are compared, and both are compared with
parameter counts of the real Hugging Face model classes built on the meta device (no weights downloaded).
Run code/ch11_ablate.py first (it caches tex/data_ch11/ablations.json)."""
import json, math, pathlib, sys, importlib.util
ROOT = pathlib.Path(__file__).resolve().parent.parent
M = []
def m(k, v): M.append(f"\\newcommand{{\\nEleven{k}}}{{{v}}}")
def big(x, unit=""):
    for s, n in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if x >= s: return f"{x/s:.3g} {n}{unit}"
    return f"{x:.3g}{unit}"
def comma(x): return f"{int(round(x)):,}".replace(",", "{,}")
def sci(x):
    mant, ex = f"{x:.2e}".split("e"); return f"{mant}\\times10^{{{int(ex)}}}"

# ---------- the calculator: a decoder-only transformer, counted piece by piece ----------
def count(V, d, L, H, Hkv, hd, F, swiglu=True, rms=True, qk_norm=False, tied=True, n_pos=0, bias=False):
    nrm = d if rms else 2 * d
    attn = d * H * hd + 2 * d * Hkv * hd + H * hd * d + (2 * hd if qk_norm else 0)
    attn += (H * hd + 2 * Hkv * hd + d) if bias else 0
    mlp = (3 if swiglu else 2) * d * F + ((F + d) if bias else 0)
    layer = attn + mlp + 2 * nrm
    emb = V * d + n_pos * d
    total = emb + L * layer + nrm + (0 if tied else V * d)
    return dict(total=total, embed=V * d, nonemb=L * layer + nrm, layer=layer, attn=attn, mlp=mlp)
def kv_bytes(L, Hkv, hd, ctx, bytes_per=2): return 2 * L * Hkv * hd * ctx * bytes_per
def train_flops(N, D): return 6 * N * D                         # Kaplan et al. (2020)

GPT2 = dict(V=50257, d=768, L=12, H=12, Hkv=12, hd=64, F=3072, swiglu=False, rms=False, tied=True, n_pos=1024, bias=True)
QWEN = dict(V=151936, d=1024, L=28, H=16, Hkv=8, hd=128, F=3072, swiglu=True, rms=True, qk_norm=True, tied=True)
g, q = count(**GPT2), count(**QWEN)

# ---------- verification 1: real model classes on the meta device ----------
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM
with torch.device("meta"):
    hg = GPT2LMHeadModel(GPT2Config())
    hq = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
          num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True))
def unique_numel(model):
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen: seen.add(id(p)); n += p.numel()
    return n
assert unique_numel(hg) == g["total"], (unique_numel(hg), g["total"])
assert unique_numel(hq) == q["total"], (unique_numel(hq), q["total"])
# ---------- verification 2: sibling calculator code/calc.py, if present ----------
calc_ok = "absent"
cp = ROOT / "code" / "calc.py"
if cp.exists():
    spec = importlib.util.spec_from_file_location("calc", cp); calc = importlib.util.module_from_spec(spec); spec.loader.exec_module(calc)
    assert calc.params(calc.GPT2_SMALL)["total"] == g["total"] and calc.params(calc.QWEN3_06B)["total"] == q["total"]
    assert calc.kv_bytes(calc.QWEN3_06B, 8192) == kv_bytes(28, 8, 128, 8192)
    calc_ok = "agrees"
m("GptParams", comma(g["total"])); m("QwenParams", comma(q["total"])); m("QwenNonEmb", comma(q["nonemb"]))
m("GptEmbShare", f"{100*(g['embed']+1024*768)/g['total']:.0f}"); m("QwenEmbShare", f"{100*q['embed']/q['total']:.0f}")
m("QwenLayer", comma(q["layer"])); m("QwenAttn", comma(q["attn"])); m("QwenMlp", comma(q["mlp"]))
m("QwenMlpShare", f"{100*q['mlp']/q['layer']:.0f}")
m("CalcCheck", calc_ok)

# ---------- scenario A: what one CPU thread did in our ablations ----------
ab = json.loads((ROOT / "tex" / "data_ch11" / "ablations.json").read_text())
base = [r for r in ab["runs"] if r["name"] == "baseline"]
tokens = ab["steps"] * ab["B"] * ab["T"]
Nab = base[0]["params"]
flops_s = train_flops(Nab, tokens) / base[0]["seconds"]
C_hour = flops_s * 3600
N_opt = math.sqrt(C_hour / (6 * 20)); D_opt = 20 * N_opt
m("AbParams", comma(Nab)); m("AbTokens", comma(tokens)); m("AbSeconds", f"{base[0]['seconds']:.0f}")
m("AbTokPerParam", f"{tokens/Nab:.0f}")
m("CpuFlops", sci(flops_s)); m("CpuHourFlops", sci(C_hour)); m("CpuNopt", comma(N_opt)); m("CpuDopt", big(D_opt))

# ---------- scenario B: 1,000 GPU-hours on a data-centre GPU ----------
PEAK, MFU, HOURS = 989e12, 0.40, 1000      # H100 SXM dense BF16 spec; utilisation is an assumption
C = HOURS * 3600 * PEAK * MFU
N_b = math.sqrt(C / 120); D_b = 20 * N_b
m("GpuC", sci(C)); m("GpuNopt", big(N_b)); m("GpuDopt", big(D_b)); m("GpuPeak", "989"); m("GpuMfu", "40")
# choose a shape near N_b: width, depth, heads, KV heads, d_ff, vocab
best = None
for d in (2048, 2560, 3072, 3584):
    for L in range(16, 49, 2):
        hd = 128; H = d // hd; Hkv = max(H // 4, 1); F = int(3 * d)
        c = count(V=151936, d=d, L=L, H=H, Hkv=Hkv, hd=hd, F=F, tied=False)
        ratio = d / L
        if 60 <= ratio <= 140:
            score = abs(math.log(c["total"] / N_b))
            if best is None or score < best[0]: best = (score, d, L, H, Hkv, F, c)
_, d, L, H, Hkv, F, c = best
m("BD", d); m("BL", L); m("BH", H); m("BKv", Hkv); m("BF", comma(F)); m("BTotal", big(c["total"]))
m("BEmbShare", f"{100*2*c['embed']/c['total']:.0f}"); m("BRatio", f"{d/L:.0f}")
m("BKvGB", f"{kv_bytes(L, Hkv, 128, 8192)/1e9:.2f}"); m("BKvGBmha", f"{kv_bytes(L, H, 128, 8192)/1e9:.2f}")

# ---------- scenario C: a phone with 4 GB for the model ----------
for nm, by in (("Sixteen", 2), ("Four", 0.5)):
    m(f"PhoneW{nm}", f"{q['total']*by/1e9:.2f}")
for ctx, nm in ((8192, "Eight"), (32768, "ThirtyTwo")):
    m(f"PhoneKv{nm}", f"{kv_bytes(28, 8, 128, ctx)/1e9:.2f}")
    m(f"PhoneKvMha{nm}", f"{kv_bytes(28, 16, 128, ctx)/1e9:.2f}")
    m(f"PhoneKvMqa{nm}", f"{kv_bytes(28, 1, 128, ctx)/1e9:.2f}")
m("PhoneKvPerTok", comma(kv_bytes(28, 8, 128, 1)))
for nm, h in (("Mha", 16), ("Gqa", 8), ("Mqa", 1)): m(f"KvTokRaw{nm}", kv_bytes(28, h, 128, 1))

# ---------- Chinchilla vs Qwen3 ----------
QWEN_TOKENS = 36e12   # Qwen3 technical report (2025): pre-training corpus of about 36 trillion tokens
m("QwenTokPerParam", comma(QWEN_TOKENS / q["total"])); m("QwenChinTokens", big(20 * q["total"]))
m("QwenTrainC", sci(train_flops(q["total"], QWEN_TOKENS)))

# ---------- ablation table macros ----------
names = {"baseline": "Base", "no position": "NoPos", "no residual": "NoRes", "no normalisation": "NoNorm",
         "1 head": "OneHead", "4 heads": "FourHeads", "4 heads, 2 KV (GQA)": "Gqa"}
for r in ab["runs"]:
    k = names[r["name"]] + ("SeedTwo" if r["seed"] == 1 else "")
    m("V" + k, f"{r['val']:.3f}"); m("P" + k, comma(r["params"]))
spread = abs(base[0]["val"] - base[1]["val"])
m("Spread", f"{spread:.3f}"); m("Uniform", f"{ab['uniform_loss']:.2f}"); m("Vocab", ab["vocab"])
m("Steps", comma(ab["steps"])); m("TrainChars", comma(ab["train_chars"])); m("ValChars", comma(ab["val_chars"]))
bars = [(names[r["name"]], r["val"]) for r in ab["runs"] if r["seed"] == 0]
(ROOT / "tex" / "data_ch11" / "ablations.dat").write_text("i label val\n" + "\n".join(
    f"{i} {n} {v:.4f}" for i, (n, v) in enumerate(bars)) + "\n")
# compare every variant with the noise band of the baseline
for r in ab["runs"]:
    if r["seed"] == 0 and r["name"] != "baseline":
        diff = r["val"] - base[0]["val"]
        m("D" + names[r["name"]], f"{diff:+.3f}")
# ---------- exercise answers ----------
C1 = 6e20; N1 = math.sqrt(C1 / 120); D1 = 20 * N1
assert abs(train_flops(N1, D1) - C1) / C1 < 1e-9
m("ExOneN", sci(N1)); m("ExOneD", sci(D1))
e2 = count(V=0, d=512, L=1, H=8, Hkv=2, hd=64, F=1536)
wq, wkv = 512 * 512, 2 * 512 * 2 * 64
assert e2["attn"] == 2 * wq + wkv
m("ExTwoWq", comma(wq)); m("ExTwoWkv", comma(wkv)); m("ExTwoAttn", comma(e2["attn"])); m("ExTwoMlp", comma(e2["mlp"]))
m("ExTwoBlock", comma(e2["attn"] + e2["mlp"])); m("ExTwoShare", f"{100*e2['attn']/(e2['attn']+e2['mlp']):.0f}")
kv3 = kv_bytes(28, 8, 128, 4096); left = 4e9 - 1e9 - kv3
m("ExThreeKv", f"{kv3/1e9:.2f}"); m("ExThreeLeft", f"{left/1e9:.2f}"); m("ExThreeN", big(left / 0.5))
(ROOT / "tex" / "num_ch11.tex").write_text("% AUTO-GENERATED by code/ch11_budget.py\n" + "\n".join(M) + "\n")
print("ok; GPT-2", g["total"], "Qwen3-0.6B", q["total"], "calc.py:", calc_ok, "| B:", d, L, H, Hkv, F, big(c["total"]))
