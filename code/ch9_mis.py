"""Write and verify design/mis_ch9.json and design/mis_ch10.json (quotes checked verbatim)."""
import re, json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\n[0-9:]+\n", " ", s)).lower()
SRC = {w: norm(open(R / "data" / f"ai_week{w}.txt").read()) for w in range(1, 10)}
CH9 = [
 dict(id="9.1", wrong="Writing each new token means running the whole sentence through the model again.",
      tempting="The model reads its whole context every time it guesses.", fixed_in="9.2, 9.3", week=None, quote=None),
 dict(id="9.2", wrong="The cache is a shortcut that changes the answer slightly.",
      tempting="Caches in everyday software can go stale.", fixed_in="9.3 (difference measured as exactly 0)", week=None, quote=None),
 dict(id="9.3", wrong="The cache must also store the queries.",
      tempting="Queries, keys and values are made together.", fixed_in="9.3", week=None, quote=None),
 dict(id="9.4", wrong="What a model costs to run is set by its weights alone; context length is free.",
      tempting="Model size is the number everyone quotes.", fixed_in="9.4, 9.5", week=9,
      quote="should it run in gpu, cpu, mobile phone?"),
 dict(id="9.5", wrong="Fewer key-value heads means the model looks at fewer things.",
      tempting="Heads were introduced as different ways of looking.", fixed_in="9.6", week=None, quote=None),
]
CH10 = [
 dict(id="10.1", wrong="A bigger width, more heads or more layers is always better.",
      tempting="Bigger models score better on leaderboards.", fixed_in="10.4, 10.6", week=9,
      quote="it is not that the more the number is the better it is"),
 dict(id="10.2", wrong="More attention heads always help.", tempting="Each head is one more way of looking.",
      fixed_in="10.3", week=9, quote="it is not always it helps"),
 dict(id="10.3", wrong="The embedding table has exactly one row per tokenizer entry.",
      tempting="Each token needs one row, so the counts should match.", fixed_in="10.2", week=9,
      quote="they have around 151,669 address"),
 dict(id="10.4", wrong="Most of a transformer's parameters sit in attention.",
      tempting="Attention is the famous part.", fixed_in="10.4", week=None, quote=None),
 dict(id="10.5", wrong="A model with billions of weights will overfit.",
      tempting="In classic machine learning more parameters than examples meant overfitting.", fixed_in="10.5", week=9,
      quote="billions of weights means too much"),
 dict(id="10.6", wrong="Number of heads times head width must equal the model width.",
      tempting="It does in the original transformer and in GPT-2.", fixed_in="10.3", week=None, quote=None),
]
for name, L in (("mis_ch9.json", CH9), ("mis_ch10.json", CH10)):
    for e in L:
        if e["quote"]: assert e["quote"] in SRC[e["week"]], e["id"]
    (R / "design" / name).write_text(json.dumps(L, indent=1))
print("ok")
