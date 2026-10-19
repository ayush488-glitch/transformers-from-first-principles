"""Misconceptions for Chapter 11, with transcript quotes checked verbatim."""
import json, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\n[0-9:]+\n", " ", s)).lower()
SRC = {w: norm(open(ROOT / "data" / f"ai_week{w}.txt").read()) for w in range(1, 10)}
M = [
 dict(id="11.1", wrong="More attention heads means more capacity and a better model.",
      tempting="More heads sounds like more ways of looking, and more of anything sounds better.",
      fixed_in="11.4", week=9, quote="what does it means that you have number more number of attention heads?"),
 dict(id="11.2", wrong="Making the model bigger always helps, whatever the data.",
      tempting="The famous models are huge.", fixed_in="11.2", week=None, quote=None),
 dict(id="11.3", wrong="Compute depends only on the number of parameters.",
      tempting="Parameter count is the number everyone quotes.", fixed_in="11.2", week=None, quote=None),
 dict(id="11.4", wrong="The parameter count tells you how much memory the model needs to run.",
      tempting="Weights are the obvious thing to store.", fixed_in="11.5", week=None, quote=None),
 dict(id="11.5", wrong="The vocabulary is free; only the layers cost anything.",
      tempting="The embedding table is a lookup, not a computation.", fixed_in="11.3", week=None, quote=None),
 dict(id="11.6", wrong="If a change lowers the loss in one small run, the change is better.",
      tempting="A lower number looks like a win.", fixed_in="11.6", week=None, quote=None),
 dict(id="11.7", wrong="Width is free: wider only means more ways of looking at a token.",
      tempting="Wider vectors hold more information.", fixed_in="11.3", week=9,
      quote="if the hidden dimensions are too much, then your more parameters will come into the place or more computations"),
]
for e in M:
    if e["quote"]: assert e["quote"] in SRC[e["week"]], e["id"]
(ROOT / "design" / "mis_ch11.json").write_text(json.dumps(M, indent=1))
print(len(M), "misconceptions, quotes verbatim")
