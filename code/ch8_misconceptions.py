"""Misconceptions for Chapter 8, with transcript quotes checked verbatim. Writes design/mis_ch8.json."""
import re, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\n[0-9:]+\n", " ", s)).lower()
SRC = {w: norm(open(ROOT / "data" / f"ai_week{w}.txt").read()) for w in range(1, 10)}
M = [
 ("8.1", "The loss is the share of wrong answers, like a test score.", "Every exam you have taken was scored that way.", "8.1", None, None),
 ("8.2", "For text, the error is the prediction minus the answer, as for a price.", "Earlier models (house prices, churn scores) were trained exactly that way.", "8.1", 9,
  "you are subtracting yhat minus y and then you are getting a score"),
 ("8.3", "A bigger step size always learns faster.", "Bigger steps cover more ground per step.", "8.3", 3,
  "but my loss started to increase over the bit of a time."),
 ("8.4", "Training adjusts the arrow of each word inside each sentence directly.", "Chapter 3 drew the arrows, so it feels as if they are what moves.", "8.4", None, None),
 ("8.5", "A sentence gives the model one exercise: predict the word after it.", "We generate one token at a time.", "8.6", 9,
  "because we are predicting one token at a time, right?"),
 ("8.6", "Each step on a batch moves exactly downhill on the whole dataset.", "The step is computed, so it must be exact.", "8.5", None, None),
 ("8.7", "Training finds the best possible weights.", "The bowl picture has one bottom.", "8.3", 9,
  "to reach your local global minima"),
 ("8.8", "A low training loss means the model learned what we wanted.", "Lower loss is the goal we set.", "8.4, 8.7", 9,
  "it will overfit. you need to have a substantial training data with consistent patterns into it"),
]
out = []
for i, wrong, tempt, fix, wk, q in M:
    if q: assert q in SRC[wk], f"quote not verbatim: {i}"
    out.append(dict(id=i, wrong=wrong, tempting=tempt, fixed_in="\\S" + fix, week=wk, quote=q))
(ROOT / "design" / "mis_ch8.json").write_text(json.dumps(out, indent=1)); print(len(out), "ok, quotes verbatim")
