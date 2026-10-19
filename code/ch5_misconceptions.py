"""Misconception inventories for Chapters 5, 6 and 7 -> design/mis_ch5.json, mis_ch6.json, mis_ch7.json.
Every quote is asserted to be a verbatim substring of its transcript (same normalisation as design/misconceptions.py)."""
import re, json, pathlib
R = pathlib.Path(__file__).resolve().parent.parent
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\n[0-9:]+\n", " ", s)).lower()
SRC = {w: norm(open(R / "data" / f"ai_week{w}.txt").read()) for w in range(1, 10)}
MIS = {5: [
 ("5.1", "More heads is always better.", "More heads sounds like more ways of looking.", "\\S5.2", 9, "it is not always it helps."),
 ("5.2", "One head can attend to two tokens just as well, by splitting its attention.", "Softmax can give 50\\% to each.", "\\S5.1", None, None),
 ("5.3", "The MLP is where tokens exchange information.", "It comes right after attention and is the bigger part of the block.", "\\S5.4", 9, "why doesn't your ffn looks back to other tokens?"),
 ("5.4", "Each layer replaces a token's arrow with a new one.", "Diagrams draw layers as boxes in a row.", "\\S5.5", 9, "it forgots the original in representation without combining the effects of it"),
 ("5.5", "Each head has a fixed job, like ``the grammar head''.", "Our toy heads have tidy jobs.", "\\S5.2", None, None)],
 6: [
 ("6.1", "Normalisation is there so that no token can beat another.", "The word sounds like making things equal.", "\\S6.2", 9, "to remove comparison bias so that nobody beats each other"),
 ("6.2", "Normalising throws away the information in an arrow.", "Every arrow comes out the same length.", "\\S6.2", None, None),
 ("6.3", "LayerNorm makes every arrow length 1.", "Unit vectors are what ``normalised'' means in school maths.", "\\S6.2", None, None),
 ("6.4", "More layers always means a better model.", "Bigger models are better, so deeper must be too.", "\\S6.1, \\S6.4", None, None),
 ("6.5", "Whether the norm sits before or after a layer is a detail.", "Both versions compute the same kinds of things.", "\\S6.4", None, None)],
 7: [
 ("7.1", "The model can write any word, even one it has never seen.", "It writes new sentences all the time.", "\\S7.1", 9, "do you really think my output token can be different from what is up in the embedding table of quen? nope."),
 ("7.2", "The probability the model gives is how sure it is that the answer is true.", "It is a number between 0 and 1 that sounds like confidence.", "\\S7.6", 8, "that is not calibrated. that is next token prediction after all"),
 ("7.3", "Turning the temperature down makes the model more correct.", "Lower temperature gives steadier answers.", "\\S7.3", 6, "they will control it with the help of um temperature or other small small things"),
 ("7.4", "Picking the top word every time gives the best reply.", "Each step makes the best choice.", "\\S7.4", None, None),
 ("7.5", "A model that gives different answers to the same prompt is broken.", "Programs should be repeatable.", "\\S7.5", 6, "it should give a different answer but there is no patterns to it"),
 ("7.6", "One wrong word is a small, local mistake.", "It is one word out of many.", "\\S7.7", None, None)]}
for ch, rows in MIS.items():
    out = []
    for mid, wrong, tempt, fix, wk, q in rows:
        if q: assert q in SRC[wk], (mid, q)
        out.append(dict(id=mid, wrong=wrong, tempting=tempt, fixed_in=fix, week=wk, quote=q))
    (R / "design" / f"mis_ch{ch}.json").write_text(json.dumps(out, indent=1))
print("misconceptions ok")
