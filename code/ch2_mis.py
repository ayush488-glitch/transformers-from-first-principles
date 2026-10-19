"""Misconception inventories for Chapters 2 and 4; design notes only, not printed in the book."""
import re, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
def norm(s): return re.sub(r"\s+", " ", re.sub(r"\n[0-9:]+\n", " ", s)).lower()
SRC = {w: norm(open(ROOT/"data"/f"ai_week{w}.txt").read()) for w in range(1, 10)}
CH2 = [
 ("2.1", "A model reads words, or letters, the way a person does.", "We see words when we read.", "S2.2, S2.3", None, None),
 ("2.2", "A word the model has never seen makes it fail or print ``unknown''.", "Old systems did exactly that.", "S2.3", 9,
  "it will try to go to the nearest token or maybe it will split further or maybe it will write unknown."),
 ("2.3", "A token is a word.", "For common words the two coincide.", "S2.3", 9, "tokenizing it into the words"),
 ("2.4", "A token id is a meaningful number: id 5 is closer in meaning to id 6 than to id 900.", "Ids are numbers, and numbers have an order.", "S2.4", None, None),
 ("2.5", "The vector of a word changes from sentence to sentence inside the embedding table.", "Meaning does change with context.", "S2.4, S2.6", 9,
  "embeddings are same, assume that embeddings are same."),
 ("2.6", "Each number in a word vector means something nameable, like `sentiment'.", "In a churn model each clue had a name.", "S2.5", None, None),
 ("2.7", "Similar words get similar vectors because someone listed the synonyms.", "Dictionaries list synonyms.", "S2.5", None, None),
]
CH4 = [
 ("4.1", "The causal mask already tells attention the order of the words.", "The mask is about who came before whom.", "S4.1", 9,
  "do you really think any of the transformation takes that positional thing"),
 ("4.2", "Adding the position number (0, 1, 2, \\ldots) to every coordinate is enough.", "It is the simplest way to write position down.", "S4.2", None, None),
 ("4.3", "Positions have to be learned, like every other weight.", "Most things in the model are learned.", "S4.2, S4.3", 9,
  "we want you to also learn positions."),
 ("4.4", "RoPE changes the token's embedding or its value.", "Older methods add position to the embedding.", "S4.4", None, None),
 ("4.5", "What matters to a word is its absolute position in the text.", "Positions are counted from the start.", "S4.5", None, None),
 ("4.6", "Because rotations repeat, a far word is confused with a near one.", "One clock hand alone does repeat.", "S4.6", None, None),
]
for name, L in (("mis_ch2.json", CH2), ("mis_ch4.json", CH4)):
    out = []
    for i, wrong, tempt, fix, wk, q in L:
        if q: assert q in SRC[wk], (i, q)
        out.append(dict(id=i, wrong=wrong, tempting=tempt, fixed_in=fix, week=wk, quote=q))
    (ROOT/"design"/name).write_text(json.dumps(out, indent=1))
print("ok: quotes verbatim")
