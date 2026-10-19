"""Chapter 1 picture: one pass over a batch makes a guess at every position (inputs vs. targets)."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from figlib import matrix_fig, write
T = pathlib.Path(__file__).resolve().parent.parent / "tex"
def stretch(pic, f):
    return pic.replace("\\begin{tikzpicture}", f"\\begin{{tikzpicture}}[x={f}cm]", 1)
tt = lambda w: f"\\texttt{{{w}}}"
inp = np.array([[tt(w) for w in ["waste", "of", "time"]], [tt(w) for w in ["not", "very", "useful"]]], dtype=object)
tgt = np.array([[tt(w) for w in ["of", "time", "Sorry"]], [tt(w) for w in ["very", "useful", "."]]], dtype=object)
rows = ["sentence 1", "sentence 2"]
write(T / "fig_ch1_batch.tex",
      stretch(matrix_fig(inp, title="what the model reads", rows=rows, cols=["1", "2", "3"], shape="$B\\times T=2\\times3$"), 1.7),
      stretch(matrix_fig(tgt, title="the word it must guess next", rows=rows, cols=["1", "2", "3"], shape="$B\\times T=2\\times3$ guesses"), 1.7),
      sep="\\qquad")
print("ch1 figures written")
