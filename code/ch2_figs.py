"""Pictures for Chapter 2: token ids, the one-hot lookup as a matrix product, and a batch of ids -> arrows."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vocab import E, ID, VOCAB
from figlib import matmul_fig, matrix_fig, vec_fig, stack_fig, write
T = pathlib.Path(__file__).resolve().parent.parent / "tex"
def stretch(pic, f):
    return pic.replace("\\begin{tikzpicture}", f"\\begin{{tikzpicture}}[x={f}cm]", 1)
s1, s2 = ["waste", "of", "time"], ["not", "very", "useful"]
ids = np.array([[ID[t] for t in s1]]); ids2 = np.array([[ID[t] for t in s1], [ID[t] for t in s2]])
# token ids of one sentence and of a batch
write(T / "fig_ch2_ids.tex",
      stretch(matrix_fig(ids, title="token ids of \\texttt{waste of time}", cols=s1, shape="$1\\times T=1\\times3$"), 1.3),
      stretch(matrix_fig(ids2, title="a batch of token ids", rows=["sentence 1", "sentence 2"],
                         shape="$B\\times T=2\\times3$"), 1.3), sep="\\qquad\\qquad")
# one-hot row times the embedding table picks out one row
oh = np.zeros((1, len(VOCAB))); oh[0, ID["time"]] = 1
x = oh @ E; assert np.allclose(x[0], E[ID["time"]])
write(T / "fig_ch2_onehot.tex", matmul_fig(oh, E, 0, 3, "\\text{one-hot}(\\texttt{time})", "E", "\\bm{x}_{\\text{time}}",
      rowsA=["time"], colsA=[str(i) for i in range(len(VOCAB))], rowsB=None, colsB=["1", "2", "3", "4"], expand=False))
write(T / "fig_ch2_E.tex", matrix_fig(E, title="$E$", rows=VOCAB, cols=["1", "2", "3", "4"],
      shape="$V\\times d=12\\times4$", hl_rows=(ID["time"],)))
# the batch of ids becomes a batch of arrows
Xb = [E[ids2[0]], E[ids2[1]]]
pic = stack_fig(Xb, titles=["sentence 1", "sentence 2"], rows=s1, name="X", batch_label=True)
write(T / "fig_ch2_batch.tex", pic)
print("ch2 figures written")
