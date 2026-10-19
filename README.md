# Transformers from First Principles

**From one word to designing your own model.**
Lecture notes by Ayush Singh and Kirti Goyal · Antern Studies · October 2026

[Read online](https://www.antern.co/blogs/transformers-from-first-principles/) · [PDF](transformers-from-first-principles.pdf)

---

Why does a transformer look the way it does? These notes build one a block at a time, from the next-word prediction problem to designing your own model. Each component arrives as the fix for a failure you can see in real numbers.

One running example, *"waste of time"*, is worked by hand on a tiny model of width 4. The same steps are then replayed on GPT-2 small and Qwen3-0.6B.

148 pages · 12 chapters · 2 appendices

## What's here

```
tex/        LaTeX source of the notes
  main.tex      master file
  front.tex     title page and "How to read these notes"
  ch0–ch11.tex  chapters
  appA, appB    appendices
  refs.tex      references
  notes.sty     shared style: boxes, the three rungs, "Where we are in the machine"
  num_*.tex     generated numbers
  fig_*.tex     generated figures (69)
code/       Python scripts that compute every number and draw every figure
design/     misconception sets for each chapter, as JSON
Makefile    regenerate numbers, then build the PDF
```

No number in the text is typed by hand. Each one is a macro written by a script in `code/`, so the notes and the computations cannot drift apart. The `num_*.tex` and `fig_*.tex` files are generated, so don't edit them directly.

## Build the PDF

The generated files are committed, so you only need a LaTeX distribution:

```bash
cd tex
latexmk -pdf main.tex
```

## Regenerate everything

Rebuilding the numbers and figures from scratch needs Python 3 with `numpy`, `torch` and `transformers`, plus the lecture transcripts (`data/ai_week1.txt` to `data/ai_week9.txt`), which are not included in this repository.

```bash
make            # numbers + PDF
make numbers    # numbers and figures only
make pdf        # PDF only
```

---

© 2026 Ayush Singh and Kirti Goyal. All rights reserved.
