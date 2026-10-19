# Regenerate every number, then build the book.  `make` = numbers + PDF.
PY = python3
SCRIPTS = code/toy.py code/corpus.py code/history.py code/ch1_numbers.py code/ch2_bpe.py code/ch2_mis.py \
  code/ch3_numbers.py code/ch4_rope.py code/ch5_random_blocks.py code/ch5_numbers.py code/ch5_misconceptions.py \
  code/ch6_numbers.py code/ch7_numbers.py code/ch8_train.py code/ch8_numbers.py code/ch8_misconceptions.py \
  code/appA_check.py code/ch9_numbers.py code/ch9_mis.py code/ch10_verify.py code/ch10_numbers.py \
  code/ch11_ablate.py code/ch11_budget.py code/ch11_mis.py code/ch1_figs.py code/ch2_figs.py code/ch3_figs.py code/ch4_figs.py code/ch5_figs.py \
  code/ch6_figs.py code/ch7_figs.py code/ch8_figs.py code/ch9_figs.py code/appA_figs.py code/ch10_figs.py code/ch11_figs.py \
  code/ch11_appB_index.py
all: numbers pdf
numbers:
	@for s in $(SCRIPTS); do echo "== $$s"; $(PY) $$s > /dev/null || exit 1; done
	cat tex/num_toy.tex tex/num_corpus.tex tex/num_history.tex tex/num_ch*.tex tex/num_appA.tex > tex/num_all.tex
pdf:
	cd tex && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
