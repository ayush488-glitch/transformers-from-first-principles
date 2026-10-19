"""Builds the script index for Appendix B from the code/ directory itself, so the appendix never goes
stale: for every script, its first docstring line, the files it writes, and how many checks (assert
statements) it contains. Also records software versions. Writes tex/data_ch11/scripts.tex, versions.tex."""
import ast, pathlib, re, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
def tex(s):
    m = {"\\": "\\textbackslash{}", "_": "\\_", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "~": "\\textasciitilde{}",
         "^": "\\textasciicircum{}", "{": "\\{", "}": "\\}", "<": "$<$", ">": "$>$"}
    return "".join(m.get(ch, ch) for ch in s.replace("->", "\x00")).replace("\x00", "$\\to$")
SUBS = [(r", with transcript quotes checked verbatim", " (design inventory, not printed in the book)"),
        (r"; every transcript quote is asserted verbatim", " (design inventory, not printed in the book)"),
        (r" \(quotes checked verbatim\)", " (design inventory, not printed in the book)"),
        (r"the author's own lecture transcripts|the author's transcripts|the lecture transcripts|the transcripts", "the talk corpus"),
        (r"transcripts?", "talk corpus")]
def clean(d):
    for a, b in SUBS: d = re.sub(a, b, d)
    return d
rows, total = [], 0
for p in sorted((ROOT / "code").glob("*.py"), key=lambda q: (re.sub(r"\D", "", q.stem.split("_")[0]).zfill(3) if q.stem.startswith("ch") else "zzz", q.name)):
    src = p.read_text(); tree = ast.parse(src)
    doc = clean((ast.get_docstring(tree) or "").strip().split("\n")[0])
    n_assert = sum(isinstance(n, ast.Assert) for n in ast.walk(tree)); total += n_assert
    outs = sorted(set(re.findall(r"(num_[A-Za-z0-9_]+\.tex|[a-z_]+\.(?:json|dat))", src)))
    rows.append(f"\\texttt{{{tex(p.name)}}} & {tex(doc)} & {tex(', '.join(outs)) or '---'} & {n_assert}\\\\")
(ROOT / "tex" / "data_ch11" / "scripts.tex").write_text("\n".join(rows) + "\n")
import numpy, torch, transformers
texv = subprocess.run(["pdflatex", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
V = {"Python": sys.version.split()[0], "NumPy": numpy.__version__, "PyTorch": torch.__version__,
     "Transformers": transformers.__version__, "TeX": texv}
(ROOT / "tex" / "data_ch11" / "versions.tex").write_text(
    "\n".join(f"{k} & \\texttt{{{tex(v)}}}\\\\" for k, v in V.items()) + "\n")
(ROOT / "tex" / "data_ch11" / "counts.tex").write_text(
    f"\\newcommand{{\\nElevenScriptCount}}{{{len(rows)}}}\\newcommand{{\\nElevenAssertCount}}{{{total}}}\n")
print(len(rows), "scripts,", total, "asserts")
