"""
Word copy of the manuscript for co-author review:  python paper/make_docx.py  ->  paper/main.docx

pandoc cannot convert algorithmic environments or resized tables faithfully, so every algorithm and table is compiled
on its own with the manuscript preamble, cropped and inserted as an image; text, equations, theorems and citations are
converted by pandoc (--citeproc). The LaTeX source stays the reference version.
"""
import re, subprocess, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_docx"


TYPES = {"section": ("Section", "Sections"), "subsection": ("Section", "Sections"), "equation": ("Eq.", "Eqs."),
         "algorithm": ("Algorithm", "Algorithms"), "theorem": ("Theorem", "Theorems"), "proposition": ("Proposition", "Propositions"),
         "lemma": ("Lemma", "Lemmas"), "definition": ("Definition", "Definitions"), "table": ("Table", "Tables"),
         "figure": ("Fig.", "Figs."), "appendix": ("", ""), "remark": ("Remark", "Remarks")}


def labels():
    """label -> (number, type) from main.aux (cleveref entries give the type)."""
    aux = (HERE / "main.aux").read_text() if (HERE / "main.aux").exists() else ""
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}@]+)@cref\}\{\{\[([a-z]+)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}", aux):
        out[m.group(1)] = (" ".join(m.group(3).replace("~", " ").split()), m.group(2))
    return out


def cites():
    """bib key -> (authors, year) from main.bbl (elsarticle-harv)."""
    bbl = (HERE / "main.bbl").read_text() if (HERE / "main.bbl").exists() else ""
    out = {}
    for m in re.finditer(r"\\bibitem\[\{(.*?)\((\d{4}[a-z]?)\)[^}]*\}\]\{([^}]+)\}", bbl):
        out[m.group(3)] = (m.group(1).replace("et~al.", "et al.").replace("{", "").replace("}", "").strip(), m.group(2))
    return out


def resolve_refs(tex, lab, cit, inline_cites=False):
    def one(key):
        return lab.get(key, ("??", "section"))
    def cref(m, cap=True):
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = [one(k)[0] for k in keys]
        typ = one(keys[0])[1]
        name = TYPES.get(typ, ("", ""))[0 if len(keys) == 1 else 1]
        return (name + " " if name else "") + " and ".join(nums)
    tex = re.sub(r"\\[cC]ref\{([^}]+)\}", cref, tex)
    tex = re.sub(r"\\eqref\{([^}]+)\}", lambda m: "(" + one(m.group(1))[0] + ")", tex)
    tex = re.sub(r"\\ref\{([^}]+)\}", lambda m: one(m.group(1))[0], tex)
    # numbered displays: turn equation labels into tags so the Word copy shows the same numbers
    tex = re.sub(r"\\label\{(eq:[^}]+)\}", lambda m: "\\tag{" + one(m.group(1))[0] + "}", tex)
    if inline_cites:
        def c(m, paren):
            parts = []
            for k in m.group(1).split(","):
                a, y = cit.get(k.strip(), (k.strip(), ""))
                parts.append(f"{a}, {y}" if paren else f"{a} ({y})")
            return "(" + "; ".join(parts) + ")" if paren else "; ".join(parts)
        tex = re.sub(r"\\citep\{([^}]+)\}", lambda m: c(m, True), tex)
        tex = re.sub(r"\\citet\{([^}]+)\}", lambda m: c(m, False), tex)
    return tex


def flatten(tex):
    # \IfFileExists{f}{\input{f}}{...} -> content of f (or nothing)
    def repl(m):
        f = HERE / m.group(1)
        return f.read_text() if f.exists() else ""
    tex = re.sub(r"\\IfFileExists\{(tables/[^}]+)\}\{\\input\{[^}]+\}\}\{\}", repl, tex)
    tex = re.sub(r"\\IfFileExists\{tables/numbers.tex\}\{\\input\{tables/numbers.tex\}\}\{\}",
                 lambda m: (HERE / "tables/numbers.tex").read_text() if (HERE / "tables/numbers.tex").exists() else "", tex)
    return tex


def _trim(png, pad=12):
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if bbox:
        l, t, r, b = bbox
        im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))).save(png)


def render_blocks(tex, preamble):
    """Replace algorithm/table environments by images rendered with the manuscript preamble."""
    BUILD.mkdir(exist_ok=True)
    pat = re.compile(r"\\begin\{(algorithm|table\*?)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)
    out, k, last = [], 0, 0
    for m in pat.finditer(tex):
        k += 1
        body = m.group(3)
        cap = re.search(r"\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body)
        caption = cap.group(1) if cap else ""
        lab = re.search(r"\\label\{([^}]+)\}", body)
        env = "algorithm" if m.group(1) == "algorithm" else "table"
        blk = body.replace(cap.group(0), "") if cap else body
        blk = re.sub(r"\\label\{[^}]+\}", "", blk)
        blk = resolve_refs(blk, LAB, CIT, inline_cites=True)
        if env == "algorithm":           # keep the "Algorithm n" rule block without a caption line
            blk = "\\caption*{}" + blk if False else blk
        src = (preamble + "\n\\pagestyle{empty}\\begin{document}\\nolinenumbers\n"
               + f"\\begin{{{env}}}[H]" + blk + f"\\end{{{env}}}\n\\end{{document}}\n")
        src = src.replace("\\linenumbers", "")
        name = f"block{k:02d}"
        (BUILD / f"{name}.tex").write_text(src)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=BUILD, capture_output=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=BUILD, capture_output=True)
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", str(BUILD / f"{name}.pdf"), str(BUILD / name)],
                       capture_output=True)
        _trim(BUILD / f"{name}.png")
        num, typ = LAB.get(lab.group(1), ("", env)) if lab else ("", env)
        head = TYPES.get(typ, (env.capitalize(), ""))[0]
        cap_txt = resolve_refs(caption, LAB, CIT)
        out.append(tex[last:m.start()])
        out.append(f"\n\n\\textbf{{{head} {num}.}} {cap_txt}\n\n\\includegraphics[width=\\textwidth]{{build_docx/{name}.png}}\n\n")
        last = m.end()
    out.append(tex[last:])
    return "".join(out)


LAB, CIT = {}, {}


def main():
    global LAB, CIT
    LAB, CIT = labels(), cites()
    tex = (HERE / "main.tex").read_text()
    tex = flatten(tex)
    pre_end = tex.index("\\begin{document}")
    preamble = tex[:pre_end]
    # standalone blocks use the article class with the same packages (elsarticle adds a title page)
    blk_pre = preamble.replace("\\documentclass[preprint,12pt,authoryear]{elsarticle}",
                               "\\documentclass[12pt]{article}\\usepackage{natbib}\\usepackage[margin=2cm]{geometry}")
    blk_pre = re.sub(r"\\journal\{[^}]*\}", "", blk_pre)
    body = render_blocks(tex[pre_end:], blk_pre)
    body = resolve_refs(body, LAB, CIT)
    for w in ("mean", "point-wise", "sign"):
        body = body.replace("\\text{(%s)}" % w, "\\text{%s:}" % w)
    body = body.replace("\\Bigl", "\\left").replace("\\Bigr", "\\right").replace("\\linenumbers", "")
    body = re.sub(r"\\begin\{frontmatter\}|\\end\{frontmatter\}", "", body)
    flat = preamble + body
    (HERE / "main_flat.tex").write_text(flat)
    r = subprocess.run(["pandoc", "main_flat.tex", "--citeproc", "--bibliography=refs.bib", "--resource-path=.",
                        "-M", "title=Constructive neural networks with certified partial monotonicity for binary classification",
                        "-o", "main.docx"], cwd=HERE, capture_output=True, text=True)
    print(r.returncode, "\n".join(l for l in r.stderr.splitlines() if "WARNING" in l)[:2000])


if __name__ == "__main__":
    main()
