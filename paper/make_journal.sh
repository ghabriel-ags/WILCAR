#!/usr/bin/env bash
# Preview of the article in the final two-column layout of the journal (elsarticle 5p, Times), for page estimates.
#   bash paper/make_journal.sh   ->  paper/main_journal.pdf   (needs a previous build of main.tex and supplement.tex)
set -e
cd "$(dirname "$0")"
B=build_journal; rm -rf $B; mkdir -p $B
cp -r refs.bib tables figures supplement.aux $B/ 2>/dev/null || true
sed -e '1s/.*/\\documentclass[5p,times,authoryear]{elsarticle}/' -e 's/^\\linenumbers$//' main.tex > $B/main.tex
(cd $B && latexmk -pdf -interaction=nonstopmode main.tex >/dev/null 2>&1 || true)
cp $B/main.pdf main_journal.pdf
grep -E "Output written" $B/main.log
