#!/bin/bash
# make.sh
# This script compiles a LaTeX report using BibTeX in the correct order

# File name without extension
FILE="report"

echo "Step 1: Running pdflatex..."
pdflatex $FILE.tex

echo "Step 2: Running bibtex..."
bibtex $FILE

echo "Step 3: Running pdflatex again..."
pdflatex $FILE.tex

echo "Step 4: Final pdflatex run..."
pdflatex $FILE.tex

echo "Compilation finished! Output: $FILE.pdf"