@echo off
cd /d "%~dp0"
xelatex -interaction=nonstopmode -synctex=0 paper.tex
xelatex -interaction=nonstopmode -synctex=0 paper.tex
xelatex -interaction=nonstopmode -synctex=0 paper.tex
del *.aux *.log *.out *.toc *.auxlock 2>nul
start paper.pdf
