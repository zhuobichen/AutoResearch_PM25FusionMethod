# -*- coding: utf-8 -*-
"""
LaTeX Paper Compiler - 将LaTeX论文源码编译为PDF

编译流程：
1. xelatex paper.tex (生成辅助文件)
2. bibtex paper (处理参考文献)
3. xelatex paper.tex (生成交叉引用)
4. xelatex paper.tex (最终输出)

触发场景：
- 用户要求"编译论文"、"生成PDF"、"LaTeX转PDF"
- 用户修改了.tex文件后需要重新生成
- 用户问"如何把LaTeX转成PDF"

使用方法：
from paper_compiler import compile_paper
compile_paper(paper_dir='path/to/paper_dir')
"""

import os
import subprocess
import shutil
from pathlib import Path


class PaperCompiler:
    """LaTeX论文编译器"""

    def __init__(self, paper_dir, paper_name='paper'):
        self.paper_dir = Path(paper_dir)
        self.paper_name = paper_name
        self.tex_path = self.paper_dir / f"{paper_name}.tex"
        self.log_path = self.paper_dir / f"{paper_name}.log"
        self.aux_path = self.paper_dir / f"{paper_name}.aux"

    def compile(self, clean=True):
        if not self.tex_path.exists():
            raise FileNotFoundError(f"找不到TeX文件: {self.tex_path}")

        if clean:
            self._clean_aux_files()

        print(f"开始编译: {self.tex_path}")
        print("=" * 50)

        # Step 1: 第一次xelatex
        print("[1/4] 运行 xelatex (第1次)...")
        ret1 = self._run_xelatex()
        if ret1 != 0:
            print(f"[错误] xelatex 第1次失败 (返回码: {ret1})")
            return False

        # Step 2: bibtex
        print("[2/4] 运行 bibtex (处理参考文献)...")
        ret2 = self._run_bibtex()
        if ret2 != 0:
            print(f"[警告] bibtex 失败 (返回码: {ret2})")

        # Step 3: 第二次xelatex
        print("[3/4] 运行 xelatex (第2次)...")
        ret3 = self._run_xelatex()
        if ret3 != 0:
            print(f"[错误] xelatex 第2次失败 (返回码: {ret3})")
            return False

        # Step 4: 第三次xelatex
        print("[4/4] 运行 xelatex (第3次 - 最终输出)...")
        ret4 = self._run_xelatex()
        if ret4 != 0:
            print(f"[错误] xelatex 第3次失败 (返回码: {ret4})")
            return False

        pdf_path = self.paper_dir / f"{self.paper_name}.pdf"
        if pdf_path.exists():
            size = os.path.getsize(pdf_path) / 1024
            print("=" * 50)
            print(f"[成功] PDF已生成: {pdf_path.name} ({size:.1f} KB)")
            return True
        else:
            print("[错误] PDF文件未生成")
            return False

    def _run_xelatex(self):
        cmd = ['xelatex', '-interaction=batchmode', self.tex_path]
        result = subprocess.run(
            cmd,
            cwd=str(self.paper_dir),
            capture_output=True,
            text=True
        )
        return result.returncode

    def _run_bibtex(self):
        cmd = ['bibtex', self.paper_name]
        result = subprocess.run(
            cmd,
            cwd=str(self.paper_dir),
            capture_output=True,
            text=True
        )
        return result.returncode

    def _clean_aux_files(self):
        extensions = ['.aux', '.bbl', '.blg', '.log', '.out', '.toc', '.ind', '.idx', '.fls', '.fdb_latexmk']
        for ext in extensions:
            f = self.paper_dir / f"{self.paper_name}{ext}"
            if f.exists():
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"  [警告] 无法删除 {f.name}: {e}")

    def get_errors(self):
        if not self.log_path.exists():
            return []
        errors = []
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'error' in line.lower() or '!' in line:
                    errors.append(line.strip())
        return errors[:20]


def compile_paper(paper_dir=None, paper_name='paper'):
    if paper_dir is None:
        paper_dir = os.getcwd()
    compiler = PaperCompiler(paper_dir, paper_name)
    return compiler.compile()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        paper_dir = sys.argv[1]
    else:
        paper_dir = os.getcwd()
    compiler = PaperCompiler(paper_dir)
    success = compiler.compile()
    if not success:
        errors = compiler.get_errors()
        if errors:
            print("\n检测到的错误:")
            for e in errors:
                print(f"  - {e}")
        sys.exit(1)
