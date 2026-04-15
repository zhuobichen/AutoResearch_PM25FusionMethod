# Paper Compiler Skill

将LaTeX论文源码编译为PDF的专业工具。

## 目录结构

```
paper_compiler/
├── config.json      # 配置文件
├── compiler.py      # 编译脚本
├── SKILL.md         # Skill核心文档
└── README.md        # 本文件
```

## 快速使用

在Claude Code中直接说：
- "编译论文"
- "生成PDF"
- "LaTeX转PDF"

## 编译流程

```
xelatex (生成辅助文件) → bibtex (处理文献) → xelatex x2 (交叉引用+最终输出)
```

## 核心功能

1. **自动多次编译**：确保交叉引用和文献引用正确
2. **错误检测**：从.log文件中提取错误信息
3. **辅助文件清理**：强制重新编译时清除缓存

## 避坑要点

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 引用显示"??" | 编译次数不足 | 执行3次xelatex |
| 引用显示"[?]" | bibtex失败 | 检查bib文件和key |
| 绿色链接 | hyperref彩色 | `colorlinks=false` |
| 非上标引用 | 未用natbib | `\usepackage[super]{natbib}` |
| 无段落缩进 | 未设parindent | `\setlength{\parindent}{2em}` |

## 配置

编辑`config.json`自定义：
```json
{
  "citation_style": "super",
  "output_format": "pdf"
}
```
