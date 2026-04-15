# Paper Compiler Skill

## 触发描述

当用户要求**编译LaTeX论文、生成PDF、LaTeX转PDF**时触发本Skill。

典型场景：
- "编译论文"
- "生成PDF"
- "LaTeX转PDF"
- "帮我编译这个tex文件"
- 修改了`.tex`文件后需要重新生成PDF

## 编译流程

LaTeX交叉引用和参考文献需要**多次编译**才能正确生成：

```
xelatex (第1次) → 生成.aux交叉引用文件
bibtex (处理参考文献) → 生成.bbl
xelatex (第2次) → 交叉引用就位
xelatex (第3次) → 最终输出
```

**最少需要3次xelatex + 1次bibtex**

## 避坑清单 (Gotchas)

### 1. 交叉引用显示为"??"

原因：编译次数不足
- 检查是否执行了3次xelatex
- 检查`.aux`文件是否存在

### 2. 参考文献显示为"[?]"

原因：bibtex未执行或失败
- 检查`references.bib`文件是否存在
- 检查bibkey是否匹配（大小写敏感）
- 检查bib文件格式是否正确（特别是中文作者名的`and`连接）

### 3. 绿色圆圈/彩色链接

原因：hyperref默认彩色链接
- 设置`colorlinks=false`
- 或设置`colorlinks=true, linkcolor=black, citecolor=black`

### 4. 引用不是上标

原因：未使用natbib包
- 添加`\usepackage[super]{natbib}`
- 或`\usepackage[square,super]{natbib}`

### 5. 编译报错"File not found"

原因：文件路径包含空格或中文
- 使用绝对路径
- 路径避免空格和特殊字符

### 6. 中文字体缺失

原因：xeCJK配置问题
- 确保使用`\usepackage{xeCJK}`
- 设置`\setCJKmainfont{SimSun}`

### 7. 段落没有缩进

原因：未设置`\parindent`
- 添加`\setlength{\parindent}{2em}`
- 中文论文通常需要首行缩进两格

## 配置文件

`config.json`:
```json
{
  "name": "paper_compiler",
  "citation_style": "super",
  "output_format": "pdf"
}
```

## 使用方法

### Python调用
```python
from compiler import compile_paper
success = compile_paper(paper_dir='E:/path/to/paper_output')
```

### 命令行
```bash
python compiler.py "E:/path/to/paper_output"
```

### Claude Code中触发
直接说"编译论文"或"生成PDF"

## 编译检查清单

编译完成后检查：
- [ ] PDF文件已生成
- [ ] 交叉引用正确显示（不是"???"）
- [ ] 参考文献正确显示（不是"[?]"）
- [ ] 引用格式正确（上标/作者年）
- [ ] 链接颜色正确（黑色非绿色）
- [ ] 段落缩进正确
