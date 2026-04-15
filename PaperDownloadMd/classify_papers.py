# -*- coding: utf-8 -*-
"""
使用 PyPDF2 解析摘要并打分
"""
import os
import json
import shutil
import PyPDF2
import sys

ROOT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
PAPER_DOWNLOAD = f'{ROOT_DIR}/PaperDownload'
PAPER_LIST_JSON = f'{ROOT_DIR}/PaperDownloadMd/paper_list.json'

# 重新设置：先把所有PDF移回根目录
print("Resetting all PDFs to root...")
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.pdf'):
                src = os.path.join(folder, f)
                dst = os.path.join(PAPER_DOWNLOAD, f)
                try:
                    shutil.move(src, dst)
                except:
                    pass

def extract_abstract(pdf_path):
    """从PDF提取摘要"""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for i in range(min(3, len(reader.pages))):
                page = reader.pages[i]
                t = page.extract_text()
                if t:
                    text += t + '\n'

        if not text:
            return None

        # 找 Abstract 章节
        lines = text.split('\n')
        in_abstract = False
        abstract_lines = []

        for line in lines:
            line_clean = line.strip().lower()
            if 'abstract' in line_clean and len(line_clean) < 30:
                in_abstract = True
                continue
            if in_abstract:
                if line.startswith('1 ') or line.startswith('2 ') or line.startswith('Introduction') or line.startswith('1.'):
                    break
                abstract_lines.append(line.strip())

        if abstract_lines:
            abstract = ' '.join(abstract_lines[:20])
            return abstract[:1000]

        return text[:800]

    except Exception as e:
        return None

def score_by_abstract(abstract):
    """根据摘要内容打分"""
    if not abstract:
        return 2

    text = abstract.lower()

    # 5分：PM2.5 + CMAQ/融合/克里金
    if 'pm2.5' in text or 'pm 2.5' in text:
        if any(kw in text for kw in ['cmaq', 'chemical transport', 'ctm', 'downscaling', 'kriging', 'fusion', 'bayesian', 'variational']):
            return 5
        return 4

    if 'air quality' in text and any(kw in text for kw in ['pm', 'pollution', 'aerosol', 'prediction', 'estimation']):
        return 4

    if any(kw in text for kw in ['machine learning', 'deep learning', 'neural network', 'neural',
                                    'interpolation', 'spatial', 'temporal', 'prediction',
                                    'atmospheric', 'weather', 'satellite', 'aerosol', 'aod']):
        return 3

    return 2

# 创建文件夹
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    os.makedirs(folder, exist_ok=True)

# 获取所有PDF
all_pdfs = [f for f in os.listdir(PAPER_DOWNLOAD) if f.endswith('.pdf')]

print(f"Total PDFs to classify: {len(all_pdfs)}")

stats = {2: 0, 3: 0, 4: 0, 5: 0}

for i, fname in enumerate(all_pdfs):
    src = os.path.join(PAPER_DOWNLOAD, fname)
    fname_display = fname[:50]

    # 提取摘要
    abstract = extract_abstract(src)

    # 打分
    score = score_by_abstract(abstract)

    # 移动
    dst = os.path.join(PAPER_DOWNLOAD, f'score_{score}', fname)
    try:
        shutil.move(src, dst)
        stats[score] += 1
    except Exception as e:
        print(f"Error moving {fname}: {e}")
        stats[2] += 1

    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{len(all_pdfs)}")

print()
print("=" * 50)
print("Classification Complete")
print("=" * 50)
print(f"Score 5: {stats[5]}篇")
print(f"Score 4: {stats[4]}篇")
print(f"Score 3: {stats[3]}篇")
print(f"Score 2: {stats[2]}篇")
print(f"Total: {len(all_pdfs)}篇")