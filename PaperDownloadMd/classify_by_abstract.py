# -*- coding: utf-8 -*-
"""
使用 pdfplumber 解析摘要并打分
"""
import os
import json
import shutil
import pdfplumber

ROOT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
PAPER_DOWNLOAD = f'{ROOT_DIR}/PaperDownload'
PAPER_LIST_JSON = f'{ROOT_DIR}/PaperDownloadMd/paper_list.json'

def extract_abstract(pdf_path):
    """从PDF提取摘要"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages[:3]:  # 只读前3页
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
                # 遇到新章节或太长的行就停止
                if line.startswith('1 ') or line.startswith('2 ') or line.startswith('Introduction') or line.startswith('1. '):
                    break
                abstract_lines.append(line.strip())

        if abstract_lines:
            abstract = ' '.join(abstract_lines[:15])
            return abstract[:800] if abstract else None

        # 没找到Abstract，取前500字
        return text[:500] if text else None

    except Exception as e:
        return None

def score_by_abstract(abstract):
    """根据摘要内容打分"""
    if not abstract:
        return 2

    text = abstract.lower()

    # 5分：PM2.5 + CMAQ/融合/克里金
    if 'pm2.5' in text and ('cmaq' in text or 'chemical transport' in text or 'ctm' in text):
        return 5
    if 'pm2.5' in text and 'downscaling' in text and ('kriging' in text or 'fusion' in text):
        return 5
    if 'pm2.5' in text and 'data fusion' in text:
        return 5
    if 'pm2.5' in text and 'kriging' in text and ('spatiotemporal' in text or 'spatial' in text):
        return 5
    if 'pm2.5' in text and 'spatiotemporal' in text and ('monitor' in text or 'ground' in text):
        return 5
    if 'pm2.5' in text and 'bayesian' in text and 'fusion' in text:
        return 5
    if 'pm2.5' in text and 'variational' in text and 'fusion' in text:
        return 5

    # 4分：PM2.5 或空气质量相关
    if 'pm2.5' in text:
        return 4
    if 'air quality' in text and ('pm' in text or 'pollution' in text):
        return 4
    if 'aerosol optical depth' in text and ('prediction' in text or 'estimation' in text):
        return 4
    if 'surface pm2.5' in text or 'ground-level pm2.5' in text:
        return 4
    if 'pm2.5' in text and 'estimation' in text:
        return 4

    # 3分：机器学习/深度学习/气象相关
    if any(kw in text for kw in ['prediction', 'forecasting', 'machine learning', 'deep learning',
                                   'neural network', 'interpolation', 'atmospheric',
                                   'weather', 'climate', 'satellite', 'aod', 'aerosol']):
        return 3

    return 2

# 创建文件夹
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    os.makedirs(folder, exist_ok=True)

# 获取所有PDF（包括score_X文件夹里的）
all_pdfs = []
for root, dirs, files in os.walk(PAPER_DOWNLOAD):
    for f in files:
        if f.endswith('.pdf'):
            all_pdfs.append(os.path.join(root, f))

# 先把之前的分类移回根目录
print("Resetting folders...")
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.pdf'):
                src = os.path.join(folder, f)
                dst = os.path.join(PAPER_DOWNLOAD, f)
                if src != dst:
                    shutil.move(src, dst)

print(f"Total PDFs to classify: {len(all_pdfs)}")
print()

stats = {2: 0, 3: 0, 4: 0, 5: 0}
errors = 0

for i, pdf_path in enumerate(all_pdfs):
    fname = os.path.basename(pdf_path)
    print(f"[{i+1}/{len(all_pdfs)}] {fname[:60]}...")

    try:
        # 提取摘要
        abstract = extract_abstract(pdf_path)

        # 打分
        score = score_by_abstract(abstract)

        print(f"  Score: {score}")
        if abstract:
            print(f"  Abstract: {abstract[:80]}...")
    except Exception as e:
        print(f"  Error: {e}")
        score = 2
        errors += 1

    # 移动到对应文件夹
    dst = os.path.join(PAPER_DOWNLOAD, f'score_{score}', fname)
    if pdf_path != dst:
        shutil.move(pdf_path, dst)

    stats[score] += 1

print()
print("=" * 50)
print("Classification Complete")
print("=" * 50)
print(f"Score 5: {stats[5]}篇")
print(f"Score 4: {stats[4]}篇")
print(f"Score 3: {stats[3]}篇")
print(f"Score 2: {stats[2]}篇")
print(f"Errors: {errors}")
print(f"Total: {len(all_pdfs)}篇")