# -*- coding: utf-8 -*-
"""
对 PaperDownload 下所有未分类的 PDF 进行打分和分类
"""
import os
import json
import shutil

ROOT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
PAPER_DOWNLOAD = f'{ROOT_DIR}/PaperDownload'
PAPER_LIST_JSON = f'{ROOT_DIR}/PaperDownloadMd/paper_list.json'

# 评分关键词
SCORE_KEYWORDS = {
    5: ['PM2.5', 'CMAQ', 'downscaling', 'kriging', 'data fusion', 'spatiotemporal'],
    4: ['air quality', 'PM2.5', 'spatial', 'temporal', 'pollution', 'aerosol', 'monitoring'],
    3: ['kriging', 'interpolation', 'prediction', 'machine learning', 'deep learning', 'neural'],
}

def score_paper(filename):
    """根据文件名打分"""
    filename_lower = filename.lower()

    # 5分：直接相关
    if all(kw in filename_lower for kw in ['pm2.5', 'cmaq']):
        return 5
    if all(kw in filename_lower for kw in ['pm2.5', 'downscaling']):
        return 5
    if all(kw in filename_lower for kw in ['pm2.5', 'kriging']):
        return 5
    if all(kw in filename_lower for kw in ['pm2.5', 'fusion']):
        return 5
    if all(kw in filename_lower for kw in ['pm2.5', 'spatiotemporal']):
        return 5

    # 4分：高度相关
    if 'pm2.5' in filename_lower:
        return 4
    if 'air quality' in filename_lower:
        return 4
    if 'cmaq' in filename_lower:
        return 4
    if 'downscaling' in filename_lower:
        return 4
    if 'kriging' in filename_lower:
        return 4
    if 'aerosol' in filename_lower:
        return 4

    # 3分：部分相关
    if any(kw in filename_lower for kw in ['spatial', 'temporal', 'prediction', 'machine learning', 'neural', 'deep learning']):
        return 3
    if any(kw in filename_lower for kw in ['pollution', 'atmospheric', 'weather']):
        return 3

    return 2  # 不相关

# 读取已有的 paper_list
existing = {}
try:
    with open(PAPER_LIST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for paper in data.get('papers', []):
        pdf_file = paper.get('pdf_file', '')
        score = paper.get('score', 0)
        if pdf_file:
            # 提取文件名
            fname = os.path.basename(pdf_file)
            existing[fname] = score
except:
    existing = {}

# 创建评分文件夹
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    os.makedirs(folder, exist_ok=True)

# 扫描所有PDF
all_pdfs = [f for f in os.listdir(PAPER_DOWNLOAD) if f.endswith('.pdf')]

stats = {2: 0, 3: 0, 4: 0, 5: 0}
skipped = 0

for pdf in all_pdfs:
    src = f'{PAPER_DOWNLOAD}/{pdf}'

    # 跳过已有分类的
    if '/score_' in src or '\\score_' in src:
        skipped += 1
        continue

    # 从文件名提取判断分数
    score = score_paper(pdf)

    # 移动到对应文件夹
    dst = f'{PAPER_DOWNLOAD}/score_{score}/{pdf}'
    if src != dst:
        shutil.move(src, dst)

    stats[score] += 1

print("=" * 50)
print("PDF 分类完成")
print("=" * 50)
print(f"Score 5 (直接相关): {stats[5]} 篇")
print(f"Score 4 (高度相关): {stats[4]} 篇")
print(f"Score 3 (部分相关): {stats[3]} 篇")
print(f"Score 2 (不相关): {stats[2]} 篇")
print(f"已分类: {sum(stats.values())} 篇")
print(f"跳过(已分类): {skipped} 篇")
print(f"总计: {sum(stats.values()) + skipped} 篇")