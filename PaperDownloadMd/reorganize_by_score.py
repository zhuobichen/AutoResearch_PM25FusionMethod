# -*- coding: utf-8 -*-
"""
按评分整理论文PDF到不同文件夹
Score 3, 4, 5 -> 各自文件夹
"""
import os
import json
import shutil

ROOT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
PAPER_LIST = f'{ROOT_DIR}/PaperDownloadMd/paper_list.json'
PAPER_DOWNLOAD = f'{ROOT_DIR}/PaperDownload'

# 读取 paper_list.json
with open(PAPER_LIST, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建评分文件夹
for score in [3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    os.makedirs(folder, exist_ok=True)

# 统计
stats = {3: 0, 4: 0, 5: 0}
moved = []
not_found = []

for paper in data['papers']:
    score = paper.get('score')
    if score not in [3, 4, 5]:
        continue

    pdf_rel = paper.get('pdf_file', '')
    if not pdf_rel:
        continue

    # 转换相对路径为绝对路径
    pdf_src = pdf_rel.replace('PaperDownload/', f'{PAPER_DOWNLOAD}/')
    pdf_src = os.path.join(ROOT_DIR, pdf_rel.replace('PaperDownload/', ''))
    pdf_src = f'{PAPER_DOWNLOAD}/{os.path.basename(pdf_rel)}'

    if not os.path.exists(pdf_src):
        not_found.append(os.path.basename(pdf_src))
        continue

    # 目标文件夹
    dst_folder = f'{PAPER_DOWNLOAD}/score_{score}'
    pdf_dst = f'{dst_folder}/{os.path.basename(pdf_src)}'

    # 移动
    shutil.move(pdf_src, pdf_dst)
    stats[score] += 1
    moved.append((os.path.basename(pdf_src), score))

print("=" * 50)
print("按评分整理论文PDF")
print("=" * 50)
print(f"Score 3: {stats[3]} 篇")
print(f"Score 4: {stats[4]} 篇")
print(f"Score 5: {stats[5]} 篇")
print(f"总计: {sum(stats.values())} 篇")
print()

if not_found:
    print(f"未找到文件 ({len(not_found)}):")
    for f in not_found[:10]:
        print(f"  - {f}")

print(f"\n整理完成！PDF已移动到:")
print(f"  {PAPER_DOWNLOAD}/score_3/")
print(f"  {PAPER_DOWNLOAD}/score_4/")
print(f"  {PAPER_DOWNLOAD}/score_5/")