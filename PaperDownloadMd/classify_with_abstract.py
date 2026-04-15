# -*- coding: utf-8 -*-
"""
使用 opendataloader-pdf 解析摘要并打分
"""
import os
import json
import shutil
import opendataloader_pdf

ROOT_DIR = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'
PAPER_DOWNLOAD = f'{ROOT_DIR}/PaperDownload'
PAPER_LIST_JSON = f'{ROOT_DIR}/PaperDownloadMd/paper_list.json'
TEMP_DIR = f'{ROOT_DIR}/PaperDownloadMd/temp_extract'

os.makedirs(TEMP_DIR, exist_ok=True)

# 评分关键词（更精确）
SCORE5_KEYWORDS = ['pm2.5', 'cmaq', 'data fusion', 'downscaling', 'kriging', 'spatiotemporal pm2.5']
SCORE4_KEYWORDS = ['pm2.5', 'air quality', 'pollution', 'aerosol', 'spatial interpolation', 'ground monitoring']
SCORE3_KEYWORDS = ['prediction', 'machine learning', 'deep learning', 'neural network', 'interpolation',
                   'atmospheric', 'weather', 'climate', 'satellite']

def score_by_abstract(abstract):
    """根据摘要内容打分"""
    if not abstract:
        return 2

    text = abstract.lower()

    # 5分：PM2.5 + CMAQ/融合/克里金
    if 'pm2.5' in text and ('cmaq' in text or 'chemical transport' in text):
        return 5
    if 'pm2.5' in text and 'downscaling' in text and ('kriging' in text or 'fusion' in text):
        return 5
    if 'pm2.5' in text and 'data fusion' in text:
        return 5
    if 'pm2.5' in text and 'kriging' in text:
        return 5
    if 'pm2.5' in text and 'spatiotemporal' in text and 'monitor' in text:
        return 5

    # 4分：PM2.5 或空气质量相关
    if 'pm2.5' in text:
        return 4
    if 'air quality' in text:
        return 4
    if 'aerosol optical depth' in text and ('prediction' in text or 'estimation' in text):
        return 4

    # 3分：机器学习/深度学习/气象相关
    if any(kw in text for kw in SCORE3_KEYWORDS):
        return 3

    return 2

def extract_abstract(pdf_path):
    """从PDF提取摘要"""
    try:
        # 解析PDF
        opendataloader_pdf.run(
            input_path=pdf_path,
            output_folder=TEMP_DIR,
            generate_markdown=True,
            generate_html=False,
            generate_annotated_pdf=False,
        )

        # 读取生成的markdown
        fname = os.path.basename(pdf_path).replace('.pdf', '.md')
        md_path = os.path.join(TEMP_DIR, fname)

        if os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 尝试提取摘要（找 Abstract 章节）
            abstract = extract_abstract_from_markdown(content)

            # 清理临时文件
            for ext in ['.md', '.json', '.txt']:
                try:
                    os.remove(os.path.join(TEMP_DIR, fname.replace('.pdf', ext)))
                except:
                    pass

            return abstract

        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def extract_abstract_from_markdown(content):
    """从markdown提取摘要"""
    lines = content.split('\n')

    # 找 Abstract 关键词
    in_abstract = False
    abstract_lines = []

    for line in lines:
        line_clean = line.strip().lower()
        if 'abstract' in line_clean and len(line_clean) < 20:
            in_abstract = True
            continue
        if in_abstract:
            if line.startswith('#'):
                break
            abstract_lines.append(line.strip())

    if abstract_lines:
        return ' '.join(abstract_lines[:10])  # 取前10行

    # 如果没找到Abstract，尝试取前500字
    text = content.replace('\n', ' ')[:500]
    return text if text else None

# 读取已有的评分
existing = {}
try:
    with open(PAPER_LIST_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for paper in data.get('papers', []):
        fname = os.path.basename(paper.get('pdf_file', ''))
        score = paper.get('score', 0)
        if fname:
            existing[fname] = score
except:
    pass

# 创建文件夹
for score in [2, 3, 4, 5]:
    folder = f'{PAPER_DOWNLOAD}/score_{score}'
    os.makedirs(folder, exist_ok=True)

# 获取所有PDF
all_pdfs = [f for f in os.listdir(PAPER_DOWNLOAD) if f.endswith('.pdf')]

# 过滤未分类的
unclassified = [f for f in all_pdfs if '/score_' not in f and '\\score_' not in f]

print(f"Total PDFs: {len(all_pdfs)}")
print(f"Unclassified: {len(unclassified)}")
print()

stats = {2: 0, 3: 0, 4: 0, 5: 0}
processed = 0

for i, pdf in enumerate(unclassified):
    src = f'{PAPER_DOWNLOAD}/{pdf}'

    print(f"[{i+1}/{len(unclassified)}] {pdf[:50]}...")

    # 提取摘要
    abstract = extract_abstract(src)

    # 打分
    score = score_by_abstract(abstract)

    print(f"  Score: {score}")
    if abstract:
        print(f"  Abstract: {abstract[:100]}...")

    # 移动到对应文件夹
    dst = f'{PAPER_DOWNLOAD}/score_{score}/{pdf}'
    if src != dst:
        shutil.move(src, dst)

    stats[score] += 1
    processed += 1

print()
print("=" * 50)
print("Classification Complete")
print("=" * 50)
print(f"Score 5: {stats[5]}篇")
print(f"Score 4: {stats[4]}篇")
print(f"Score 3: {stats[3]}篇")
print(f"Score 2: {stats[2]}篇")
print(f"Processed: {processed}篇")