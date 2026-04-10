#!/usr/bin/env python3
"""
Download papers from arXiv based on search keywords.
Avoids duplicates by checking existing IDs.
"""

import os
import urllib.request
import urllib.error
import time
import re

DOWNLOAD_DIR = "E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch/PaperDownload"

# Search queries
SEARCH_KEYWORDS = [
    'abs:downscaling AND abs:PM2.5',
    'abs:kriging AND abs:air+quality',
    'abs:CMAQ AND abs:air+quality',
    'abs:PM2.5 AND abs:exposure AND abs:assessment',
    'abs:spatiotemporal AND abs:air+pollution',
    'abs:neural+network AND abs:air+quality',
    'abs:data+fusion AND abs:air+quality',
    'abs:interpolation AND abs:air+pollution',
    'abs:bias+correction AND abs:air+quality',
    'abs:deep+learning AND abs:PM2.5',
]

def load_existing_ids():
    existing = set()
    for f in os.listdir(DOWNLOAD_DIR):
        match = re.match(r'(\d{4}\.\d+)', f)
        if match:
            existing.add(match.group(1))
    return existing

def search_arxiv(query, max_results=50):
    query_encoded = query.replace(' ', '%20')
    url = f"https://export.arxiv.org/api/query?search_query={query_encoded}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Error searching: {e}")
        return []
    ids = re.findall(r'<id>http://arxiv\.org/abs/(\d+\.\d+)v\d+</id>', content)
    return list(set(ids))

def download_paper(paper_id, download_dir):
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    output_path = os.path.join(download_dir, f"{paper_id}.pdf")
    if os.path.exists(output_path):
        return False
    try:
        urllib.request.urlretrieve(pdf_url, output_path)
        print(f"  Downloaded: {paper_id}")
        return True
    except Exception as e:
        print(f"  Failed: {paper_id}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    existing_ids = load_existing_ids()
    print(f"Existing papers: {len(existing_ids)}")

    all_new_ids = set()
    for keyword in SEARCH_KEYWORDS:
        print(f"\nSearching: {keyword}")
        ids = search_arxiv(keyword, max_results=50)
        new_ids = [id_ for id_ in ids if id_ not in existing_ids]
        print(f"  Found {len(ids)}, new: {len(new_ids)}")
        all_new_ids.update(new_ids)

    print(f"\nTotal new papers to download: {len(all_new_ids)}")

    downloaded = 0
    for paper_id in sorted(all_new_ids):
        if download_paper(paper_id, DOWNLOAD_DIR):
            downloaded += 1
            time.sleep(2)

    print(f"\nDownloaded {downloaded} new papers")
    total = len(os.listdir(DOWNLOAD_DIR))
    print(f"Total papers now: {total}")

    # Update list
    all_ids = sorted(existing_ids | all_new_ids)
    list_file = os.path.join(DOWNLOAD_DIR, "paper_list.txt")
    with open(list_file, 'w') as f:
        for id_ in all_ids:
            f.write(f"{id_}\n")
    print(f"Updated: {list_file}")

if __name__ == '__main__':
    main()