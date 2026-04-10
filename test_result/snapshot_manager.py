# -*- coding: utf-8 -*-
"""
快照管理器
=========
负责：
1. 创建/恢复快照
2. 追踪去重状态
3. 元数据管理
"""

import os
import json
import shutil
import hashlib
from datetime import datetime


class SnapshotManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.snapshot_dir = os.path.join(root_dir, 'test_result', 'snapshots')
        self.state_dir = os.path.join(root_dir, 'test_result', '.state')

        # 当前指针文件
        self.current_file = os.path.join(self.state_dir, 'current.txt')

        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)

    # ========== 快照基础 ==========

    def create_snapshot(self, round_num, note=""):
        """创建新快照"""
        snapshot_name = f"round_{round_num}"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)

        if os.path.exists(snapshot_path):
            print(f"快照 {snapshot_name} 已存在，覆盖")
        else:
            os.makedirs(snapshot_path)

        # 创建元数据
        metadata = {
            "round": round_num,
            "created": datetime.now().isoformat(),
            "note": note,
            "parent": self.get_current_round(),
            "downloaded_papers": [],
            "analyzed_methods": [],
            "method_fingerprints": [],
            "best_method": None,
            "best_r2": None
        }

        # 保存元数据
        metadata_file = os.path.join(snapshot_path, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 更新当前指针
        self._update_current(snapshot_name)

        print(f"快照已创建: {snapshot_name}")
        return snapshot_path

    def restore_snapshot(self, round_num=None):
        """恢复快照"""
        if round_num is None:
            round_num = self.get_current_round()

        snapshot_name = f"round_{round_num}"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)

        if not os.path.exists(snapshot_path):
            print(f"快照 {snapshot_name} 不存在")
            return None

        print(f"恢复快照: {snapshot_name}")
        return snapshot_path

    def get_current_round(self):
        """获取当前轮次"""
        if os.path.exists(self.current_file):
            with open(self.current_file, 'r') as f:
                return f.read().strip()
        return None

    def get_latest_snapshot(self):
        """获取最新快照"""
        snapshots = [d for d in os.listdir(self.snapshot_dir)
                     if d.startswith('round_')]
        if not snapshots:
            return None
        # 按数字排序
        snapshots.sort(key=lambda x: int(x.split('_')[1]))
        return snapshots[-1]

    def _update_current(self, snapshot_name):
        """更新当前指针"""
        with open(self.current_file, 'w') as f:
            f.write(snapshot_name)

    # ========== 去重追踪 ==========

    def load_dedup_state(self):
        """加载去重状态"""
        snapshot_path = self.restore_snapshot()
        if not snapshot_path:
            return {
                "downloaded_papers": [],
                "analyzed_methods": [],
                "method_fingerprints": []
            }

        metadata_file = os.path.join(snapshot_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_dedup_state(self, state):
        """保存去重状态"""
        snapshot_name = self.get_current_round()
        if not snapshot_name:
            print("错误：没有当前快照")
            return

        snapshot_path = os.path.join(self.snapshot_dir, snapshot_name)
        metadata_file = os.path.join(snapshot_path, 'metadata.json')

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        metadata.update(state)

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def check_paper_dedup(self, title, authors):
        """检查论文是否已下载"""
        state = self.load_dedup_state()
        dedup_key = self._compute_dedup_key(title, authors)
        return dedup_key in state.get("downloaded_papers", [])

    def add_downloaded_paper(self, title, authors):
        """标记论文已下载"""
        state = self.load_dedup_state()
        dedup_key = self._compute_dedup_key(title, authors)
        if "downloaded_papers" not in state:
            state["downloaded_papers"] = []
        if dedup_key not in state["downloaded_papers"]:
            state["downloaded_papers"].append(dedup_key)
        self.save_dedup_state(state)

    def check_method_analyzed(self, method_name):
        """检查方法是否已分析"""
        state = self.load_dedup_state()
        return method_name in state.get("analyzed_methods", [])

    def add_analyzed_method(self, method_name):
        """标记方法已分析"""
        state = self.load_dedup_state()
        if "analyzed_methods" not in state:
            state["analyzed_methods"] = []
        if method_name not in state["analyzed_methods"]:
            state["analyzed_methods"].append(method_name)
        self.save_dedup_state(state)

    def check_fingerprint_exists(self, fingerprint):
        """检查指纹是否已存在"""
        state = self.load_dedup_state()
        return fingerprint in state.get("method_fingerprints", [])

    def add_fingerprint(self, fingerprint):
        """添加方法指纹"""
        state = self.load_dedup_state()
        if "method_fingerprints" not in state:
            state["method_fingerprints"] = []
        if fingerprint not in state["method_fingerprints"]:
            state["method_fingerprints"].append(fingerprint)
        self.save_dedup_state(state)

    # ========== 结果更新 ==========

    def update_best_method(self, method_name, metrics):
        """更新最佳方法"""
        state = self.load_dedup_state()
        state["best_method"] = method_name
        state["best_r2"] = metrics.get("R2")
        self.save_dedup_state(state)

    def update_note(self, note):
        """更新备注"""
        state = self.load_dedup_state()
        state["note"] = note
        self.save_dedup_state(state)

    # ========== 工具 ==========

    @staticmethod
    def _compute_dedup_key(title, authors):
        """计算论文去重指纹"""
        content = f"{title}{authors}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @staticmethod
    def compute_method_fingerprint(method_doc):
        """计算方法指纹（基于方法名+核心公式）"""
        content = json.dumps(method_doc, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def print_status(self):
        """打印当前状态"""
        state = self.load_dedup_state()
        snapshot = self.get_current_round()

        print("=== Snapshot Status ===")
        print(f"Current: {snapshot or 'None'}")
        print(f"Downloaded papers: {len(state.get('downloaded_papers', []))}")
        print(f"Analyzed methods: {len(state.get('analyzed_methods', []))}")
        print(f"Method fingerprints: {len(state.get('method_fingerprints', []))}")
        print(f"Best method: {state.get('best_method', 'None')} (R2={state.get('best_r2', 'N/A')})")
        print(f"Note: {state.get('note', '')}")


if __name__ == '__main__':
    root = r'E:\CodeProject\ClaudeRoom\Data_Fusion_AutoResearch'
    manager = SnapshotManager(root)
    manager.print_status()
