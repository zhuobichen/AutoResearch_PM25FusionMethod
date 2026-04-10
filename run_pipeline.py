# -*- coding: utf-8 -*-
"""
PM2.5 CMAQ融合方法自动研究流程
================================
一键执行完整的工作流

使用方式：
    python run_pipeline.py

退出：
    Ctrl+C 或 在出现提示时输入 q
"""

import os
import sys
import time
import signal
from datetime import datetime

# 确保可以导入agents
sys.path.insert(0, os.path.dirname(__file__))

from agents.spawn_executor import SpawnExecutor
from test_result.snapshot_manager import SnapshotManager


class PipelineRunner:
    """工作流运行器"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.executor = SpawnExecutor(project_root)
        self.snapshot_mgr = SnapshotManager(project_root)
        self.running = True

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理Ctrl+C"""
        print("\n\n[中断] 收到停止信号，正在保存状态...")
        self.running = False
        self._save_and_exit()

    def _save_and_exit(self):
        """保存状态并退出"""
        print(f"\n[状态] 已保存快照")
        print(f"[下一步] 下次运行时 python run_pipeline.py 继续\n")
        sys.exit(0)

    def print_banner(self):
        """打印banner"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║     PM2.5 CMAQ融合方法自动研究流程 v10.1                     ║
║     自动化文献搜索 → 方法设计 → 代码实现 → 测试验证          ║
╚══════════════════════════════════════════════════════════════╝
""")

    def print_status(self):
        """打印当前状态"""
        state = self.executor.get_state()
        snapshot_state = self.snapshot_mgr.load_dedup_state()

        print(f"""
=== 当前状态 ===
快照轮次: {state.get('round', 'N/A')}
已分析方法: {len(snapshot_state.get('analyzed_methods', []))}
已有指纹: {len(snapshot_state.get('method_fingerprints', []))}
当前最佳: {snapshot_state.get('best_method', 'None')} (R2={snapshot_state.get('best_r2', 'N/A')})
""")

    def wait_for_agent(self, agent_id: str, timeout: int = 300) -> bool:
        """等待Agent完成（轮询方式）"""
        print(f"  [等待] {agent_id} 运行中...")
        start = time.time()

        while self.running:
            elapsed = time.time() - start
            if elapsed > timeout:
                print(f"  [超时] {agent_id} 运行超过 {timeout} 秒")
                return False

            # 检查状态
            if self.executor.wait_and_check([agent_id]):
                print(f"  [完成] {agent_id}")
                return True

            time.sleep(5)  # 每5秒检查一次

        return False

    def wait_for_agents(self, agent_ids: list, timeout: int = 600) -> bool:
        """等待多个Agent完成"""
        print(f"  [等待] {len(agent_ids)} 个Agent运行中...")

        start = time.time()
        while self.running:
            elapsed = time.time() - start
            if elapsed > timeout:
                print(f"  [超时] 运行超过 {timeout} 秒")
                return False

            if self.executor.wait_and_check(agent_ids):
                print(f"  [完成] 全部Agent")
                return True

            time.sleep(5)

        return False

    def phase1_download(self):
        """Phase 1: 并行下载"""
        print("\n" + "="*60)
        print("Phase 1: 并行文献下载")
        print("="*60)

        results = self.executor.phase1_download()

        for agent_id, info in results.items():
            print(f"  [Spawn] {agent_id}")
            print(f"    - Role: {info['role']}")
            print(f"    - Background: {info.get('background', False)}")

        # 并行等待
        agent_ids = list(results.keys())
        self.wait_for_agents(agent_ids, timeout=600)

        return results

    def phase2_analyze(self):
        """Phase 2: 文献分析"""
        print("\n" + "="*60)
        print("Phase 2: 文献分析")
        print("="*60)

        result = self.executor.phase2_analyze()
        print(f"  [Spawn] analyzer")

        self.wait_for_agent('analyzer', timeout=600)

        return result

    def phase3_design(self):
        """Phase 3: 方案设计"""
        print("\n" + "="*60)
        print("Phase 3: 方案设计")
        print("="*60)

        result = self.executor.phase3_design()
        print(f"  [Spawn] designer")

        self.wait_for_agent('designer', timeout=600)

        return result

    def phase4_code(self):
        """Phase 4: 代码实现"""
        print("\n" + "="*60)
        print("Phase 4: 代码实现")
        print("="*60)

        result = self.executor.phase4_code()
        print(f"  [Spawn] engineer")

        self.wait_for_agent('engineer', timeout=600)

        return result

    def phase5_test(self):
        """Phase 5: 测试验证"""
        print("\n" + "="*60)
        print("Phase 5: 测试验证")
        print("="*60)

        result = self.executor.phase5_test()
        print(f"  [Spawn] verifier")

        self.wait_for_agent('verifier', timeout=600)

        return result

    def check_innovation(self) -> bool:
        """检查创新是否成立"""
        state = self.executor.get_state()

        if state.get('innovation_established'):
            print("\n  [OK] 创新已成立！")
            return True

        no_improve = state.get('no_improvement_count', 0)
        if no_improve >= 3:
            print(f"\n  [WARN] 连续 {no_improve} 次无显著提升")

        return False

    def phase6_write(self):
        """Phase 6: 技术写作"""
        print("\n" + "="*60)
        print("Phase 6: 技术写作")
        print("="*60)

        result = self.executor.phase6_write()
        print(f"  [Spawn] writer")
        print(f"  [提示] writer在后台运行，完成后论文将输出到 paper_output/")

        return result

    def run_iteration(self, iteration: int):
        """运行一次完整迭代"""
        print(f"\n{'='*60}")
        print(f"迭代 #{iteration}")
        print(f"{'='*60}")

        # 创建新快照
        snapshot_state = self.snapshot_mgr.load_dedup_state()
        note = snapshot_state.get('note', '')
        self.snapshot_mgr.create_snapshot(round_num=iteration, note=note)
        self.executor.state['round'] = iteration
        self.executor._save_state()

        # Phase 1-5
        self.phase1_download()
        self.phase2_analyze()
        self.phase3_design()
        self.phase4_code()
        self.phase5_test()

        # 检查创新
        if self.check_innovation():
            self.phase6_write()
            return True

        return False

    def run(self):
        """运行完整流程"""
        self.print_banner()
        self.print_status()

        iteration = 1
        max_iterations = 20

        while self.running and iteration <= max_iterations:
            try:
                # 运行一次迭代
                innovation_done = self.run_iteration(iteration)

                if innovation_done:
                    print("\n" + "="*60)
                    print("[完成] 研究完成！创新已成立，论文正在生成。")
                    print("="*60)
                    break

                # 打印提示
                print("\n" + "="*60)
                print("本次迭代未达到创新标准")
                print("="*60)
                print(f"\n当前最佳: {self.snapshot_mgr.load_dedup_state().get('best_method', 'N/A')}")
                print(f"R2: {self.snapshot_mgr.load_dedup_state().get('best_r2', 'N/A')}")
                print(f"\n是否继续下一轮迭代？")
                print(f"  - 输入 y + 回车: 继续下一轮")
                print(f"  - 输入 n + 回车: 退出（下次可继续）")
                print(f"  - 直接回车: 继续下一轮")

                response = input("\n请输入 (y/n): ").strip().lower()

                if response == 'n':
                    print("\n[退出] 状态已保存，下次运行可继续")
                    break

                iteration += 1

            except Exception as e:
                print(f"\n[错误] 迭代 {iteration} 执行出错: {e}")
                import traceback
                traceback.print_exc()
                print("\n是否继续？ (y/n)")
                response = input().strip().lower()
                if response == 'n':
                    break
                iteration += 1

        print(f"\n[完成] 共运行 {iteration} 轮")
        print(f"[状态] 快照已保存到 snapshots/")


def main():
    """主函数"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    print(f"项目目录: {project_root}")

    runner = PipelineRunner(project_root)
    runner.run()


if __name__ == '__main__':
    main()
