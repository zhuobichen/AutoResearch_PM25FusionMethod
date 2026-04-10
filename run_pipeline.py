"""
一键启动完整 Agent 工作流
=========================

用法：
    python run_pipeline.py

效果：
    自动按顺序执行 Phase 0 → 1 → 2 → 3 → 4 → 5 → 6
    每个 Phase 完成后自动触发下一个
"""

import subprocess
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from agents.spawn_executor import SpawnExecutor


def run_command(cmd: str) -> str:
    """执行命令并返回输出"""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or '.'
    )
    return result.stdout + result.stderr


def trigger_and_print(executor: SpawnExecutor) -> bool:
    """触发下一个 Agent，打印信息"""
    result = executor.trigger_next()

    if result is None:
        print("\n" + "="*60)
        print("所有 Phase 已完成！工作流结束。")
        print("="*60)
        return False  # 停止循环

    print("\n" + "="*60)
    print(f"下一步: {result['agent_id']} ({result['role']})")
    print("="*60)

    # 返回 True 表示有下一步，继续循环
    return True


def main():
    project_root = os.path.dirname(__file__) or '.'

    print("="*60)
    print("PM2.5 CMAQ 融合方法研究 Agent 工作流")
    print("="*60)
    print()
    print("启动自动执行...")
    print()

    executor = SpawnExecutor(project_root)

    # 循环触发直到完成
    while trigger_and_print(executor):
        print("\n" + "-"*40)
        print("提示: 上述 Agent 执行完成后，再次运行本脚本继续下一个 Phase")
        print("-"*40)
        break  # 暂时需要手动再次运行，因为无法自动检测 Agent 完成

    print("\n" + "="*60)
    print("当前状态:")
    print("="*60)
    status = executor.get_status()
    for agent_id, info in status['agents'].items():
        skipped = " (skipped)" if info.get('skipped') else ""
        print(f"  {agent_id}: {info['status']}{skipped}")


if __name__ == "__main__":
    main()
