"""
Agent Spawn 执行器
================
在主会话中实际执行 Agent spawn 的脚本

使用方式：
    from agents.spawn_executor import SpawnExecutor
    executor = SpawnExecutor(project_root)

    # Phase 1: 并行下载
    executor.phase1_download()

    # Phase 2: 文献分析
    executor.phase2_analyze()

    # ... 后续 Phase

注意：实际 spawn 需要在 Claude Code 主会话中通过 Agent 工具执行
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from agents.role_templates import get_spawn_prompt


class SpawnExecutor:
    """
    Agent Spawn 执行器

    管理工作流状态，并在主会话中协调 Agent spawn
    """

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.state_file = os.path.join(project_root, '.agent_state.json')
        self.error_dir = os.path.join(project_root, 'error')

        # 确保目录存在
        os.makedirs(self.error_dir, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'round': 0,
            'agents': {},
            'innovation_established': False,
            'iteration_count': 0,
            'no_improvement_count': 0,
            'terminated': False,
            'last_run': None,
        }

    def _save_state(self):
        self.state['last_run'] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def get_agent_prompt(self, role: str) -> str:
        """获取指定角色的 spawn prompt"""
        return get_spawn_prompt(role, self.project_root)

    def spawn(self, agent_id: str, role: str, background: bool = False) -> Dict:
        """
        Spawn 一个 Agent

        返回 spawn 信息，供主会话使用 Agent 工具调用
        """
        prompt = self.get_agent_prompt(role)

        self.state['agents'][agent_id] = {
            'role': role,
            'status': 'spawned',
            'spawned_at': datetime.now().isoformat(),
            'background': background,
        }
        self._save_state()

        return {
            'agent_id': agent_id,
            'role': role,
            'background': background,
            'prompt': prompt,
        }

    def mark_completed(self, agent_id: str, result: any = None):
        """标记 Agent 完成"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'completed'
            self.state['agents'][agent_id]['completed_at'] = datetime.now().isoformat()
            if result:
                self.state['agents'][agent_id]['result'] = result
            self._save_state()

    def mark_failed(self, agent_id: str, error: str):
        """标记 Agent 失败"""
        if agent_id in self.state['agents']:
            self.state['agents'][agent_id]['status'] = 'failed'
            self.state['agents'][agent_id]['failed_at'] = datetime.now().isoformat()
            self.state['agents'][agent_id]['error'] = error
            self._save_state()

    def wait_and_check(self, agent_ids: List[str]) -> bool:
        """检查所有 Agent 是否完成（基于状态文件，非阻塞）"""
        for agent_id in agent_ids:
            state = self.state['agents'].get(agent_id, {}).get('status')
            if state != 'completed':
                return False
        return True

    def get_pending_agents(self) -> List[str]:
        """获取所有未完成的 Agent ID 列表"""
        pending = []
        for agent_id, info in self.state['agents'].items():
            if info.get('status') not in ['completed', 'failed']:
                pending.append(agent_id)
        return pending

    def is_any_running(self) -> bool:
        """检查是否有任何 Agent 还在运行"""
        for info in self.state['agents'].values():
            if info.get('status') not in ['completed', 'failed']:
                return True
        return False

    def get_state(self) -> Dict:
        """获取当前状态"""
        return self.state

    def verify_agent_output(self, agent_id: str, output_paths: List[str]) -> bool:
        """
        健康检查：验证 Agent 是否真正产生了预期输出

        Parameters:
        -----------
        agent_id : str
            Agent ID
        output_paths : List[str]
            预期输出文件路径列表

        Returns:
        --------
        bool : 所有文件都存在返回 True，否则返回 False
        """
        missing = []
        for path in output_paths:
            if not os.path.exists(path):
                missing.append(path)

        if missing:
            print(f"  [警告] {agent_id} 缺少输出文件:")
            for p in missing:
                print(f"    - {p}")
            return False
        return True

    def retry_agent(self, agent_id: str, role: str, max_retries: int = 3) -> Dict:
        """
        重试失败的 Agent

        Parameters:
        -----------
        agent_id : str
            Agent ID
        role : str
            角色名
        max_retries : int
            最大重试次数

        Returns:
        --------
        Dict : spawn 结果
        """
        # 标记当前为失败
        self.mark_failed(agent_id, "需要重试")

        for attempt in range(max_retries):
            print(f"  [重试] {agent_id} 第 {attempt + 1} 次重试...")

            # 生成新的 agent_id
            new_agent_id = f"{agent_id}_retry_{attempt + 1}"
            result = self.spawn(new_agent_id, role, background=True)

            # 更新原 agent_id 的重试信息
            if agent_id in self.state['agents']:
                self.state['agents'][agent_id]['retry_count'] = attempt + 1
                self.state['agents'][agent_id]['latest_retry_id'] = new_agent_id
                self._save_state()

            return result

        print(f"  [错误] {agent_id} 重试 {max_retries} 次后仍失败")
        return None

    def check_and_retry(self, agent_id: str, role: str, output_paths: List[str], max_retries: int = 3) -> Dict:
        """
        检查 Agent 输出，必要时重试

        Parameters:
        -----------
        agent_id : str
            Agent ID
        role : str
            角色名
        output_paths : List[str]
            预期输出文件路径
        max_retries : int
            最大重试次数

        Returns:
        --------
        Dict : spawn 结果（如果是新的 retry）
        """
        if self.verify_agent_output(agent_id, output_paths):
            return None  # 输出正常，不需要重试

        print(f"  [触发重试] {agent_id} 输出验证失败")
        return self.retry_agent(agent_id, role, max_retries)

    # ====== Phase 执行方法 ======

    def phase0_organize(self) -> Dict:
        """
        Phase 0: 项目整理
        进入项目后首先执行，整理前人遗留，生成盘点报告
        """
        print("\n" + "="*60)
        print("Phase 0: 项目整理")
        print("="*60)

        result = self.spawn('organizer', 'organizer', background=True)
        print(f"  [准备 Spawn] organizer (background=True)")
        print(f"    Role: organizer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase1_download(self) -> Dict[str, Dict]:
        """
        Phase 1: 并行下载
        Spawn 3个下载Agent
        """
        print("\n" + "="*60)
        print("Phase 1: 并行文献下载")
        print("="*60)

        results = {}
        for i in range(1, 4):
            agent_id = f'dl_{i}'
            result = self.spawn(agent_id, 'literature_downloader', background=True)
            results[agent_id] = result
            print(f"  [准备 Spawn] {agent_id} (background=True)")
            print(f"    Role: literature_downloader")
            print(f"    Prompt 长度: {len(result['prompt'])} chars")

        print(f"\n  请在主会话中使用 Agent 工具执行上述 Agent")
        print(f"  执行完成后调用 mark_completed() 更新状态")

        return results

    def phase2_analyze(self) -> Dict:
        """
        Phase 2: 文献分析
        等下载完成后执行
        """
        print("\n" + "="*60)
        print("Phase 2: 文献分析")
        print("="*60)

        # 检查下载是否完成
        if not self.wait_and_check(['dl_1', 'dl_2', 'dl_3']):
            print("  错误: 下载阶段未完成")
            return None

        result = self.spawn('analyzer', 'literature_analyzer', background=True)
        print(f"  [准备 Spawn] analyzer (background=True)")
        print(f"    Role: literature_analyzer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase3_design(self) -> Dict:
        """
        Phase 3: 方案设计
        等分析完成后执行
        """
        print("\n" + "="*60)
        print("Phase 3: 方案设计")
        print("="*60)

        if not self.wait_and_check(['analyzer']):
            print("  错误: 分析阶段未完成")
            return None

        result = self.spawn('designer', 'method_designer', background=True)
        print(f"  [准备 Spawn] designer (background=True)")
        print(f"    Role: method_designer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase4_code(self) -> Dict:
        """
        Phase 4: 代码实现
        等设计完成后执行
        """
        print("\n" + "="*60)
        print("Phase 4: 代码实现")
        print("="*60)

        if not self.wait_and_check(['designer']):
            print("  错误: 设计阶段未完成")
            return None

        result = self.spawn('engineer', 'code_engineer', background=True)
        print(f"  [准备 Spawn] engineer (background=True)")
        print(f"    Role: code_engineer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase5_test(self) -> Dict:
        """
        Phase 5: 测试验证
        等代码完成后执行
        """
        print("\n" + "="*60)
        print("Phase 5: 测试验证")
        print("="*60)

        if not self.wait_and_check(['engineer']):
            print("  错误: 代码阶段未完成")
            return None

        result = self.spawn('verifier', 'test_verifier', background=True)
        print(f"  [准备 Spawn] verifier (background=True)")
        print(f"    Role: test_verifier")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result

    def phase6_write(self) -> Optional[Dict]:
        """
        Phase 6: 技术写作
        创新成立时执行
        """
        print("\n" + "="*60)
        print("Phase 6: 技术写作")
        print("="*60)

        if not self.state.get('innovation_established'):
            print("  跳过: 创新未成立")
            return None

        if not self.wait_and_check(['verifier']):
            print("  错误: 验证阶段未完成")
            return None

        result = self.spawn('writer', 'technical_writer', background=True)
        print(f"  [准备 Spawn] writer (background=True)")
        print(f"    Role: technical_writer")
        print(f"    Prompt 长度: {len(result['prompt'])} chars")

        return result


def print_spawn_guide():
    """打印 Agent spawn 执行指南"""
    guide = """
================================================================================
                    Agent Spawn 执行指南
================================================================================

【核心概念】

  Claude Code 的 Agent 工具可以 spawn 子 Agent独立运行
  主会话负责任务协调，子 Agent负责具体执行

【执行模式】

  1. 初始化执行器
  2. 调用 Phase 方法获取 spawn 信息
  3. 使用 Agent 工具 spawn 子 Agent
  4. 子 Agent 完成后调用 mark_completed()
  5. 继续下一个 Phase

【代码示例】

  from agents.spawn_executor import SpawnExecutor

  executor = SpawnExecutor(project_root)

  # Phase 1: 并行下载
  spawns = executor.phase1_download()
  for agent_id, info in spawns.items():
      # 使用 Agent 工具 spawn
      Agent(tool_call="...", prompt=info['prompt'])

  # ... 等待完成后

  # Phase 2: 文献分析
  executor.mark_completed('dl_1')
  executor.mark_completed('dl_2')
  executor.mark_completed('dl_3')

  info = executor.phase2_analyze()
  Agent(tool_call="...", prompt=info['prompt'])

【Agent 工具调用格式】

  Agent(
      description="文献下载 Agent dl_1",
      prompt="你是一个专业的学术论文搜索专家。..."
  )

【注意事项】

  - 后台 Agent (background=True) 可以并行运行
  - 前台 Agent (background=False) 顺序执行
  - 每个 Phase 完成后需要调用 mark_completed()
  - 状态保存在 .agent_state.json

================================================================================
"""
    print(guide)


if __name__ == '__main__':
    project_root = 'E:/CodeProject/ClaudeRoom/Data_Fusion_AutoResearch'

    print(f"初始化项目: {project_root}")
    executor = SpawnExecutor(project_root)

    print("\n当前状态:")
    print(json.dumps(executor.state, indent=2, ensure_ascii=False))

    print_spawn_guide()
