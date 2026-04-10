"""
研究状态追踪器 (State Tracker)
参考 AutoSOTA 架构改造用于 PM2.5 CMAQ 融合方法研究

功能：
1. 追踪当前最佳方法及指标
2. 记录失败方法及原因
3. 记录活跃假设和待验证方向
4. 支持结构化 Ledger 输出
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pathlib import Path


class ResearchStatus(Enum):
    """研究状态枚举"""
    RUNNING = "running"
    CONVERGED = "converged"           # 创新成立，收敛
    FAILED = "failed"                 # 连续失败，终止
    EXHAUSTED = "exhausted"           # 创新力耗尽
    MAX_ITERATIONS = "max_iterations" # 达到迭代上限


class MutationStatus(Enum):
    """变异/优化状态"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    NEEDS_ADJUSTMENT = "needs_adjustment"


@dataclass
class MetricRecord:
    """单条指标记录"""
    name: str
    value: float
    timestamp: str
    iteration: int
    method: str = ""


@dataclass
class MutationRecord:
    """方法优化记录"""
    mutation_id: str
    method_name: str
    description: str
    metric_before: Dict[str, float]
    metric_after: Dict[str, float]
    status: str  # accepted, rejected, rolled_back, needs_adjustment
    iteration: int
    failure_reason: str = ""  # 如果被拒绝，记录原因
    code_diff: str = ""       # 代码变更


@dataclass
class HypothesisRecord:
    """待验证假设记录"""
    hypothesis_id: str
    description: str
    source: str              # 来自哪篇论文/分析
    status: str              # pending, validated, rejected
    validation_result: str = ""
    iteration_submitted: int = 0


@dataclass
class ResearchState:
    """全局研究状态"""

    # 基线和当前指标
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    current_best_metrics: Dict[str, float] = field(default_factory=dict)
    current_best_method: str = ""

    # 迭代控制
    iteration: int = 0
    max_iterations: int = 20
    consecutive_no_improvement: int = 0
    max_consecutive_no_improvement: int = 5

    # 状态
    status: str = ResearchStatus.RUNNING.value
    target_improvement: float = 0.01  # R² 提升阈值

    # 历史记录
    accepted_mutations: List[MutationRecord] = field(default_factory=list)
    rejected_mutations: List[MutationRecord] = field(default_factory=list)
    metric_history: List[MetricRecord] = field(default_factory=list)
    active_hypotheses: List[HypothesisRecord] = field(default_factory=list)

    # 失败方法记录（已验证不可行的方法）
    failed_methods: List[Dict[str, Any]] = field(default_factory=list)

    # 当前正在验证的方法
    current_method: str = ""
    current_mutation_id: str = ""


class StateTracker:
    """研究状态追踪器"""

    def __init__(self, output_dir: str = "test_result/.state"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.output_dir / "research_state.json"
        self.ledger_file = self.output_dir / "ledger.jsonl"

        # 加载已有状态或创建新状态
        self.state = self._load_state()

        # Ledger 条目计数器
        self._mutation_counter = 0

    def _load_state(self) -> ResearchState:
        """从文件加载状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 重建 ResearchState
                return ResearchState(**data)
        return ResearchState()

    def _save_state(self):
        """保存状态到文件"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.state), f, indent=2, ensure_ascii=False)

    def _record_ledger(self, event_type: str, data: Dict[str, Any]):
        """记录到 Ledger"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "iteration": self.state.iteration,
            **data
        }
        with open(self.ledger_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ========== 核心方法 ==========

    def initialize(self, baseline_metrics: Dict[str, float], baseline_method: str):
        """初始化研究状态"""
        self.state.baseline_metrics = baseline_metrics.copy()
        self.state.current_best_metrics = baseline_metrics.copy()
        self.state.current_best_method = baseline_method
        self.state.status = ResearchStatus.RUNNING.value
        self.state.iteration = 0

        self._record_ledger("initialize", {
            "baseline_method": baseline_method,
            "baseline_metrics": baseline_metrics
        })
        self._save_state()

    def set_baseline(self, method: str, metrics: Dict[str, float]):
        """设置基准方法（用于对比）"""
        self.state.baseline_metrics = metrics.copy()
        self._record_ledger("set_baseline", {
            "method": method,
            "metrics": metrics
        })
        self._save_state()

    def start_iteration(self, method_name: str) -> str:
        """开始新的方法验证迭代"""
        self.state.iteration += 1
        self.state.current_method = method_name
        self.state.current_mutation_id = f"mut_{self.state.iteration}_{self._mutation_counter}"

        self._record_ledger("iteration_start", {
            "mutation_id": self.state.current_mutation_id,
            "method": method_name,
            "iteration": self.state.iteration
        })
        self._save_state()
        return self.state.current_mutation_id

    def update_metrics(self, metrics: Dict[str, float], method: str = ""):
        """更新当前指标"""
        for name, value in metrics.items():
            record = MetricRecord(
                name=name,
                value=value,
                timestamp=datetime.now().isoformat(),
                iteration=self.state.iteration,
                method=method or self.state.current_method
            )
            self.state.metric_history.append(asdict(record))

        if not method:
            method = self.state.current_method

        self._record_ledger("metrics_update", {
            "method": method,
            "metrics": metrics
        })
        self._save_state()

    def accept_mutation(self, method_name: str, metrics: Dict[str, float],
                       description: str = "", code_diff: str = ""):
        """接受优化，更新最佳方法"""
        self._mutation_counter += 1

        mutation = MutationRecord(
            mutation_id=self.state.current_mutation_id,
            method_name=method_name,
            description=description,
            metric_before=self.state.current_best_metrics.copy(),
            metric_after=metrics.copy(),
            status=MutationStatus.ACCEPTED.value,
            iteration=self.state.iteration,
            code_diff=code_diff
        )

        self.state.accepted_mutations.append(asdict(mutation))
        self.state.current_best_metrics = metrics.copy()
        self.state.current_best_method = method_name
        self.state.consecutive_no_improvement = 0

        self._record_ledger("mutation_accepted", asdict(mutation))
        self._check_convergence()
        self._save_state()

    def reject_mutation(self, method_name: str, metrics: Dict[str, float],
                       reason: str, description: str = ""):
        """拒绝优化，记录失败原因"""
        self._mutation_counter += 1

        mutation = MutationRecord(
            mutation_id=self.state.current_mutation_id,
            method_name=method_name,
            description=description,
            metric_before=self.state.current_best_metrics.copy(),
            metric_after=metrics.copy(),
            status=MutationStatus.REJECTED.value,
            iteration=self.state.iteration,
            failure_reason=reason
        )

        self.state.rejected_mutations.append(asdict(mutation))
        self.state.consecutive_no_improvement += 1

        # 记录失败方法
        self.state.failed_methods.append({
            "method": method_name,
            "reason": reason,
            "iteration": self.state.iteration,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })

        self._record_ledger("mutation_rejected", {
            **asdict(mutation),
            "failure_reason": reason
        })
        self._check_exhaustion()
        self._save_state()

    def rollback(self, mutation_id: str, reason: str = ""):
        """回滚优化"""
        # 找到要回滚的mutation并标记
        for mut in self.state.rejected_mutations:
            if mut.get("mutation_id") == mutation_id:
                mut["status"] = MutationStatus.ROLLED_BACK.value

        self._record_ledger("rollback", {
            "mutation_id": mutation_id,
            "reason": reason
        })
        self._save_state()

    def add_hypothesis(self, description: str, source: str) -> str:
        """添加待验证假设"""
        hypothesis_id = f"hyp_{len(self.state.active_hypotheses) + 1}"
        hypothesis = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            description=description,
            source=source,
            status="pending",
            iteration_submitted=self.state.iteration
        )
        self.state.active_hypotheses.append(asdict(hypothesis))

        self._record_ledger("hypothesis_added", asdict(hypothesis))
        self._save_state()
        return hypothesis_id

    def validate_hypothesis(self, hypothesis_id: str, result: str, status: str = "validated"):
        """验证假设结果"""
        for hyp in self.state.active_hypotheses:
            if hyp.get("hypothesis_id") == hypothesis_id:
                hyp["status"] = status
                hyp["validation_result"] = result

        self._record_ledger("hypothesis_validated", {
            "hypothesis_id": hypothesis_id,
            "result": result,
            "status": status
        })
        self._save_state()

    def _check_convergence(self):
        """检查是否收敛（创新成立）"""
        if not self.state.baseline_metrics:
            return

        r2_baseline = self.state.baseline_metrics.get("R2", 0)
        r2_current = self.state.current_best_metrics.get("R2", 0)

        improvement = r2_current - r2_baseline

        if improvement >= self.state.target_improvement:
            self.state.status = ResearchStatus.CONVERGED.value

    def _check_exhaustion(self):
        """检查是否创新力耗尽"""
        if self.state.consecutive_no_improvement >= self.state.max_consecutive_no_improvement:
            self.state.status = ResearchStatus.EXHAUSTED.value

        if self.state.iteration >= self.state.max_iterations:
            self.state.status = ResearchStatus.MAX_ITERATIONS.value

    # ========== 查询方法 ==========

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前状态摘要"""
        return {
            "iteration": self.state.iteration,
            "status": self.state.status,
            "current_best_method": self.state.current_best_method,
            "current_best_r2": self.state.current_best_metrics.get("R2", 0),
            "baseline_r2": self.state.baseline_metrics.get("R2", 0),
            "improvement": self.state.current_best_metrics.get("R2", 0) - self.state.baseline_metrics.get("R2", 0),
            "consecutive_no_improvement": self.state.consecutive_no_improvement,
            "accepted_count": len(self.state.accepted_mutations),
            "rejected_count": len(self.state.rejected_mutations),
            "failed_methods": [f["method"] for f in self.state.failed_methods[-5:]]  # 最近5个
        }

    def get_failed_methods_summary(self) -> List[Dict[str, Any]]:
        """获取失败方法摘要（用于排除策略）"""
        return self.state.failed_methods[-10:]  # 最近10个

    def is_method_failed(self, method_name: str) -> bool:
        """检查方法是否已被验证失败"""
        return any(f["method"] == method_name for f in self.state.failed_methods)

    def get_failed_reason(self, method_name: str) -> Optional[str]:
        """获取方法失败原因"""
        for f in self.state.failed_methods:
            if f["method"] == method_name:
                return f["reason"]
        return None

    def should_continue(self) -> bool:
        """判断是否应该继续迭代"""
        if self.state.status in [ResearchStatus.CONVERGED.value,
                                  ResearchStatus.EXHAUSTED.value,
                                  ResearchStatus.MAX_ITERATIONS.value]:
            return False
        return True

    def get_next_optimization_direction(self) -> str:
        """基于历史分析，给出下一个优化方向建议"""
        failed_reasons = [f["reason"] for f in self.state.failed_methods[-5:]]

        # 分析失败原因模式
        if any("参数迁移" in r for r in failed_reasons):
            return "建议：优先选择具有物理锚点的固定参数方法，减少数据依赖"
        elif any("数据不足" in r for r in failed_reasons):
            return "建议：引入多源数据（气象、土地利用）增强建模"
        elif any("过拟合" in r for r in failed_reasons):
            return "建议：增加正则化或简化模型复杂度"

        return "建议：继续探索集成学习方法"

    # ========== 报告生成 ==========

    def generate_report(self) -> str:
        """生成研究状态报告"""
        state = self.get_current_state()

        report = f"""
# 研究状态报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 当前状态
- 迭代轮次: {state['iteration']}
- 研究状态: {state['status']}
- 当前最佳方法: {state['current_best_method']}
- 当前最佳 R²: {state['current_best_r2']:.4f}
- 基准 R²: {state['baseline_r2']:.4f}
- R² 提升: {state['improvement']:+.4f}

## 优化历史
- 已接受优化: {state['accepted_count']}
- 已拒绝优化: {state['rejected_count']}
- 连续无提升轮次: {state['consecutive_no_improvement']}

## 失败方法（最近5个）
"""
        for f in self.state.failed_methods[-5:]:
            report += f"- {f['method']}: {f['reason']}\n"

        report += f"""
## 下一个优化方向
{self.get_next_optimization_direction()}

## Ledger 文件
{self.ledger_file}
"""
        return report

    def save_report(self, output_path: str = "test_result/.state/research_status_report.md"):
        """保存报告到文件"""
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)


# ========== 工具函数 ==========

def convert_inventory_to_ledger(inventory_path: str, output_path: str = "test_result/.state/ledger_from_inventory.jsonl"):
    """
    将 INVENTORY.md 格式的研究记录转换为 Ledger JSONL 格式

    用于：初始化 StateTracker 时导入历史数据
    """
    # 读取 INVENTORY.md
    with open(inventory_path, 'r', encoding='utf-8') as f:
        content = f.read()

    ledger_entries = []

    # 解析 INVENTORY.md 中的方法记录
    # 这是一个简化版本，实际使用时需要更复杂的解析逻辑

    # 示例：将已有的失败方法记录转换为 ledger 格式
    # 实际实现需要根据 INVENTORY.md 的具体格式来解析

    # 写入 ledger
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in ledger_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return output_path


# ========== 主程序测试 ==========

if __name__ == "__main__":
    # 测试 StateTracker
    tracker = StateTracker()

    # 初始化基准
    tracker.initialize(
        baseline_metrics={"R2": 0.8100, "MAE": 7.5, "RMSE": 11.0},
        baseline_method="eVNA"
    )

    print("=== 初始化状态 ===")
    print(tracker.get_current_state())

    # 模拟第一个方法验证
    tracker.start_iteration("ResidualKriging")
    tracker.update_metrics({"R2": 0.8273, "MAE": 7.2, "RMSE": 10.8})
    tracker.accept_mutation(
        "ResidualKriging",
        {"R2": 0.8273, "MAE": 7.2, "RMSE": 10.8},
        description="残差克里金方法"
    )

    print("\n=== 第一次优化后 ===")
    print(tracker.get_current_state())

    # 模拟失败的方法
    tracker.start_iteration("CSP-RK")
    tracker.update_metrics({"R2": 0.8520, "MAE": 7.1, "RMSE": 10.9})
    tracker.reject_mutation(
        "CSP-RK",
        {"R2": 0.8520, "MAE": 7.1, "RMSE": 10.9},
        reason="参数迁移性差，跨天平均Δ=-0.00003 << 0.01阈值",
        description="浓度分层多项式克里金"
    )

    print("\n=== CSP-RK 失败后 ===")
    print(tracker.get_current_state())
    print(f"\n失败原因: {tracker.get_failed_reason('CSP-RK')}")
    print(f"\n下一个方向: {tracker.get_next_optimization_direction()}")

    # 保存报告
    tracker.save_report()
    print("\n报告已保存")

    # 显示 Ledger
    print("\n=== Ledger 内容 ===")
    with open(tracker.ledger_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            print(json.dumps(entry, indent=2, ensure_ascii=False))
