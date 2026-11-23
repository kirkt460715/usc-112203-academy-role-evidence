"""青训晋级证据系统。

模块划分：
- records:   资料装载与完整性、输入关系校验
- model:     分层效应模型（剥离整队/对手效应）与可辨认性连通分量
- evidence:  选手-角色证据档案、版本桥接与排名
- rounds:    决策时点回看与后续窗口检验
- trials:    试训席位安排、组合约束与版本重判
- report:    按人按角色的可解释证据报告
"""

__all__ = [
    "records",
    "model",
    "evidence",
    "rounds",
    "trials",
    "report",
]
