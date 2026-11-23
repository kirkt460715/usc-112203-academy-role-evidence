"""选手-岗位证据档案、版本桥接与排名。

证据组织原则（对应业务要求）：

1. 证据的最小归属是 ``(选手, 岗位)``。同一选手换岗位后产生与新岗位有关的
   独立证据；资料没有证明相通的指标绝不直接拼接（cover 与 sync 从不合并）。
2. 同岗位跨版本只在 ``compatibility.json`` *显式登记* 的
   ``(指标, 地图, 岗位)`` 范围内，沿兼容链借用旧版本证据，并逐条标注来源；
   未登记的组合缺少可借用依据，旧证据只作历史参照、不进入合并效应。
3. 合并采用逆方差加权（各分层效应来自 :mod:`coaching.model`，已剥离场次
   情境），权重 w = 1/se²，并保留每个分层的独立条目供追溯。
4. 可辨认性按分层连通分量逐对判断，再对所有参与合并的分层取交集：
   只要在某分层二人分属不同分量，其相对次序就附带不可辨认提示。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .model import StratumFit
from .records import ROLE_METRIC, Dataset

# 证据充分性门槛（在报告中明示，便于教练组复核与调整）。
# 数值依据资料中每人-岗位的分层片段分布标定：目标版本单岗片段中位数约
# 8-15，8 个片段（约 64 次尝试）配合逆方差合并 se<=0.45 时，效应方向
# 在后续窗口保持；低于该组合的资格一律进入观察并生成试训求证需求。
# - 目标版本累计片段数不少于该值；
# - 合并效应标准误不大于该值（对数几率口径）；
# - 与该岗位主比较分量（含选手最多的分量）可辨认。
MIN_TARGET_FRAGMENTS = 8
MAX_POOLED_SE = 0.45


@dataclass
class StratumEvidence:
    """单个分层对某 (选手, 岗位) 的证据条目。"""

    role: str
    version: str
    map_name: str
    metric: str
    fragments: int
    attempts: int
    successes: int
    raw_rate: float
    effect: float
    std_error: float
    component_id: int
    component_regions: set[str]
    bridged: bool  # 是否经 compatibility 登记从非目标版本借用


@dataclass
class Dossier:
    """一个 (选手, 岗位) 的完整晋级证据档案。"""

    player: str
    role: str
    metric: str
    strata: list[StratumEvidence] = field(default_factory=list)
    pooled_effect: float | None = None
    pooled_se: float | None = None
    target_fragments: int = 0       # 目标版本（不含桥接）片段数
    target_attempts: int = 0
    target_successes: int = 0
    in_main_component: bool = False
    sufficient: bool = False
    # 证据本身充分但受每岗席位数限制未入选：属于备选，不需要新增试训求证。
    seat_limited_alternate: bool = False
    insufficient_reasons: list[str] = field(default_factory=list)
    # 与哪些 (选手, 岗位) 的相对次序不可辨认，及原因
    unidentifiable_with: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.player, self.role)


@dataclass
class BridgeBook:
    """compatibility.json 的可查询视图。"""

    # (metric, map, role) -> 无向兼容边集合中的版本对
    edges: dict[tuple[str, str, str], set[tuple[str, str]]]

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "BridgeBook":
        edges: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
        for c in ds.compatibility:
            for map_name in c["maps"]:
                for role in c["roles"]:
                    key = (c["metric"], map_name, role)
                    edges.setdefault(key, set()).add(
                        tuple(sorted((c["left"], c["right"])))  # type: ignore[arg-type]
                    )
        return cls(edges=edges)

    def reachable_versions(self, role: str, map_name: str, target: str) -> set[str]:
        """沿兼容链从目标版本可达的全部版本（含目标自身）。"""
        key = (ROLE_METRIC[role], map_name, role)
        adj: dict[str, set[str]] = {}
        for a, b in self.edges.get(key, set()):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        seen = {target}
        stack = [target]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):  # 图沿链传播，兼容即可达
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen


def _inverse_variance(pairs: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """对 (效应, 标准误) 序列做逆方差加权，返回 (合并效应, 合并标准误)。"""
    weight_sum = 0.0
    est_sum = 0.0
    for effect, se in pairs:
        w = 1.0 / (se * se)
        weight_sum += w
        est_sum += w * effect
    pooled = est_sum / weight_sum
    return pooled, 1.0 / weight_sum ** 0.5


def build_dossiers(
    ds: Dataset,
    fits: dict[tuple[str, str, str], StratumFit],
    target_version: str,
    bridges: BridgeBook,
) -> dict[tuple[str, str], Dossier]:
    """为资料中出现过的每个 (选手, 岗位) 建立证据档案。"""
    dossiers: dict[tuple[str, str], Dossier] = {}
    # 先确定每个岗位的"主比较分量"（目标版本各地图分量中含节点最多者）。
    main_component: dict[str, set[tuple[str, str]]] = {}
    for role in {k[0] for k in fits}:
        best: set[tuple[str, str]] = set()
        for (r, version, map_name), fit in fits.items():
            if r != role or version != target_version:
                continue
            for members in fit.components.values():
                if len(members) > len(best):
                    best = members
        main_component[role] = best

    for (role, version, map_name), fit in fits.items():
        metric = ROLE_METRIC[role]
        allowed = bridges.reachable_versions(role, map_name, target_version)
        for node, effect in fit.effects.items():
            dossier = dossiers.setdefault(node, Dossier(player=node[0], role=role, metric=metric))
            in_target = version == target_version
            # 非目标版本且无兼容登记：只留在原始分层里，不进入档案合并。
            if not in_target and version not in allowed:
                continue
            entry = StratumEvidence(
                role=role,
                version=version,
                map_name=map_name,
                metric=metric,
                fragments=fit.fragments[node],
                attempts=fit.attempts[node],
                successes=fit.successes[node],
                raw_rate=fit.raw_rates[node],
                effect=effect,
                std_error=fit.std_errors[node],
                component_id=fit.component_of[node],
                component_regions=set(fit.component_regions[fit.component_of[node]]),
                bridged=not in_target,
            )
            dossier.strata.append(entry)
            if in_target:
                dossier.target_fragments += entry.fragments
                dossier.target_attempts += entry.attempts
                dossier.target_successes += entry.successes

    for dossier in dossiers.values():
        dossier.strata.sort(key=lambda e: (e.map_name, e.version))
        if dossier.strata:
            dossier.pooled_effect, dossier.pooled_se = _inverse_variance(
                (e.effect, e.std_error) for e in dossier.strata
            )
        main = main_component.get(dossier.role, set())
        dossier.in_main_component = dossier.key in main
        reasons: list[str] = []
        if dossier.target_fragments < MIN_TARGET_FRAGMENTS:
            reasons.append(
                f"目标版本片段不足：{dossier.target_fragments} < {MIN_TARGET_FRAGMENTS}"
            )
        if dossier.pooled_se is None or dossier.pooled_se > MAX_POOLED_SE:
            reasons.append(
                f"合并效应精度不足：se={None if dossier.pooled_se is None else round(dossier.pooled_se, 3)}"
                f" > {MAX_POOLED_SE}"
            )
        if not dossier.in_main_component:
            reasons.append("不在该岗位主比较分量内，缺少与主要候选群的同期桥接")
        dossier.sufficient = not reasons
        dossier.insufficient_reasons = reasons

    _annotate_identifiability(dossiers, fits, target_version, bridges)
    return dossiers


def _annotate_identifiability(
    dossiers: dict[tuple[str, str], Dossier],
    fits: dict[tuple[str, str, str], StratumFit],
    target_version: str,
    bridges: BridgeBook,
) -> None:
    """逐对标注相对次序不可辨认的选手-岗位及原因。"""
    by_role: dict[str, list[tuple[str, str]]] = {}
    for node, dossier in dossiers.items():
        by_role.setdefault(dossier.role, []).append(node)

    for role, nodes in by_role.items():
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                blockers: list[str] = []
                for (r, version, map_name), fit in fits.items():
                    if r != role or a not in fit.component_of or b not in fit.component_of:
                        continue
                    allowed = bridges.reachable_versions(role, map_name, target_version)
                    if version != target_version and version not in allowed:
                        continue
                    if fit.component_of[a] != fit.component_of[b]:
                        blockers.append(
                            f"{version}/{map_name} 分属分量 "
                            f"{sorted(fit.component_regions[fit.component_of[a]])} vs "
                            f"{sorted(fit.component_regions[fit.component_of[b]])}，无同期桥接"
                        )
                if blockers:
                    reason = "；".join(blockers)
                    dossiers[a].unidentifiable_with[b] = reason
                    dossiers[b].unidentifiable_with[a] = reason


def rank_role(
    dossiers: dict[tuple[str, str], Dossier], role: str
) -> list[Dossier]:
    """同一岗位按合并效应排序；不可辨认的对在报告层另行提示。"""
    items = [d for d in dossiers.values() if d.role == role and d.pooled_effect is not None]
    return sorted(items, key=lambda d: d.pooled_effect, reverse=True)
