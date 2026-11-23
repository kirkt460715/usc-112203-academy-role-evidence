"""决策时点回看与后续窗口检验。

每个轮次 ``rounds.jsonl`` 给出 ``decision_week``（决策时点）与
``follow_until``（回看截止）。关键纪律：

- 决策只能使用 ``available_week <= decision_week`` 的比赛；晚到的赛况
  （例如 G057 在第 6 周才可用）即使比赛周更早，也不得提前进入结论。
- ``decision_week < available_week <= follow_until`` 的随后训练表现作为
  检验样本：只在 *可比岗位*（同指标、同地图，且版本兼容登记允许）上
  检查决策时解释的方向是否保持；没有可比依据时显式标注"无法检验"。
- 目标版本尚无任何观测的轮次（如 v3 轮次）不产生晋级结论，全部资格
  进入观察并交给试训安排模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import (
    BridgeBook,
    Dossier,
    build_dossiers,
    rank_role,
)
from .model import StratumFit, fit_all_strata
from .records import ROLE_METRIC, Dataset, Indexes


@dataclass
class FollowCheck:
    """一个晋级档案在后续窗口上的检验结果。"""

    player: str
    role: str
    checked_strata: list[str]
    comparable: bool
    direction_held: bool | None
    detail: str
    # 后续窗口内可辨认对照中的名次（1 为最佳），无法检验时为 None
    follow_rank: int | None = None
    follow_candidates: int | None = None


@dataclass
class RoundDecision:
    round_id: str
    decision_week: int
    follow_until: int | None
    target_version: str
    cohort_size: int
    dossiers: dict[tuple[str, str], Dossier]
    promotions: list[Dossier]          # 已有充分证据、按岗位取前 cohort_size
    observations: list[Dossier]        # 必须再观察的资格（附原因）
    follow_checks: list[FollowCheck]
    lagged_excluded: list[str]         # 比赛周已过但决策时点之后才可用的晚到赛况
    upcoming_games: list[str]          # 决策时点之后才进行的比赛
    target_has_data: bool
    unidentifiable_notes: list[str] = field(default_factory=list)


def _games_visible(games, week: int):
    return [g for g in games if g["available_week"] <= week]


def evaluate_round(
    ds: Dataset, idx: Indexes, round_record: dict, bridges: BridgeBook
) -> RoundDecision:
    dw = round_record["decision_week"]
    target = round_record["target_version"]
    cohort = round_record["cohort_size"]

    visible = _games_visible(ds.games, dw)
    visible_ids = {g["id"] for g in visible}
    # 晚到赛况：比赛周已到、资料在决策时点之后才可用（信息纪律的核心对象）。
    lagged = sorted(
        g["id"] for g in ds.games
        if g["available_week"] > dw and g["week"] <= dw
    )
    # 决策时点之后才进行的比赛，单独列出以示区别。
    upcoming = sorted(
        g["id"] for g in ds.games if g["week"] > dw
    )

    # 用过滤后的比赛清单驱动分层拟合（drills 经 game 关联，天然同步过滤）。
    filtered_ds = Dataset(
        players=ds.players,
        games=visible,
        drills=[d for d in ds.drills if d["game"] in visible_ids],
        compatibility=ds.compatibility,
        rounds=ds.rounds,
        seats=ds.seats,
    )
    filtered_idx = filtered_ds.index()
    fits = fit_all_strata(filtered_ds, filtered_idx)
    target_has_data = any(v == target for (_, v, _) in fits)

    dossiers = build_dossiers(filtered_ds, fits, target, bridges) if target_has_data else {}

    promotions: list[Dossier] = []
    observations: list[Dossier] = []
    if target_has_data:
        for role in {d.role for d in dossiers.values()}:
            ranked = rank_role(dossiers, role)
            promoted_role = 0
            for rank_pos, dossier in enumerate(ranked, start=1):
                if dossier.sufficient and promoted_role < cohort:
                    promotions.append(dossier)
                    promoted_role += 1
                elif dossier.sufficient:
                    # 证据本身充分，但每岗席位有限：这是"排队中的备选"，
                    # 与证据不足必须再观察的资格区分开。
                    dossier.insufficient_reasons.append(
                        f"证据达充分门槛但岗位仅 {cohort} 席，"
                        f"当前名次 {rank_pos} 未入选（备选，无需新增试训）"
                    )
                    dossier.sufficient = False
                    dossier.seat_limited_alternate = True
                    observations.append(dossier)
                else:
                    observations.append(dossier)
    else:
        # 目标版本尚无观测：不产生任何晋级结论，全部登记资格进入观察，
        # 由试训安排模块按席位约束生成求证组合。
        dossiers = {}
        for p in ds.players:
            for role in {p["main"], p["secondary"]} - {None}:
                dossier = Dossier(
                    player=p["id"], role=role, metric=ROLE_METRIC[role],
                    sufficient=False,
                    insufficient_reasons=[
                        f"目标版本 {target} 在决策周 {dw} 前没有任何观测，"
                        "不存在可引用的同版本证据"
                    ],
                )
                dossiers[dossier.key] = dossier
                observations.append(dossier)

    notes = _unidentifiable_notes(dossiers)
    checks = _follow_window(ds, idx, round_record, promotions, bridges)

    return RoundDecision(
        round_id=round_record["id"],
        decision_week=dw,
        follow_until=round_record["follow_until"],
        target_version=target,
        cohort_size=cohort,
        dossiers=dossiers,
        promotions=sorted(promotions, key=lambda d: (d.role, -d.pooled_effect)),
        observations=sorted(observations, key=lambda d: (d.role, d.player)),
        follow_checks=checks,
        lagged_excluded=lagged,
        upcoming_games=upcoming,
        target_has_data=target_has_data,
        unidentifiable_notes=notes,
    )


def _unidentifiable_notes(dossiers: dict[tuple[str, str], Dossier]) -> list[str]:
    notes: list[str] = []
    seen: set[frozenset] = set()
    for dossier in dossiers.values():
        for other, reason in dossier.unidentifiable_with.items():
            token = frozenset((dossier.key, other))
            if token in seen:
                continue
            seen.add(token)
            notes.append(
                f"{dossier.role}: {dossier.player} 与 {other[0]} 的相对次序不可辨认——{reason}"
            )
    return sorted(notes)


def _follow_window(
    ds: Dataset,
    idx: Indexes,
    round_record: dict,
    promotions: list[Dossier],
    bridges: BridgeBook,
) -> list[FollowCheck]:
    """用随后窗口的训练表现检验晋级解释在可比岗位上是否成立。"""
    fu = round_record["follow_until"]
    dw = round_record["decision_week"]
    if fu is None or not promotions:
        return []

    window = [
        g for g in ds.games
        if dw < g["available_week"] <= fu
    ]
    window_ids = {g["id"] for g in window}
    if not window:
        return []
    follow_ds = Dataset(
        players=ds.players,
        games=window,
        drills=[d for d in ds.drills if d["game"] in window_ids],
        compatibility=ds.compatibility,
        rounds=ds.rounds,
        seats=ds.seats,
    )
    follow_idx = follow_ds.index()
    follow_fits = fit_all_strata(follow_ds, follow_idx)

    checks: list[FollowCheck] = []
    for dossier in promotions:
        role = dossier.role
        metric = ROLE_METRIC[role]
        comparable_strata: list[StratumFit] = []
        for (r, version, map_name), fit in follow_fits.items():
            if r != role:
                continue
            allowed = bridges.reachable_versions(role, map_name, version)
            # 决策版本的证据能否进入该后续分层，取决于反向兼容是否登记。
            decision_version = _decision_version_of(dossier, map_name)
            if decision_version is not None and decision_version in allowed:
                comparable_strata.append(fit)

        labels = sorted({f"{f.version}/{f.map_name}" for f in comparable_strata})
        if not comparable_strata:
            checks.append(FollowCheck(
                player=dossier.player, role=role, checked_strata=[],
                comparable=False, direction_held=None,
                detail="后续窗口没有同指标同地图且版本兼容的分层，原解释无法检验",
            ))
            continue

        held_all = True
        ranks: list[int] = []
        sizes: list[int] = []
        for fit in comparable_strata:
            node = dossier.key
            if node not in fit.effects:
                # 该选手在这个后续分层没有出场，不能用于检验。
                labels = [x for x in labels if x != f"{fit.version}/{fit.map_name}"]
                continue
            cid = fit.component_of[node]
            peers = sorted(
                fit.components[cid],
                key=lambda n: fit.effects[n],
                reverse=True,
            )
            rank = peers.index(node) + 1
            ranks.append(rank)
            sizes.append(len(peers))
            # 方向保持：在可辨认对照中仍处于上半区。
            if rank > len(peers) / 2:
                held_all = False
        if not ranks:
            checks.append(FollowCheck(
                player=dossier.player, role=role, checked_strata=[],
                comparable=False, direction_held=None,
                detail="兼容分层内该选手在后续窗口没有出场记录，原解释无法检验",
            ))
            continue
        labels = sorted({f"{f.version}/{f.map_name}" for f in comparable_strata
                         if dossier.key in f.effects})
        detail = (
            f"在 {', '.join(labels)} 的可辨认对照中名次为 "
            f"{ranks}（对照规模 {sizes}）"
        )
        checks.append(FollowCheck(
            player=dossier.player, role=role, checked_strata=labels,
            comparable=True, direction_held=held_all, detail=detail,
            follow_rank=min(ranks) if ranks else None,
            follow_candidates=max(sizes) if sizes else None,
        ))
    return checks


def _decision_version_of(dossier: Dossier, map_name: str) -> str | None:
    """档案中某地图的目标版本证据来自哪个版本（优先非桥接条目）。"""
    candidates = [e for e in dossier.strata if e.map_name == map_name]
    if not candidates:
        return None
    native = [e for e in candidates if not e.bridged]
    return (native or candidates)[0].version
