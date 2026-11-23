"""试训席位安排、组合约束与版本重判。

业务约束（seats.jsonl + README）：
- 每场试训需要十个互异选手，两边各五人五岗；
- 同一时段（slot 相同）的两场不能共用选手，同一人的两个岗位资格
  也不能在同一场同时使用；
- ``confirmed`` 是已通知/已承诺的集训席位，必须保留，且在版本变更
  影响中单列；
- 可安排人选只能取自该场 ``available``，岗位以训练登记（主/替补）为准。

设计目标：对"必须再观察"的资格，安排能区分 *个人效应与整队效应* 的
组合——每个待观察岗位尽量与其主要对照（不可辨认对或相邻名次）同场
对垒，并给两边配置强度不同的搭档，使个人效应与情境效应在统计上可分。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import BridgeBook, Dossier
from .records import ROLE_METRIC, ROLES, Dataset
from .rounds import RoundDecision


@dataclass
class PlannedSeat:
    session_id: str
    side: str
    role: str
    player: str
    status: str          # confirmed（已承诺）/ observation（待观察资格）/ contrast（对照）/ filler
    note: str = ""


@dataclass
class SessionPlan:
    session_id: str
    week: int
    slot: str
    version: str
    seats: list[PlannedSeat] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # 本场无法安排的求证需求

    @property
    def players(self) -> set[str]:
        return {s.player for s in self.seats}


@dataclass
class TrialPlan:
    sessions: list[SessionPlan]
    # 每个待观察 (选手, 岗位) -> 安排说明或未安排原因
    case_resolution: dict[tuple[str, str], str]
    version_change: "VersionReassessment"


@dataclass
class VersionReassessment:
    """赛前换到新版本后的重判结论。"""

    old_version: str
    new_version: str
    # 仍然站得住的结论（有显式兼容登记支持的 指标/地图/岗位）
    conclusions_standing: list[str] = field(default_factory=list)
    # 站不住、无法验证原疑点的安排
    conclusions_broken: list[str] = field(default_factory=list)
    # 已通知集训席位的变更影响（单列）
    confirmed_seat_impact: list[str] = field(default_factory=list)


def _role_eligibility(ds: Dataset) -> dict[str, set[str]]:
    """选手 -> 可打岗位集合（主位置 + 已登记替补位置）。"""
    eligible: dict[str, set[str]] = {}
    for p in ds.players:
        eligible[p["id"]] = {p["main"]}
        if p["secondary"]:
            eligible[p["id"]].add(p["secondary"])
    return eligible


def plan_trials(
    ds: Dataset,
    decision: RoundDecision,
    bridges: BridgeBook,
    *,
    new_version: str | None = None,
) -> TrialPlan:
    """依据某轮次的观察名单与 seats.jsonl 生成试训安排。

    new_version 给定时执行版本重判：目标版本之后出现的新版本会使
    无兼容依据的原疑点无法在新试训中验证。
    """
    eligible = _role_eligibility(ds)
    idx = ds.index()
    seats_records = {s["id"]: s for s in ds.seats}
    sessions = [
        SessionPlan(session_id=s["id"], week=s["week"], slot=s["slot"], version=s["version"])
        for s in ds.seats
    ]

    observations = decision.observations
    case_resolution: dict[tuple[str, str], str] = {}
    # 席位受限的备选不需要新增试训求证，单独登记去向；只有证据真正不足的
    # 资格才进入组合安排（它们的疑点需要个人 vs 整队的对照设计）。
    trial_cases: list[Dossier] = []
    for dossier in observations:
        if dossier.seat_limited_alternate:
            case_resolution[dossier.key] = (
                "席位受限的备选：证据已达充分门槛，无需新增试训，"
                "出现空缺时按名次依次递补"
            )
        else:
            trial_cases.append(dossier)
    # 待观察资格按岗位分组，并为每个资格找首选对照。
    by_role: dict[str, list[Dossier]] = {}
    for dossier in trial_cases:
        by_role.setdefault(dossier.role, []).append(dossier)

    # 占用记录：
    # - used_in_session[session]：本场已上场的十人（一场内一人只能占一个资格）；
    # - used_in_slot[slot]：同一时段各场次的并集（同时段两场不得共用选手）。
    used_in_session: dict[str, set[str]] = {s.session_id: set() for s in sessions}
    used_in_slot: dict[str, set[str]] = {s.slot: set() for s in sessions}

    # 1) 先锁定 confirmed 席位（岗位优先取登记主位置；若主位置该场已被
    # 更早的承诺占用，则使用其登记替补位置，并在说明中标注）。
    for session in sessions:
        rec = seats_records[session.session_id]
        for pid in rec["confirmed"]:
            player = idx.player[pid]
            role = player["main"]
            note = "已通知的集训承诺"
            if _role_taken(session, "blue", role) or _role_taken(session, "red", role):
                if player["secondary"] and not (
                    _role_taken(session, "blue", player["secondary"])
                    or _role_taken(session, "red", player["secondary"])
                ):
                    role = player["secondary"]
                    note = "已通知的集训承诺（主位置冲突，按登记替补位置出场）"
                else:
                    session.unresolved.append(
                        f"已承诺选手 {pid} 的登记岗位在 {session.session_id} 均已占用，"
                        "需要人工调整席位"
                    )
                    continue
            _place(session, used_in_session, used_in_slot, "blue", role, pid,
                   status="confirmed", note=note)

    # 2) 为待观察资格安排对照组合。
    for role in ROLES:
        dossiers = by_role.get(role, [])
        for dossier in dossiers:
            target = _find_session_with_contrast(
                ds, sessions, used_in_session, used_in_slot, eligible, dossier, role
            )
            if target is None:
                # 退而求其次：先安排进任意可行场次，对照缺失记入未解决事项。
                fallback = _find_session_for(ds, sessions, used_in_session, used_in_slot,
                                             eligible, dossier.player, role)
                if fallback is None:
                    reason = _blocker_reason(ds, sessions, used_in_slot, eligible,
                                             dossier.player, role)
                    case_resolution[dossier.key] = f"未能安排：{reason}"
                    continue
                session, side = target = fallback
                partner = None
            else:
                session, side, partner = target
            version_note = _version_note(decision.target_version, session.version)
            _place(session, used_in_session, used_in_slot, side, role, dossier.player,
                   status="observation",
                   note=f"检验 {dossier.player} 的 {role} 效应（个人 vs 整队）")
            other_side = "red" if side == "blue" else "blue"
            if partner is not None:
                _place(session, used_in_session, used_in_slot, other_side, role, partner,
                       status="contrast",
                       note=f"与 {dossier.player} 同岗对垒的主要对照")
                case_resolution[dossier.key] = (
                    f"{session.session_id}（{session.version}）{side}/{role}，"
                    f"对照 {partner}（{other_side}）{version_note}"
                )
            else:
                case_resolution[dossier.key] = (
                    f"{session.session_id}（{session.version}）{side}/{role}，"
                    f"该场无可用同岗对照，需后续补排{version_note}"
                )
                session.unresolved.append(
                    f"{dossier.player}/{role} 缺少同岗对照，个人与整队效应尚不能完全分离"
                )

    # 3) 填满每场十人（每岗两边各一人）。
    for session in sessions:
        rec = seats_records[session.session_id]
        availability = set(rec["available"])
        _fill_session(session, ds, availability, eligible, used_in_session, used_in_slot)
        if len(session.players) != 10:
            session.unresolved.append(
                f"受席位/时段互斥限制，本场仅能安排 {len(session.players)} 人，不足十人"
            )

    reassessment = _reassess(ds, decision, sessions, bridges, new_version)
    return TrialPlan(sessions=sessions, case_resolution=case_resolution,
                     version_change=reassessment)


def _place(session: SessionPlan, used_session, used_slot, side, role, player,
           *, status: str, note: str = "") -> None:
    session.seats.append(PlannedSeat(session.session_id, side, role, player, status, note))
    used_session[session.session_id].add(player)
    used_slot[session.slot].add(player)


def _can_play_in(ds, session: SessionPlan, used_session, used_slot,
                 eligible, player: str, role: str) -> bool:
    rec_available = player in {p for p in _available(ds, session.session_id)}
    return (
        rec_available
        and role in eligible.get(player, set())
        and player not in used_session[session.session_id]
        and player not in used_slot[session.slot]
    )


def _available(ds: Dataset, session_id: str) -> list[str]:
    return next(s["available"] for s in ds.seats if s["id"] == session_id)


def _version_note(target_version: str, session_version: str) -> str:
    """场次版本与轮次目标版本不一致时，附加可验证性提示。"""
    if session_version == target_version:
        return ""
    return f"；注意场次为 {session_version}，不能直接验证 {target_version} 疑点（见版本重判）"


def _role_taken(session: SessionPlan, side: str, role: str) -> bool:
    return any(s.side == side and s.role == role for s in session.seats)


def _find_session_for(ds, sessions, used_session, used_slot, eligible, player, role):
    """选择一个能让该选手打该岗位的场次与边。

    同一时段的两场互斥；不同时段可以重复出场。该选手已在某场占用资格时
    跳过该场（一场内一人只能占一个岗位），但不影响其他场次。
    """
    for session in sessions:
        if player not in set(_available(ds, session.session_id)):
            continue
        if role not in eligible.get(player, set()):
            continue
        if player in used_session[session.session_id]:
            continue
        if player in used_slot[session.slot]:
            continue
        for side in ("blue", "red"):
            if not _role_taken(session, side, role):
                return session, side
    return None


def _find_session_with_contrast(ds, sessions, used_session, used_slot,
                                eligible, dossier, role):
    """找一个"本人 + 对边同岗对照"都能落位的场次。

    优先保证个人效应与整队效应可分：只有能配上对照的场次才返回。
    """
    player = dossier.player
    for session in sessions:
        if player not in set(_available(ds, session.session_id)):
            continue
        if role not in eligible.get(player, set()):
            continue
        if player in used_session[session.session_id]:
            continue
        if player in used_slot[session.slot]:
            continue
        for side in ("blue", "red"):
            if _role_taken(session, side, role):
                continue
            partner = _pick_contrast(ds, dossier, sessions, used_session,
                                     used_slot, eligible, session, side, role)
            if partner is not None:
                return session, side, partner
    return None


def _blocker_reason(ds, sessions, used_slot, eligible, player, role) -> str:
    if role not in eligible.get(player, set()):
        return f"{player} 无 {role} 岗位登记"
    reachable = [s.session_id for s in sessions
                 if player in set(_available(ds, s.session_id))]
    if not reachable:
        return "该选手不在任何试训可安排名单内"
    clashes = [sid for sid in reachable if player in used_slot.get(
        next(s.slot for s in sessions if s.session_id == sid), set())]
    return f"可到场次 {reachable} 的同时段席位已被占用（{clashes}）"


def _pick_contrast(ds, dossier, sessions, used_session, used_slot, eligible,
                   session, side, role):
    """从不可辨认对/相邻候选中挑一个可在对边同岗出场的对照。

    对照必须是另一个选手：同一人的两个岗位资格不能在同一场同时使用，
    更不能把本人排到对边充当自己的对照。
    """
    other_side = "red" if side == "blue" else "blue"
    preferred = [node for node in dossier.unidentifiable_with if node[0] != dossier.player]
    for other in preferred:
        pid = other[0]
        if _can_play_in(ds, session, used_session, used_slot, eligible, pid, role) \
                and not _role_taken(session, other_side, role):
            return pid
    # 退而求其次：任意可在对边同岗出场的合格选手（排除本人）。
    for pid in _available(ds, session.session_id):
        if pid == dossier.player:
            continue
        if _can_play_in(ds, session, used_session, used_slot, eligible, pid, role) \
                and not _role_taken(session, other_side, role):
            return pid
    return None


def _fill_session(session, ds, availability, eligible, used_session, used_slot) -> None:
    """补齐每边每岗，保证十人互异；同一场不允许同一人占两个资格。"""
    for side in ("blue", "red"):
        for role in ROLES:
            if _role_taken(session, side, role):
                continue
            for pid in availability:
                if (role in eligible.get(pid, set())
                        and pid not in used_session[session.session_id]
                        and pid not in used_slot[session.slot]):
                    _place(session, used_session, used_slot, side, role, pid,
                           status="filler", note="补齐阵容的可用选手")
                    break


def _reassess(ds: Dataset, decision: RoundDecision, sessions: list[SessionPlan],
              bridges: BridgeBook, new_version: str | None) -> VersionReassessment:
    """版本重判：比较决策目标版本与试训场次版本，核对兼容登记。

    重判是 *按场次版本* 进行的：同一批席位里 v2 场与 v3 场对原疑点的
    可验证性不同。已通知席位无论有无历史结论都单列变更影响。
    """
    old = decision.target_version
    new = new_version or max((s.version for s in sessions), default=old)
    result = VersionReassessment(old_version=old, new_version=new)

    # 从新版本沿兼容链反向可达旧版本，才算"仍站得住"。
    roles_seen = sorted({d.role for d in decision.dossiers.values()})
    maps_by_role: dict[str, set[str]] = {}
    for d in decision.dossiers.values():
        maps_by_role.setdefault(d.role, set()).update(e.map_name for e in d.strata)
    for role in roles_seen:
        metric = ROLE_METRIC[role]
        for map_name in sorted(maps_by_role[role]):
            label = f"{role}/{metric}/{map_name}"
            reachable = bridges.reachable_versions(role, map_name, new)
            if old in reachable:
                result.conclusions_standing.append(
                    f"{label}：存在显式跨版本兼容登记，{old} 结论可带入 {new} 继续检验"
                )
            else:
                result.conclusions_broken.append(
                    f"{label}：{old}→{new} 无兼容依据，原疑点在新版本试训中无法验证"
                )

    # 目标版本本身没有历史结论时（如 v3 轮次），按岗位登记显式说明
    # 新版本试训能验证什么、不能验证什么。此时 old 是尚未开赛的新版本，
    # 要看它能否沿兼容链接到任一已有数据的历史版本。
    historical_versions = sorted(
        {g["version"] for g in ds.games if g["version"] != old}
    )
    if not any(maps_by_role.values()):
        for role in ROLES:
            metric = ROLE_METRIC[role]
            for map_name in sorted({g["map"] for g in ds.games}):
                label = f"{role}/{metric}/{map_name}"
                reachable = bridges.reachable_versions(role, map_name, new)
                linked = [v for v in historical_versions if v in reachable]
                if linked:
                    result.conclusions_standing.append(
                        f"{label}：{old} 无历史证据，但与 {','.join(linked)} 的"
                        f"兼容登记允许在 {new} 沿用相邻版本口径积累新证据"
                    )
                else:
                    result.conclusions_broken.append(
                        f"{label}：{old} 无历史证据且无兼容登记，{new} 试训"
                        "只能建立全新基线，不能回答旧版本疑点"
                    )

    # 已通知席位单列变更影响（即使版本未变也逐条留痕）。
    for session in sessions:
        rec = next(s for s in ds.seats if s["id"] == session.session_id)
        for pid in rec["confirmed"]:
            if session.version == old == new:
                result.confirmed_seat_impact.append(
                    f"{pid} 的已通知席位（{session.session_id} {session.slot}）"
                    f"在目标版本 {old} 举行，不受版本重判影响，承诺保留"
                )
            elif session.version == new:
                result.confirmed_seat_impact.append(
                    f"{pid} 的已通知席位（{session.session_id} {session.slot}）"
                    f"改在 {new} 版本举行：承诺保留，但其 {old} 证据需按兼容范围重新解释"
                )
            else:
                result.confirmed_seat_impact.append(
                    f"{pid} 的已通知席位（{session.session_id} {session.slot}）"
                    f"仍在 {session.version} 版本，不能验证 {new} 疑点，承诺保留但需另排新版本场次"
                )
    return result
