"""按人按角色的可解释证据报告。

报告对每个 ``(选手, 岗位)`` 给出三层可追溯信息：
1. 支撑结论的比赛片段：逐分层列出版本/地图、片段数、成功率、剥离情境后
   的效应与标准误、所属连通分量及赛区构成；
2. 仍然未知的对照：相对次序不可辨认的选手对及具体原因；
3. 下一轮能实施的安排：试训场次、边/岗位、对照与未解决事项。

同时提供机器可读 JSON（供后续训练周期复用同一套办法）与 Markdown
（供教练组直接阅读）两种输出。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evidence import Dossier
from .records import Dataset, Indexes
from .rounds import RoundDecision
from .trials import TrialPlan


def _dossier_payload(dossier: Dossier) -> dict[str, Any]:
    return {
        "player": dossier.player,
        "role": dossier.role,
        "metric": dossier.metric,
        "sufficient": dossier.sufficient,
        "insufficient_reasons": dossier.insufficient_reasons,
        "pooled_effect": dossier.pooled_effect,
        "pooled_se": dossier.pooled_se,
        "target_fragments": dossier.target_fragments,
        "target_attempts": dossier.target_attempts,
        "target_successes": dossier.target_successes,
        "in_main_component": dossier.in_main_component,
        "strata": [
            {
                "version": e.version,
                "map": e.map_name,
                "metric": e.metric,
                "fragments": e.fragments,
                "attempts": e.attempts,
                "successes": e.successes,
                "raw_rate": round(e.raw_rate, 4),
                "effect": round(e.effect, 4),
                "std_error": round(e.std_error, 4),
                "component_id": e.component_id,
                "component_regions": sorted(e.component_regions),
                "bridged": e.bridged,
            }
            for e in dossier.strata
        ],
        "unidentifiable_with": [
            {"player": other[0], "role": other[1], "reason": reason}
            for other, reason in sorted(dossier.unidentifiable_with.items())
        ],
    }


def build_json_report(
    ds: Dataset,
    idx: Indexes,
    decisions: list[RoundDecision],
    trial_plans: dict[str, TrialPlan],
) -> dict[str, Any]:
    """汇总全部轮次的机器可读报告。"""
    players = {p["id"]: {"main": p["main"], "secondary": p["secondary"],
                        "region": p["region"]} for p in ds.players}
    rounds_out = []
    for decision in decisions:
        plan = trial_plans.get(decision.round_id)
        rounds_out.append({
            "round": decision.round_id,
            "decision_week": decision.decision_week,
            "follow_until": decision.follow_until,
            "target_version": decision.target_version,
            "cohort_size": decision.cohort_size,
            "target_has_data": decision.target_has_data,
            "lagged_excluded_at_decision": decision.lagged_excluded,
            "upcoming_games_at_decision": decision.upcoming_games,
            "promotions": [
                {"player": d.player, "role": d.role,
                 "pooled_effect": round(d.pooled_effect, 4),
                 "pooled_se": round(d.pooled_se, 4),
                 "target_fragments": d.target_fragments}
                for d in decision.promotions
            ],
            "observations": [
                {"player": d.player, "role": d.role,
                 "seat_limited_alternate": d.seat_limited_alternate,
                 "reasons": d.insufficient_reasons}
                for d in decision.observations
            ],
            "unidentifiable_notes": decision.unidentifiable_notes,
            "follow_checks": [asdict(c) for c in decision.follow_checks],
            "trials": None if plan is None else {
                "sessions": [
                    {
                        "session": s.session_id,
                        "week": s.week,
                        "slot": s.slot,
                        "version": s.version,
                        "lineup": [
                            {"side": seat.side, "role": seat.role,
                             "player": seat.player, "status": seat.status,
                             "note": seat.note}
                            for seat in sorted(s.seats, key=lambda x: (x.side, x.role))
                        ],
                        "unresolved": s.unresolved,
                    }
                    for s in plan.sessions
                ],
                "case_resolution": {
                    f"{p}|{r}": note for (p, r), note in plan.case_resolution.items()
                },
                "version_change": asdict(plan.version_change),
            },
        })

    # 按 (选手, 岗位) 的追溯视图：合并各轮次档案。目标版本无数据的轮次
    # （如 v3）只产生"待观察"资格、不含分层片段，不能覆盖早期轮次积累
    # 的证据；每个轮次的结论状态分别保留。
    trace: dict[str, Any] = {}
    for decision in decisions:
        for dossier in decision.dossiers.values():
            key = f"{dossier.player}|{dossier.role}"
            payload_entry = _dossier_payload(dossier)
            round_status = (
                "晋级" if dossier in decision.promotions
                else ("席位备选" if dossier.seat_limited_alternate else "再观察")
            )
            if key not in trace:
                trace[key] = {
                    "registration": players[dossier.player],
                    "latest_round": decision.round_id,
                    "round_status": {decision.round_id: round_status},
                    **payload_entry,
                }
            else:
                trace[key]["latest_round"] = decision.round_id
                trace[key]["round_status"][decision.round_id] = round_status
                if payload_entry["strata"]:
                    # 去掉轮次专属字段后更新证据本体。
                    payload_entry.pop("sufficient", None)
                    payload_entry.pop("insufficient_reasons", None)
                    trace[key].update(payload_entry)
                else:
                    trace[key]["sufficient"] = payload_entry["sufficient"]
                    trace[key]["insufficient_reasons"] = payload_entry[
                        "insufficient_reasons"
                    ]
    return {"players": players, "rounds": rounds_out, "player_role_trace": trace}


def write_json_report(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def render_markdown(payload: dict[str, Any]) -> str:
    """把机器可读报告渲染成教练组阅读的 Markdown。"""
    lines: list[str] = ["# 青训晋级证据报告", ""]
    for rd in payload["rounds"]:
        lines.append(f"## 轮次 {rd['round']}（决策周 {rd['decision_week']}，"
                     f"目标版本 {rd['target_version']}，每岗席位 {rd['cohort_size']}）")
        if rd["lagged_excluded_at_decision"]:
            lines.append(f"- 决策时点尚不可用、已排除的晚到比赛："
                         f"{', '.join(rd['lagged_excluded_at_decision'])}")
        if not rd["target_has_data"]:
            lines.append("- 目标版本在决策时点没有任何观测：不产生晋级结论，"
                         "全部资格进入试训观察。")
        lines.append("")
        lines.append("### 已有充分证据的晋级")
        if rd["promotions"]:
            lines.append("| 岗位 | 选手 | 合并效应(logit) | 标准误 | 目标版本片段 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for p in rd["promotions"]:
                lines.append(
                    f"| {p['role']} | {p['player']} | {p['pooled_effect']:.3f} | "
                    f"{p['pooled_se']:.3f} | {p['target_fragments']} |"
                )
        else:
            lines.append("（无）")
        lines.append("")
        lines.append("### 必须再观察的资格及不足")
        if rd["observations"]:
            lines.append("| 岗位 | 选手 | 不足之处 |")
            lines.append("| --- | --- | --- |")
            for o in rd["observations"]:
                lines.append(f"| {o['role']} | {o['player']} | {'；'.join(o['reasons'])} |")
        else:
            lines.append("（无）")
        lines.append("")
        if rd["unidentifiable_notes"]:
            lines.append("### 不可辨认的相对次序")
            lines.extend(f"- {n}" for n in rd["unidentifiable_notes"])
            lines.append("")
        if rd["follow_checks"]:
            lines.append("### 后续窗口检验")
            lines.append("| 选手 | 岗位 | 可比分层 | 方向是否保持 | 说明 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for c in rd["follow_checks"]:
                held = "—" if c["direction_held"] is None else (
                    "保持" if c["direction_held"] else "未保持")
                lines.append(
                    f"| {c['player']} | {c['role']} | "
                    f"{', '.join(c['checked_strata']) or '无'} | {held} | {c['detail']} |"
                )
            lines.append("")
        trials = rd["trials"]
        if trials:
            lines.append("### 下一轮试训安排")
            vc = trials["version_change"]
            if vc["new_version"] != vc["old_version"]:
                lines.append(f"#### 版本重判 {vc['old_version']} → {vc['new_version']}")
                lines.append("**仍站得住的结论**")
                lines.extend(f"- {x}" for x in vc["conclusions_standing"])
                lines.append("**无法验证原疑点的结论**")
                lines.extend(f"- {x}" for x in vc["conclusions_broken"])
                lines.append("**已通知集训席位的变更影响（单列）**")
                lines.extend(f"- {x}" for x in vc["confirmed_seat_impact"])
                lines.append("")
            for s in trials["sessions"]:
                lines.append(f"#### {s['session']}（第 {s['week']} 周 {s['slot']}，"
                             f"版本 {s['version']}，{len({x['player'] for x in s['lineup']})} 人）")
                lines.append("| 边 | 岗位 | 选手 | 身份 | 说明 |")
                lines.append("| --- | --- | --- | --- | --- |")
                for seat in s["lineup"]:
                    lines.append(
                        f"| {seat['side']} | {seat['role']} | {seat['player']} | "
                        f"{seat['status']} | {seat['note']} |"
                    )
                if s["unresolved"]:
                    lines.append("未解决：")
                    lines.extend(f"- {u}" for u in s["unresolved"])
                lines.append("")
            if trials["case_resolution"]:
                lines.append("#### 待观察资格的安排去向")
                lines.append("| 资格 | 安排 |")
                lines.append("| --- | --- |")
                for key, note in sorted(trials["case_resolution"].items()):
                    lines.append(f"| {key} | {note} |")
                lines.append("")

    lines.append("## 按人按角色追溯")
    for key, item in sorted(payload["player_role_trace"].items()):
        lines.append(f"### {key}（登记主岗 {item['registration']['main']}，"
                     f"替补 {item['registration']['secondary']}，赛区 {item['registration']['region']}）")
        status_text = "，".join(f"{r}={s}" for r, s in sorted(item["round_status"].items()))
        lines.append(f"- 各轮结论状态：{status_text}")
        lines.append(f"- 最近一轮（{item['latest_round']}）判断："
                     f"{'证据充分' if item['sufficient'] else '再观察'}"
                     f"（{('；'.join(item['insufficient_reasons'])) or '满足全部充分性门槛'}）")
        if item["pooled_effect"] is not None:
            lines.append(f"- 合并效应 {item['pooled_effect']:.3f}（se {item['pooled_se']:.3f}），"
                         f"目标版本片段 {item['target_fragments']}")
        lines.append("- 支撑片段：")
        lines.append("| 版本 | 地图 | 指标 | 片段 | 成功率 | 效应 | se | 分量赛区 | 桥接 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in item["strata"]:
            lines.append(
                f"| {e['version']} | {e['map']} | {e['metric']} | {e['fragments']} | "
                f"{e['raw_rate']:.3f} | {e['effect']:.3f} | {e['std_error']:.3f} | "
                f"{','.join(e['component_regions'])} | {'是' if e['bridged'] else '否'} |"
            )
        if item["unidentifiable_with"]:
            lines.append("- 仍然未知的对照：")
            for u in item["unidentifiable_with"]:
                lines.append(f"  - 与 {u['player']}/{u['role']}：{u['reason']}")
        lines.append("")
    return "\n".join(lines)
