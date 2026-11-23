"""训练资料装载与校验。

所有资料均为离线合成业务记录（UTF-8 JSONL）。装载时核对：
1. catalog.json 登记的行数与 sha256；
2. id 在各表内唯一；
3. 跨表外键、阵容与练习记录的一致性；
4. 业务约定（每场每边五人五岗、练习指标与角色绑定等）。

数据本身是不受信输入：本模块只做结构校验，不执行其中任何内容；
调用方应使用 ``python -I`` 运行，路径以参数传入。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ROLES = ("top", "jungle", "mid", "support", "carry")
SIDES = ("blue", "red")
# 资料中指标与角色的固定绑定关系（同时由校验代码确认）。
ROLE_METRIC = {
    "top": "cover",
    "support": "cover",
    "jungle": "sync",
    "mid": "sync",
    "carry": "sync",
}

TABLES = ("players", "games", "drills", "compatibility", "rounds", "seats")


class RecordError(ValueError):
    """资料不满足发行约定时抛出。"""


@dataclass(frozen=True)
class Dataset:
    """一次装载得到的全部业务记录及索引。"""

    players: list[dict[str, Any]]
    games: list[dict[str, Any]]
    drills: list[dict[str, Any]]
    compatibility: list[dict[str, Any]]
    rounds: list[dict[str, Any]]
    seats: list[dict[str, Any]]

    def index(self) -> "Indexes":
        return Indexes.build(self)


@dataclass(frozen=True)
class Indexes:
    """常用跨表索引，避免重复扫描。"""

    player: dict[str, dict[str, Any]]
    game: dict[str, dict[str, Any]]
    drills_by_game: dict[str, list[dict[str, Any]]]
    drills_by_key: dict[tuple[str, str, str], dict[str, Any]]

    @classmethod
    def build(cls, ds: "Dataset") -> "Indexes":
        drills_by_game: dict[str, list[dict[str, Any]]] = {}
        drills_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for d in ds.drills:
            drills_by_game.setdefault(d["game"], []).append(d)
            drills_by_key[(d["game"], d["side"], d["role"])] = d
        return cls(
            player={p["id"]: p for p in ds.players},
            game={g["id"]: g for g in ds.games},
            drills_by_game=drills_by_game,
            drills_by_key=drills_by_key,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - 防御性
            raise RecordError(f"{path.name} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return rows


def load_dataset(records_dir: str | Path, *, verify_checksum: bool = True) -> Dataset:
    """装载并校验 records 目录。

    verify_checksum=False 时跳过 catalog 摘要核对（便于在生成的新资料上
    复用同一套装载逻辑），其余结构校验仍然执行。
    """
    root = Path(records_dir)
    catalog_path = root / "catalog.json"
    if not catalog_path.exists():
        raise RecordError(f"缺少 catalog.json: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    tables: dict[str, list[dict[str, Any]]] = {}
    for name in TABLES:
        path = root / f"{name}.jsonl"
        if not path.exists():
            raise RecordError(f"缺少资料表: {path}")
        raw = path.read_bytes()
        if verify_checksum and name in catalog:
            expected = catalog[name]
            if sha256(raw).hexdigest() != expected["sha256"]:
                raise RecordError(f"{name}.jsonl 的 sha256 与 catalog 不符")
        rows = _read_jsonl(path)
        if verify_checksum and name in catalog and len(rows) != catalog[name]["rows"]:
            raise RecordError(f"{name}.jsonl 行数 {len(rows)} 与 catalog {catalog[name]['rows']} 不符")
        ids = [row.get("id") for row in rows]
        if any(i is None for i in ids):
            raise RecordError(f"{name}.jsonl 存在缺少 id 的记录")
        if len(set(ids)) != len(ids):
            raise RecordError(f"{name}.jsonl 的 id 不唯一")
        tables[name] = rows

    ds = Dataset(**tables)  # type: ignore[arg-type]
    validate_relations(ds)
    return ds


def validate_relations(ds: Dataset) -> None:
    """核对跨表外键与业务输入关系。"""
    player_ids = {p["id"] for p in ds.players}
    game_ids = {g["id"] for g in ds.games}

    for p in ds.players:
        if p["main"] not in ROLES:
            raise RecordError(f"选手 {p['id']} 主位置非法: {p['main']}")
        if p["secondary"] is not None and p["secondary"] not in ROLES:
            raise RecordError(f"选手 {p['id']} 替补位置非法: {p['secondary']}")
        if p["region"] not in ("east", "west"):
            raise RecordError(f"选手 {p['id']} 赛区非法: {p['region']}")

    for g in ds.games:
        if g["available_week"] < g["week"]:
            raise RecordError(f"比赛 {g['id']} available_week 早于比赛周")
        if set(g["lineups"]) != set(SIDES):
            raise RecordError(f"比赛 {g['id']} 阵容必须包含 blue/red 两边")
        seen: set[str] = set()
        for side in SIDES:
            lu = g["lineups"][side]
            if set(lu) != set(ROLES):
                raise RecordError(f"比赛 {g['id']} {side} 边五个岗位不完整")
            for role, pid in lu.items():
                if pid not in player_ids:
                    raise RecordError(f"比赛 {g['id']} 引用未知选手 {pid}")
                if pid in seen:
                    raise RecordError(f"比赛 {g['id']} 中选手 {pid} 同时占两个岗位")
                seen.add(pid)

    for d in ds.drills:
        if d["game"] not in game_ids:
            raise RecordError(f"练习 {d['id']} 引用未知比赛 {d['game']}")
        if d["side"] not in SIDES or d["role"] not in ROLES:
            raise RecordError(f"练习 {d['id']} 边/岗位非法")
        if d["metric"] != ROLE_METRIC[d["role"]]:
            raise RecordError(f"练习 {d['id']} 指标 {d['metric']} 与岗位 {d['role']} 不匹配")
        if not isinstance(d["attempts"], int) or d["attempts"] <= 0:
            raise RecordError(f"练习 {d['id']} attempts 非法")
        if not 0 <= d["completed"] <= d["attempts"]:
            raise RecordError(f"练习 {d['id']} completed 超出 attempts 范围")

    idx = ds.index()
    for g in ds.games:
        rows = idx.drills_by_game.get(g["id"], [])
        if len(rows) != 10:
            raise RecordError(f"比赛 {g['id']} 应有 10 条分岗练习，实际 {len(rows)}")
        for d in rows:
            expected_player = g["lineups"][d["side"]][d["role"]]
            if d["player"] != expected_player:
                raise RecordError(
                    f"练习 {d['id']} 的选手 {d['player']} 与阵容 {expected_player} 不一致"
                )

    for c in ds.compatibility:
        if c["metric"] not in set(ROLE_METRIC.values()):
            raise RecordError(f"兼容登记 {c['id']} 指标非法")
        if any(r not in ROLES for r in c["roles"]):
            raise RecordError(f"兼容登记 {c['id']} 岗位非法")

    for r in ds.rounds:
        if r["cohort_size"] <= 0:
            raise RecordError(f"轮次 {r['id']} cohort_size 非法")
        if r["follow_until"] is not None and r["follow_until"] < r["decision_week"]:
            raise RecordError(f"轮次 {r['id']} follow_until 早于 decision_week")

    for s in ds.seats:
        if len(s["confirmed"]) > 10:
            raise RecordError(f"试训 {s['id']} 锁定人数超过一场十人")
        unknown = set(s["confirmed"]) - player_ids
        if unknown:
            raise RecordError(f"试训 {s['id']} 锁定未知选手 {sorted(unknown)}")
        unknown = set(s["available"]) - player_ids
        if unknown:
            raise RecordError(f"试训 {s['id']} 可安排名单含未知选手 {sorted(unknown)}")
        if set(s["confirmed"]) - set(s["available"]):
            raise RecordError(f"试训 {s['id']} 锁定选手不在可安排名单内")
