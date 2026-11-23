"""晋级证据系统的性质测试（与发行资料的完整性测试互补）。

运行：python3 -I -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coaching.records import load_dataset, Dataset, ROLE_METRIC, ROLES  # noqa: E402
from coaching.model import fit_all_strata, fit_stratum  # noqa: E402
from coaching.evidence import (  # noqa: E402
    BridgeBook,
    build_dossiers,
    rank_role,
)
from coaching.rounds import evaluate_round  # noqa: E402
from coaching.trials import plan_trials  # noqa: E402
from coaching.report import build_json_report, render_markdown  # noqa: E402

RECORDS = ROOT / "records"


class EvidenceTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset(RECORDS)
        cls.idx = cls.ds.index()
        cls.bridges = BridgeBook.from_dataset(cls.ds)
        cls.round_by_id = {r["id"]: r for r in cls.ds.rounds}


class TestRecords(EvidenceTestBase):
    def test_checksum_loaded(self):
        # 能装载即说明 catalog 校验、外键与阵容一致性全部通过。
        self.assertEqual(len(self.ds.games), 144)
        self.assertEqual(len(self.ds.drills), 1440)

    def test_metric_role_binding_never_mixed(self):
        # 指标与岗位绑定：cover/sync 绝不跨指标拼接的前提。
        for d in self.ds.drills:
            self.assertEqual(d["metric"], ROLE_METRIC[d["role"]])
        self.assertEqual(
            {ROLE_METRIC[r] for r in ("top", "support")}, {"cover"}
        )
        self.assertEqual(
            {ROLE_METRIC[r] for r in ("jungle", "mid", "carry")}, {"sync"}
        )

    def test_tamper_detected(self):
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(RECORDS, Path(tmp) / "records")
            target = Path(tmp) / "records" / "players.jsonl"
            rows = target.read_text(encoding="utf-8").splitlines()
            rows[0] = rows[0].replace('"main":"top"', '"main":"mid"')
            target.write_text("\n".join(rows), encoding="utf-8")
            with self.assertRaises(Exception):
                load_dataset(Path(tmp) / "records")


class TestModel(EvidenceTestBase):
    def test_all_strata_converge_with_bounded_effects(self):
        fits = fit_all_strata(self.ds, self.idx)
        self.assertEqual(len(fits), 30)  # 5 角色 × 2 版本 × 3 地图
        for fit in fits.values():
            self.assertTrue(fit.converged, f"分层未收敛: {fit.role}/{fit.version}/{fit.map_name}")
            # 岭正则 + 步长控制后，效应应保持在有界、可解释的范围内。
            self.assertLess(max(abs(v) for v in fit.effects.values()), 5.0)

    def test_effect_direction_matches_known_signal(self):
        # 合成资料中 P11/jungle 是强信号、P00/top 是弱信号。
        strong = fit_stratum(self.ds, self.idx, "jungle", "v2", "ridge")
        weak = fit_stratum(self.ds, self.idx, "top", "v1", "ridge")
        self.assertGreater(strong.effects[("P11", "jungle")], 1.0)
        self.assertLess(weak.effects[("P00", "top")], -0.5)

    def test_opponent_edge_bridges_regions_on_ridge(self):
        # 决策周 9 的 v2 切片中，ridge 桥接赛（G072/G084）仍在窗口内。
        # 可辨认性是岗位特异的：jungle 在桥接赛中有跨赛区对位，
        # 因而出现跨赛区分量；top 在桥接赛里恰好都是同赛区对位，仍分裂。
        visible = [g for g in self.ds.games if g["available_week"] <= 9]
        ids = {g["id"] for g in visible}
        sliced = Dataset(
            self.ds.players, visible,
            [d for d in self.ds.drills if d["game"] in ids],
            self.ds.compatibility, self.ds.rounds, self.ds.seats,
        )
        bridged = fit_stratum(sliced, sliced.index(), "jungle", "v2", "ridge")
        self.assertTrue(
            any(len(regs) == 2 for regs in bridged.component_regions.values())
        )
        unbridged = fit_stratum(sliced, sliced.index(), "top", "v2", "ridge")
        self.assertTrue(
            all(len(regs) == 1 for regs in unbridged.component_regions.values())
        )

    def test_identifiability_pair_rule(self):
        fit = fit_stratum(self.ds, self.idx, "top", "v2", "canal")
        nodes = list(fit.effects)
        east = next(n for n in nodes if self.idx.player[n[0]]["region"] == "east")
        west = next(n for n in nodes if self.idx.player[n[0]]["region"] == "west")
        self.assertFalse(fit.identifiable_pair(east, west))
        east2 = next(n for n in nodes if n != east
                     and self.idx.player[n[0]]["region"] == "east")
        self.assertTrue(fit.identifiable_pair(east, east2))


class TestBridges(EvidenceTestBase):
    def test_only_registered_combinations_bridge(self):
        # compatibility: cover 仅 v1-v2/ridge/support；sync 仅 v2-v3/canal/jungle。
        self.assertEqual(
            self.bridges.reachable_versions("support", "ridge", "v2"),
            {"v1", "v2"},
        )
        self.assertEqual(
            self.bridges.reachable_versions("jungle", "canal", "v3"),
            {"v2", "v3"},
        )
        # 未登记组合：看不到任何跨版本可达性。
        self.assertEqual(
            self.bridges.reachable_versions("top", "canal", "v2"), {"v2"}
        )
        self.assertEqual(
            self.bridges.reachable_versions("jungle", "ridge", "v3"), {"v3"}
        )

    def test_role_switch_creates_separate_dossiers(self):
        # P00 既打 top 又打 jungle：两个岗位证据独立，绝不合并。
        dossiers = build_dossiers(
            self.ds, fit_all_strata(self.ds, self.idx), "v2", self.bridges
        )
        self.assertIn(("P00", "top"), dossiers)
        self.assertIn(("P00", "jungle"), dossiers)
        top_metrics = {e.metric for e in dossiers[("P00", "top")].strata}
        jungle_metrics = {e.metric for e in dossiers[("P00", "jungle")].strata}
        self.assertEqual(top_metrics, {"cover"})
        self.assertEqual(jungle_metrics, {"sync"})


class TestPointInTime(EvidenceTestBase):
    def test_lagged_game_not_visible_before_available_week(self):
        # G133：week12 比赛、available_week13。R3 决策周 12 必须排除它。
        r3 = evaluate_round(self.ds, self.idx, self.round_by_id["R3"], self.bridges)
        self.assertIn("G133", r3.lagged_excluded)
        g133 = self.idx.game["G133"]
        # 直接验证决策切片里没有 G133 的任何练习。
        visible = [g for g in self.ds.games if g["available_week"] <= r3.decision_week]
        self.assertNotIn(g133["id"], {g["id"] for g in visible})

    def test_r3_has_no_promotion_and_all_qualities_observe(self):
        r3 = evaluate_round(self.ds, self.idx, self.round_by_id["R3"], self.bridges)
        self.assertFalse(r3.target_has_data)
        self.assertEqual(r3.promotions, [])
        # 38 名选手：28 人单岗 + 10 人双岗 = 48 个选手-岗位资格。
        self.assertEqual(len(r3.observations), 48)

    def test_r1_r2_promotion_cap_per_role(self):
        for rid in ("R1", "R2"):
            decision = evaluate_round(self.ds, self.idx, self.round_by_id[rid], self.bridges)
            per_role = {role: 0 for role in ROLES}
            for d in decision.promotions:
                per_role[d.role] += 1
            self.assertTrue(all(n <= decision.cohort_size for n in per_role.values()))
            self.assertTrue(decision.promotions)
            self.assertTrue(all(d.sufficient for d in decision.promotions))

    def test_follow_window_only_uses_comparable_strata(self):
        r1 = evaluate_round(self.ds, self.idx, self.round_by_id["R1"], self.bridges)
        # R1→R2 是 v1→v2：除 support/ridge(cover) 外都没有兼容登记，
        # 这些晋级档案的后续检验必须显式标注不可比。
        for check in r1.follow_checks:
            if check.role != "support":
                self.assertFalse(check.comparable)
                self.assertIsNone(check.direction_held)
        support = [c for c in r1.follow_checks if c.role == "support"]
        self.assertTrue(any(c.comparable for c in support))


class TestTrials(EvidenceTestBase):
    def setUp(self):
        super().setUp()
        self.r3 = evaluate_round(self.ds, self.idx, self.round_by_id["R3"], self.bridges)
        self.plan = plan_trials(self.ds, self.r3, self.bridges)

    def test_each_session_ten_distinct_players_full_lineup(self):
        for session in self.plan.sessions:
            players = session.players
            self.assertEqual(len(players), 10)
            # 两边五岗各一人。
            slots = {(seat.side, seat.role) for seat in session.seats}
            self.assertEqual(len(slots), 10)
            for side in ("blue", "red"):
                self.assertEqual({r for s, r in slots if s == side}, set(ROLES))

    def test_same_slot_sessions_do_not_share_players(self):
        by_slot: dict[str, set[str]] = {}
        for session in self.plan.sessions:
            by_slot.setdefault(session.slot, set()).update(session.players)
        # 直接检查 Tue-AM 的 S1/S2 互斥。
        s1 = next(s for s in self.plan.sessions if s.session_id == "S1")
        s2 = next(s for s in self.plan.sessions if s.session_id == "S2")
        self.assertEqual(s1.slot, s2.slot)
        self.assertTrue(s1.players.isdisjoint(s2.players))

    def test_no_player_takes_two_qualities_in_one_session(self):
        for session in self.plan.sessions:
            self.assertEqual(len(session.players), len(session.seats))

    def test_confirmed_seats_preserved(self):
        for session in self.plan.sessions:
            rec = next(s for s in self.ds.seats if s["id"] == session.session_id)
            for pid in rec["confirmed"]:
                self.assertIn(pid, session.players)
                seat = next(x for x in session.seats if x.player == pid)
                self.assertEqual(seat.status, "confirmed")

    def test_availability_and_registration_respected(self):
        eligible = {p["id"]: {p["main"]} | ({p["secondary"]} if p["secondary"] else set())
                    for p in self.ds.players}
        for session in self.plan.sessions:
            rec = next(s for s in self.ds.seats if s["id"] == session.session_id)
            for seat in session.seats:
                self.assertIn(seat.player, rec["available"])
                self.assertIn(seat.role, eligible[seat.player])

    def test_version_reassessment_lists_impact_and_confirmed_seats(self):
        vc = self.plan.version_change
        # R3 目标 v3，场次含 v2 与 v3：重判必然产生站得住/站不住两类。
        self.assertEqual(vc.old_version, "v3")
        standing_roles = {x.split("/")[0] for x in vc.conclusions_standing}
        # 只有 jungle（sync/canal 有 v2-v3 登记）可能站得住。
        self.assertTrue(standing_roles <= {"jungle"})
        # 已通知席位单列，且三场的 confirmed 全部覆盖。
        confirmed_total = sum(len(s["confirmed"]) for s in self.ds.seats)
        self.assertEqual(len(vc.confirmed_seat_impact), confirmed_total)

    def test_r3_reassessment_only_jungle_canal_bridges(self):
        # 无历史版本结论的轮次也要逐条重判：仅 jungle/sync/canal 能借 v2 口径，
        # 其余组合在 v3 只能建立全新基线。
        vc = self.plan.version_change
        standing = set(vc.conclusions_standing)
        self.assertEqual(len(standing), 1)
        self.assertIn("jungle/sync/canal", next(iter(standing)))
        broken_labels = {x.split("：")[0] for x in vc.conclusions_broken}
        self.assertEqual(len(broken_labels), 14)
        self.assertIn("top/cover/ridge", broken_labels)
        self.assertIn("carry/sync/canal", broken_labels)

    def test_capacity_shortfall_is_explained_not_silently_dropped(self):
        # 48 个观察资格、30 个席位：安排不下的资格必须带原因。
        resolved = set()
        for key, note in self.plan.case_resolution.items():
            if "未能安排" not in note:
                resolved.add(key)
        # 每个资格都有去向说明（安排或阻塞原因），没有静默丢失。
        self.assertEqual(
            set(self.plan.case_resolution),
            {d.key for d in self.r3.observations},
        )
        self.assertLess(len(resolved), len(self.r3.observations))


class TestReport(EvidenceTestBase):
    def test_report_is_traceable_and_serializable(self):
        decisions = [evaluate_round(self.ds, self.idx, r, self.bridges)
                     for r in self.ds.rounds]
        plans = {d.round_id: plan_trials(self.ds, d, self.bridges) for d in decisions}
        payload = build_json_report(self.ds, self.idx, decisions, plans)
        # JSON 可序列化（机器可读，供下一周期复用）。
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("player_role_trace", json.loads(encoded))
        md = render_markdown(payload)
        self.assertIn("按人按角色追溯", md)
        self.assertIn("不可辨认的相对次序", md)
        # 双岗选手的两个岗位档案都可追溯到分层片段；即使 R3 目标版本
        # 没有数据，早期轮次积累的证据仍保留在追溯视图中。
        trace = payload["player_role_trace"]
        self.assertIn("P00|top", trace)
        self.assertIn("P00|jungle", trace)
        self.assertTrue(trace["P00|top"]["strata"])
        self.assertEqual(trace["P00|top"]["latest_round"], "R3")
        # R3 无目标版本数据：观察原因显式记录。
        r3 = next(rd for rd in payload["rounds"] if rd["round"] == "R3")
        self.assertTrue(all(o["reasons"] for o in r3["observations"]))


if __name__ == "__main__":
    unittest.main()
