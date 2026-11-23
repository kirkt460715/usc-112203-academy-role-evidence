"""命令行入口。

用法（在仓库根目录）：

    python3 -I -m coaching.cli analyze --records records --out output
        生成全部轮次的 JSON + Markdown 证据报告

    python3 -I -m coaching.cli verify --records records
        只做资料完整性与输入关系校验
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evidence import BridgeBook
from .records import load_dataset
from .report import build_json_report, render_markdown, write_json_report
from .rounds import evaluate_round
from .trials import plan_trials


def cmd_verify(args: argparse.Namespace) -> int:
    ds = load_dataset(args.records)
    print(f"校验通过：{', '.join(f'{n}={len(getattr(ds, n))}' for n in ds.__dataclass_fields__)}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    ds = load_dataset(args.records)
    idx = ds.index()
    bridges = BridgeBook.from_dataset(ds)

    decisions = []
    for record in ds.rounds:
        decision = evaluate_round(ds, idx, record, bridges)
        decisions.append(decision)

    trial_plans = {}
    for decision in decisions:
        # 试训资料只有一批（week 13），对每个轮次都给出"若现在安排"的方案；
        # 版本重判以场次实际版本与轮次目标版本的差异自动触发。
        trial_plans[decision.round_id] = plan_trials(ds, decision, bridges)

    payload = build_json_report(ds, idx, decisions, trial_plans)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "evidence_report.json"
    md_path = out / "evidence_report.md"
    write_json_report(payload, json_path)
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    for decision in decisions:
        print(f"[{decision.round_id}] 目标版本 {decision.target_version} "
              f"决策周 {decision.decision_week}: "
              f"晋级 {len(decision.promotions)} 人/岗，观察 {len(decision.observations)} 个资格，"
              f"不可辨认提示 {len(decision.unidentifiable_notes)} 条，"
              f"后续检验 {len(decision.follow_checks)} 项")
    print(f"报告已写入 {json_path} 与 {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="青训选手跨版本协作能力的晋级证据系统")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="校验资料完整性与输入关系")
    p_verify.add_argument("--records", default="records", help="records 目录路径")
    p_verify.set_defaults(func=cmd_verify)

    p_analyze = sub.add_parser("analyze", help="生成晋级证据与试训安排报告")
    p_analyze.add_argument("--records", default="records", help="records 目录路径")
    p_analyze.add_argument("--out", default="output", help="报告输出目录")
    p_analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
