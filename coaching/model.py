"""分层效应模型与可辨认性。

为什么不能用全训练期个人均分：
- 一条分岗练习的成败同时取决于个人、同队四人（整队效应）与对手；
- 游戏版本/地图更换后任务环境改变，旧打法可能被版本淘汰。

本模块把每条练习视为二项观测，在 *同一分层* ``(角色, 版本, 地图)`` 内拟合
加性 logistic 模型：

    logit P(单次成功) = alpha[选手,角色] + beta[场次情境]

其中 ``beta`` 吸收该场该边的整队/对手共同处境，``alpha`` 是剥离情境后的
个人岗位效应。指标天然由角色决定（top/support=cover，其余=sync），
不同角色的指标不同，模型从不把不同指标的观测放进同一分层。

可辨认性：构造 (选手-角色) 与场次情境的二部共现图，只有同一连通分量内的
两个选手-岗位效应之差可被资料辨认；跨分量（例如后期各自赛区内部循环、
缺少同期桥接赛）的相对次序必须显式报告为不可辨认。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .records import ROLES, ROLE_METRIC, Dataset, Indexes

# IRLS 迭代控制（纯标准库实现，数据规模下收敛很快）。
MAX_ITER = 100
TOL = 1e-7
# 对数几率的数值截断，避免 exp 溢出。
LOGIT_CLIP = 12.0
# 步长减半的最大次数。
MAX_STEPSIZE_HALVINGS = 12


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, LOGIT_CLIP))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -LOGIT_CLIP))
    return z / (1.0 + z)


def _solve_dense(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """带部分选主元的高斯消元，解稠密线性方程组。"""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            # 岭正则保证矩阵正定；走到这里说明数据异常，对该列做轻微扰动。
            a[pivot][col] = 1e-8 if a[pivot][col] >= 0 else -1e-8
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / piv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (a[i][n] - sum(a[i][j] * out[j] for j in range(i + 1, n))) / a[i][i]
    return out


@dataclass
class StratumFit:
    """一个 (角色, 版本, 地图) 分层的拟合结果。"""

    role: str
    version: str
    map_name: str
    metric: str
    # 选手-岗位 -> 效应（对数几率，分层内零均值口径由连通分量内部中心化给出）
    effects: dict[tuple[str, str], float]
    # 选手-岗位 -> 效应标准误（近似 Fisher 信息对角元）
    std_errors: dict[tuple[str, str], float]
    # 选手-岗位 -> 原始片段数 / 尝试数 / 成功数
    fragments: dict[tuple[str, str], int]
    attempts: dict[tuple[str, str], int]
    successes: dict[tuple[str, str], int]
    # 场次情境 id（"Gxxx-blue" 等）-> 情境效应
    context_effects: dict[str, float]
    # 连通分量 id -> 分量内选手-岗位集合
    components: dict[int, set[tuple[str, str]]]
    # 选手-岗位 -> 连通分量 id
    component_of: dict[tuple[str, str], int]
    # 每个分量的赛区构成，便于报告"哪些相对次序不可辨认"的原因
    component_regions: dict[int, set[str]]
    iterations: int
    converged: bool
    # 原始成功率（对照展示用，不参与跨层合并）
    raw_rates: dict[tuple[str, str], float] = field(default_factory=dict)

    def identifiable_pair(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        """两个选手-岗位的相对次序是否可被本分层资料辨认。"""
        return self.component_of.get(a) == self.component_of.get(b) and a in self.component_of


def _union_find(nodes: list[tuple[str, str]], contexts: list[str],
                context_members: dict[str, list[tuple[str, str]]],
                opponent_pairs: list[tuple[tuple[str, str], tuple[str, str]]]
                ) -> tuple[dict[tuple[str, str], int], dict[int, set[tuple[str, str]]]]:
    """构造辨认图。

    两类边共同决定相对次序是否可辨认：
    - 选手-岗位 与 场次情境（同队五人共享的整队/对手处境）；
    - 同一场比赛同岗位的两边选手（直接对位，是跨队/跨赛区的桥接边）。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for node in nodes:
        find(f"u|{node[0]}|{node[1]}")
    for ctx in contexts:
        ckey = f"c|{ctx}"
        find(ckey)
        for node in context_members[ctx]:
            union(ckey, f"u|{node[0]}|{node[1]}")
    for a, b in opponent_pairs:
        union(f"u|{a[0]}|{a[1]}", f"u|{b[0]}|{b[1]}")

    comp_of: dict[tuple[str, str], int] = {}
    groups: dict[int, set[tuple[str, str]]] = defaultdict(set)
    root_to_cid: dict[str, int] = {}
    for node in nodes:
        root = find(f"u|{node[0]}|{node[1]}")
        cid = root_to_cid.setdefault(root, len(root_to_cid))
        comp_of[node] = cid
        groups[cid].add(node)
    return comp_of, dict(groups)


def fit_stratum(ds: Dataset, idx: Indexes, role: str, version: str,
                map_name: str) -> StratumFit | None:
    """拟合单个分层；分层内没有观测时返回 None。"""
    metric = ROLE_METRIC[role]
    # 收集本分层全部练习观测。
    obs: list[dict] = []
    contexts: list[str] = []
    context_members: dict[str, list[tuple[str, str]]] = defaultdict(list)
    opponent_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    nodes: set[tuple[str, str]] = set()
    for g in ds.games:
        if g["version"] != version or g["map"] != map_name:
            continue
        blue_node = (g["lineups"]["blue"][role], role)
        red_node = (g["lineups"]["red"][role], role)
        opponent_pairs.append((blue_node, red_node))
        for side in ("blue", "red"):
            d = idx.drills_by_key[(g["id"], side, role)]
            node = (d["player"], role)
            ctx = f"{g['id']}-{side}"
            obs.append({"node": node, "ctx": ctx, "n": d["attempts"], "y": d["completed"],
                        "game": g["id"], "side": side})
            if ctx not in context_members:
                contexts.append(ctx)
            context_members[ctx].append(node)
            nodes.add(node)

    if not obs:
        return None
    node_list = sorted(nodes)
    comp_of, groups = _union_find(node_list, contexts, context_members, opponent_pairs)

    # 观测索引（按节点/情境聚合）。
    node_idx = {node: i for i, node in enumerate(node_list)}
    ctx_idx = {ctx: i for i, ctx in enumerate(contexts)}
    node_obs: list[list[int]] = [[] for _ in node_list]
    ctx_obs: list[list[int]] = [[] for _ in contexts]
    for k, o in enumerate(obs):
        o["ni"] = node_idx[o["node"]]
        o["ci"] = ctx_idx[o["ctx"]]
        node_obs[o["ni"]].append(k)
        ctx_obs[o["ci"]].append(k)

    n_u = len(node_list)
    n_c = len(contexts)
    # 岭正则参数：对完全分离（0/n 或 n/n 成功）给出有限估计，量级取
    # 单次尝试信息量的一小部分，弱信号下几乎不改变点估计。
    ridge = 0.5
    alpha = [0.0] * n_u
    beta = [0.0] * n_c

    def penalized_loglikelihood(a: list[float], b: list[float]) -> float:
        """二项对数似然 + 岭惩罚（步长减半的单调目标）。"""
        total = -0.5 * ridge * (sum(x * x for x in a) + sum(x * x for x in b))
        for o in obs:
            eta = a[o["ni"]] + b[o["ci"]]
            # log-likelihood: y*log(p) + (n-y)*log(1-p)，稳定形式。
            total += o["y"] * eta - o["n"] * math.log1p(math.exp(min(eta, LOGIT_CLIP)))
        return total

    def center(new_alpha: list[float], new_beta: list[float]
               ) -> tuple[list[float], list[float]]:
        """分量内中心化：约束分量内选手效应均值为 0，吸收整体平移。"""
        comp_sum: dict[int, float] = defaultdict(float)
        comp_n: dict[int, int] = defaultdict(int)
        for i, node in enumerate(node_list):
            cid = comp_of[node]
            comp_sum[cid] += new_alpha[i]
            comp_n[cid] += 1
        comp_mean = {cid: comp_sum[cid] / comp_n[cid] for cid in comp_sum}
        ca = [new_alpha[i] - comp_mean[comp_of[node]]
              for i, node in enumerate(node_list)]
        cb = [
            new_beta[j] + comp_mean[comp_of[obs[ctx_obs[j][0]]["node"]]]
            for j in range(n_c)
        ]
        return ca, cb

    converged = False
    iterations = 0
    current_obj = penalized_loglikelihood(alpha, beta)
    for iterations in range(1, MAX_ITER + 1):
        old = alpha + beta
        # 工作权重与工作因变量（logistic IRLS）。
        w = [0.0] * len(obs)
        z = [0.0] * len(obs)
        for k, o in enumerate(obs):
            eta = alpha[o["ni"]] + beta[o["ci"]]
            p = _sigmoid(eta)
            w[k] = o["n"] * p * (1.0 - p)
            z[k] = eta + (o["y"] - o["n"] * p) / (o["n"] * p * (1.0 - p) + 1e-12)

        # 正规方程分块（x = [alpha; beta]）：
        # [A  C] [alpha]   [ua]
        # [Ct B] [beta ] = [ub]
        A = [ridge] * n_u
        B = [ridge] * n_c
        C: list[dict[int, float]] = [dict() for _ in range(n_u)]
        ua = [0.0] * n_u
        ub = [0.0] * n_c
        for k, o in enumerate(obs):
            i, j = o["ni"], o["ci"]
            A[i] += w[k]
            B[j] += w[k]
            C[i][j] = C[i].get(j, 0.0) + w[k]
            ua[i] += w[k] * z[k]
            ub[j] += w[k] * z[k]

        # 块消元：alpha = A^-1 (ua - C beta)，代入得 Schur 补方程。
        S = [[0.0] * n_c for _ in range(n_c)]
        rhs = ub[:]
        Ainv = [1.0 / a for a in A]
        for i, row in enumerate(C):
            for j, cij in row.items():
                rhs[j] -= cij * ua[i] * Ainv[i]
                for j2, cij2 in row.items():
                    S[j][j2] += cij * cij2 * Ainv[i]
        for j in range(n_c):
            S[j][j] = B[j] - S[j][j]
        full_beta = _solve_dense(S, rhs)
        full_alpha = [
            (ua[i] - sum(cij * full_beta[j] for j, cij in C[i].items())) * Ainv[i]
            for i in range(n_u)
        ]
        full_alpha, full_beta = center(full_alpha, full_beta)

        # 步长减半：IRLS 在完全分离附近会越跳越大，用目标函数单调性约束。
        step = 1.0
        chosen = None
        for _ in range(MAX_STEPSIZE_HALVINGS + 1):
            cand_a = [alpha[i] + step * (full_alpha[i] - alpha[i]) for i in range(n_u)]
            cand_b = [beta[j] + step * (full_beta[j] - beta[j]) for j in range(n_c)]
            cand_a, cand_b = center(cand_a, cand_b)
            obj = penalized_loglikelihood(cand_a, cand_b)
            if obj >= current_obj - 1e-10:
                chosen = (cand_a, cand_b, obj)
                break
            step *= 0.5
        if chosen is None:
            # 极小步长仍无改进，视为数值收敛。
            converged = True
            break
        alpha, beta, current_obj = chosen

        delta = max(abs((alpha + beta)[i] - old[i]) for i in range(n_u + n_c))
        if delta < TOL:
            converged = True
            break

    # 岭正则下的近似标准误：选手效应边际方差取 (信息对角元 + ridge)^-1。
    se = [0.0] * n_u
    for i, ks in enumerate(node_obs):
        info = ridge
        for k in ks:
            o = obs[k]
            p = _sigmoid(alpha[o["ni"]] + beta[o["ci"]])
            info += o["n"] * p * (1.0 - p)
        se[i] = 1.0 / math.sqrt(info)

    def _sum(field_name: str) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = defaultdict(int)
        for o in obs:
            out[o["node"]] += o[field_name]
        return dict(out)

    fragments: dict[tuple[str, str], int] = defaultdict(int)
    for o in obs:
        fragments[o["node"]] += 1
    attempts = _sum("n")
    successes = _sum("y")
    raw_rates = {node: successes[node] / attempts[node] for node in node_list}

    component_regions: dict[int, set[str]] = {}
    for cid, members in groups.items():
        component_regions[cid] = {idx.player[pid]["region"] for pid, _ in members}

    return StratumFit(
        role=role,
        version=version,
        map_name=map_name,
        metric=metric,
        effects={node: alpha[node_idx[node]] for node in node_list},
        std_errors={node: se[node_idx[node]] for node in node_list},
        fragments=dict(fragments),
        attempts=attempts,
        successes=successes,
        context_effects={ctx: beta[ctx_idx[ctx]] for ctx in contexts},
        components=groups,
        component_of=comp_of,
        component_regions=component_regions,
        iterations=iterations,
        converged=converged,
        raw_rates=raw_rates,
    )


def fit_all_strata(ds: Dataset, idx: Indexes, *, game_filter=None
                   ) -> dict[tuple[str, str, str], StratumFit]:
    """拟合资料中出现的全部 (角色, 版本, 地图) 分层。

    game_filter 接收 game 记录，返回 False 则排除（用于决策时点切片）。
    """
    versions = sorted({g["version"] for g in ds.games if (game_filter is None or game_filter(g))})
    maps = sorted({g["map"] for g in ds.games if (game_filter is None or game_filter(g))})
    out: dict[tuple[str, str, str], StratumFit] = {}
    for role in ROLES:
        for version in versions:
            for map_name in maps:
                fit = fit_stratum(ds, idx, role, version, map_name)
                if fit is not None:
                    out[(role, version, map_name)] = fit
    return out
