# 青训选手跨版本协作能力的晋级证据

合作练习按五个位置记载。v1 至 v3 是规则版本，不是周次；历史窗口的结果可能晚一个资料周进入。

records 保存可离线使用的合成业务记录。每行一个 UTF-8 JSON 对象，id 在各数据表内唯一，跨表键沿用业务记录中的标识。catalog.json 保存行数及原始文件校验值。

| 记录 | 业务含义 |
| --- | --- |
| players.jsonl | 主位置和可替补位置为训练登记。 |
| games.jsonl | available_week 是教练拿到参赛名单和结果的周次。报名与实战以实际阵容区分。 |
| drills.jsonl | 每场实际的分岗协作练习，completed 以成功处理次数为单位。 |
| compatibility.jsonl | 指标的跨版本有效范围由教练组登记。未列出的组合缺少可以借用的依据。 |
| rounds.jsonl | decision_week 决策时点。随后 follow_until 用于回看结果；cohort_size 按位置表示最多晋级席。 |
| seats.jsonl | 每次试训同时需要十个互异选手，锁定集训为已承诺时间。同一人的两个资格不能在同一场同时使用。 |

## 核对资料

运行环境为 Python 3.11，使用标准库，无需下载依赖。用 `python3 -m unittest discover -s tests -v` 核对发行资料的完整性与约定的输入关系。

## 晋级证据系统（coaching 包）

`coaching/` 在上述资料上实现可解释的晋级证据与试训安排，全部使用标准库。

| 模块 | 职责 |
| --- | --- |
| `records.py` | 装载 JSONL、核对 catalog 摘要/行数/外键/阵容与练习一致性。 |
| `model.py` | 按 `(岗位, 版本, 地图)` 分层的 logistic 加性模型：个人岗位效应与场次情境（整队/对手）效应分离；二部共现图 + 同场同岗对手边给出可辨认连通分量。带岭正则与步长减半的 IRLS 求解。 |
| `evidence.py` | 选手-岗位证据档案；只按 `compatibility.jsonl` 显式登记做跨版本桥接；逆方差合并、充分性门槛与不可辨认对照标注。 |
| `rounds.py` | 按 `available_week` 切片的决策时点回看；晋级/席位备选/再观察分流；后续窗口只在可比岗位上检验方向。 |
| `trials.py` | 十人互异、同时段互斥、岗位登记与已承诺席位约束下的试训组合安排；赛前换版本的结论重判与席位影响单列。 |
| `report.py` | 按人按岗位追溯比赛片段、未知对照与下轮安排的 JSON/Markdown 报告。 |

关键纪律：

- 证据归属到 `(选手, 岗位)`；换岗位产生独立证据，指标（cover/sync）不相通，绝不直接拼接。
- 跨版本只沿兼容登记的链借用证据，未登记组合只作历史参照。
- 晚到赛况在其 `available_week` 之后才进入结论；相对次序不可辨认的选手对显式报告原因。
- 证据充分但受每岗席位限制的资格标为"备选"，与证据不足、必须再观察的资格分开处理。

### 运行

```bash
# 资料完整性与输入关系校验
python3 -I run_analysis.py verify --records records

# 生成三个轮次的证据与试训安排（output/evidence_report.{json,md}）
python3 -I run_analysis.py analyze --records records --out output

# 全部测试（发行资料测试 + 系统性质测试）
python3 -m unittest discover -s tests -v
```

`-I` 隔离模式忽略环境变量与用户站点包；不受信资料只以路径参数读取，装载时先校验后使用。

