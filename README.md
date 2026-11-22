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
