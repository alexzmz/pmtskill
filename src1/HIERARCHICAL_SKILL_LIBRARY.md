# 从已有 SQLite 生成可在线评测的分层库

默认 `maintain` 的严格策略保持不变。覆盖优先实验用 `--success-once`：至少一条
结构有效、由 AndroidWorld evaluator 判定成功的示范即可准入 active；接受路径质量
为 candidate 的成功轨迹，但排除空动作、未提交动作和基础设施异常。这个 active
只表示教师示范准入，不等于学生模型通过标定；不生成虚假的 SR、能力或 calibration trials。

```bash
python -m src1 --config src1/config.local_all.toml maintain \
  --source-database /data/skill_library.sqlite3 \
  --output-database /data/online_library/skill_library.sqlite3 \
  --output-dir /data/online_library \
  --skip-import --success-once
```

输出数据库必须是新路径。SQLite backup 保留输入库以及原始 raw / traces；
`--skip-import` 不再读取上游技能目录，覆盖优先模式也不会调用云端 raw 编译器。
没有 `--output-database` 时，会在选中的源库上原地维护。

分层结构：

- L0：`primitive_catalog` 中的 26 个原语。
- L1：真实已提交的 GUI 操作，正文来自示范当时的 planned instruction。
- L2：连续操作组成的子流程，每块默认 3 个操作。
- L3：完整成功任务路径，保留不同语义 / 路径的变体。

`skills.level` 是语义层级，不再是原语数量。`skill_relations.composes_polished`
和 `metadata.components` 保存高层到低层的引用及有序操作 span；
`fallback_topology` 保留完整展开路径。只有 trace 明确记录输入 raw 来源才创建
`derived_from_raw`，不会凭猜测把任务轨迹归因到所有 raw 技能。

示范输入文本、姓名和数字会参数化为 `<goal_value>`，运行时由当前目标填入。
不会直接回放示范的 Action JSON、坐标、索引或最终答案。生成不需要大模型请求。
这是一份确定性的示范技能库，不是 LLM 语义总结或模型专属优化后的库。

输出有 `coverage.json`、`coverage.md`、`skills.jsonl`、`hierarchy.json`。覆盖率的
分母是输入库中出现的任务，不是整个 AndroidWorld registry；无成功示范的任务
会明确列在 `uncovered_tasks`，仍可在线评测但没有成功证明的 L3 技能。

也可以在自己的 TOML 中持久配置：

```toml
[maintenance]
admission_policy = "success_once"
build_hierarchy = true
hierarchy_block_size = 3
```

在线测评沿用你原有的 adapter / 模型配置，只覆盖技能库：

```bash
python -m src1 --config src1/config.local_all.toml evaluate-pmtskill \
  --adapter-paths /your/adapter/run \
  --skill-database /data/online_library/skill_library.sqlite3 \
  --combinations 2 --seed 123 --max-steps 25
```

不需要 `--include-candidates`。在线只从目标文本检索 L3（不使用 task ID 或 evaluator
答案），展开原语路径，并将路由候选限制为该变体的后代，以免同样的 click 原语
注入另一任务的 Delete / Save 指令。重复原语保持顺序，层级子流程不会只执行一个
动作就跳到下一路由单元。检索不匹配时回退原有 raw + 原语规划。

已有模型兼容矩阵与 deprecated 状态继续生效，不会用一次教师成功抹掉学生失败。
需要确认学生增益时仍使用 `calibrate-skills` 的 no-skill / forced-skill 配对实验。
建议换 seed / 任务参数测评；同任务示范已用于建库，不应据此宣称零样本泛化。
