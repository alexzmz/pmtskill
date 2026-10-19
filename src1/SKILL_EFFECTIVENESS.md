# Polished Skill Effectiveness：PORTS 固定模型三组对照

本实验对应论文的 “Polished Skill Effectiveness”，不是五/八个方法的 end-to-end 对比。

当前实验协议为 `ports-skill-effectiveness-v2`：修复 CalendarEvent 参数恢复，并为三组提供相同的应用名称约束。
旧版 v1 目录不能续跑为 v2；保留旧结果，使用新的 `--output-dir`。协议不匹配会在部署 GPU 服务前拒绝。

固定同一个 adapter、同一个 AndroidWorld 任务实例，仅改变技能的表示：

| 条件 | 给模型的技能指导 |
|---|---|
| `raw` | 人工指定的等价原始 raw skill 原文，使用 PORTS 的技能注入路径 |
| `primitives` | 对应 polished skill 拓扑的完整扁平原语序列、原语定义和参数；不注入 raw/polished 原文 |
| `polished` | polished skill，复用 PORTS `RoutedVLWrapper`，已有 stepwise 技能沿用逐步指导 |

三组均使用 M3A 执行单个真实 GUI 动作、AndroidWorld task evaluator 判分。只部署一个 adapter，
不运行 DP 选路、不切换模型、不自动检索别的技能、不改变技能状态或模型标定。
这能隔离技能表示的影响，但不是完整 PORTS 系统的性能测试，也不是对片段的独立判分。

特别注意：每组仍然会调用动作模型与总结模型。**一个 polished skill 不是“一次模型调用执行完所有 GUI 操作”**。
是否减少步骤、调用或延迟，必须由实验结果决定，代码不预设 polished 获胜。

## 1. 准备环境与冻结输入

在服务器的项目根目录运行，使用已有 `pmtskill` 环境、AndroidWorld 模拟器及 ms-swift/vLLM。
本功能不需要新增 submodule，也不需要额外教师模型或 LLM judge。

以下示例使用 control adapter，CUDA 0 / 推理端口 8002 / emulator 5554 / gRPC 8554。
如果这些资源正在跑 V-Droid 或其他实验，请更换空闲资源；不能并发操控同一个模拟器。

确认 TOML：

```toml
[android_world]
console_port = 5554
grpc_port = 8554
```

这两个端口以你的 emulator 启动命令 `-port` / `-grpc` 为准，不通过 GPU 编号推导。
基座模型、adb 等路径继续使用你现有 TOML；不需要先执行 `calibrate-skills`。

```bash
cd /home/zmz/Workspace/gui

CFG=src1/config.local_test.toml
OUT=/home/zmz/Workspace/gui/src1/runtime/evaluations/skill_effectiveness_control_v2
DB=/home/zmz/Workspace/gui/src1/runtime/evaluations/compare_five_control/ports/skill_library.sqlite3
CONTROL=/home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01

python -m src1 --config "$CFG" prepare-skill-effectiveness \
  --skill-database "$DB" \
  --output-dir "$OUT"
```

产物：

```text
skill_effectiveness_control/
  inputs/
    skill_library.sqlite3      # SQLite backup 快照，不污染原库
    skills_catalog.json        # raw/polished 的 ID、原文、拓扑和元数据
    pairs.example.json         # active L3 配对候选，默认全部未启用、未确认
```

重复准备不会覆盖已有快照或模板。希望使用新版技能库时，创建新的实验目录。

## 2. 指定真正目标等价的技能对

```bash
cp "$OUT/inputs/pairs.example.json" "$OUT/inputs/pairs.json"
```

编辑 `pairs.json`，先保留 1 个明确的完整任务技能对进行检查，结构示例：

```json
{
  "pairs": [
    {
      "pair_id": "contacts-add",
      "enabled": true,
      "equivalence_confirmed": true,
      "raw_skill_id": "填写 catalog 中确实存在的 raw skill_id",
      "polished_skill_id": "填写 catalog 中对应的 polished skill_id",
      "tasks": ["ContactsAddContact"]
    }
  ]
}
```

要求：

- 三组针对同一执行目标，例如“添加联系人”，不能用“搜索联系人”的 raw 与“添加联系人”的 polished 配对。
- 优先选择 L3 完整任务技能；L1/L2 也能指定，但其结果是**使用片段指导时的整个任务 SR**，不是该片段独立 SR。
- 模板中的 `provenance_raw_candidates` 来自来源元数据，仍需检查语义等价。
- `taxonomy_only_raw_candidates` 只是分类关联，**不能当作原始 raw → polished 的生成关系或等价证明**。
- 确认后才将 `enabled`、`equivalence_confirmed` 设为 `true`；缺少可确认的 raw 时不要伪造对应关系。
- 原语条件严格展开 polished 的原始拓扑，不做去重；顺序、重复原语、节点参数均保留。
- 当前实验要求线性、无条件的技能拓扑；不能把条件 DAG 随意展开并声称等价，遇到此类技能会拒绝运行。

如果库中没有 active L3，模板可能为空。可以根据 `skills_catalog.json` 手动添加有明确等价关系的 pair。
本实验不会自动将 candidate 改为 active；显式配对允许测 candidate，但论文应披露其筛选状态。

## 3. 先做不启动 GPU/模拟器的检查

```bash
python -m src1 --config "$CFG" evaluate-skill-effectiveness \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --skill-database "$OUT/inputs/skill_library.sqlite3" \
  --pairs-file "$OUT/inputs/pairs.json" \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --no-store-trajectories \
  --output-dir "$OUT" --dry-run
```

检查输出中解析到的 checkpoint、技能 ID、任务集合及每组 episode 数。此时不执行标定或测评。

## 4. 启动三组配对测试

```bash
python -m src1 --config "$CFG" \
  --log-dir "$OUT/cli_logs" \
  evaluate-skill-effectiveness \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --skill-database "$OUT/inputs/skill_library.sqlite3" \
  --pairs-file "$OUT/inputs/pairs.json" \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories \
  --output-dir "$OUT"
```

首次建议 1 个 pair、1 个任务、1 个组合，共 3 个 episode。确认三组都产生真实有效结果后，再增加配对及
`--combinations`，并使用新的输出目录重新准备输入。正式实验可按预算选 5 或 10 个组合；不保证这些样本已足以证明统计显著性。

`--tasks` 可进一步限制 pairs 文件中的任务范围，但不会把不相关技能配到未声明的任务。
`--guidance-max-chars 12000` 控制原文/扁平序列的预算；超长时会停止提示，增大此值和服务上下文预算后重开实验，
不静默删除原语或截断 raw 原文。stepwise polished 使用技能原有逐步提示预算。

## 5. 如何保证配对与续跑

- 同一个 pair 的三组共享 `suites/<hash>.json` 中保存的**完整任务参数和 goal**，不仅是相同 seed。
- 日历目标与噪声事件按带类型 JSON 保存，恢复时重建 `CalendarEvent`，保留字段和属性；不是将对象转为字符串或普通字典。
  同时支持原生日期/时间、tuple、set 等参数；仅允许重建固定 AndroidWorld SQLite schema 模块中的数据类型，不使用 pickle/eval 或任意模块加载。
- 因此像 `MarkorCreateFolder` 这种含当前时间戳的参数，后两组及续跑均恢复第一次保存的值，不会重新生成。
- 每组分别调用 task 初始化/清理流程；这不是模拟器完整状态快照，也不声称所有随机背景状态完全一致。
- 所有组执行相同的 `max-steps` 固定预算及相同的 stop-on-success 配置。
- 三组顺序在不同 pair 间确定性轮换，不根据成功率挑选顺序；服务只部署一次，GPU 配置相同。
- 三组收到完全相同的公开应用名称提示，名称从当前 harness 的映射读取。例如文件应用是 `files`，不是 raw 技能名 `file-manager`；
  日历应用是 `simple calendar pro`。不会根据任务答案替模型选动作，也不改写模型输出、补装应用或伪造成功。模型继续选错时仍按真实执行结果判分。
- `--resume` 跳过已正常完成的实例，包括真实失败；异常实例重跑。已有参数、模型权重、技能定义或实验设置变化时拒绝混合结果。
- 默认不存截图/完整轨迹，也不写回来源 SQLite；轻量 checkpoint 始终保留计数器。
- 基础设施故障会中止，保留已有结果并返回失败；全部无效不能当 SR=0。修复后用完全相同的命令续跑。

注意最后一条指的是**相同 v2 协议下**的中断恢复。此次 v1→v2 同时改变了三组提示，不能继续复用旧六个文件任务结果。

## 6. 报告与论文表格

```text
experiment_manifest.json          # 固定实验身份
suites/                          # 三组共享的任务实例参数
conditions/<pair-hash>/raw/       # 每组独立报告/checkpoint
conditions/<pair-hash>/primitives/
conditions/<pair-hash>/polished/
summary.json                     # 三组汇总、逐 pair、共同成功子集
comparison.csv                   # 论文主表的三行数值
trials.jsonl / trials.csv         # 每个 pair × task × instance × condition
report.md                        # 人可读总览
table_skill_effectiveness.tex    # booktabs 表格，可 \input 到论文
```

主表仅使用**三组均有效**的相同实例，以相同分母计算 SR。真实失败保留在分母中；任何一组异常，
该三元组不进入主表，但异常和已完成条件仍保留在 trials。全部配对未完成时 CLI 返回失败，不能当完整结果使用。

指标定义：

- **Execution success**：AndroidWorld 官方任务 evaluator 的 whole-task SR；不是模型自报成功，不是启发式 H。
- **Interaction length**：实际 M3A 决策步数，包括终止/回答步骤；不把一个宏技能人为计为一个交互。
- **Model invocations**：动作和总结阶段的 HTTP 请求尝试总数，包含客户端重试；同时输出 `action_calls` / `summary_calls`。
- **Latency**：M3A step 的累计墙钟秒数，包含模型与设备交互，不含服务部署和 task 初始化。`model_latency_seconds` 另给出模型调用等待时间。
- **all_three_successful_table**：仅比较同一批三组都成功的实例的效率，避免“提前失败因为步数短，看起来更高效”的误解。

多 pair 汇总按测试实例 micro-average，每个 pair 的结果另存 `summary.json.per_pair`。
如果在多个 seed 下重复实验，用不同输出目录，并在论文中报告均值/方差；当前不会自动做显著性检验。

可在论文中准确描述为：

> We compare manually matched raw skills, their polished counterparts, and the
> flat primitive implementations of the polished skills, using a fixed model
> and identical saved AndroidWorld task parameters. We disable model routing
> and skill retrieval to isolate the effect of skill representation.
