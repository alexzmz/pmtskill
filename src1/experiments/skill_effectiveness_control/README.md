# 当前技能库的人工配对结果

输入是你提供的 `skills_catalog.json`：106 raw、334 polished；之前的模板包含 43 个 active L3 候选。
本次只创建配对文件，不改动已有技能正文、数据库、程序或测试结果。

## 已配置的 4 对

| pair_id | Raw | Polished 对应任务 | 原语节点数 |
|---|---|---|---:|
| files-delete | skvm:file-manager | FilesDeleteFile | 7 |
| files-move | skvm:file-manager | FilesMoveFile | 12 |
| calendar-relative-day | skvm:calendar | SimpleCalendarAddOneEventRelativeDay | 16 |
| calendar-repeating | skvm:calendar | SimpleCalendarAddRepeatingEvent | 18 |

这些原语序列已经在目录中存在，未删除重复动作。每对均使用既有 raw/polished ID。
配对依据是：当前 Android raw 的能力范围覆盖该 AndroidWorld 目标，polished 中存在该任务的完整工作流。
删除/移动的源技能说明确实有删除/移动功能；日历源技能说明有相对日期解析和重复事件字段。
没有根据成功率挑选配对或填造实验结果，选择阶段没有调用目标模型。

**重要限制：不是已验证的同一原始技能编译前后版本。**

- 这 4 个 polished 的 `metadata.source_skill_ids` 都为空，没有直接生成链证据。
- `related_raw_skill_ids` 是分类关联，不证明来源；本文件不把它当成生成链。
- 文件中的 `equivalence_confirmed=true` 仅表示审阅确认“同一个具体任务目标处于 raw 的能力范围内”，
  让当前 loader 可以运行；不是形式化程序等价、逐步骤等价或原始来源确认。
- 当前程序实际注入的是已有 `SkillRecord.body`。文件管理 raw 正文仅 23 个字符，日历 raw 仅 17 个字符；
  它们已经由之前的 `llm-raw-skill-compiler-v1` 适配为 Android 指令。
- `metadata.original_body` 保存的完整来源文本分别有 2239、1351 个字符，但当前实验不会使用该字段。
  **不要把结果表述为“完整原始 SKILL.md 与其编译优化后的同一技能”的比较。**
- 如果论文要主张“同一技能优化前后”，还需要准备具有真实 raw→polished 来源链的技能对，而不是放宽上述事实。

其余 39 个 active L3 不启用，具体任务与原因分组记录在 `pairs.json.excluded_active_l3_candidates`。
例如没有 Contacts 的 raw，不能用任意点击/填表技能伪装成联系人新增技能；Obsidian 专用 raw 也不强配 Markor。
此选择只覆盖两个能力族、四种任务，不能推广为整个 AndroidWorld 的全面结果。

## 如何使用

将本文件夹同步到服务器 `src1/experiments/skill_effectiveness_control/`。
你的原命令只需把 `--pairs-file` 改成：

```bash
--pairs-file /home/zmz/Workspace/gui/src1/experiments/skill_effectiveness_control/pairs.json
```

完整命令（沿用你这次的 CUDA 2 / 模型服务 8003）：

```bash
python -m src1 --config /home/zmz/Workspace/gui/src1/config.local_test_effectiveness.toml \
  --log-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/skill_effectiveness_control/cli_logs \
  evaluate-skill-effectiveness \
  --adapter-path /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
  --model-id adapter_control01 \
  --skill-database /home/zmz/Workspace/gui/src1/runtime/evaluations/skill_effectiveness_control/inputs/skill_library.sqlite3 \
  --pairs-file /home/zmz/Workspace/gui/src1/experiments/skill_effectiveness_control/pairs.json \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 2 --deploy-port 8003 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories \
  --output-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/skill_effectiveness_control
```

第一次可先追加 `--dry-run`。这 4 对 × 3 条件 × 1 组合，一共 12 个 episode，不是测试全部 334 个 polished。
如果之前只有“没有启用配对”的报错，还没产生 experiment_manifest/conditions，可继续使用上述目录。
如果这个目录已经开始跑过另一组配对或配置，必须使用新 `--output-dir`，保留旧记录，不要删除 manifest 绕过校验。
设备 console/grpc 继续读取 `config.local_test_effectiveness.toml`，它们与 CUDA 2、推理端口 8003 无直接映射关系。

如先做最小检查，用 `pairs.smoke.json` 替换上面的 `--pairs-file`，并同时把 `--output-dir` / `--log-dir` 改到单独的
`.../skill_effectiveness_control_smoke` 目录。其余参数不变，共 3 个 episode。不要用同一个结果目录先测 1 对再续跑 4 对。

## 更准确的论文表述

> We compare existing Android-adapted raw guidance, task-specific polished
> workflows, and their flat primitive implementations under the same saved
> AndroidWorld task goals and a fixed model. Raw guidance is manually matched
> by task-objective coverage; direct raw-to-polished compilation lineage is
> not established for these pairs.

这组实验是当前库中“通用 raw 指导 vs 原语序列 vs 任务级 polished 指导”的受控比较。
任何成功率提升都必须等实际评测后才能给出；此处仅做静态输入配置与审核。
