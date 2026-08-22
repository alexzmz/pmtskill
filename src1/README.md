# PMT-Skill v2：AndroidWorld VL 蒸馏与动态技能路由

本目录是一个与旧 `src` 完全隔离的新框架。它按照 PPT 的思路实现三条闭环：

1. **离线监督蒸馏**：教师 VL 模型在 AndroidWorld 生成带截图的
   `[Reason-Action-Observation]` 轨迹，学生 VL 模型通过 ms-swift LoRA 学习。
2. **在线动态执行**：任务先转换为 raw skill topology，再展开为 26 个原语，
   路由器在“模型/LoRA 池 × polished-skill 图”上动态选择执行路径。
3. **backend 技能优化**：设备上传成功/失败轨迹；backend 挖掘高频成功子路径，
   生成候选 polished skill，经过真实 trial 后晋升，退化时自动回滚。

旧目录 `src` 不会被导入或修改。新框架只读取：

- `libs/android_world`：环境、M3A agent、任务 evaluator 和 checkpointer；
- `libs/skvm/skvm-data/skills`：raw skill 初始来源；
- `libs/ms-swift`：学生 VL 模型 LoRA 训练。

## 1. 架构

```text
离线
  AndroidWorld tasks
       ↓ 教师 VL + M3A
  .pkl.gz episodes（截图、SoM、prompt、Reason/Action）
       ↓ DatasetBuilder（按 episode 切分）
  ms-swift 多图 JSONL
       ↓ MSSwiftLoraTrainer
  通用 LoRA / grounding LoRA / planning LoRA / action LoRA

在线设备
  goal → raw skill topology → 26 原语 topology
       → DP Router（成功率、延迟、skill 降级、adapter 切换）
       → M3A + 动态 model/adapter + polished skill
       → AndroidWorld evaluator → SR / trace

云侧 backend
  trace → 高频成功子序列 → candidate polished skill
       → 灰度 trial → promote / rollback → 更新技能图和模型画像
```

## 2. 目录和替换点

```text
src1/
├── config.example.toml             # 唯一运行配置入口
├── resources/primitives.json       # 26 个原语，可扩展
├── pmtskill_v2/
│   ├── core/                        # 数据契约、配置、原子 IO
│   ├── inference/                   # OpenAI-compatible VL 与模型池
│   ├── offline/                     # 教师采集、数据转换、ms-swift 训练
│   ├── online/                      # 规划、DP 路由、执行、backend
│   ├── skills/                      # SQLite 技能库、SKVM 导入、维护
│   └── evaluation/                  # AndroidWorld runner 与报告
└── tests/                           # 不需要 emulator/GPU 的单元测试
```

主要模块化接口如下：

- 改路由算法：实现 `online.router.RoutingAlgorithm.route()`；
- 改训练算法：实现 `offline.trainer.TrainingAlgorithm`；
- 改任务分解：实现 `online.planner.SkillTopologyPlanner`；
- 改技能生成方式：实现 `skills.maintenance.SkillCompiler`；
- 改模型服务：实现 `inference.vlm.VLModelClient`；
- 改设备执行层：实现 `online.executor.ExecutionBackend`。

## 3. 配置

先复制配置样例并修改 Linux 机器上的模型、adb 和服务地址：

```bash
cp src1/config.example.toml src1/config.local.toml
```

关键项：

- `offline.teacher_model_id` 必须匹配一个 `[[models]].model_id`；
- `offline.student_model_path` 是 ms-swift 训练的学生 VL 基座；
- 每个 `[[models]]` 是在线可选的基座或 LoRA served model；
- `[models.capabilities]` 保存该模型在每个原语上的成功率画像；
- `android_world.max_steps = 0` 表示不设置 agent steps 上限；任务 evaluator
  一旦成功仍会因 `stop_on_task_success = true` 正常停止；
- API key 只通过 `api_key_env` 指定的环境变量读取，不写入 TOML。

初始化数据库、导入 SKVM 技能并登记模型：

```bash
python -m src1 --config src1/config.local.toml init
python -m src1 --config src1/config.local.toml doctor
```

SKVM 中的技能来源广泛，导入后状态是 `imported`，**不会直接在线执行**。
当前仓库的 SKVM skills 没有显式 Android/ADB 技能，因此不能用 `screen/click`
之类宽泛词盲目激活。可以让云模型逐批做相关性判断、原语映射和移动端适配：

```bash
python -m src1 --config src1/config.local.toml compile-skills \
  --model-id teacher-glm-vl --limit 8
```

也可在 `[maintenance]` 中设置
`raw_skill_compiler_model_id = "teacher-glm-vl"`，之后每次 `maintain` 自动编译一批。
只有编译器明确批准的 raw skill 才进入规划候选；polished skill 还需真实 trial
才能 `active`。

## 4. 离线监督蒸馏

### 4.1 启动教师 VL 服务

配置示例假设教师服务兼容 OpenAI `/v1/chat/completions`。可用 vLLM、SGLang
或已有服务启动 `/home/zmz/Workspace/models/glm4.1-9b`，并让
`teacher-glm-vl.base_url` 指向它。框架会以两个图像输入调用教师：原始截图和
带 UI 索引的 SoM 截图。

### 4.2 指定任务采集教师轨迹

```bash
python -m src1 --config src1/config.local.toml collect \
  --tasks ContactsAddContact SimpleCalendarAddOneEvent \
  --combinations 10 --seed 42
```

`--tasks` 正是“选择哪些任务做蒸馏”的参数；也接受逗号分隔。不传时运行整个
family。输出沿用 AndroidWorld `IncrementalCheckpointer` 的 `.pkl.gz`，因此中断后
用同一目录可续跑。

需要启动emulator和教师模型服务

### 4.3 生成 ms-swift 多模态数据

```bash
python -m src1 --config src1/config.local.toml build-dataset
```

每个有效 step 生成：

```json
{
  "messages": [
    {"role": "user", "content": "<image><image>\n<AndroidWorld action prompt>"},
    {"role": "assistant", "content": "Reason: ...\nAction: {...}"}
  ],
  "images": ["/absolute/raw.png", "/absolute/som.png"],
  "metadata": {"task_name": "...", "episode_id": "...", "primitives": ["action.click"]}
}
```

默认只学习成功 episode；训练/验证按完整 episode 切分，避免一条轨迹的相邻
截图同时出现在两边。

### 4.4 训练通用或定向 LoRA

先检查实际 ms-swift 命令：

```bash
python -m src1 --config src1/config.local.toml train --dry-run
```

训练一个通用 AndroidWorld LoRA：

```bash
python -m src1 --config src1/config.local.toml train \
  --adapter-name android_world_all
```

按 PPT 的 split/branch-merge 思路训练定向 adapter：

```bash
python -m src1 --config src1/config.local.toml train \
  --adapter-name ui_grounding \
  --primitives ground.text ground.icon ground.bbox action.click

python -m src1 --config src1/config.local.toml train \
  --adapter-name planning \
  --primitives reason.intent reason.decompose reason.verify reason.recover
```

`--extra-arg` 可重复传给 ms-swift，例如
`--extra-arg=--bf16 --extra-arg=true`。训练完成后把 adapter 作为新的 `[[models]]`
登记，并给出初始能力画像。

## 5. 在线动态模型与技能选择

先不连 emulator 查看路由解释：

```bash
python -m src1 --config src1/config.local.toml plan \
  --goal "Open Contacts and add Alice with phone number 123456"
```

路由每个 step 都会输出：

- 选中的 `model_id` 与 `skill_id`；
- 覆盖的原语和预计成功率；
- `log_success`、延迟惩罚、adapter 切换惩罚、polished 奖励和降级代价；
- 总分和切换次数。

当前 DP 的目标近似为：

```text
Σ [ w_success × log(P(success))
  - w_latency × latency
  - w_switch × adapter_switch_ms
  + polished_bonus
  - primitive_fallback_cost ]
```

真实技能统计会逐渐覆盖配置先验。polished skill 失败后，执行器会禁用当前技能，
展开其 fallback 原语 topology 并重新路由。

## 6. AndroidWorld 在线评测

```bash
python -m src1 --config src1/config.local.toml evaluate \
  --tasks ContactsAddContact SimpleCalendarAddOneEvent \
  --combinations 5 --seed 42
```

默认用本地关键词规划，拓扑转换开销很低。若希望让某个在线模型做第一阶段任务
分解，可增加：

```bash
--planner-model student-base
```

候选 polished skill 默认不参与正式评测。灰度验证时显式增加
`--include-candidates`。每次评测目录包含：

- `summary.json`：完整机器可读指标；
- `report.md`：Micro/Macro SR、每任务 SR、平均步数、模型切换、技能使用、失败分类；
- `traces.jsonl`：backend 可直接消费的轻量轨迹；
- `checkpoints/*.pkl.gz`：AndroidWorld 原始 episode。

## 7. 技能库自主维护

```bash
python -m src1 --config src1/config.local.toml maintain
python -m src1 --config src1/config.local.toml skills --kind polished
```

作为常驻 backend 自主循环（每 5 分钟同步、编译、画像更新和技能优化）：

```bash
python -m src1 --config src1/config.local.toml maintain \
  --watch --interval-seconds 300
```

一次维护周期会：

1. 幂等同步 `libs/skvm/skvm-data/skills`；
2. 若配置了 compiler model，逐批编译尚未处理的 raw skills；
3. 用新轨迹增量更新每个模型/LoRA 的原语能力画像；
4. 消费尚未处理的成功/失败轨迹；
5. 按 episode 去重挖掘 2～5 个原语的高频成功子序列；
6. 生成带 fallback 的 `candidate` polished skill；
7. 根据真实 trial 数、成功率和 Wilson 置信下界晋升为 `active`；
8. active 技能低于回滚阈值时标记 `deprecated`；
9. 把本轮结果写到 `runtime/reports/maintenance_latest.json`。

生产环境可以把 `TemplateSkillCompiler` 换成云端 LLM compiler。无论编译器多强，
候选技能都不会绕过 AndroidWorld trial 自动晋升，避免技能库被错误经验污染。

## 8. 更新模型能力画像

离线 profile 或一轮评测后可更新原语能力：

```bash
python -m src1 --config src1/config.local.toml profile \
  --model-id student-ui-grounding-lora \
  --capability ground.text=0.81 \
  --capability ground.icon=0.76 \
  --capability action.click=0.79
```

在线 polished skill 的模型专属成功率和延迟由轨迹自动累计，不需要手工填写。

## 9. 测试

单元测试不需要 GPU、模型服务或 Android emulator：

```bash
python -m unittest discover -s src1/tests -v
```

真正跑 SR 前仍需完成 AndroidWorld 的 emulator/app 初始化，并启动配置中的模型
服务。`doctor` 只检查静态路径；它不会擅自启动 emulator 或占用 GPU。
