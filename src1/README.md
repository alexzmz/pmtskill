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

- `paths.log_dir` 是所有 CLI 运行日志的根目录；
- `offline.teacher_model_id` 必须匹配一个 `[[models]].model_id`；
- `offline.student_model_path` 是 ms-swift 训练的学生 VL 基座；
- 每个 `[[models]]` 是在线可选的基座或 LoRA served model；
- `[models.capabilities]` 保存该模型在每个原语上的成功率画像；
- `android_world.max_steps = 50` 表示每个 episode 最多执行 50 步；collector
  会在 AndroidWorld 的 episode runner 外层再次强制这一硬上限，任务完成时仍会提前停止；
- `android_world.infrastructure_recovery_attempts = 1` 会在 a11y/ADB 失效时恢复
  飞行模式、网络和 accessibility forwarder，必要时只重启 emulator guest，并重试当前任务；
- API key 只通过 `api_key_env` 指定的环境变量读取，不写入 TOML。

### 3.1 每次 CLI 调用的日志

所有实际子命令都会自动创建独立目录：

```text
runtime/logs/20260824T153012_collect_ContactsAddContact_a1b2c3d4/
├── runtime.log   # print、进度、INFO/WARNING/ERROR、训练子进程输出
├── errors.log    # stderr、warning、error 和 traceback
├── run.json      # 命令、参数、主机、时间、状态、退出码
├── result.json   # 完整机器可读最终结果
└── result.md     # 适合实验结束后直接阅读的最终报告
```

日志根目录可在 TOML 中设置，也可临时覆盖：

```bash
python -m src1 --config src1/config.local.toml \
  --log-dir /data/pmtskill-logs --log-level DEBUG collect --tasks TaskA
```

即使命令报错，CLI 也会尽量生成 `result.json`/`result.md`，并把完整 traceback
写入运行日志。`collect` 的最终报告包含 Micro/Macro TSR、每任务 SR、平均步数、
耗时、异常数、达到步数上限次数，以及按最终 episode 成功做保守信用分配的
primitive 成功率。AndroidWorld 每个 episode 后打印的累计表仍完整保留在
`runtime.log`。

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

每个 episode 默认最多执行 50 步。可以临时调低，例如 `--max-steps 30`，但不能
把上限提高到 50 以上；到达上限时只结束当前 episode，checkpoint 保存后继续下一个。

`--tasks` 正是“选择哪些任务做蒸馏”的参数；也接受逗号分隔。不传时运行整个
family。输出沿用 AndroidWorld `IncrementalCheckpointer` 的 `.pkl.gz`，因此中断后
用同一目录可续跑。

需要启动emulator和教师模型服务

### 4.3 生成 ms-swift 多模态数据

```bash
python -m src1 --config src1/config.local.toml build-dataset
```

默认同时使用成功、失败和结果未知的 episode，并逐 step 过滤：只要存在可辨认的
教师输出、任务 prompt（缺失时回退到 goal）以及至少一张有效截图，就生成训练样本。
episode 的结果保存在 `metadata.episode_outcome`，不会再作为默认丢弃条件。

如果旧的 `config.local.toml` 仍写着 `successful_only = true`，可以改成 `false`，
或直接覆盖：

```bash
python -m src1 --config src1/config.local.toml build-dataset --include-failed
```

只有确实需要严格成功数据集时才使用 `--successful-only`。

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

训练/验证仍按完整 episode 切分，避免一条轨迹的相邻截图同时出现在两边。
`manifest.json` 会记录 episode 结果分布、候选/接受/拒绝 step 数量和逐项拒绝原因。

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

### 4.5 可选：训练前、逐 epoch 与训练后 AndroidWorld 评测

普通 `train` 仍然只训练，不会占用 emulator 做 SR 测试。需要完整闭环时显式增加：

```bash
python -m src1 --config src1/config.local.toml train \
  --adapter-name android_world_all \
  --with-evaluation \
  --eval-task-count 30 \
  --eval-every-epochs 1 \
  --checkpoint-every-epochs 1 \
  --eval-seed 42
```

这条命令严格复用同一批任务、同一 seed 和同一组合数，顺序为：

1. 学生基座裸模型（原生 M3A，不使用技能库）；
2. 同一个学生基座 + 当前技能库；
3. 累计训练到第 1、2、… 个 epoch，每个节点部署真实 LoRA checkpoint，测试裸模型；
4. 最终 checkpoint 裸模型 + 最终 checkpoint 与技能库的组合效果。

为避免训练进程与评测服务争用 GPU，每个阶段会先结束 ms-swift SFT 进程，再用
`swift deploy --adapters <checkpoint>` 临时启动 OpenAI-compatible 服务；评测后关闭
服务，从该 checkpoint 恢复 optimizer、随机种子和训练进度，继续下一段。训练命令
会自动补充 `save_strategy=epoch`、`save_total_limit=1`、`add_version=false` 和
checkpoint symlink 参数；同时显式设置 `load_args=false`、`load_data_args=false`，
不允许旧 checkpoint 的 `args.json` 覆盖本次数据参数。每个累计 epoch 节点使用独立
目录，避免后续阶段覆盖记录。

正式启动基线评测前，框架会把 train/validation JSONL 冻结到本次 run 的
`dataset_snapshot/`。如果合并数据集后样本仍引用旧的
`.../dataset/images/...`，会按 `images/` 后的相对路径重定位到当前 TOML 的
`dataset_dir`；缺失或损坏的图片会在预检报告中记录，只剔除完全没有可读图片的样本。
之后所有训练阶段只使用这份固定索引，不会回退到默认 `./runtime/dataset`。

若希望明确只训练（包括 TOML 中设置了 `enabled = true` 的情况），使用：

```bash
python -m src1 --config src1/config.local.toml train --without-evaluation
```

可用 `--eval-tasks TaskA TaskB` 代替随机抽样；不传时读取
`[training_evaluation].task_count`，推荐 20～50。先用 `--dry-run --with-evaluation`
可以查看全部分段训练命令、部署命令和评测顺序而不启动 GPU/emulator。

首次带评测训练使用独立目录；再次对同名 adapter 执行命令会自动复用最近一次运行目录
和原 `dataset_snapshot`，从其中最新的完整 checkpoint 恢复。`history.json` 已记录的
基线、epoch 训练和评测不会重复执行；若原计划已经完成且总 epoch 没有增加，这次调用
会直接返回已有结果；提高 TOML 中的 `offline.epochs` 后则继续训练新增 epoch。若上次
在某轮评测中被直接 `Killed`，残缺评测目录会改名为
`*.interrupted_<time>` 后重跑该轮，已完成结果仍保留。显式传入同一个
`--eval-output-dir` 也会触发相同行为。使用 `--no-resume` 可关闭自动续训；此时显式
输出目录非空会直接报错，未指定目录则创建新的时间戳运行。

目录结构如下：

```text
runtime/checkpoints/<adapter>/training_runs/<time>/
├── sample_manifest.json              # 固定任务、seed、计划 episode 数
├── training_stage_commands.json      # 各累计 epoch 的命令与恢复来源
├── history.json                      # 完整机器可读 SR 历史和最佳 checkpoint
├── history.csv                       # 方便表格/画图的 SR 曲线
├── comparison.md                     # 基座、逐 epoch、最终技能增益总览
├── checkpoints.json                  # epoch→checkpoint、是否永久保留
├── dataset_snapshot/                 # 固定 JSONL、路径修复/坏图过滤报告
├── training/
│   ├── epoch_001/                    # 该累计 epoch 的命令、checkpoint、trainer state
│   └── epoch_002/
└── evaluations/
    ├── baseline/
    │   ├── standalone/               # summary.json/report.md/traces/checkpoints
    │   └── skills/
    ├── epoch_001/
    │   └── standalone/
    └── epoch_002/
        ├── standalone/
        └── skills/                   # 最终 checkpoint + 技能库
```

周期评测阶段只测裸模型；最终阶段再测一次技能库，减少 emulator 时间。训练评测产生
的 trace 不会写回技能数据库，避免测试集结果污染在线路由统计。

AndroidWorld 把任务初始化异常保存成 `episode_data = NaN` 等标量时，评测后处理会将其
记录为异常 episode 和空事件 trace，不会让整轮训练退出。临时 `swift deploy` 默认关闭
请求级 verbose，避免把完整 prompt/base64 截图写入日志；持久化的 runtime/error/result
文件还会按敏感字段名和常见密钥格式统一脱敏。

评测会在每个 task 开始前检查 accessibility tree。若模型误触飞行模式、a11y forwarder
失联或 ADB/emulator 短暂异常，框架先恢复网络和转发，软恢复失败后执行 `adb reboot`
重启 Android guest，等待 `sys.boot_completed=1`，重新连接后重试当前 task。默认只尝试
1 次；若仍不可用会立即结束本轮并保留已有 checkpoint，而不会把之后数百个跳过任务
写成 `episodes=0, SR=0`。可用 `infrastructure_recovery_attempts = 0` 关闭此行为。

训练评测的 M3A 每一步原本会同时保存原图、标框前后截图、UI tree 和完整 prompt；长
suite 会让这些对象一直留在 Python 内存中，最终可能被 Linux OOM killer 直接
`Killed`，且来不及写 traceback。评测模式现在只保留动作、reason、summary 和路由
元数据，逐步释放截图/UI tree；`collect` 仍保留完整截图轨迹，因此不会影响数据集构建。

`--checkpoint-every-epochs N` 控制永久保留间隔，默认 `1`（逐 epoch 保留）；设为
`2` 时保留第 2、4、… epoch 和最终结果，设为 `0` 时仅保留最终 checkpoint。为完成
中间评测/续训而创建的临时 checkpoint 会在整个流程成功后删除；若流程失败则保留，
便于定位问题和手动恢复。也可在 `[training_evaluation]` 中配置
`checkpoint_every_epochs = 1`。

#### 显存、上下文和训练/评测 GPU 分配

若模型原始配置声明了 262K 上下文，vLLM 会按该长度预留 KV cache。即使 AndroidWorld
实际 prompt 远小于 262K，也可能在服务启动阶段直接 OOM。建议在本地 TOML 中明确设置：

```toml
[offline]
# 其余训练配置省略；物理 GPU 2 专用于 ms-swift SFT。
cuda_visible_devices = "2"

[training_evaluation]
# 物理 GPU 1 专用于基座/checkpoint 的临时评测服务。
cuda_visible_devices = "1"
max_model_len = 32768
gpu_memory_utilization = 0.90
```

框架会分别向两个子进程注入：

```text
训练进程：CUDA_VISIBLE_DEVICES=2 ... swift sft
评测进程：CUDA_VISIBLE_DEVICES=1 ... swift deploy \
          --vllm_max_model_len 32768 \
          --vllm_gpu_memory_utilization 0.9
```

注意，设置 `CUDA_VISIBLE_DEVICES=2` 后，该进程内部通常把物理 GPU 2 映射成
`cuda:0`，这是 CUDA 的正常行为。训练和评测在当前编排中不会同时运行，但分别指定
GPU 可以避免继承错误的 shell 显卡环境，也方便后续改成并行部署。

以上参数也可以临时从 CLI 覆盖：

```bash
python -m src1 --config src1/config.local.toml train \
  --with-evaluation \
  --train-cuda-visible-devices 2 \
  --eval-cuda-visible-devices 1 \
  --eval-max-model-len 32768 \
  --eval-gpu-memory-utilization 0.90
```

如果 32768 仍不够，可依次降到 24576 或 16384；AndroidWorld 一般不需要 262K。
启动失败后出现的 `destroy_process_group() was not called` 通常是 vLLM/NCCL 在前述
OOM 异常退出后的清理警告，不是另一处独立故障。

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
