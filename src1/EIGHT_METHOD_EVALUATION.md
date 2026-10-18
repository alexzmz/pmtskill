# AndroidWorld 八入口对比实验

以下为服务器 Linux / Bash 指令，在 `/home/zmz/Workspace/gui` 执行。不会预设方法排名，不把运行故障记为真实 SR=0。

## 1. 方法、模型与实验边界

| 名称 | CLI | 使用的 adapter | 说明 |
|---|---|---|---|
| standalone | `evaluate-standalone` | control | 已有原生 M3A，不使用技能 |
| simple-skill | `evaluate-simple-skills` | control | 已有简单技能检索 |
| SKVM | `evaluate-skvm` | control | 真实 SKVM TCP / AOT / 运行时变体选择 |
| V-Droid-Qwen | `evaluate-vdroid-qwen` | control | 已有明确标注的 Qwen 适配版，不是原版训练 verifier |
| AndroidWorld | `evaluate-androidworld` | control | 默认 ICLR2025 官方 M3A；可选 T3A |
| EchoTrail-GUI-Qwen | `evaluate-echotrail-gui` | control | 论文适配复现，探索 / Critic / 记忆检索注入 |
| Random routing + SKVM | `evaluate-random-skvm` | control + crossapp + all | 每个 GUI step 均匀随机选择 adapter，再使用该模型对应的 SKVM |
| PORTS | `evaluate-pmtskill` | control + crossapp + all | 已有规划和 DP 模型 / 技能路由 |

AndroidWorld ICLR2025 提出的 agent 是 **M3A**，不是一个与当前 standalone 不同的神秘 agent。因此默认 AndroidWorld 与 standalone 是同 agent 的两个实验入口，不应计作两种独立算法，也不保证两次生成结果逐步一致。要比较另一种官方 agent，可以把 AndroidWorld 改为 `--androidworld-agent t3a`，并在表格中明确称为 T3A。[官方项目](https://google-research.github.io/android_world/)、[官方 agent 说明](https://github.com/google-research/android_world/blob/main/android_world/agents/README.md)。

PORTS 和随机 SKVM 同为三个 adapter 池，其余方法为单 adapter；这种对比不是等模型资源的纯路由消融。所有方法均保留任务真值、统一任务集合 / seed / combinations / max-steps，不为降低 baseline SR 而跳过任务或削弱实现。

## 2. 依赖与模拟器

不需要新增 Git submodule。使用已有 `libs/android_world`、`libs/skvm`、`libs/v-droid`（或 `libs/V-Droid`）。尚未确认 EchoTrail 官方代码仓库，本实现根据[论文](https://arxiv.org/abs/2512.19396)编写，并非官方代码或原论文模型。

在已有能运行 standalone、SKVM、V-Droid 的 Python 环境中，仅 EchoTrail 新增语义检索依赖：

```bash
cd /home/zmz/Workspace/gui
python -m pip install -r src1/baselines/requirements-echotrail.txt
```

需要自行准备一个完整的、本地 `sentence-transformers` 兼容 embedding 模型目录。不会自动下载模型，不执行远程模型代码，不使用随机 / 哈希伪向量。默认在 CPU 编码，避免挤占 vLLM 的 GPU。实际 embedding 权重身份会写进记忆和实验 manifest；更换权重需要重建记忆。目录不能只有 tokenizer / config，必须包含模型权重。[加载接口](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。

同一时间不要让两套评测共享一个 emulator。你的端口对应关系是启动时确定的，不是 vLLM port 推导出来的：

| emulator console | emulator gRPC | 常用模型端口 | GPU |
|---:|---:|---:|---:|
| 5554 | 8554 | 8002 | 0 |
| 5556 | 8555 | 8003 | 1 |
| 5558 | 8556 | 8004 | 2 |
| 5560 | 8557 | 8005 | 3 |

下面统一使用 **5554 / 8554 / 8002 / CUDA 0**。先复制你已验证可用的配置，保持新 TOML 在 `src1`，以免改变相对路径基准：

```bash
cp src1/config.local_control.toml src1/config.local_baselines.toml
```

编辑新配置的已有 `[android_world]` 段，不要重复添加同名段：

```toml
[android_world]
console_port = 5554
grpc_port = 8554
# 其余 adb_path / emulator_setup / freeze_datetime / 等参数保留原配置
```

如果 `config.local_control.toml` 不是你的有效完整配置，请以目前能正确运行的 TOML 为模板。三个 adapter 也必须共享同一个可加载基座。

仅在模拟器尚未启动时手动启动；如果已启动，不要重复启动同一个 AVD：

```bash
export PATH=/home/zmz/Workspace/gui/Android/Sdk/platform-tools:$PATH
/home/zmz/Workspace/gui/Android/Sdk/emulator/emulator \
  -avd AndroidWorldAvd -no-window -no-snapshot -no-boot-anim \
  -memory 2048 -accel on -grpc 8554 -gpu off -port 5554
```

在另一个终端核对：

```bash
adb -s emulator-5554 get-state
ss -ltnp | grep -E ':(5554|8554|8002)\b'
```

评测前 8002 应空闲，CLI 会自行部署模型。`--dry-run` 是模型/模拟器无调用预检，不证明实际设备、显存、服务兼容性已经通过。

## 3. 一次性准备 EchoTrail 冻结记忆

### 推荐：自主探索 + Critic

先准备路径变量；`EMBED` 必须改成你的真实本地目录：

```bash
cd /home/zmz/Workspace/gui
CFG=src1/config.local_baselines.toml
CONTROL=/home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01
CROSSAPP=/home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_crossapp01
ALL=/home/zmz/Workspace/gui/src1/runtime/checkpoints/deprecated/adapter_all02
EMBED=/absolute/path/to/local/embedding-model
MEMROOT=/home/zmz/Workspace/gui/src1/runtime/echotrail_memory_control_seed123
OUT=/home/zmz/Workspace/gui/src1/runtime/evaluations/compare_eight_control
```

独立采集 seed=123，正式评测 seed=42。app 名称须为当前 AndroidWorld 支持的已安装 app 名称；先用少量 app 验证环境，再扩大覆盖。探索会操作 app 数据，应在实验模拟器而非个人手机上运行。它不使用 benchmark 成功判定器，也不计算 benchmark SR。

```bash
python -m src1 --config "$CFG" collect-echotrail-memory \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --apps contacts markor camera \
  --episodes-per-app 3 --focus-after-steps 4 \
  --seed 123 --max-steps 20 --critic-threshold 4 \
  --embedding-model-path "$EMBED" \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --output-dir "$MEMROOT"
```

流程：好奇探索 → 根据当前屏幕与部分轨迹提出具体子目标 → 继续执行 → 真实 Critic 以 1–5 分评价连贯性 / 效率 / 可复用性 → 达到阈值 4 才存入正式记忆。处理库保留近期失败 / 成功的探索与 Critic 反馈，后续探索使用这些反馈与当前历史。

输出：

- `$MEMROOT/exploration_state.json`：处理库与已完成探索，支持按 app / episode 续跑。
- `$MEMROOT/curated/build_state.json`：Critic / embedding 构建进度。
- `$MEMROOT/curated/memory.json`：在线评测用的冻结记忆。

Critic 可能不接受任何轨迹；此时会明确报错，不会把空记忆当成 EchoTrail。调高覆盖应增加探索，不应仅为得到结果降低标准或伪造成功。`--no-store-trajectories` 控制正式 evaluation 的轨迹，不会删除用于建库的探索记录。

```bash
MEMORY="$MEMROOT/curated/memory.json"
```

### 可选：已有成功轨迹导入（必须标注历史记忆适配）

如果已有独立采集 seed=123 的成功 traces，也可以不重新探索，用本地 control 做 Critic：

```bash
python -m src1 --config "$CFG" prepare-echotrail-memory \
  --source-database /absolute/path/to/source_skill_library.sqlite3 \
  --source-seed 123 --limit 500 --critic-threshold 4 \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --embedding-model-path "$EMBED" \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume \
  --output-dir src1/runtime/echotrail_historical_memory_seed123
```

此方式只读源 SQLite，不修改已有 traces，不运行新的 benchmark。也可将 `--adapter-path` / `--model-id` 替换为 `--critic-api-model-id teacher-qwen3.7-plus`，使用 TOML API Critic；不要再提供本地部署参数。API 调用会产生费用。

`--source-seed` 会核对 trace 中实际的 `metadata.collection_context.suite_seed` 等字段，不能把 seed42 的数据声明成 seed123。若现有 242 条都是 seed42，请另采 seed123；或者让**全部对比方法**改用新的评测 seed，且承认同一任务模板仍有重合。在线检索额外排除与当前 goal 完全相同的记录。不同 seed / 排除同 goal 只是基本隔离，不保证训练数据无泄漏；论文仍须说明 adapter 训练集和记忆与测试的划分。

### 与原论文的差异

原论文包括探索、Critic、双记忆、混合检索和记忆增强推理。本实现是 **Qwen 论文适配版**：探索、子目标提出、Critic 使用同一 control adapter；使用现有 M3A action / summary 和 UI 表示；处理库为近期 8 条探索的有界反馈；不复现论文所有独立专用模型。可选历史导入会另外标注 `exploration_reproduced=false`。[论文方法及实现设置](https://arxiv.org/html/2512.19396v2)。

检索采用真实本地 embedding 的 cosine 与 BM25：`score = alpha * cosine + (1-alpha) * BM25`，默认 alpha=0.5、K=2；小规模库使用精确 cosine 扫描替代 FAISS。注入文本含过去的界面上下文、意图和真实动作；只作为演示，不直接重放坐标 / 索引。长演示在边界处缩短并明确标注 partial，不截断动作 JSON。测试期记忆只读冻结，不用成功率、任务答案更新。

## 4. 先准备完整八方法实验目录，再开始运行

第一次用新目录，准备阶段会解析真实 checkpoint、冻结一致性 SQLite 快照与 EchoTrail memory、抽样并冻结任务、创建八个方法目录，输出可独立执行的命令及比较表，并做所有入口的 dry-run。不会加载 GPU 模型或操作模拟器。需要三个 adapter / 基座 / 技能库 / embedding 路径均存在，V-Droid 和 SKVM 原有依赖可用。

```bash
BATCH=(
  --config "$CFG" --output-dir "$OUT"
  --adapter-paths "$CONTROL" "$CROSSAPP" "$ALL"
  --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02
  --echotrail-memory "$MEMORY" --embedding-model-path "$EMBED"
  --cuda-visible-devices 0 --deploy-port 8002
  --combinations 2 --seed 42 --max-steps 30
  --resume --no-store-trajectories
)

python -m src1.compare_eight "${BATCH[@]}" --prepare-only
```

需要覆盖 TOML 的 SQLite 时在 BATCH 中加 `--skill-database /absolute/path/to/skill_library.sqlite3`。默认不把 candidate 自动改成 active；与原测试一致，仅使用满足原有在线筛选条件的技能。需要候选技能时统一加 `--include-candidates`。

然后去掉 `--prepare-only` 启动串行评测：

```bash
python -m src1.compare_eight "${BATCH[@]}"
```

中断后重复同一命令。任务级恢复保留真实成功和失败结果，跳过已经正常测完的实例，异常实例重新执行。单方法失败默认停止；希望继续其余方法时加 `--continue-on-error`，失败项仍是 N/A，而非伪造 SR=0。

先冒烟测试时，用**另一个输出目录**，在 BATCH 中固定 `--tasks ContactsAddContact ClockTimerEntry --combinations 1`；正式全量换新目录并去掉 tasks。不要在同一批次目录里改 tasks / seed / 权重 / 端口 / 记忆 / 配置，否则会拒绝混合结果。

仅预览八条指令、不要求模型路径或配置文件存在、也不创建任何输出：

```bash
python -m src1.compare_eight "${BATCH[@]}" --dry-run
```

目录结构：

```text
compare_eight_control/
  inputs/skill_library.sqlite3       # 不给任一方法直接写的冻结快照
  inputs/echotrail_memory.json       # 冻结记忆
  inputs/config.reference.toml       # 配置归档，不是可移动的运行配置
  inputs/environment.json            # 真实配置位置、Python、端口与 harness 路径
  standalone/  simple_skill/  skvm/  vdroid/
  androidworld/  echotrail_gui/  random_skvm/  ports/
  batch_manifest.json  batch_progress.json
  commands.json  commands.sh  preflight.log
  comparison.md  comparison.csv  comparison.json
```

各技能方法有独立 SQLite 副本。随机 SKVM 的 `models/model-<id>/skvm/` 为每个 adapter 保存独立 TCP、AOT、catalog、能力/编译产物及累计统计。SKVM 变体失败采用已有显式回退策略，报告区分 AOT 与原技能回退；API/设备故障不伪装成适配成功。

`config.reference.toml` 只是归档：相对路径仍以原 TOML 目录解析，运行命令故意引用原配置并校验内容 SHA。不要直接用归档文件运行。adapter 权重、embedding 模型与 harness 不会复制几 GB 到目录中；应保持这些依赖固定。

## 5. 八种方法分别运行的 CLI

先执行第 2–4 节准备输入和目录，再设置公共参数（Bash 数组）：

```bash
COMMON=(--combinations 2 --seed 42 --max-steps 30
        --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm
        --no-require-training-state --resume --no-store-trajectories)
POOL=(--adapter-paths "$CONTROL" "$CROSSAPP" "$ALL"
      --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02)
```

不要和批量运行同时执行以下命令。若准备阶段因旧 SKVM 协议隔离使用了嵌套目录，以 `commands.json` / `commands.sh` 为准；新建干净八方法目录通常无嵌套。

### standalone

```bash
python -m src1 --config "$CFG" evaluate-standalone \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  "${COMMON[@]}" --output-dir "$OUT/standalone"
```

### simple-skill

```bash
python -m src1 --config "$CFG" evaluate-simple-skills \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --skill-database "$OUT/simple_skill/skill_library.sqlite3" --no-record-traces \
  "${COMMON[@]}" --output-dir "$OUT/simple_skill"
```

### SKVM

```bash
python -m src1 --config "$CFG" evaluate-skvm \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --skill-database "$OUT/skvm/skill_library.sqlite3" --no-record-traces \
  --skvm-prepare auto --skvm-profile-instances 3 \
  --skvm-failure-policy fallback --skvm-compile-retries 1 \
  "${COMMON[@]}" --output-dir "$OUT/skvm"
```

### V-Droid-Qwen

```bash
python -m src1 --config "$CFG" evaluate-vdroid-qwen \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --vdroid-score-workers 4 --vdroid-search-iterations 1 \
  "${COMMON[@]}" --output-dir "$OUT/vdroid"
```

上游 fork 找不到时，加 `--vdroid-root /home/zmz/Workspace/gui/libs/v-droid`。不要错用只支持原生 Llama verifier 的 `evaluate-vdroid`。

### AndroidWorld 官方 M3A（默认与 standalone 重合）

```bash
python -m src1 --config "$CFG" evaluate-androidworld \
  --adapter-path "$CONTROL" --model-id adapter_control01 --androidworld-agent m3a \
  "${COMMON[@]}" --output-dir "$OUT/androidworld"
```

要用 T3A，把 m3a 改成 t3a，并创建新实验目录；不能沿用已有 M3A 结果。

### EchoTrail-GUI-Qwen 论文适配版

```bash
python -m src1 --config "$CFG" evaluate-echotrail-gui \
  --adapter-path "$CONTROL" --model-id adapter_control01 \
  --memory-file "$OUT/inputs/echotrail_memory.json" --embedding-model-path "$EMBED" \
  --memory-alpha 0.5 --memory-top-k 2 --memory-max-chars 12000 \
  "${COMMON[@]}" --output-dir "$OUT/echotrail_gui"
```

### Random routing + SKVM

```bash
python -m src1 --config "$CFG" evaluate-random-skvm \
  "${POOL[@]}" --routing-seed 42 \
  --skill-database "$OUT/random_skvm/skill_library.sqlite3" --no-record-traces \
  --skvm-prepare auto --skvm-profile-instances 3 \
  --skvm-failure-policy fallback --skvm-compile-retries 1 \
  "${COMMON[@]}" --output-dir "$OUT/random_skvm"
```

每 step 独立均匀采样，允许连续选中同一个 adapter，不强制制造切换；动作和总结固定使用该 step 的同一 adapter。随机流由 routing-seed 与实际 task / instance / goal / params 决定，跳过已完成实例不会把后面的随机序列错位。不读 PORTS 的能力估计或 DP 分数。三个模型各自完成真实 SKVM profiling / AOT，绝不共用一个模型的 TCP 冒充三个模型。

### PORTS

```bash
python -m src1 --config "$CFG" evaluate-pmtskill \
  "${POOL[@]}" \
  --skill-database "$OUT/ports/skill_library.sqlite3" --no-record-traces \
  "${COMMON[@]}" --output-dir "$OUT/ports"
```

## 6. 报告、稳定性与复现实验注意事项

每个方法输出 `summary.json`、`report.md`、轻量实例 checkpoint；三类能力仍仅 reasoning / coordination / manipulation。EchoTrail 附记忆来源 / Critic / embedding / 覆盖调用数，随机 SKVM 附每模型编译覆盖与适配统计。没有完整 trajectories 不代表没有恢复信息。

通过八方法 batch 运行会持续更新 `comparison.*`。直接执行单方法 CLI 后，batch 自身的完成状态不会凭空变更；再次运行原 batch 命令会利用实例 checkpoint 复用结果并更新汇总。准备阶段记录的任务列表与实际 command 以 `commands.json` 为准。

配置 / 权重 / 协议变更使用新目录；不要为了绕过检查删除 manifest 后拼接旧结果。现有五方法入口 `python -m src1.batch_evaluate` 默认仍是原来的五种，不影响旧指令。

稳定性保障是有界步数、有限重试、明确回退、输入冻结、实例恢复和诚实的 unavailable 状态；**不是保证所有依赖都能启动，更不是保证 PORTS 必然最高**。本地自动测试使用 mock 验证边界；真实 GPU / AndroidWorld / 模型服务仍需服务器冒烟和全量实验确认。
