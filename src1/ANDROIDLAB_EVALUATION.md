# AndroidLab 原生 AVD：先跑 standalone，各方法独立运行

默认使用宿主机 Android SDK + 预置 AndroidLab AVD，**不需要 Docker**。
每次只启动指定的一种方法；先检查 standalone 单任务，再测完整 standalone。
AndroidLab 提供任务、初始应用数据和判定器；这里复用现有 agent/PORTS，不是给 AndroidWorld 任务改名。

## 1. 下载源码和正确架构的预置 AVD

在 Linux 服务器 `/home/zmz/Workspace/gui` 下：

```bash
git clone https://github.com/THUDM/Android-Lab.git libs/Android-Lab
# 可选：只在刚克隆且没有本地修改的仓库固定本接口核对的版本。
git -C libs/Android-Lab checkout 4bfe108cf3883b710e8e5c6fba288161c830e504
```

希望主仓库管理依赖时，可将 clone 换成 `git submodule add https://github.com/THUDM/Android-Lab.git libs/Android-Lab`。
本次没有新增 submodule、下载大镜像或修改现有 SDK。
需保留现有 AndroidWorld agent、ms-swift/vLLM 依赖，并安装官方源码依赖：

```bash
# 官方 requirements 可能有旧版本约束，先备份/克隆现有 conda 环境。
pip install -r libs/Android-Lab/requirements.txt
```

### 两个官方资源入口不能混用

| 宿主机 | 官方资源 | 限制 |
|---|---|---|
| macOS ARM64 | [avd.zip / ini 下载](https://drive.google.com/file/d/19FnCaEh6VzR99Za-pvUoR-60iNe8mgom/view) | 官方标注 ARM64，不能直接用于 Linux x86_64 |
| Linux x86_64（你的服务器） | [官方 Linux 环境资源包](https://drive.google.com/file/d/1SJ79gdO7whgUod3HnuS87aOKihRk1i-U/view) | 官方称为 Docker 构建资源；需从包或其 Dockerfile 引用资源取得 x86_64 预置 AVD，不要求构建/运行 Docker |

依据：[官方 Linux 说明](https://github.com/THUDM/Android-Lab/blob/main/docs/prepare_for_linux.md)、
[ARM64 AVD 说明](https://github.com/THUDM/Android-Lab/blob/main/docs/prepare_for_mac.md)。
后者还有 Android Studio 更新导致原获取方式可能失效的警告。
**目前未核实到官方单独发布的 Linux x86_64 avd.zip 直链，不能把 ARM64 地址冒充 Linux 下载地址。**

Linux 可先下载并检查资源包，不执行 Docker：

```bash
pip install gdown
mkdir -p downloads/androidlab
gdown 'https://drive.google.com/uc?id=1SJ79gdO7whgUod3HnuS87aOKihRk1i-U' \
  -O downloads/androidlab/docker-file.zip
unzip -l downloads/androidlab/docker-file.zip
unzip downloads/androidlab/docker-file.zip -d downloads/androidlab/extracted
find downloads/androidlab/extracted -type d -name '*.avd'
find downloads/androidlab/extracted -type f -name Dockerfile -print
```

若有预置 `.avd`，使用 x86_64 版本；若只有 Dockerfile/下载脚本，阅读其中的 AVD 地址再下载。
这里未检查 Google Drive 大包内部，**不保证包内直接包含完整 AVD**。
找不到时可提供解压文件清单/Dockerfile 来定位资源；不要用空 AVD 凑数。
Android Instruct 是训练数据，不是这里需要的测试环境，不必下载。

### SDK 与模板目录

Linux 原生加速需要 KVM 权限。官方 x86_64 配置为 API 33 / google_apis / x86_64：
[官方镜像配置说明](https://github.com/THUDM/Android-Lab/blob/main/docs/modify_androidlab.md)。

```bash
/home/zmz/Workspace/gui/Android/Sdk/cmdline-tools/latest/bin/sdkmanager \
  'platform-tools' 'emulator' 'system-images;android-33;google_apis;x86_64' \
  'build-tools;33.0.0' 'platforms;android-33'
```

已有匹配版本不必更新。命名快照要求兼容的 emulator/system image 版本，盲目升级可能使快照不可用。
下载的 **已安装应用、填充初始数据** 的 AVD 按以下结构放置，名字与 TOML 一致：

```text
/home/zmz/Workspace/gui/AndroidLab/avd/
  Pixel_7_Pro_API_33.ini
  Pixel_7_Pro_API_33.avd/
    config.ini
    userdata-qemu.img
    ...原始数据盘/加密文件/SD 卡等完整配套文件...
    snapshots/                 # 仅选命名快照时需要
```

`ini` 应有 `target=android-33`；旧绝对路径会在**临时副本**中重定位。
必须保留完整配套文件；单独 system.img、同名 APK 或 AndroidWorld AVD 都不能替代它。
模板不要启动使用，不要保留运行锁；程序拒绝克隆含 `.lock` 或符号链接的模板。

## 2. 配置原生 AVD

```bash
cp src1/config.androidlab.example.toml src1/config.local_androidlab.toml
```

编辑模型路径、已有模型配置及下段。相对路径以 TOML 所在目录为基准：

```toml
[androidlab]
root = "../libs/Android-Lab"
backend = "avd"
sdk_root = "/home/zmz/Workspace/gui/Android/Sdk"
avd_home = "/home/zmz/Workspace/gui/AndroidLab/avd"
avd_name = "Pixel_7_Pro_API_33"
avd_work_dir = "./runtime/androidlab_avd_work"
console_port = 5562
avd_gpu_mode = "off"
avd_headless = true
clone_timeout_seconds = 600.0
startup_timeout_seconds = 240.0
command_timeout_seconds = 45.0
episode_timeout_seconds = 1800.0
wait_after_action_seconds = 1.0
app_start_wait_seconds = 15.0
judge_model_id = "teacher-qwen3.7-plus"
# avd_snapshot = "default_boot"  # 仅当模板确实包含且能恢复该快照时启用
```

`5562/5563` 要空闲，避免占用已有的 5554、5556、5558、5560。`8002` 是模型 API 端口。
**不需要 grpc_port，也不用 Docker 的 6060。** 程序不会连接或关闭已有模拟器。
全量回答类任务需要独立 judge；设置其 `api_key_env` 对应环境变量，不写明文密钥。
judge 只比较回答，不向执行 agent 提供答案/提示。

只读检查与查看任务，不启动设备/模型，不调用 API：

```bash
python -m src1 --config src1/config.local_androidlab.toml inspect-androidlab-avd
python -m src1 --config src1/config.local_androidlab.toml list-androidlab-tasks
```

检查架构、API、数据盘、SDK、快照存在性；**静态检查不能证明应用内部数据正确**，还需真实单任务检查。
`inspect-androidlab-avd --hash-images` 可额外计算大镜像摘要，会耗时；正式测试会自动计算以固定 resume 身份。

### 每个任务如何重置

每个 episode 从模板复制独立 AVD，启动专属进程；所有 ADB 指令带 `-s emulator-5562` 并核对专属 AVD 名。
Linux 优先 reflink/sparse 复制，不支持时真实复制，需要充足磁盘和启动时间。
正常/异常退出只回收本次自建进程和带所有权标记的临时副本；不修改模板、不执行全局 kill-server。

默认从**复制的预填充 userdata 冷启动**，不执行 `-wipe-data`。
仅 `-no-snapshot-save` 不能防止 userdata 污染，因此不复用同一个可写 AVD 连跑全部任务。
显式选 `avd_snapshot` 时，强制恢复命名快照，失败标为 unavailable，绝不静默退回冷启动。
报告注明重置协议；冷启动保留磁盘数据，但不等同于原 RAM/UI 快照恢复。

## 3. 优先 standalone：单任务与全量

先只测 control + `clock_1`，不需要 SQLite、SKVM、EchoTrail memory 或其他 adapter：

```bash
python -m src1 --config src1/config.local_androidlab.toml \
  --log-dir src1/runtime/evaluations/androidlab_standalone_smoke/cli_logs \
  evaluate-androidlab-standalone \
  --adapter-path /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
  --adapter-model-ids adapter_control01 \
  --tasks clock_1 --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories --no-record-traces \
  --output-dir src1/runtime/evaluations/androidlab_standalone_smoke
```

通过后完整 standalone 用下条；不要在 smoke 目录改变任务集合后 resume：

```bash
python -m src1 --config src1/config.local_androidlab.toml \
  --log-dir src1/runtime/evaluations/androidlab_standalone/cli_logs \
  evaluate-androidlab-standalone \
  --adapter-path /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
  --adapter-model-ids adapter_control01 \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories --no-record-traces \
  --output-dir src1/runtime/evaluations/androidlab_standalone
```

不指定 `--tasks/--task-count` 就读取全部官方任务（核对版本 138 项）。
`--tasks clock_1 contacts_1` 可选多项，`--task-count 5` 按 seed 抽样。
`--combinations N` 是每个固定任务重复 N 次，不是 AndroidWorld 的参数随机化。
`--max-steps` 限制 GUI action/决策步数，另有包含设备准备的 episode 超时。
`--dry-run` 不启动 AVD/GPU/API，只打印计划；正式测试先做依赖/判定器预检再部署模型。

## 4. 可选：预先准备完整实验文件夹

只冻结任务/建立 standalone 目录，不测试、不复制无关技能库：

```bash
python -m src1 --config src1/config.local_androidlab.toml prepare-androidlab \
  --method standalone --combinations 1 --seed 42 --max-steps 30 \
  --output-dir src1/runtime/evaluations/androidlab_control
```

生成 `inputs/tasks.json` 与 `standalone/cli_logs/`。然后使用相同 seed/combinations/max-steps，
在 standalone 指令中加 `--suite-file src1/runtime/evaluations/androidlab_control/inputs/tasks.json`，
输出目录改为 `.../androidlab_control/standalone`。
准备技能方法时换 `--method` 并给 `--skill-database`；已有数据库快照不覆盖。

## 5. 其他方法各自独立运行

以下每条**独立选择执行**，不是一键八方法脚本。先在 Bash 设置公共参数（不测试）：

```bash
LAB_CONFIG=src1/config.local_androidlab.toml
LAB_OUT=src1/runtime/evaluations/androidlab_control
CONTROL=/home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01
CROSSAPP=/home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_crossapp01
ALL=/home/zmz/Workspace/gui/src1/runtime/checkpoints/deprecated/adapter_all02
SKILL_DB=src1/runtime/skill_library.sqlite3
EVAL_COMMON=(--combinations 1 --seed 42 --max-steps 30 --cuda-visible-devices 0
  --deploy-port 8002 --infer-backend vllm --no-require-training-state
  --resume --no-store-trajectories --no-record-traces)
CONTROL_ARGS=(--adapter-path "$CONTROL" --adapter-model-ids adapter_control01)
MULTI_ARGS=(--adapter-paths "$CONTROL" "$CROSSAPP" "$ALL"
  --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02)
```

Simple-skill（control，关键词技能检索/注入）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/simple-skill/cli_logs" \
  evaluate-androidlab-simple-skill "${CONTROL_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --skill-database "$SKILL_DB" --output-dir "$LAB_OUT/simple-skill"
```

SKVM（control，真实内核 TCP/AOT/在线技能适配）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/skvm/cli_logs" \
  evaluate-androidlab-skvm "${CONTROL_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --skill-database "$SKILL_DB" --output-dir "$LAB_OUT/skvm"
```

V-Droid（control，Qwen Yes/No verifier 改造版，不是原 Llama value head）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/vdroid/cli_logs" \
  evaluate-androidlab-vdroid "${CONTROL_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --vdroid-root libs/v-droid --output-dir "$LAB_OUT/vdroid"
```

AndroidWorld agent（control；任务/评分仍为 AndroidLab；M3A 与 standalone 相同，T3A 为文本观察模式）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/androidworld/cli_logs" \
  evaluate-androidlab-androidworld "${CONTROL_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --androidworld-agent t3a --output-dir "$LAB_OUT/androidworld"
```

EchoTrail-GUI（control，论文适配版；替换为实际独立收集的 memory、本地 embedding 路径）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/echotrail-gui/cli_logs" \
  evaluate-androidlab-echotrail-gui "${CONTROL_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --memory-file /absolute/path/to/memory.json \
  --embedding-model-path /absolute/path/to/embedding_model --output-dir "$LAB_OUT/echotrail-gui"
```

Random routing + SKVM（三个 adapter，每步随机模型，再用该模型自己的 SKVM）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/random-skvm/cli_logs" \
  evaluate-androidlab-random-skvm "${MULTI_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --routing-seed 42 --skill-database "$SKILL_DB" --output-dir "$LAB_OUT/random-skvm"
```

PORTS（三个 adapter，已有 PlannerPipeline/DP 路由/在线适配）：

```bash
python -m src1 --config "$LAB_CONFIG" --log-dir "$LAB_OUT/ports/cli_logs" \
  evaluate-androidlab-ports "${MULTI_ARGS[@]}" "${EVAL_COMMON[@]}" \
  --skill-database "$SKILL_DB" --output-dir "$LAB_OUT/ports"
```

任意一条可加 `--tasks clock_1` 只测一个任务，使用独立输出目录。
旧通用入口 `evaluate-androidlab --method NAME` 保留。
模型已部署时，用 TOML `--model-id NAME`（多模型 `--model-ids ...`）替代 adapter 参数，不另起服务。
EchoTrail 依赖/记忆准备见 `EIGHT_METHOD_EVALUATION.md`，没有记忆时报错，不暗中退回 standalone。

可选 `python -m src1.compare_androidlab` 也改为**默认仅 standalone**。
只有明确传 `--methods standalone simple-skill ...` 才串行测多个方法；`--prepare-only` 只生成所选方法的目录/命令。
不要并发用同一 console/model 端口；并行测试须不同配置、空闲端口和输出目录。

## 6. 输出、续跑与解释范围

- 各方法保存 `report.md`、`summary.json`、`episodes.csv`、`episodes/*.json`、manifest；原生模拟器日志在 `avd_logs/任务ID__repeat/`。
- `--resume` 跳过正常完成的成功/失败任务，异常任务重试。改变任务集、权重、模板/SDK、源码或协议需新目录；不能混入旧 Docker/v1 结果。
- 正式运行固定模板、SDK、系统镜像摘要，克隆前后检查模板未变；不要在测试时修改/启动模板。
- 默认不保留截图/XML/完整轨迹；临时判定后释放。可用 `--store-trajectories` 审计，不将测试轨迹写入技能库。
- 设备、模型 API、judge 异常标为 unavailable，完整 SR 为 N/A，不伪装成失败。`--continue-on-error` 继续后续任务，但最终仍非零退出。
- 官方判定保留最后一个 eligible page 结果，不因 agent finish 就成功；本桥接额外检查最终 post-action 页面，是明确的采集扩展。
- SR 是这里实现的正式 AndroidLab 指标；未重新实现 Sub-SR/RRR/ROR。另有步骤、调用、延迟、模型切换、技能使用诊断。
- 执行延迟排除部署、AVD 克隆/启动、judge 和回收；steps 为提交的 JSON GUI action，M3A 决策步数单列。V-Droid 远程 verifier 不在普通生成调用数内。
- 当前桥接支持 portrait；旋转显式 unavailable。默认按应用报告，可另配三分类映射，不把 AndroidWorld 任务名硬套过来。
- AndroidWorld adapter/技能/记忆用于 AndroidLab 是跨基准迁移，不是 AndroidLab 标定；不保证 PORTS 必然更高。

## 7. 验证范围与旧后端

```bash
python -m unittest src1.tests.test_androidlab src1.tests.test_androidlab_avd
python -m compileall -q src1/androidlab src1/compare_androidlab.py
```

上述是无设备/无 GPU 的协议与回归测试；本机没有真实预置 AVD，尚未验证服务器端到端表现。
Windows 支持 prepare/inspect/dry-run；真实评测需 Linux/macOS 的 POSIX 超时机制。
若要旧 Docker 后端，显式设置 `backend="docker"`、`docker_image`、`docker_port`；它不是原生模式依赖，也不作为静默后备。
