# PORTS 与八种方法的 AndroidLab 评测

本接口运行 **AndroidLab 的任务、应用快照与官方任务判定器**，不是换一个名称运行 AndroidWorld。
它复用本项目现有的 agent / skill / routing 实现，用 AndroidLab XML 与截图桥接 M3A、T3A 和 V-Droid。

支持：`standalone`、`simple-skill`、`skvm`、`vdroid`、`androidworld`、`echotrail-gui`、`random-skvm`、`ports`。
单模型方法默认只用传入列表第一个 control adapter；PORTS 与 random-SKVM 使用全部三个。

## 1. 下载的究竟是哪套数据

官方项目：<https://github.com/THUDM/Android-Lab>。
本次接口核对的官方 commit 是 `4bfe108cf3883b710e8e5c6fba288161c830e504`；运行时还会记录源码内容摘要。

评测需要两部分：

1. 仓库中的 `evaluation/config/*.yaml`：九个应用的任务定义（官方介绍为 138 项），以及 `evaluation/tasks/` 中的判定器。
2. **官方已经安装应用、填好初始数据的 AVD/Docker 构建包**。只有 YAML 不够，普通 AndroidWorld AVD 也不能代替。

Android Instruct 是另一套训练数据，本次测试不需要下载，更不能直接作为测试集使用。

### Linux / x86_64（本实现支持的执行平台）

在 `/home/zmz/Workspace/gui` 下执行：

```bash
git clone https://github.com/THUDM/Android-Lab.git libs/Android-Lab
# 可选：固定为本接口核对的版本（在刚克隆的、无本地修改的仓库执行）。
git -C libs/Android-Lab checkout 4bfe108cf3883b710e8e5c6fba288161c830e504

pip install -r libs/Android-Lab/requirements.txt
```

如希望主仓库跟踪依赖，第一条可以改成：

```bash
git submodule add https://github.com/THUDM/Android-Lab.git libs/Android-Lab
```

两种方式选一个；本次代码没有替你修改 submodule 或下载大体积镜像。
还需保持现有 AndroidWorld agent 运行依赖、ms-swift/vLLM、SKVM CLI、V-Droid 依赖可用。
EchoTrail 额外安装 `src1/baselines/requirements-echotrail.txt`，并使用预先下载的本地 embedding 模型。
官方 requirements 可能与现有训练环境存在版本约束，安装前建议备份 conda 环境。

官方 Linux 环境说明：<https://github.com/THUDM/Android-Lab/blob/main/docs/prepare_for_linux.md>。
其中提供的 **Docker 构建包**：

<https://drive.google.com/file/d/1SJ79gdO7whgUod3HnuS87aOKihRk1i-U/view>

可通过浏览器下载；也可在服务器安装 gdown 后尝试下载公开文件（访问权限、网络或 Google Drive 配额仍可能限制下载）：

```bash
pip install gdown
mkdir -p downloads/androidlab
gdown 'https://drive.google.com/uc?id=1SJ79gdO7whgUod3HnuS87aOKihRk1i-U' \
  -O downloads/androidlab/docker-file.zip
unzip downloads/androidlab/docker-file.zip -d downloads/androidlab
# 进入解压出来的、包含 Dockerfile 的 docker-file 目录。
cd downloads/androidlab/docker-file
docker build -t android_eval:latest .
cd /home/zmz/Workspace/gui
```

构建前确认 Linux 支持 `/dev/kvm`，Docker daemon 可用且当前账号有权限。Docker 权限接近 root 权限；只对可信的官方镜像启用。
官方运行方式需要 privileged container，本接口也会使用它，但 HTTP 仅绑定 `127.0.0.1`。
每个 episode 都从同一个镜像 ID 新建独立容器，完成或异常时只删除该次自己创建的容器。
需要充足磁盘与内存；首次构建、镜像下载可能很慢，不由 prepare 命令自动执行。

可选的 **Android Instruct 训练数据**（不用于本次评测）：
<https://drive.google.com/file/d/1s0b74VEOww9n1kMocd6RJivwaUCymEs4/view>，
对应官方说明 <https://github.com/THUDM/Android-Lab/blob/main/docs/instruction_tuning.md>。

## 2. 配置

```bash
cp src1/config.androidlab.example.toml src1/config.local_androidlab.toml
```

检查 `student_model_path`、`skill_library_db`，以及 `[androidlab]` 中的源码目录、镜像名和 AVD 名。
设置 judge 的 API 密钥环境变量，例如 `DASHSCOPE_API_KEY`。**不要把实际密钥写入 TOML。**
也可以在你现有的 TOML 中追加示例的 `[androidlab]` 段，保留原来的模型能力画像。

这里不需要启动 5554/5556/5558/5560 的 AndroidWorld emulator，也不使用它们的 gRPC 端口。
端口含义是：`8002` 为 vLLM 模型服务；`6060` 为本次 AndroidLab Docker 内的设备 HTTP 接口。
可以修改 `[androidlab].docker_port` 以运行另一批实验，但不要让两批实验共用输出目录。

官方一部分任务通过页面/系统查询直接判定，一部分回答任务还需要 judge。这里保留官方答案判断提示和题目内的标准答案，
把请求发送到你配置的 `judge_model_id`，**不使用仓库里的默认凭据**。
judge 只接触评估阶段的答案对比，不向执行 agent 提供标准答案。更换 judge 是协议变化，应使用新输出目录。
使用非原论文 judge 时，报告是明确的 judge-adapted 测量，不能冒称原论文相同配置。

## 3. 一次性准备八方法目录

```bash
python -m src1.compare_androidlab \
  --config src1/config.local_androidlab.toml \
  --output-dir src1/runtime/evaluations/androidlab_compare_eight \
  --skill-database src1/runtime/skill_library.sqlite3 \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 \
  --prepare-only
```

这一步只创建：

```text
androidlab_compare_eight/
  inputs/tasks.json
  inputs/skill_library.sqlite3
  commands.json
  commands.sh
  standalone/  simple-skill/  skvm/  vdroid/
  androidworld/  echotrail-gui/  random-skvm/  ports/
```

它不跑测试、不启动 Docker/GPU、不调用 judge。已有快照保留，不会被新的源技能库覆盖。
AndroidLab 任务是固定任务，`--combinations N` 表示每项任务重复 N 次，**不是生成 N 组新参数**。
未指定 `--tasks` 或 `--task-count` 时读取全部官方任务；选择示例是 `--tasks clock_1 contacts_1`，不是 `ContactsAddContact`。
所有方法共用冻结的任务、repeat、seed 和步数上限。

## 4. 先做单任务 smoke test

不建议一开始就跑完整八方法。先验证 control + 官方 Clock 任务能够读截图、执行动作并产出判定结果：

```bash
python -m src1 --config src1/config.local_androidlab.toml \
  --log-dir src1/runtime/evaluations/androidlab_smoke/cli_logs \
  evaluate-androidlab --method standalone \
  --adapter-path /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
  --adapter-model-ids adapter_control01 \
  --tasks clock_1 --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories \
  --output-dir src1/runtime/evaluations/androidlab_smoke
```

`--dry-run` 只检查任务/配置/checkpoint 并输出部署计划，不连接 Docker、不启动模型、不执行付费 API。
正式运行会先做独立进程的依赖和任务判定器预检，再启动模型服务。

## 5. 一次串行跑全部八组

准备好真实的 EchoTrail memory 文件和对应 embedding 模型后：

```bash
python -m src1.compare_androidlab \
  --config src1/config.local_androidlab.toml \
  --output-dir src1/runtime/evaluations/androidlab_compare_eight \
  --skill-database src1/runtime/skill_library.sqlite3 \
  --adapter-paths \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_crossapp01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/deprecated/adapter_all02 \
  --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02 \
  --memory-file /absolute/path/to/independently_collected/memory.json \
  --embedding-model-path /absolute/path/to/local/embedding_model \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 \
  --resume --no-store-trajectories --continue-on-error
```

不要原样保留两个 `/absolute/path/...` 占位路径。EchoTrail 复用现有的真实 Critic + embedding 记忆库；
没有 memory 时明确报错，不会暗中改为 standalone。构建现有记忆的流程见 `EIGHT_METHOD_EVALUATION.md`。
这次没有新增 AndroidLab 自主探索训练流程；使用 AndroidWorld 探索记忆时会记录为跨基准迁移，不能宣称在 AndroidLab 上学习过。

如果暂时没有 EchoTrail 资源，可先指定：

```bash
--methods standalone simple-skill skvm vdroid androidworld random-skvm ports
```

以后相同冻结任务可单独补跑 `--methods echotrail-gui`。比较表会对应本次选定的方法集合。
默认脚本中各单模型方法只部署 control；仅 PORTS / random-SKVM 部署三个 adapter。
SKVM 复用真实内核的 TCP/AOT/技能选择，编译失败按现有稳定策略回退并明确记录，不删除失败任务。

## 6. 各方法单独调用

通用入口是 `python -m src1 --config ... evaluate-androidlab --method METHOD ...`。
准备步骤生成的 `commands.sh` 已包含八条完整指令，可直接复制其中任意一条。

| `METHOD` | 模型参数 | 实现 |
|---|---|---|
| `standalone` | control | 原有 M3A，无技能 |
| `simple-skill` | control | 原有关键词技能检索/注入 |
| `skvm` | control | 真实 SKVM 内核、该模型的 TCP/AOT |
| `vdroid` | control | 现有 V-Droid/Qwen Yes/No verifier 改造版；不是原 Llama value head |
| `androidworld` | control | 官方 M3A 或 `--androidworld-agent t3a`，桥接到 AndroidLab |
| `echotrail-gui` | control | 现有 Qwen 论文适配版，须另给 memory / embedding |
| `random-skvm` | 全部三个 | 每 GUI step 随机 adapter，再用该模型自己的 SKVM |
| `ports` | 全部三个 | 原有 PlannerPipeline、DP 路由、RoutedVLWrapper、在线适配 |

注意：`androidworld=m3a` 和 standalone 是同一个 agent，不能解释为两个不同算法；T3A 是文本观察模式。
这里的 `androidworld` 是 agent 名称，任务和评分仍来自 **AndroidLab**。

PORTS 单独运行完整示例：

```bash
python -m src1 --config src1/config.local_androidlab.toml \
  --log-dir src1/runtime/evaluations/androidlab_compare_eight/ports/cli_logs \
  evaluate-androidlab --method ports \
  --adapter-paths \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_crossapp01 \
    /home/zmz/Workspace/gui/src1/runtime/checkpoints/deprecated/adapter_all02 \
  --adapter-model-ids adapter_control01 adapter_crossapp01 adapter_all02 \
  --skill-database src1/runtime/evaluations/androidlab_compare_eight/inputs/skill_library.sqlite3 \
  --suite-file src1/runtime/evaluations/androidlab_compare_eight/inputs/tasks.json \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories --no-record-traces \
  --output-dir src1/runtime/evaluations/androidlab_compare_eight/ports
```

如模型已经在 TOML 配置的 OpenAI-compatible 服务上运行，可不传 adapter 路径，改用 `--model-id NAME`，
或者 PORTS / random-SKVM 的 `--model-ids NAME1 NAME2 NAME3`。此时不会启动本地模型服务。

## 7. 结果、恢复和解释范围

- 每个方法有 `summary.json`、`report.md`、`episodes.csv`、`episodes/*.json`、`evaluation_manifest.json`。
- 总目录有 `comparison.json/.csv/.md`。目前正式复用的 AndroidLab 指标是 **SR**；没有伪造 Sub-SR/RRR/ROR。
- 记录步骤、延迟、模型使用分布、切换次数、技能使用情况和 PORTS 分数来源。V-Droid 的远程 verifier 请求不计入普通生成调用数，报告明确注明该限制。
- 执行延迟排除 GPU 部署、设备启动、judge 与资源回收；包含策略初始化/规划及实际模型、设备交互。judge 用时另存，不能混作模型推理耗时。
- 可比的 `steps` 为已提交的 JSON GUI action 数，不包含 summary/finish，不等于底层触摸事件数；M3A 的决策轮次单独存为 `decision_steps`。
- 同目录 resume 会跳过**正常结束的成功和失败任务**；异常、超时或损坏的 checkpoint 会重跑。
- 任务、模型权重、技能库、判定器、镜像 ID、运行协议改变时拒绝混合续跑。要改实验设置请换输出目录。
- 默认不保留截图/XML/完整轨迹，但判定时会临时使用它们；可用 `--store-trajectories` 保留原生格式的审计记录。
- 原判定器语义为“最后一个可判定页面的结果”，不是看到一次 `complete` 就永久成功。执行 agent 的 finish 也不是成功依据。
- 为避免漏掉最后一个动作的结果，本桥接额外评估最终截图/页面；这是明示的 trace-collection 扩展，不承诺与官方 leaderboard 原始采集流程数值完全一致。
- 模拟器、judge、接口异常标记 unavailable；完整 SR 显示 N/A。`--continue-on-error` 会继续剩余任务/方法，但最终仍非零退出，绝不把异常写成真实任务 SR=0。
- AndroidLab 没有你之前的 AndroidWorld 三分类任务映射。默认按应用报告；可以提供 task_id→三类之一的 JSON。未映射项目保留 unclassified，不强行猜测。
- 沿用 AndroidWorld 学到的 adapter、技能和兼容分数时，这是**跨基准迁移实验**。不同应用包、UI、任务分布会影响命中与切换，不能保证 PORTS 必然高于单模型。
- 当前桥接支持官方 portrait snapshot；检测到旋转会显式报 unavailable，不静默使用错误坐标。

## 8. 本地验证范围

无 GPU/Docker 的协议与回归测试：

```bash
python -m unittest src1.tests.test_androidlab
python -m compileall -q src1/androidlab src1/compare_androidlab.py
```

Windows 开发机只能 prepare/dry-run/单元测试；真实 Docker/KVM + vLLM 端到端结果必须在服务器验证。
