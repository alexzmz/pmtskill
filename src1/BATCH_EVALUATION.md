# 五方法批量评测

四个 baseline **只用 adapter_control01**；PORTS 使用 adapter_control01、
adapter_crossapp01、deprecated/adapter_all02 三个 adapter。三个 adapter 必须兼容
同一基座，脚本会提前检查，不强行覆盖成错误的基座。模型 ID 与离线 profile 保持一致。

```bash
cd /home/zmz/Workspace/gui

python -m src1.batch_evaluate \
  --config src1/config.local_control.toml \
  --skill-database /home/zmz/Workspace/gui/src1/runtime/skill_library_sqlite/v2/skill_library.sqlite3 \
  --cuda-visible-devices 3 --deploy-port 8005 \
  --combinations 2 --seed 42 --max-steps 30 \
  --output-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/compare_five_control
```

先追加 `--dry-run` 打印全部命令，不读取权重、启动 GPU、连接模拟器或写文件。
实际运行还会调用五个入口的 dry-run 做 checkpoint/命令预检，再开始第一项。
默认测试全部任务；建议先用 `--tasks ContactsAddContact` 跑一轮冒烟测试。
也支持 `--task-count 10` 固定抽样，任务列表冻结在 batch_manifest.json。

运行顺序：standalone → simple-skill → SKVM → V-Droid-Qwen → PORTS。
共用同一个 TOML、GPU、端口、任务列表、组合数、seed、步数上限，每项结束后释放
服务再开始下一项。默认不保存完整轨迹；保留轻量 checkpoint、报告、日志和 SKVM
编译/能力评估产物。要保存完整轨迹，加 `--store-trajectories`。

默认开启 resume，再次运行同一命令即可跳过已完成的方法，以及中断方法内已完成的
任务实例。正常任务失败也算已完成，不会只重测失败任务来抬高 SR。权重或配置变化
需要新的 output-dir。技能库使用首次运行时的冻结快照，不跟随源库更新。
默认遇到失败停止；`--continue-on-error` 可继续后续方法。错误/不完整结果显示 N/A，
不伪装成 SR=0。Ctrl+C 先通知子进程退出，让部署上下文释放服务。

## 输出

```text
compare_five_control/
  batch_manifest.json
  batch_progress.json
  commands.json / commands.sh
  preflight.log
  comparison.md / comparison.csv / comparison.json
  inputs/skill_library.sqlite3
  standalone/              # report.md、summary.json、checkpoints、console.log
  simple_skill/            # 独立技能库副本
  skvm/                    # 另有 skvm_report.md/json、编译产物
  vdroid/                  # 明确标注 Qwen 改造版
  ports/                   # 三 adapter 池、独立技能库副本
```

汇总包含 Micro/Macro SR、三类能力、相对 standalone 的 Micro SR 百分点变化，
以及步数、耗时、模型切换。原始技能库不被修改。默认只允许 active polished；
`--include-candidates` 可额外放开 candidate，不自动晋升。
Simple-skill 与 SKVM 使用相同可用技能集合（active polished + 现有检索逻辑认可的 raw）；
PORTS 使用现有分层规划/路由，不改变生命周期或离线估算记录。

## 新增 baseline 入口与依赖

- `evaluate-skvm`：复用旧 runner 的真实 SKVM profiler/AOT/runtime，但使用和
  standalone 一样的 src1 M3A、报告和恢复协议。默认 profiler + p1/p1p3 编译，
  每次请求选择适配变体，无 PORTS 模型切换，多模态截图传给模型。缓存按实际权重和
  技能内容隔离，不把 PORTS prior 当作 TCP。需要 `skvm` 可执行文件，或 Bun 与已安装
  依赖的 `libs/skvm`。首次准备可能较慢。`--skvm-prepare reuse` 要求完整缓存；
  `--skvm-command /path/to/skvm` 指定程序。
- `evaluate-vdroid-qwen`：**改造版，不是原论文结果**。保留原生 UI 候选枚举、动作
  补全、工作记忆和搜索，固定 Qwen adapter 替代专用 Llama verifier 和 GPT。
  分数是 `P(Yes)/(P(Yes)+P(No))`，不是原版训练的 value head。
  服务需支持 chat `logprobs=true, top_logprobs=20`，且 Yes/No 同时有有效概率；
  不支持在 warm-up 报错，不伪造分数。请求关闭 thinking 以取得直接 Yes/No token。
  需要 V-Droid 源码、Ray 等依赖，但不下载 Llama/verifier、不调用外部教师 API。
  `--vdroid-root /path/to/v-droid` 覆盖源码目录。

原 `evaluate-vdroid` 原生 Llama 入口不变。新入口均读取主配置 TOML。

## 解读限制

PORTS 有三个 adapter，其他方法只有一个，增益包含模型池资源的贡献，不是等模型
数量的纯算法消融。V-Droid 使用独立 AndroidWorld fork，同名任务/seed 不保证两个
版本的参数实现完全相同，需要检查 checkpoint 中的 goal/task_seed。
不删掉 baseline 成功任务，不把基础设施故障计成任务失败。
本地单元测试不等于 Linux 真实 GPU/模拟器端到端验证。
