# V-Droid baseline

This package runs the upstream `libs/V-Droid` verifier-driven agent as an
AndroidWorld baseline. It intentionally does not import the PORTS skill store,
planner, dynamic-programming router, model pool, or online skill adaptation.

The worker process loads V-Droid's vendored AndroidWorld fork in isolation. The
only shared contract with PORTS is the evaluation protocol and output schema:
task list, combinations, seed, maximum steps, Micro/Macro SR, per-task results,
three capability scores (Reasoning, Coordination, Manipulation), traces, and checkpoints.

```bash
cp src1/baseline_vdroid/config.example.toml \
  src1/baseline_vdroid/config.local.toml

export OPENAI_API_KEY=...

python -m src1 \
  --config src1/baseline_vdroid/config.local.toml \
  evaluate-vdroid \
  --tasks ContactsAddContact SimpleCalendarAddOneEvent \
  --combinations 5 --seed 42 --max-steps 30
```

Use `--dry-run` first to inspect the fixed models, paths, dependency status, and
worker command without loading CUDA or connecting to the emulator. The native
V-Droid environment, including its vLLM 0.7.3 compatibility patch, must be
installed as documented in `libs/V-Droid/README.md`.

Use `--resume --output-dir /path/to/the/same/run` to skip completed task parameter
instances after an interruption. `[evaluation].resume = true` enables the same
policy in TOML; `--no-resume` overrides it. Normal task failures are retained,
while exception/corrupt checkpoints are retried. The common `progress.json` and
`evaluation_manifest.json` record completion and experiment identity. Reports
merge old and new complete episodes. Changed experiment settings or pre-update
directories without a manifest require a new output directory; no old results
are deleted. An interrupted episode restarts from step zero, not from the middle
of an emulator interaction.

`--no-store-trajectories` (or `[evaluation].store_trajectories = false`) disables
the shared trace JSONL and full episode pickle files. Only reports and small
result-only JSON checkpoints are saved, with resume and verifier statistics
preserved. It also overrides screenshot saving. Existing artifacts and native
V-Droid/CLI diagnostic logs are not deleted.

## Qwen adapter: smoke test and invalid-run diagnosis

`evaluate-vdroid-qwen` is the explicitly labelled Qwen adaptation; it does not
use the original trained Llama verifier. For the 5554/8554 emulator, confirm
these settings in `src1/config.local_test.toml` (the inference port is separate):

```toml
[android_world]
console_port = 5554
grpc_port = 8554
```

Run two tasks on CUDA 0 / inference port 8002 before a full suite:

```bash
python -m src1 --config src1/config.local_test.toml evaluate-vdroid-qwen \
  --adapter-path /home/zmz/Workspace/gui/src1/runtime/checkpoints/adapter_control01 \
  --model-id adapter_control01 \
  --tasks AudioRecorderRecordAudio ContactsAddContact \
  --combinations 1 --seed 42 --max-steps 30 \
  --cuda-visible-devices 0 --deploy-port 8002 --infer-backend vllm \
  --no-require-training-state --resume --no-store-trajectories \
  --output-dir /home/zmz/Workspace/gui/src1/runtime/evaluations/vdroid_control_smoke_fixed
```

Check that `summary.json` has `evaluation_available = true`,
`episodes_evaluated = 2`, and nonzero `vdroid.decision_count`. This checks that
the baseline actually ran; it does not require either task to succeed. For the
full suite, remove `--tasks AudioRecorderRecordAudio ContactsAddContact` and use
a **different** output directory such as `.../vdroid_control_fixed`. Do not
resume a two-task smoke test as a full-suite experiment in the same directory.

The previous `AttributeError: 'StrictFixedClient' object has no attribute
'model_name'` occurred at the end of upstream `search()`, after model requests
had already succeeded. The upstream task harness swallowed it and discarded
each episode; 116 such exceptions mean **unavailable evaluation, not SR=0**.
The wrapper now supplies the deployed adapter identity and optional timing
arguments, and is checked before GUI tasks start. Backend/interface/import
errors stop the suite immediately; ordinary completed task failures remain
valid failures and do not stop evaluation. Zero-valid-episode runs cannot
return CLI success; diagnostic reports display N/A and V-Droid JSON SR values
are null. Existing files are not rewritten or deleted. Normal resume retries
exception episodes rather than treating them as completed tasks.

