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

