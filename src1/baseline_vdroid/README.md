# V-Droid baseline

This package runs the upstream `libs/V-Droid` verifier-driven agent as an
AndroidWorld baseline. It intentionally does not import the PORTS skill store,
planner, dynamic-programming router, model pool, or online skill adaptation.

The worker process loads V-Droid's vendored AndroidWorld fork in isolation. The
only shared contract with PORTS is the evaluation protocol and output schema:
task list, combinations, seed, maximum steps, Micro/Macro SR, per-task results,
seven capability scores, traces, and checkpoints.

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

