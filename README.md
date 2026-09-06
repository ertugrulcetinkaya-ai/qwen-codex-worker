# Qwen Codex Worker

This worker provides two intentionally separate modes:

* `qwen-patch` is the legacy read-only patch-artifact worker. It asks Qwen for a
  unified diff and validates that diff.
* `qwen-agent` is the v3 sandbox filesystem worker. It asks Qwen for one JSON
  action at a time and makes the validated filesystem state the artifact.

The v3 agent never asks Qwen for a unified diff and never invokes `git apply`.

The OpenAI-compatible endpoint is supplied through the existing `QWEN_BASE_URL`
worker configuration. The worker does not contain a default private endpoint;
configure `QWEN_BASE_URL` with the developer's local endpoint before starting a
task.

The dedicated virtual environment is `.venv`. Codex launches the server with:

```text
/Users/ertugrulcetinkaya/Tools/qwen-codex-worker/.venv/bin/python
/Users/ertugrulcetinkaya/Tools/qwen-codex-worker/qwen_worker_server.py
```

The `qwen-patch` executable exposes the same worker through strict JSON on stdin:

```sh
echo '{"task":"Change alpha to beta","repo_root":"/tmp/example","files":["example.txt"]}' \
  | /Users/ertugrulcetinkaya/Tools/qwen-codex-worker/qwen-patch
```

It also accepts `--request-file /path/to/request.json`. Successful output is only the worker's
final response on stdout, with a terminal newline added when required for a valid text diff.
Invalid requests and worker failures produce a diagnostic on stderr and a nonzero exit status.

Patch generation uses a mechanical non-thinking profile by default, including a 4096-token output
guard. Set `QWEN_DEBUG=1` to emit generation telemetry on stderr while keeping stdout diff-only.

## v3 sandbox agent

Run the agent with an explicit disposable sandbox and writable allowlist:

`@sh
qwen-agent \
  --sandbox /tmp/qwen-agent/weather-tracker/WT-001 \
  --allow mobile/App.tsx \
  --allow mobile/src/components/LocationControls.tsx \
  --task-file /tmp/qwen-agent/weather-tracker/WT-001/task.txt
`

`--task-file` reads bounded UTF-8 task text without shell interpolation. The
short `--task` form remains available for small tasks; JSON stdin and
`--request-file` remain available for orchestrators.

The model can use the deterministic actions `read`, `grep`, `replace`, `write`,
`run`, and `done`. Paths are sandbox-relative. Reads and writes are bounded,
symlink-checked, and allowlist-enforced; replacements require an exact match.
Commands are disabled unless explicitly allowlisted with `--allow-command`, and
the worker rejects network commands, package installation, git commit/push, and
destructive git operations.

The JSON stdin form supports repair in the same existing sandbox:

`json
{
  "task": "Fix the current validation failure in the existing sandbox.",
  "sandbox": "/tmp/qwen-agent/weather-tracker/WT-001",
  "allow": ["mobile/App.tsx"],
  "validation_failure": "TypeScript reports an unused import.",
  "max_actions": 30
}
`

Successful agent stdout is a concise JSON status document containing action,
file-change, and telemetry counts. Stage diagnostics and failure telemetry go
to stderr without printing source contents, prompts, or credentials. The
worker performs a bounded `/v1/models` preflight before model calls and reports
endpoint, model, parser, validation, and execution failures by stage. Git
creates the final diff from the sandbox after local validation.
