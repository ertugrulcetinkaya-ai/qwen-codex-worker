# Qwen Codex Worker

This worker provides two intentionally separate modes:

* `qwen-patch` is the legacy read-only patch-artifact worker. It asks Qwen for a
  unified diff and validates that diff.
* `qwen-agent` is the v3 sandbox filesystem worker. It asks Qwen for one JSON
  action at a time and makes the validated filesystem state the artifact.

The v3 agent never asks Qwen for a unified diff and never invokes `git apply`.

Before `qwen-patch` returns a unified diff, its validator checks every file
header against the canonical repository-relative files supplied by the caller.
Only same-file modifications are accepted. New files, deletions, renames,
cross-file targets, traversal, absolute paths, and quoted Git paths are
rejected; `NEED_FILES` keeps its separate relative-path contract. The worker
never applies the returned patch or mutates the repository.

Request documents are capped at 2,000,000 UTF-8 bytes and are read with a
bounded `limit+1` guard before JSON parsing. Task text, validation feedback,
test output, paths, commands, and model identifiers have separate byte and
character limits; list inputs also have cardinality limits. Oversized input is
rejected before a model or preflight request and is never silently truncated.
The combined model prompt is capped at 500,000 characters and 2,400,000
UTF-8 bytes. Repeated writable/read-only scopes and allowed commands are
deduplicated in execution order; repeated validation commands remain repeated
because they are an ordered host-validation contract.

The OpenAI-compatible endpoint is supplied through the existing `QWEN_BASE_URL`
worker configuration. The worker does not contain a default private endpoint;
configure `QWEN_BASE_URL` with a loopback `http://` or `https://` endpoint before
starting a task. Remote endpoints, URL credentials, query strings, fragments,
and proxy environment variables are not supported; model traffic is direct to
the validated loopback endpoint.

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
guard. Both generation paths enforce a client-side bounded SSE transport: at most 2,000,000 raw
stream bytes, 10,000 SSE event blocks, 256,000 bytes per event, and 300 seconds from request start
through headers, retries, and stream completion. Patch final content is capped at 512,000 characters
and 2,000,000 UTF-8 bytes; agent final action text is capped at 20,000 characters and 80,000 bytes.
Hidden reasoning is never returned and is capped at 200,000 characters and 800,000 bytes. Non-2xx
streaming error bodies are read only up to 4,096 bytes. Any limit, malformed stream, deadline, or
mid-stream failure fails the generation closed without returning partial output; only connection
failures explicitly approved by the operation policy are retried. The parser accepts blank lines, CRLF, comments/heartbeats,
and multiple `data:` fields, and treats `[DONE]` as terminal. Set `QWEN_DEBUG=1` to emit bounded
generation telemetry on stderr while keeping stdout diff-only.

Preflight model discovery is an idempotent GET and may retry bounded transient transport failures.
Generation is a POST: only failures known to occur before the request can reach the model server
(`ConnectTimeout` and `PoolTimeout`) may retry, within the same 300-second invocation deadline.
Generation requests are never retried after an ambiguous send/read failure, HTTP response, stream
failure, or client-enforced limit, so a server-side inference cannot be duplicated by transport
recovery.

## v3 sandbox agent

Run the agent with an explicit disposable sandbox and writable allowlist:

`@sh
qwen-agent \
  --sandbox /tmp/qwen-agent/weather-tracker/WT-001 \
  --allow mobile/App.tsx \
  --allow mobile/src/components/LocationControls.tsx \
  --allow-command "python -m unittest" \
  --validation-command "python -m unittest" \
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

`grep` is a bounded observation: it scans at most 500 files, returns at most 100
matches, and caps matching output at 20,000 characters and 80,000 UTF-8 bytes.
Each response reports `files_scanned`, `files_skipped`, `truncated`, and a
deterministic `truncation_reason`; binary, invalid-UTF-8, and oversized files
are skipped and reported. Directory traversal is sorted. A `grep` never counts
as a full read. Rewriting an existing non-empty file requires an explicit,
complete `read` whose raw-byte SHA-256 still matches at write time. Any
successful `replace` or `write` invalidates that proof, so the file must be
read again before a later full overwrite.

The JSON stdin form supports repair in the same existing sandbox:

`json
{
  "task": "Fix the current validation failure in the existing sandbox.",
  "sandbox": "/tmp/qwen-agent/weather-tracker/WT-001",
  "allow": ["mobile/App.tsx"],
  "allowed_commands": ["python -m unittest"],
  "validation_commands": ["python -m unittest"],
  "validation_failure": "TypeScript reports an unused import.",
  "max_actions": 30
}
`

`validation_commands` is an explicit host-validation contract. Each command
must also be present as the same exact argv in `allowed_commands`; it passes
the existing command parser and command sandbox. When the model says `done`,
that means only that it has finished editing. With no required validation
commands the result is `status="done", validated=false`. With required
commands, the host runs them sequentially after `done`; only when all return
zero does the result become `status="validated", validated=true`. A failed
validation is returned to the model as bounded feedback so it can repair the
sandbox, with at most three validation rounds. A subsequent `done` after any
successful `replace` or `write` mutation runs the required validation again;
model-selected `run` results are never treated as proof of completion.

Commands use the macOS `sandbox-exec` backend fail-closed when it is available.
Model-selected `run` commands can read only the declared readable scopes and
trusted runtime paths, write only declared writable scopes, have network access
denied, cannot access `.env*` or configured secret basenames, and receive a
per-invocation scratch directory for temporary state. Host validation commands
use the same parser and exact argv allowlist, but the project tree is read-only:
they can read only declared readable scopes and can write only to their dedicated
scratch directory. Command-created changes inside writable project scopes are
recorded before validation; scratch state is removed after each invocation.

Agent model context is deterministic and character-bounded: it keeps the stable
task/scopes context, metadata-only filesystem state, and at most six recent
interactions within a 200,000-character message budget. Authorization state
remains host-side in the sandbox agent.

Successful agent stdout is a concise JSON status document containing action,
file-change, completion, validation, and telemetry counts. Stage diagnostics
and failure telemetry go to stderr without printing source contents, prompts,
task text, validation stdout/stderr, or credentials. The worker performs a
bounded `/v1/models` preflight before model calls and reports endpoint, model,
parser, validation, and execution failures by stage. Git creates the final
diff from the sandbox after local validation.

## Development and CI

The project targets the developer-supported Python 3.11 line. Runtime
dependencies are declared only in `pyproject.toml`; `uv.lock` records the exact
resolved versions. The repository is script-style and is not packaged for PyPI.

Set up the locked development environment and run the local gates with:

```sh
uv sync --frozen --group dev
uv run ruff check .
uv run mypy
uv run python -m unittest discover -v
uv run python -m compileall -q qwen_agent.py qwen_agent_cli.py qwen_input_limits.py qwen_patch_cli.py qwen_worker_server.py test_qwen_agent.py test_qwen_patch_cli.py test_qwen_worker_server.py
uv run pip-audit
```

When a dependency declaration changes, regenerate and verify the lockfile with
`uv lock`, then use `uv lock --check` in CI. CI never updates or commits the
lockfile. Pushes and pull requests run the same locked quality and security
gates; the macOS job additionally runs the real `sandbox-exec` integration
tests and fails closed if that backend is unavailable.
