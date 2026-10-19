# Qwen Codex Worker

This worker provides two intentionally separate modes:

* `qwen-patch` is the legacy read-only patch-artifact worker. It asks Qwen for a
  unified diff and validates that diff.
* `qwen-agent` is the v3 sandbox filesystem worker. It asks Qwen for one JSON
  action at a time and makes the validated filesystem state the artifact.

The MCP server exposes both modes. Prefer `qwen_agent` for small implementation
tasks in an isolated temporary checkout: Qwen reads and edits files through the
JSON action protocol, and the host runs required validation commands after each
successful file mutation and again before returning `status="validated"`. Its MCP entry point rejects sandbox roots that
are not children of the system temporary directory. `qwen_patch` remains useful
for read-only diff proposals when a caller specifically needs a patch artifact.

The `qwen_agent` action contract is shared by the model prompt, the endpoint JSON
schema, and host validation. It supports `read`, `grep`, `replace`, `write`,
`run`, and `done`; `run` accepts only caller-allowlisted commands. The agent is
prompted to plan a few steps, verify declared APIs and direct dependencies
before editing, test each edit before finishing, and reserve actions for repair
and completion. Failed compiler/test diagnostics remain in compact context after
older turns are dropped. Failed commands that changed writable files are rolled
back to their pre-command checkpoint. Repeated identical
actions produce a progress warning and stop early if the same result persists.
After 12 read/grep or failed `run` actions without a code mutation or passing
command, the agent asks the model to make progress; it stops after 22 such
actions. A mutation or passing command resets that discovery budget. The default
action limit remains 30.

Each sandbox is bound to the first sampling profile used for it. The worker
keeps this binding in private per-user state across worker restarts, and rejects
attempts to continue one sandbox with both thinking and non-thinking settings.
Use a new clean sandbox when changing profiles.

Use `qwen-agent --resume` with the same existing temporary sandbox and required
host validation commands after an interrupted run. The host reruns those checks
first; it returns validated immediately if the current files pass, or includes
the bounded failure output in the model context so Qwen can continue from the
partial files. Resume state stays in the sandbox; no conversation transcript is
written to the project.

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

For oMLX, `QWEN_MODEL` selects the model identifier returned by `/v1/models`.
The worker reads `auth.api_key` from `~/.omlx/settings.json` at startup, requires
that file to be owned by the current user and inaccessible to group/other users,
and sends it only as an in-memory Bearer header to `/v1/models` and
`/v1/chat/completions`. It never writes the key to Codex configuration, source,
or diagnostics. `QWEN_OMLX_SETTINGS_PATH` is available for isolated tests; normal
use should leave it unset so the standard oMLX settings file is used.

The dedicated virtual environment is `.venv`. Codex launches the server with:

```text
.venv/bin/python
qwen_worker_server.py
```

The `qwen-patch` executable exposes the same worker through strict JSON on stdin:

```sh
echo '{"task":"Change alpha to beta","repo_root":"/tmp/example","files":["example.txt"]}' \
  | qwen-patch
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
read again before a later full overwrite. When a `replace` target is missing,
the next action is constrained to reread that exact file before another edit.
Targeted line reads can inspect large text files without granting full-file
write authorization. `grep` accepts files up to 4 MB and caps each search at
16 MB of file data.

The JSON stdin form supports repair in the same existing sandbox:

`json
{
  "task": "Fix the current validation failure in the existing sandbox.",
  "sandbox": "/tmp/qwen-agent/weather-tracker/WT-001",
  "allow": ["mobile/App.tsx"],
  "allowed_commands": ["python -m unittest"],
  "validation_commands": ["python -m unittest"],
  "validation_read_paths": ["src", "tests"],
  "validation_output_paths": [".dart_tool", "build"],
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
use the same parser and exact argv allowlist. When `validation_read_paths` is
provided, they can read only those paths; otherwise, they use the model's
declared readable scopes (`allow` and `read_only`). They write only to the
explicitly declared `validation_output_paths` directories for generated build
state and also use a per-invocation scratch directory. Supplied validation read
paths must already exist. Output directories must not overlap declared source
scopes, and neither path list expands the model's writable allowlist. Network access remains denied, `.env*`
and configured secret basenames remain inaccessible, and scratch state is
removed after each invocation. Command-created changes inside model-writable
project scopes are recorded and rolled back when a command fails.

On macOS, host validation may read and execute from the active Apple developer
toolchain after the host resolves it and confirms it is Command Line Tools or an
Xcode developer directory under Applications. When a host validation command
resolves to an installed Flutter SDK, it may also write to that SDK's existing
`bin/cache` directory so Flutter can update its stamp files. This SDK access is
limited to host validation; model-selected commands do not receive it. If a
Flutter command reports a denied SDK-cache write, the worker treats it as a
failure even when Flutter exits with status zero. The
validation process receives the resolved Apple toolchain as DEVELOPER_DIR;
network denial, declared project read scopes, and scratch-only writes remain in
force outside the explicitly declared generated output paths and Flutter SDK
cache.

Agent model context is deterministic and character-bounded: it keeps the stable
task/scopes context, metadata-only filesystem state, and at most six recent
interactions within a 200,000-character message budget. Repeated complete reads
reuse exact content when its path and SHA-256 still match. A complete large-file
read stays pinned outside the rolling conversation window until a successful
mutation; partial or changed reads are fetched again. Authorization state remains
host-side in the sandbox agent.

If oMLX rejects a generation with `prefill_memory_exceeded`, the worker retries
once with a 2,048-token output cap. It keeps the selected model and server memory
settings unchanged, and reports request attempts and the retry count.

Successful agent output is a concise JSON status document containing action,
file-change, completion, validation, model-call, prompt-token, completion-token,
total-token, repeated-action, protocol-error, response-parse-error, and elapsed-time
metrics, including generation request attempts and prefill retries. Token fields
are `null` when any model response omits that usage field; partial totals are not
reported as complete.
Stage diagnostics and failure telemetry go to stderr without printing source
contents, prompts, task text, validation stdout/stderr, or credentials. The
worker performs a bounded `/v1/models` preflight before model calls and reports
endpoint, model, parser, validation, and execution failures by stage. Git can
create a final diff from the sandbox after local validation.

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
