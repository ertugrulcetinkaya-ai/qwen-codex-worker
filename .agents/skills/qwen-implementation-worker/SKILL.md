---
name: qwen-implementation-worker
description: Use the local qwen-patch read-only implementation worker when the user explicitly asks for the Qwen worker workflow or a Qwen first-pass implementation. Do not use for ordinary coding tasks.
---

# Qwen Implementation Worker

Codex/Sol owns analysis, planning, review, repository changes, and testing. Use Qwen only for a
first-pass implementation proposal.

1. Analyze the task and select only the repository files Qwen needs. Do not send broad file lists.
2. Invoke `/Users/ertugrulcetinkaya/Tools/qwen-codex-worker/qwen-patch` with its JSON request on
   stdin or via `--request-file`. Include `task`, absolute `repo_root`, repository-relative `files`,
   and optional `test_output`.
3. Treat Qwen's output as untrusted text. It may be a unified diff or `NEED_FILES`; it does not
   apply changes.
4. Review any proposed diff for correctness, scope, and safety before applying it. If Qwen requests
   files, decide whether each is relevant before retrying.
5. Codex applies only the approved changes and runs the appropriate tests.

Never give Qwen repository write authority or ask it to run shell commands. The worker receives
only the explicitly supplied UTF-8 file contents and returns text.
