#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:?Usage: $0 /absolute/path/to/repo}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repository directory not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

mkdir -p .ralph
touch .ralph/progress.txt

PROMPT=$(cat <<EOF
You are running one Ralph iteration in this repository.

Read:
- .ralph/prd.json
- .ralph/progress.txt
- AGENTS.md, if it exists

Treat .ralph/prd.json as the overall source of truth for scope.
Treat AGENTS.md as binding repository-local guidance for testing, feedback loops, and engineering constraints.
If .ralph/prd.json exists, use it as the small-task execution board and choose the highest-priority incomplete item from it unless the PRD clearly says otherwise. This should be the one YOU decide is the highest priority - DO NOT assume this is the first incomplete task on the list. Do NOT blindly pick the first task in prd.json

IF NEEDED use lightweight models for subagents to explore targeted parts of the code or prior work done by others - particularly in progress.txt and git history are good starting points.

Prefer risky architectural and cross-module work ahead of cleanup or polish.

Work on exactly one Ralph-sized task, or one tightly related pair of subtasks only if they must land together.
Run the repository feedback loops before finishing.

## Environment

This project uses UV for dependency management.
- Install/sync dependencies: \`uv sync\`
- Run the CLI: \`uv run concam <command>\`
- Run tests: \`uv run pytest\`
- Add a dependency: \`uv add <package>\`
Do NOT use pip, conda, or any other package manager. Do NOT activate a virtualenv manually.

## Tracking files

- .ralph/progress.txt is a local-only file (gitignored). Update it with concise notes for the next session: task completed, files changed, key decisions, blockers or next step.
- .ralph/prd.json is tracked in git. Update the completed task's entry with passes=true and the git commit hash. Do NOT commit prd.json after each session — it will be committed once at the very end when all tasks are marked done.

Before you finish, update .ralph/progress.txt with:
- task completed
- files changed
- key decisions
- blockers or next step

Commit your code changes on the current branch with meaningful explanations of what was changed and importantly, why it was done.

Do not push.
Do not change git remotes.
Do not merge into the default branch.

If the PRD is complete, commit prd.json as the final tracking record, then output <promise>COMPLETE</promise>.
EOF
)

claude --model opus --permission-mode acceptEdits "@.ralph/prd.json @.ralph/progress.txt $PROMPT"
