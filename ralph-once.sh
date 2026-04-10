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

Before you finish, update .ralph/progress.txt with concise notes for the next person working on this code that include:
- task completed
- files changed
- key decisions
- blockers or next step

Commit your changes on the current branch if you made a meaningful completed step. Include meaningful explanations of what was changed and importantly, why it was done. 

Update the particular task you completed in .ralph/json with the a field with the git commit hash so the next person can see the progress and compare against git history.

Do not push.
Do not change git remotes.
Do not merge into the default branch.

If the PRD is complete, output <promise>COMPLETE</promise>.
EOF
)

claude --permission-mode acceptEdits "@.ralph/prd.json @.ralph/progress.txt $PROMPT" 
