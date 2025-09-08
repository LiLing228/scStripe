#!/usr/bin/env bash
set -euo pipefail

# Repository root (directory where this script lives)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Commit message (default is timestamp if not provided as argument)
MSG="${1:-"Update: $(date '+%F %T')"}"

# 1) Stage & commit (skip if no changes)
git add -A
if git diff --cached --quiet; then
  echo "No staged changes to commit. Skipping commit."
else
  git commit -m "$MSG"
fi

# 2) Try pull --rebase (fallback to pull if unsupported)
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  if git pull --rebase; then
    :
  else
    echo "git pull --rebase failed, falling back to normal pull."
    git pull
  fi
else
  echo "No upstream branch configured, skipping pull."
fi

# 3) Push to current branch with upstream tracking
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git push -u origin "$CURRENT_BRANCH"