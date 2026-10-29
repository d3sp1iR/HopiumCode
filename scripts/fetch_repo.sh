#!/usr/bin/env bash
# Клонирует репозиторий-цель: scripts/fetch_repo.sh  (переопределение: REPO_URL, REPO_REF, REPO_DIR)
# Клон полный, без --depth: вопросы eval про историю опираются на git log / git blame.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/pallets/flask.git}"
REPO_REF="${REPO_REF:-22d924701a6ae2e4cd01e9a15bbaf3946094af65}"
REPO_DIR="${REPO_DIR:-data/repo/flask}"
cd "$(dirname "$0")/.."

if [ ! -e "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
elif [ ! -d "$REPO_DIR/.git" ]; then
  echo "Ошибка: $REPO_DIR существует, но это не git-репозиторий" >&2
  exit 1
fi

want="$(git -C "$REPO_DIR" rev-parse -q --verify "$REPO_REF^{commit}" || true)"
if [ -z "$want" ]; then
  git -C "$REPO_DIR" fetch --tags origin
  want="$(git -C "$REPO_DIR" rev-parse -q --verify "$REPO_REF^{commit}" || true)"
  if [ -z "$want" ]; then
    echo "Ошибка: в $REPO_URL нет ref $REPO_REF" >&2
    exit 1
  fi
fi

if [ "$(git -C "$REPO_DIR" rev-parse HEAD)" != "$want" ]; then
  if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
    echo "Ошибка: в $REPO_DIR есть незакоммиченные изменения, переключение на $REPO_REF отменено" >&2
    exit 1
  fi
  git -C "$REPO_DIR" -c advice.detachedHead=false checkout "$REPO_REF"
fi
echo "Готово: $REPO_DIR на $REPO_REF ($(git -C "$REPO_DIR" rev-parse --short HEAD))"
