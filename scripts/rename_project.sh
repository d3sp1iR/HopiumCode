#!/usr/bin/env bash
# Переименование пакета: scripts/rename_project.sh НОВОЕ_ИМЯ  (латиница, строчные, без пробелов)
set -euo pipefail
NEW="${1:?укажите новое имя пакета}"
OLD="hopiumcode"
cd "$(dirname "$0")/.."
git mv "src/$OLD" "src/$NEW" 2>/dev/null || mv "src/$OLD" "src/$NEW"
grep -rl --exclude-dir=.git --exclude-dir=.venv "$OLD" . | xargs sed -i "s/$OLD/$NEW/g"
echo "Готово: пакет теперь $NEW. Выполните: pip install -e '.[dev]' && make test"
