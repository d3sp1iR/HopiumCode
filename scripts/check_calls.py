"""Ручная проверка точности рёбер CALLS (критерий AC-4: не ниже 80% на 30 рёбрах).

  python scripts/check_calls.py sample [-n 30] [--seed 1] [--out docs/calls_check.md]
      случайная выборка рёбер CALLS в виде чек-листа: у каждой строки строка кода вызова;
      откройте файл, проверьте, что на этой строке вызывается именно указанная функция,
      и замените [ ] на [x] (верно) или [-] (неверно).
  python scripts/check_calls.py score docs/calls_check.md
      считает долю верных среди размеченных.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopiumcode.config import Config  # noqa: E402
from hopiumcode.graph.extract import call_sites, extract  # noqa: E402

MARK = re.compile(r"^- \[([ x-])\] ", re.M)


def sample(n: int, seed: int, out: Path) -> int:
    root = Config.from_env().repo_root
    if not root.is_dir():
        print(f"Нет каталога {root}: сначала `make fetch`", file=sys.stderr)
        return 1
    nodes, edges = extract(root)
    lines_at = call_sites(root)
    by_id = {node.id: node for node in nodes}
    calls = sorted((e for e in edges if e.type == "CALLS"), key=lambda e: (e.src, e.dst))
    picked = random.Random(seed).sample(calls, min(n, len(calls)))
    rows = [f"# Проверка рёбер CALLS ({len(picked)} из {len(calls)}, seed={seed}, корень {root})", "",
            "Отметьте: `[x]` верно, `[-]` неверно. Верно = на указанной строке вызывается именно эту функцию.", ""]
    for e in picked:
        src, dst = by_id[e.src], by_id[e.dst]
        line = lines_at.get((e.src, e.dst))
        code = ""
        if line:
            code = (root / src.path).read_text(encoding="utf-8").splitlines()[line - 1].strip()
        rows.append(f"- [ ] `{e.src}` → `{e.dst}`  ({src.path}:{line or '?'})")
        rows.append(f"      код: `{code}`" if code else "      код: (строка не найдена)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Чек-лист записан: {out}")
    return 0


def score(path: Path) -> int:
    marks = MARK.findall(path.read_text(encoding="utf-8"))
    done = [m for m in marks if m != " "]
    if not done:
        print("Ни одно ребро не размечено")
        return 1
    right = done.count("x")
    share = right / len(done)
    print(f"Размечено {len(done)} из {len(marks)}; верных {right}; точность {share:.0%}")
    print("AC-4 выполнен" if len(done) >= 30 and share >= 0.8
          else "AC-4 не выполнен (нужно >= 30 размеченных и >= 80%); причины неточности опишите в отчёте")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", type=Path, default=Path("docs/calls_check.md"))
    c = sub.add_parser("score")
    c.add_argument("file", type=Path)
    args = ap.parse_args()
    return sample(args.n, args.seed, args.out) if args.cmd == "sample" else score(args.file)


if __name__ == "__main__":
    raise SystemExit(main())
