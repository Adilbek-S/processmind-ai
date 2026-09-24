"""Генерация `evals/golden_dataset.json` из `evals/dataset_source.py`: `python -m evals.build_dataset`."""

from __future__ import annotations

import json

from evals.dataset import DATASET_PATH, load_dataset
from evals.dataset_source import EXAMPLES


def main() -> None:
    DATASET_PATH.write_text(json.dumps(EXAMPLES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = load_dataset()  # сразу проверяем то, что записали
    kinds = {k: sum(1 for e in dataset.examples if e.kind == k) for k in ("positive", "negative", "ambiguous")}
    print(f"Записано {len(dataset.examples)} примеров в {DATASET_PATH.name}: {kinds}")


if __name__ == "__main__":
    main()
