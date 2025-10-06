# preprocessing/difficulty_merger.py
import json
from pathlib import Path
import logging

class DifficultyMerger:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger

    def merge_difficulty_labels(
        self,
        dataset_path: str,
        difficulty_path: str,
        output_path: str
    ) -> str:
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        with open(difficulty_path, encoding="utf-8") as f:
            difficulties = json.load(f)

        if len(dataset) != len(difficulties):
            msg = f"Length mismatch: dataset={len(dataset)}, difficulties={len(difficulties)}"
            (self.logger.warning if self.logger else print)(f"⚠ {msg}")

        merged = []
        for i, item in enumerate(dataset):
            difficulty = difficulties[i] if i < len(difficulties) else "unknown"
            item["instruction"] = f"<{difficulty}> {item['instruction']}"
            merged.append(item)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        if self.logger:
            self.logger.info(f"Merged difficulty into {len(merged)} records → {output_path}")
        return output_path