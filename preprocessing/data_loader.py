# preprocessing/data_loader.py
import json
from pathlib import Path
from typing import Any, List, Dict
from datasets import load_dataset, DatasetDict

class DatasetLoader:
    def load_huggingface_dataset(self, dataset_name: str, split=None, streaming=False, columns=None):
        """Load Hugging Face dataset with optional column selection."""
        if streaming:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            if columns:
                # For streaming, we'll filter columns during iteration
                return dataset  # columns handled in main loop
            return dataset
        else:
            dataset = load_dataset(dataset_name)
            if columns and split:
                # Non-streaming: select columns
                return dataset.map(lambda x: {k: x[k] for k in columns if k in x}, 
                                remove_columns=[k for k in dataset[split].column_names if k not in columns])
            return dataset

    def load_json_file(self, filepath: str) -> List[Dict]:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, data: List[Dict], filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_text(self, text: str, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)