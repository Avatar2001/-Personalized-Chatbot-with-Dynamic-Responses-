# preprocessing/formatters/dialogue_formatter.py
from typing import Any, Dict, List
from collections import defaultdict
from .base_formatter import BaseFormatter

class DialogueFormatter(BaseFormatter):
    def __init__(self, logger=None):
        self.logger = logger

    def format_records(self, dataset_config: Dict, dataset: Any, subset: str) -> List[Dict]:
        source = dataset_config["name"]
        if "persona-chat" in source:
            return self._format_persona_chat(dataset[subset], dataset_config)
        elif "daily_dialog" in source or "better_daily_dialog" in source:
            return self._format_daily_dialog(dataset, subset, dataset_config)
        else:
            raise ValueError(f"Unsupported dialogue dataset: {source}")

    def _format_persona_chat(self, records, config: Dict) -> List[Dict]:
        dialogues = []
        prefix_map = config["role_prefixes"]
        for record in records:
            utterances = []
            for line in record[config["text_key"]]:
                for prefix, role in prefix_map.items():
                    if line.startswith(prefix):
                        text = line[len(prefix):].strip()
                        utterances.append({"role": role, "text": text})
                        break
                else:
                    if self.logger:
                        self.logger.debug(f"Unrecognized line: {line}...")
            if utterances:
                dialogues.append({"source": config["name"], "dialogue": utterances})
        return dialogues

    def _format_daily_dialog(self, dataset, subset: str, config: Dict) -> List[Dict]:
        grouped = self._group_by_id(dataset, subset, config["id_key"], config["text_key"])
        dialogues = []
        for dialog_id, utterances in grouped.items():
            dialogue = []
            for idx, text in enumerate(utterances):
                role = "user" if idx % 2 == 0 else "bot"
                dialogue.append({"role": role, "text": text})
            dialogues.append({"source": config["name"], "dialogue": dialogue})
        return dialogues

    def _group_by_id(self, dataset, subset: str, id_key: str, text_key: str) -> Dict[str, List[str]]:
        dialogues = defaultdict(list)
        for record in dataset[subset]:
            dialogues[record[id_key]].append(record[text_key])
        return dict(dialogues)