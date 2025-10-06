# preprocessing/formatters/instruction_formatter.py
from typing import List, Dict, Any
from .base_formatter import BaseFormatter

class InstructionFormatter(BaseFormatter):
    def __init__(self, logger=None):
        self.logger = logger

    def format_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for item in records:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if not instruction or not output:
                if self.logger:
                    self.logger.debug("Skipped record with empty instruction or output.")
                continue
            formatted.append({
                "source": "instruction_dataset",
                "instruction": instruction,
                "output": output
            })
        if self.logger:
            self.logger.info(f"Formatted {len(formatted)} instruction records.")
        return formatted