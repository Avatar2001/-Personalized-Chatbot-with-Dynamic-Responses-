from .base_model_formatter import BaseModelFormatter
import logging

class InstructionModelFormatter(BaseModelFormatter):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger

    def format_for_model(self, records: list) -> str:
        lines = []
        for record in records:
            lines.append(f"question: {record['instruction']}")
            lines.append(f"answer: {record['output']}")
            lines.append("")
        return "\n".join(lines)