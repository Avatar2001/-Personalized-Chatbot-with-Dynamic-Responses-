from .base_model_formatter import BaseModelFormatter
import logging

class DialogueModelFormatter(BaseModelFormatter):
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger

    def format_for_model(self, conversations: list) -> str:
        if not conversations:
            if self.logger:
                self.logger.warning("No conversations to format.")
            return ""

        lines = []
        total_turns = 0

        for conv in conversations:
            dialogue = conv.get("dialogue", [])
            for turn in dialogue:
                role = turn.get("role", "").upper()
                text = turn.get("text", "").strip()
                if text:
                    lines.append(f"{role}: {text}")
                    total_turns += 1
                elif self.logger:
                    self.logger.debug("Skipped empty utterance.")

            lines.append("")  # blank line between conversations

        if self.logger:
            self.logger.info(f"Formatted {len(conversations)} conversations with {total_turns} turns.")
        return "\n".join(lines)