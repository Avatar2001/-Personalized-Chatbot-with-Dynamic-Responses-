# preprocessing/deduplicator.py
from typing import List, Dict, Any
import logging

class Deduplicator:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger

    def remove_duplicates(
        self,
        records: List[Dict[str, Any]],
        key_fields: List[str]
    ) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for record in records:
            key = tuple(record.get(field, "") for field in key_fields)
            if key not in seen:
                seen.add(key)
                unique.append(record)
        if self.logger:
            self.logger.info(f"Deduplicated {len(records)} → {len(unique)} records.")
        return unique