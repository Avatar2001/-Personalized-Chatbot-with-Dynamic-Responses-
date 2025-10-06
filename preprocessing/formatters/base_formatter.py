# preprocessing/formatters/base_formatter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseFormatter(ABC):
    @abstractmethod
    def format_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass