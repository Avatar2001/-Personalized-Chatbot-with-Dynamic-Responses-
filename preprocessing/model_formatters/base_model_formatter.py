from abc import ABC, abstractmethod

class BaseModelFormatter(ABC):
    @abstractmethod
    def format_for_model(self, records: list) -> str:
        pass