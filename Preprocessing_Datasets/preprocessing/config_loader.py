# preprocessing/config_loader.py
from pathlib import Path
import yaml

class ConfigLoader:
    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(self.config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default if default is not None else KeyError(f"Key '{key}' not found")