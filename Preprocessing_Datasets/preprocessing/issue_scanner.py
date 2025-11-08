# preprocessing/issue_scanner.py
import re
from typing import Any, Dict, List, Callable
from .constants import DIALOUGE_And_PERSONA_SCANNING_PATTERNS

class IssueScanner:
    """
    Scans datasets for text quality issues using regex patterns.
    
    Supports both:
      - Dialogue datasets (list of utterances per record)
      - Instruction datasets (single instruction/output per record)
      
    Designed for auditing — not part of core transformation pipeline.
    """

    def __init__(
        self,
        patterns: Dict[str, re.Pattern] = None,
        logger=None
    ):
        self.patterns = patterns or DIALOUGE_And_PERSONA_SCANNING_PATTERNS
        self.logger = logger
        
    def scan_by_dataset_name(self, data_source: str, dataset, subset: str, verbose: bool):
        if "persona-chat" in data_source:
            return self.scan_dialogue_dataset(dataset, subset, "dialogue", True, verbose)
        elif "daily_dialog" in data_source:
            return self.scan_dialogue_dataset(dataset, subset, "utterance", False, verbose)
        else:
            raise ValueError(f"Unsupported dataset: {data_source}")
    
    def scan_dialogue_dataset(
        self,
        dataset: Any,
        subset: str,
        text_key: str = "dialogue",
        is_list: bool = True,
        verbose: bool = False
    ) -> Dict[str, int]:
        """Scan a dialogue dataset where text_key contains a list of strings."""
        return self._scan_dataset(
            dataset=dataset,
            subset=subset,
            text_extractor=lambda record: record[text_key],
            is_list=is_list,
            verbose=verbose
        )

    def scan_instruction_dataset(
        self,
        records: List[Dict],
        verbose: bool = False
    ) -> Dict[str, int]:
        """Scan an instruction dataset with 'instruction' and 'output' fields."""
        issue_counts = {}
        for idx, record in enumerate(records):
            for field in ["instruction", "output"]:
                text = record.get(field, "")
                if not isinstance(text, str):
                    continue
                self._scan_text(text, issue_counts, f"record_{idx}_{field}", verbose)
        if self.logger:
            self.logger.info(f"Instruction scan complete. Issues: {issue_counts}")
        return issue_counts

    def _scan_dataset(
        self,
        dataset: Any,
        subset: str,
        text_extractor: Callable,
        is_list: bool,
        verbose: bool
    ) -> Dict[str, int]:
        issue_counts = {}
        for record in dataset[subset]:
            record_id = record.get("conv_id") or record.get("dialog_id") or "unknown"
            texts = text_extractor(record)
            if not is_list:
                texts = [texts]
            for text in texts:
                if not isinstance(text, str):
                    continue
                self._scan_text(text, issue_counts, record_id, verbose)
        if self.logger:
            self.logger.info(f"Dataset scan complete. Issues: {issue_counts}")
        return issue_counts

    def _scan_text(self, text: str, counts: dict, record_id: str, verbose: bool):
        for issue_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                counts[issue_name] = counts.get(issue_name, 0) + 1
                if verbose and self.logger:
                    self.logger.info(
                        f"Issue '{issue_name}' in {record_id}: {matches} | Text: {text}..."
                    )