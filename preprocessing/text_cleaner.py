# preprocessing/text_cleaner.py
import re
import unicodedata
from typing import Dict
import logging
from .constants import CLEANING_PATTERNS

class TextCleaner:
    def __init__(self, patterns: Dict[str, re.Pattern] = None, logger=None):
        self.patterns = patterns or CLEANING_PATTERNS
        self.logger = logger

    def clean_dialogue_text(self, text: list[str]) -> list[str]:
        return [self._apply_common_cleaning(t) for t in text]
    
    def clean_dialogue_text(self, text: list[str]) -> list[str]:
        return [self._apply_common_cleaning(t) for t in text]

    def clean_instruction_text(self, text: str) -> str:
        text = self._apply_common_cleaning(text)
        # Instruction-specific
        text = self.patterns["code_blocks"].sub("", text)
        text = self.patterns["inline_code"].sub(r"\1", text)
        #text = self.patterns["markdown"].sub("", text)
        text = self.patterns["special_chars"].sub("", text)
        text = unicodedata.normalize("NFKC", text)
        return text.strip()
        
    def _apply_common_cleaning(self, text: str) -> str:
        if not isinstance(text, str):
            original = text
            text = str(text)
            if self.logger:
                self.logger.warning(f"Non-string input converted: {original}")

        try:
            # Remove structural noise
            text = self.patterns["html_tags"].sub("", text)
            text = self.patterns["urls"].sub("<URL>", text)
            text = self.patterns["emails"].sub("<EMAIL>", text)

            # Normalize quotes and contractions
            text = text.translate(str.maketrans("‘’“”", "''\"\""))
            text = re.sub(r"'\s+", "'", text)      # "don ' t" → "don' t"
            text = re.sub(r"\s+'", "'", text)      # "don 't" → "don't"
            text = re.sub(r"\\+'", "'", text)      # "don\\'t" → "don't"
            text = text.replace('\\"', '"')        # escaped quotes
            text = self.patterns["broken_contractions"].sub(r"\1'\2", text)  # "don ' t" → "don't"

            # Replace any sequence of em/en dashes (with optional surrounding spaces)
            # with a single hyphen that has exactly one space on each side
            text = re.sub(r"\s*[–—]+\s*", " - ", text)
            
            # Normalize dashes and spacing
            text = self.patterns["unicode_dashes"].sub("-", text)  # "–", "—" → "-"
            #text = re.sub(r"\s+-\s+", " ", text)   # "word - word" → "word word"

            # Only normalize hyphens that have whitespace on BOTH sides (or at least one side)
            # This preserves hyphenated words like "well-made" and "mother-in-law"
            dash_pattern = r"\s*-\s*"
            text = re.sub(dash_pattern, " ", text)

            text = re.sub(r"\s+", " ", text)       # collapse whitespace

            # Define the regex pattern to match " - - "
            double_dashes_with_spaces_pattern = r" - - "
            # Use re.sub() to replace the pattern with an empty string (effectively removing it)
            text = re.sub(double_dashes_with_spaces_pattern, "", text)

            # Fix punctuation spacing
            text = self.patterns["spaced_punctuation"].sub(r"\1", text)  # "word ." → "word."
            text = text.replace(" ,", ",")         # "word ," → "word,"
            return text.strip()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Cleaning failed: {e} | Snippet: {text}")
            return text