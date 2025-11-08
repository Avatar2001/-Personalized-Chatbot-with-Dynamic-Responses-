# preprocessing/constants.py
import re

CLEANING_PATTERNS = {
    "html_tags": re.compile(r"<.*?>"),
    "curly_quotes": re.compile(r"[“”‘’]"),
    "broken_contractions": re.compile(r"\b([A-Za-z]+)\s+['’]\s+([A-Za-z]+)\b"),
    "spaced_punctuation": re.compile(r"\s+([.,!?;:])"),
    "urls": re.compile(r"http[s]?://\S+|www\.\S+"),
    "emails": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    "unicode_dashes": re.compile(r"[–—\-]"),
    
    #"markdown": re.compile(r"[*#>]+"),
    "code_blocks": re.compile(r"```.*?```", re.DOTALL),
    "inline_code": re.compile(r"`([^`]*)`"),
    "special_chars": re.compile(r"[^\w\s.,!?;:'\"()\-+/=\u0080-\uFFFF]"),
}

DIALOUGE_And_PERSONA_SCANNING_PATTERNS = {
    # Structural noise
    "html_tags": re.compile(r"<[^>]+>"),  # safer than <.*?> (avoids greedy match across tags)
    
    "urls": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    
    "emails": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    
    # Quotes & contractions
    "curly_quotes": re.compile(r"[‘’“”]"),  # matches any curly quote
    
    "broken_contractions": re.compile(
        r"\b([A-Za-z]+)\s+['’]\s+([A-Za-z]+)\b"
    ),  # e.g., "don ' t"
    
    # Dashes: only flag **spaced hyphens** as issues (not "well-made")
    "spaced_hyphens": re.compile(r"\s+-\s+"),  # e.g., "word - word"
    
    # Unicode dashes (em/en dash) — these are normalized, so flag them as "non-standard"
    "unicode_dashes": re.compile(r"[–—]"),  # note: NOT including ASCII '-' here
    
    # Punctuation spacing
    "spaced_punctuation": re.compile(r"\s+([.,!?;:])"),  # e.g., "word ."
    
    # Excessive whitespace (optional, but useful)
    "extra_whitespace": re.compile(r"\s{2,}"),
}