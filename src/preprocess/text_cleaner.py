import os
import re
import glob
from bs4 import BeautifulSoup

class TextCleaner:
    """Cleans and structures raw text extracted from PDFs."""

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_text(self, text):
        """Apply all cleaning rules to text."""
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove front matter and metadata
        text = re.sub(r'(EARLY\s*ACCESS|NO\s*STARCH\s*PRESS|Feedback\s*Welcome|Copyright).*?(?=Introduction|Chapter|Contents)',
                      '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Table of Contents.*?(?=Introduction|Chapter\s*1)', '',
                      text, flags=re.DOTALL | re.IGNORECASE)

        # Remove non-ASCII & page numbers
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'Page\s*\d+', '', text)

        # Fix spaced letters (e.g., "A U T O M A T E" → "AUTOMATE")
        text = re.sub(r'(?<=\b)([A-Za-z])(?:\s[A-Za-z]){2,}(?=\b)',
                      lambda m: m.group(0).replace(' ', ''), text)

        # Fix double underscores and paragraph spacing
        text = re.sub(r"\s+([a-zA-Z_]+)\s+", r"\1__", text)
        text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)

        # Keep only relevant part (starting from Introduction or Chapter 1)
        match = re.search(r'(Introduction|Chapter\s*1)', text, flags=re.IGNORECASE)
        if match:
            text = text[match.start():]

        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_all_texts(self):
        """Clean all text files in input_folder."""
        for file in glob.glob(f"{self.input_folder}/*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned = self.clean_text(raw_text)
            file_name = os.path.basename(file)
            output_path = os.path.join(self.output_folder, file_name)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"✅ Cleaned text saved: {output_path}")

        print("\n✨ Cleaning completed for all files!")