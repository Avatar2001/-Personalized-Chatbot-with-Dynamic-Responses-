import fitz  # PyMuPDF
import os
import glob

class PDFExtractor:
    """Extracts raw text from all PDFs in a folder."""

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_text_from_pdf(self, path):
        """Extract all text from a single PDF."""
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def process_all_pdfs(self):
        """Extract text from all PDFs in the input folder."""
        for pdf_file in glob.glob(f"{self.input_folder}/*.pdf"):
            text = self.extract_text_from_pdf(pdf_file)
            file_name = os.path.basename(pdf_file).replace(".pdf", ".txt")
            output_path = os.path.join(self.output_folder, file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Extracted text saved: {output_path}")

        print("\n✨ Extraction completed for all PDF files!")