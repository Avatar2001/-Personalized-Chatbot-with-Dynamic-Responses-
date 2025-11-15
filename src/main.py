import os
from preprocess.pdf_extractor import PDFExtractor
from preprocess.text_cleaner import TextCleaner
from rag.data_loader import DataLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore_manager import VectorStoreManager
from rag.query_engine import QueryEngine
from llm.llm_generator import LLMGenerator

if __name__ == "__main__":
    try:
        # Paths
        RAW_PDF_DIR = "data/raw_pdfs"
        EXTRACTED_DIR = "data/extracted_texts"
        CLEANED_DIR = "data/cleaned_texts"
        # ============================================
        # Step 1: Extract PDF text
        # ============================================
        extractor = PDFExtractor(RAW_PDF_DIR, EXTRACTED_DIR)
        extractor.process_all_pdfs()

        # ============================================
        # Step 2: Clean extracted text
        # ============================================
        cleaner = TextCleaner(EXTRACTED_DIR, CLEANED_DIR)
        cleaner.process_all_texts()

        # ============================================
        # Step 3: Load, Chunk, Embed, Store, Query
        # ============================================
        text_files = [os.path.join(CLEANED_DIR, f) for f in os.listdir(CLEANED_DIR) if f.endswith(".txt")]
        print(f"Loading {len(text_files)} cleaned text files.")
        loader = DataLoader(text_files)
        documents = loader.load()

        chunker = TextChunker()
        chunks = chunker.split(documents)

        embedder = Embedder()
        embedding_model = embedder.get_model()

        vector_manager = VectorStoreManager(embedding_model)
        qdrant_store = vector_manager.create_store(chunks)

        retriever = vector_manager.get_retriever(top_k=5)
        query_engine = QueryEngine(retriever)

        generator = LLMGenerator(api_key=os.environ("GEMINI_API_KEY"))
        answer = generator.answer_generation("What is recursion in Python?", query_engine.query("What is recursion in Python?"))

        # Example Query
        print(query_engine.query("What is recursion in Python?"))
        print(answer)

    except Exception as e:
        print(f"An error occurred: {e}")