import os
from preprocess.pdf_extractor import PDFExtractor
from preprocess.text_cleaner import TextCleaner
from rag.data_loader import DataLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vectorstore_manager import VectorStoreManager
from rag.query_engine import QueryEngine
from rag.auto_merging_retriever import AutoMergingRetriever
from llm.llm_generator import LLMGenerator
from rag.retriever_adapter import RetrieverAdapter
from evaluation.evaluator import evaluate_answer
from rag.reranker  import Reranker
from dotenv import load_dotenv

if __name__ == "__main__":
    try:
        text_files = [
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/automate-the-boring-stuff-with-python-3rd-edition-early-access-3nbsped-9781718503403-9781718503410_com.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/fluent-python-2nbsped-9781492056348-9781492056287.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/learning-python-powerful-object-oriented-programming-6nbsped-1098171306-9781098171308.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/leetcode-python.txt",
            "-Personalized-Chatbot-with-Dynamic-Responses-/src/data/cleaned_texts/python-crash-course-3nbsped-1718502702-9781718502703_compress.txt"
        ]

        print("Loading documents...")
        loader = DataLoader(text_files)
        documents = loader.load()

        print("Chunking documents...")
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.split(documents)

        print("Creating embeddings...")
        embedder = Embedder()
        embedding_model = embedder.get_model()

        print("Building vector store...")
        vector_manager = VectorStoreManager(embedding_model)
        qdrant_store = vector_manager.create_store(chunks)

        print("Setting up retriever...")
        base_retriever = vector_manager.get_retriever(top_k=50)
        auto_retriever = AutoMergingRetriever(
            base_retriever=base_retriever,  
            merge_char_limit=1500,
            max_chunks_per_merge=6,
            top_k=50,
        )

        query_engine = QueryEngine(auto_retriever)

        print("Initializing LLM...")
        load_dotenv()
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        generator = LLMGenerator(api_key=gemini_key)

        q = "What is recursion in Python?"
        
        print(f"\n{'='*80}")
        print(f"Question: {q}")
        print(f"{'='*80}\n")
        
        print("Retrieving documents...")
        merged_docs = query_engine.query(q)  
        retrieved_texts = [d.page_content for d in merged_docs[:50]] 

        print("Reranking retrieved documents...")
        reranker = Reranker()
        reranked = reranker.rerank(q, retrieved_texts, top_k=5)
        docs = [doc for doc, score in reranked]
        scores = [score for doc, score in reranked]

        print("Documents:", docs)
        print("Scores:", scores)


        print("Generating answer...")
        answer = generator.answer_generation(q, docs)

        print("\n" + "="*80)
        print("GENERATED ANSWER")
        print("="*80)
        print(answer)
        print("="*80 + "\n")

        reference_answers = [
            "Recursion is a function calling itself to solve a problem",
            "Recursion occurs when a function calls itself",
            "A recursive function is one that invokes itself"
        ]
        
        print("Evaluating answer quality...\n")
        evaluation_scores = evaluate_answer(
            question=q,
            answer=answer,
            reference_answers=reference_answers,
            contexts=retrieved_texts,
            sources=retrieved_texts
        )

        print("="*80)
        print("EVALUATION SCORES")
        print("="*80)
        for metric, score in evaluation_scores.items():
            print(f"{metric.replace('_', ' ').title():20s}: {score:.2f}")
        print("="*80 + "\n")
        
        avg_score = sum(evaluation_scores.values()) / len(evaluation_scores)
        print(f"Overall Average Score: {avg_score:.2f}\n")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()