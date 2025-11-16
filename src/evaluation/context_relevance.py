from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np

def score_context_relevance(answer: str, contexts: List[str]) -> float:
    """
    Returns a score between 0 and 1 based on how relevant the contexts
    are to generating the answer. Uses TF-IDF and cosine similarity.
    """
    if not contexts or not answer:
        return 0.0
    
    try:
 
        valid_contexts = [ctx for ctx in contexts if ctx and len(ctx.strip()) > 0]
        
        if not valid_contexts:
            return 0.0

        corpus = valid_contexts + [answer]
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  
        )
        
        vectors = vectorizer.fit_transform(corpus)
        
        answer_vector = vectors[-1:]
        context_vectors = vectors[:-1]
        
        similarities = cosine_similarity(answer_vector, context_vectors)[0]
        
        top_k = min(3, len(similarities))
        top_similarities = np.sort(similarities)[-top_k:]
        avg_similarity = np.mean(top_similarities)
        
        return float(np.clip(avg_similarity, 0.0, 1.0))
        
    except Exception as e:
        print(f"Warning: Error in context_relevance calculation: {e}")
        return 0.0