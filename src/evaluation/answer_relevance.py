from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def score_answer_relevance(question: str, answer: str, reference_answers: List[str] = None) -> float:
    """
    Computes relevance score between question and answer using semantic similarity.
    If reference answers are provided, also checks similarity with them.
    Returns a float between 0 and 1.
    """
    if not answer or not question:
        return 0.0

    try:
        vectorizer = TfidfVectorizer().fit([question, answer])
        vectors = vectorizer.transform([question, answer])
        qa_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except:
        qa_similarity = 0.0

    ref_similarity = 0.0
    if reference_answers and len(reference_answers) > 0:
        try:
            answer_lower = answer.lower()
            question_lower = question.lower()

            question_words = set(question_lower.split())
            answer_words = set(answer_lower.split())
 
            common_words = question_words.intersection(answer_words)
            keyword_score = len(common_words) / max(len(question_words), 1)

            ref_scores = []
            for ref in reference_answers:
                ref_lower = ref.lower()
                ref_words = set(ref_lower.split())
                answer_ref_common = answer_words.intersection(ref_words)
                ref_score = len(answer_ref_common) / max(len(ref_words), 1)
                ref_scores.append(ref_score)
            
            ref_similarity = max(ref_scores) if ref_scores else 0.0
          
            ref_similarity = (keyword_score + ref_similarity) / 2
        except:
            ref_similarity = 0.0

    if not reference_answers:
        return float(qa_similarity)

    final_score = (qa_similarity * 0.6) + (ref_similarity * 0.4)
    return float(np.clip(final_score, 0.0, 1.0))