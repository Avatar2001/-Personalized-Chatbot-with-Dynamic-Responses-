from typing import List
import re

def score_groundedness(answer: str, sources: List[str]) -> float:
    """
    Returns a score between 0 and 1 measuring how well the answer
    is grounded in the source documents. Checks for:
    1. Word overlap between answer and sources
    2. Phrase matching
    3. Sentence-level similarity
    """
    if not sources or not answer:
        return 0.0
    
    try:
        answer_lower = answer.lower()
        
        answer_sentences = [s.strip() for s in re.split(r'[.!?]', answer_lower) if s.strip()]
        
        if not answer_sentences:
            return 0.0
        
        combined_sources = " ".join(sources).lower()
        
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        answer_words = answer_words - stop_words
        
        if not answer_words:
            return 0.0
        
        source_words = set(re.findall(r'\b\w+\b', combined_sources))
        common_words = answer_words.intersection(source_words)
        word_overlap_score = len(common_words) / len(answer_words)
 
        ngram_scores = []
        for n in [3, 4, 5]:
            answer_ngrams = extract_ngrams(answer_lower, n)
            if answer_ngrams:
                matched = sum(1 for ngram in answer_ngrams if ngram in combined_sources)
                ngram_scores.append(matched / len(answer_ngrams))
        
        ngram_score = max(ngram_scores) if ngram_scores else 0.0

        grounded_sentences = 0
        for sentence in answer_sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence)) - stop_words
            if not sentence_words:
                continue
            
            matched_words = sentence_words.intersection(source_words)
            if len(matched_words) / len(sentence_words) >= 0.5:
                grounded_sentences += 1
        
        sentence_score = grounded_sentences / len(answer_sentences)

        final_score = (word_overlap_score * 0.3) + (ngram_score * 0.4) + (sentence_score * 0.3)
        
        return min(1.0, final_score)
        
    except Exception as e:
        print(f"Warning: Error in groundedness calculation: {e}")
        return 0.0


def extract_ngrams(text: str, n: int) -> List[str]:
    """
    Extract n-grams (sequences of n words) from text.
    """
    words = re.findall(r'\b\w+\b', text)
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]