import numpy as np
from rank_bm25 import BM25Okapi

# Safely import and use NLTK tokenization
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import word_tokenize
except Exception:
    def word_tokenize(text):
        return text.split()

def tokenize(text: str) -> list[str]:
    """Lowercase and split text."""
    return [word.lower() for word in word_tokenize(text)]

class HybridSearcher:
    def __init__(self, faiss_index, chunks: list[str], doc_ids: list[str]):
        """
        Store FAISS index reference
        Build a BM25Okapi index from the raw chunk text list
        Store chunks and doc_ids in parallel arrays
        """
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.doc_ids = doc_ids
        
        # Build BM25 index
        tokenized_corpus = [tokenize(doc) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def semantic_search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """Embed query with all-MiniLM-L6-v2, query FAISS, normalize scores."""
        if self.faiss_index.ntotal == 0 or len(self.chunks) == 0:
            return []
            
        # Ensure 2D float32 array
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        # FAISS uses L2 distance (lower is better). Convert to similarity.
        raw_scores = []
        valid_indices = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx != -1 and idx < len(self.chunks):
                dist = float(distances[0][i])
                sim = 1.0 / (1.0 + dist)
                raw_scores.append(sim)
                valid_indices.append(idx)
        
        norm_scores = self._normalize_scores(raw_scores)
        
        results = []
        for rank, (idx, score) in enumerate(zip(valid_indices, norm_scores)):
            doc_id = self.doc_ids[idx] if self.doc_ids and idx < len(self.doc_ids) else str(idx)
            results.append({
                "chunk": self.chunks[idx],
                "doc_id": doc_id,
                "score": score,
                "rank": rank + 1
            })
            
        return results

    def keyword_search(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Tokenize query and score all chunks with BM25Okapi."""
        if not self.bm25 or len(self.chunks) == 0:
            return []
            
        tokenized_query = tokenize(query_text)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices efficiently
        top_n = np.argsort(doc_scores)[::-1][:top_k]
        
        # Only keep indices that actually have a positive hit
        raw_scores = [doc_scores[i] for i in top_n]
        valid_indices = list(top_n)
        
        norm_scores = self._normalize_scores(raw_scores)
        
        results = []
        for rank, (idx, score) in enumerate(zip(valid_indices, norm_scores)):
            doc_id = self.doc_ids[idx] if self.doc_ids and idx < len(self.doc_ids) else str(idx)
            results.append({
                "chunk": self.chunks[idx],
                "doc_id": doc_id,
                "score": score,
                "rank": rank + 1
            })
            
        return results

    def hybrid_search(self, query_text: str, query_vector: np.ndarray, top_k: int = 5, alpha: float = 0.7) -> list[dict]:
        """Combine semantic and keyword search scores."""
        sem_results = self.semantic_search(query_vector, top_k=20)
        key_results = self.keyword_search(query_text, top_k=20)
        
        combined_map = {}
        
        for res in sem_results:
            chunk = res["chunk"]
            combined_map[chunk] = {
                "chunk": chunk,
                "doc_id": res["doc_id"],
                "semantic": res["score"],
                "keyword": 0.0
            }
            
        for res in key_results:
            chunk = res["chunk"]
            if chunk in combined_map:
                combined_map[chunk]["keyword"] = res["score"]
            else:
                combined_map[chunk] = {
                    "chunk": chunk,
                    "doc_id": res["doc_id"],
                    "semantic": 0.0,
                    "keyword": res["score"]
                }
                
        final_list = []
        for data in combined_map.values():
            s = data["semantic"]
            k = data["keyword"]
            combined_score = (alpha * s) + ((1.0 - alpha) * k)
            
            final_list.append({
                "chunk": data["chunk"],
                "doc_id": data["doc_id"],
                "score": combined_score,
                "breakdown": {
                    "semantic": round(s, 4),
                    "keyword": round(k, 4),
                    "combined": round(combined_score, 4)
                }
            })
            
        # Sort ascending by rank, descending by score
        final_list.sort(key=lambda x: x["score"], reverse=True)
        return final_list[:top_k]
