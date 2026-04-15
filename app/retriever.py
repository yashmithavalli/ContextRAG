import os
import faiss
import numpy as np
import json
from app.search import HybridSearcher
from app.embedder import encode_text

class VectorStore:
    def __init__(self, index_path: str = "data/index.faiss", doc_path: str = "data/docs.json", dimension: int = 384):
        self.index_path = index_path
        self.doc_path = doc_path
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []  # List mapping index ID to document string
        self.searcher = None
        
        # Ensure data directory exists
        index_dir = os.path.dirname(self.index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
            
        doc_dir = os.path.dirname(self.doc_path)
        if doc_dir:
            os.makedirs(doc_dir, exist_ok=True)
        
    def add_vectors(self, vectors: np.ndarray, documents: list[str]):
        """Add vectors to the FAISS index and store corresponding documents."""
        if len(vectors) != len(documents):
            raise ValueError("Number of vectors must match number of documents.")
            
        vectors = np.asarray(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.documents.extend(documents)
        self._init_searcher()
        
    def _init_searcher(self):
        """Initialize the HybridSearcher for this index instance (handles per-user isolation implicitly)."""
        ids = [str(i) for i in range(len(self.documents))]
        self.searcher = HybridSearcher(faiss_index=self.index, chunks=self.documents, doc_ids=ids)

    def retrieve(self, query: str, mode: str = "hybrid", top_k: int = 5, alpha: float = 0.7) -> list[dict]:
        """Perform semantic, keyword, or hybrid retrieve based on the mode parameter."""
        if not self.searcher:
            return []
            
        if mode == "keyword":
            return self.searcher.keyword_search(query, top_k)
        
        query_vector = encode_text(query)
        if mode == "semantic":
            return self.searcher.semantic_search(query_vector, top_k)
            
        return self.searcher.hybrid_search(query, query_vector, top_k, alpha)

    def save(self):
        """Save the FAISS index and documents to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
    def load(self):
        """Load the FAISS index and documents from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            self._init_searcher()
        else:
            print("No existing index or documents found. Starting fresh.")
