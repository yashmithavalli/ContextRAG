from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the model (single instance loaded on module import)
model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_text(text: str) -> np.ndarray:
    """
    Encode a single text string into a numpy vector.
    """
    return model.encode(text)

def encode_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of text strings into a numpy matrix (n_samples, n_features).
    """
    return model.encode(texts)

if __name__ == "__main__":
    print("Initializing embedder.py test...")
    sample_texts = [
        "This is the first test sentence.",
        "Here is another sentence for the embedding test."
    ]
    
    embeddings = encode_texts(sample_texts)
    print(f"Encoded {len(sample_texts)} texts.")
    print(f"Output shape: {embeddings.shape} (Outputs an array of {embeddings.shape[0]} vectors, each with dimension {embeddings.shape[1]})")
    print(f"Output type: {type(embeddings)}")
    
    # Let's test single text
    single_embedding = encode_text(sample_texts[0])
    print(f"Single text shape: {single_embedding.shape}")
