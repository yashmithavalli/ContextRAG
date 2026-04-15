import io
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from a .txt file."""
    return file_content.decode("utf-8", errors="ignore")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into chunks with specified size and overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_document(file_content: bytes, filename: str) -> list[str]:
    """Extract text and chunk based on file type."""
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_content)
    elif filename.lower().endswith(".txt"):
        text = extract_text_from_txt(file_content)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or TXT.")
    
    return chunk_text(text)

if __name__ == "__main__":
    # Quick test for chunker
    print("Running chunker.py test...")
    sample_text = "This is a sample text designed to test the chunking functionality. " * 30
    print(f"Sample text total length: {len(sample_text)} characters")
    
    chunks = chunk_text(sample_text, chunk_size=500, chunk_overlap=50)
    print(f"Total chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
        print(chunk)
        print()
