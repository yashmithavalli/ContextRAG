FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for cache optimization
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time to prevent downloads at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy remaining source code
COPY . .

EXPOSE 8501

# Start the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
