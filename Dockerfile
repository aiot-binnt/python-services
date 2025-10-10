FROM python:3.10-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        faiss-cpu==1.7.4 \
        sentence-transformers==3.0.1 \
        numpy==1.26.4 \
        openai==2.2.0 \
        fastapi==0.115.0 \
        uvicorn==0.30.6 \
        python-dotenv==1.0.1 \
        packaging==25.0


COPY . .

EXPOSE 8000


CMD ["uvicorn", "faiss_service:app", "--host", "0.0.0.0", "--port", "8000"]
 