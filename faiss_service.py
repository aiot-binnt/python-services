from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import os
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
import math
import re

# Setting logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    raise Exception(f"Failed to configure OpenAI API: {e}")

# Try to load cross-encoder for re-ranking (optional)
try:
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("✓ Cross-encoder loaded successfully")
except ImportError:
    cross_encoder = None
    logger.warning("⚠ sentence-transformers not installed, re-ranking disabled")
except Exception as e:
    cross_encoder = None
    logger.warning(f"⚠ Failed to load cross-encoder: {e}")

# Try to load langdetect (optional)
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
    logger.info("✓ langdetect available")
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("⚠ langdetect not installed, using basic detection")

# Env vars for models/thresholds
EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
MIN_SCORE_THRESHOLD = float(os.getenv('MIN_SCORE_THRESHOLD', 0.5))
logger.info(f"✓ Config: Embedding model={EMBEDDING_MODEL}, Min threshold={MIN_SCORE_THRESHOLD}")

class VectorData(BaseModel):
    id: str
    values: List[float]
    metadata: Dict

class SearchRequest(BaseModel):
    query: str
    language: str
    top_k: int = 5
    threshold: float = MIN_SCORE_THRESHOLD   

class VectorSearchRequest(BaseModel):
    vector: List[float]
    language: str
    top_k: int = 5
    threshold: float = MIN_SCORE_THRESHOLD 

class DeleteRequest(BaseModel):
    ids: List[str]

class EmbeddingRequest(BaseModel):
    text: str

class ChunkRequest(BaseModel):
    text: str
    language: str
    max_chunk_length: int = 1000
    min_chunk_length: int = 20

# ==================== SMART CHUNKER ====================
class SmartChunker:
    """Improved text chunking với language-aware rules"""
    
    LANGUAGE_RULES = {
        'vi': {
            'sentence_delimiters': r'(?<=[.!?])\s+',
            'paragraph_delimiters': r'\n{2,}',
        },
        'en': {
            'sentence_delimiters': r'(?<=[.!?])\s+',
            'paragraph_delimiters': r'\n{2,}',
        },
        'ja': {
            'sentence_delimiters': r'(?<=[。！？])\s*',
            'paragraph_delimiters': r'\n{1,}',
        },
        'zh': {
            'sentence_delimiters': r'(?<=[。！？])\s*',
            'paragraph_delimiters': r'\n{1,}',
        },
        'ko': {
            'sentence_delimiters': r'(?<=[.!?])\s*',
            'paragraph_delimiters': r'\n{1,}',
        }
    }
    
    @classmethod
    def is_char_based(cls, language: str) -> bool:
        return language in ['ja', 'zh', 'ko']
    
    @classmethod
    def calculate_length(cls, text: str, language: str) -> int:
        if cls.is_char_based(language):
            return len(text)
        return len(text.split())
    
    @classmethod
    def is_special_structure(cls, text: str) -> bool:
        """Check if text is title, list, or table"""
        text = text.strip()
        # Check for lists
        if re.match(r'^\s*[-•○●]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return True
        # Check for tables
        if '|' in text or text.count('\t') > 2:
            return True
        # Check for headers
        if re.match(r'^#+\s+', text) or re.match(r'^[A-Z][A-Z\s]+:?$', text):
            return True
        return False
    
    @classmethod
    def chunk_text(cls, text: str, language: str, max_length: int = 1000, min_length: int = 20) -> List[str]:
        """
        Smart chunking với semantic awareness
        """
        rules = cls.LANGUAGE_RULES.get(language, cls.LANGUAGE_RULES['en'])
        
        # Split into paragraphs
        paragraphs = re.split(rules['paragraph_delimiters'], text.strip())
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Special structures as separate chunks
            if cls.is_special_structure(paragraph):
                if current_chunk and current_length >= min_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
                chunks.append(paragraph)
                continue
            
            para_length = cls.calculate_length(paragraph, language)
            
            # If paragraph fits
            if para_length <= max_length:
                if current_length + para_length <= max_length:
                    current_chunk += ("\n\n" if current_chunk else "") + paragraph
                    current_length += para_length
                else:
                    if current_chunk and current_length >= min_length:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                    current_length = para_length
            else:
                # Split into sentences
                if current_chunk and current_length >= min_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_length = 0
                
                sentences = re.split(rules['sentence_delimiters'], paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sent_length = cls.calculate_length(sentence, language)
                    
                    if current_length + sent_length <= max_length:
                        current_chunk += (" " if current_chunk else "") + sentence
                        current_length += sent_length
                    else:
                        if current_chunk and current_length >= min_length:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                        current_length = sent_length
        
        # Add remaining
        if current_chunk and current_length >= min_length:
            chunks.append(current_chunk.strip())
        
        logger.info(f"✓ Chunked text: {len(chunks)} chunks (language: {language})")
        return chunks

# ==================== FAISS SERVICE ====================
class FAISSService:
    def __init__(self, index_path: Optional[str] = None):
        self.dimension = 1536
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        self.index_path = index_path
        self.metadata_path = index_path + "_metadata.json"
        self.documents = []
        self.id_to_index = {}
        logger.info(f"🔧 Initializing FAISS with index_path: {index_path}")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self._load_index_and_metadata()

    def _load_index_and_metadata(self):
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"✓ Loaded FAISS index from {self.index_path}")
                
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        self.documents = json.load(f)
                    logger.info(f"✓ Loaded {len(self.documents)} documents metadata")
                    
                    self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
                    
                    if self.index.ntotal != len(self.documents):
                        logger.warning(f"⚠ Index sync issue: {self.index.ntotal} vs {len(self.documents)}")
                        self._sync_index_and_metadata()
                else:
                    self.documents = [{} for _ in range(self.index.ntotal)]
                    self.id_to_index = {f"content_{i}": i for i in range(self.index.ntotal)}
                    self._save_metadata()
            else:
                logger.info("✓ Creating new FAISS index with Inner Product (cosine similarity)")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.documents = []
                self.id_to_index = {}
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.id_to_index = {}

    def _sync_index_and_metadata(self):
        if self.index.ntotal < len(self.documents):
            self.documents = self.documents[:self.index.ntotal]
        else:
            self.documents.extend([{} for _ in range(self.index.ntotal - len(self.documents))])
        self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
        self._save_metadata()

    def _save_metadata(self):
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        try:
            if not text or not text.strip():
                raise ValueError("Empty text")

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,  
                input=text
            )
            embedding = response.data[0].embedding

            if len(embedding) != self.dimension:
                logger.warning(f"⚠ Dimension mismatch: {len(embedding)} (expected {self.dimension})")

            return embedding

        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    def upsert_vectors(self, vectors: List[VectorData]) -> List[str]:
        try:
            logger.info(f"📥 Upserting {len(vectors)} vectors")
            
            valid_vectors = []
            valid_metadatas = []
            valid_ids = []
            
            for v in vectors:
                if len(v.values) == self.dimension:
                    # Normalize for cosine similarity
                    normalized = np.array([v.values], dtype=np.float32)
                    faiss.normalize_L2(normalized)
                    valid_vectors.append(normalized[0])
                    valid_metadatas.append({**v.metadata, 'vector_id': v.id})
                    valid_ids.append(v.id)
                else:
                    logger.warning(f"⚠ Skip {v.id}: dimension {len(v.values)} (expected {self.dimension})")
            
            if not valid_vectors:
                raise ValueError("No valid vectors")
            
            embeddings = np.array(valid_vectors, dtype=np.float32)
            
            # Add to index
            self.index.add(embeddings)
            self.documents.extend(valid_metadatas)
            
            # Update mapping
            start_idx = len(self.documents) - len(valid_ids)
            for i, vid in enumerate(valid_ids):
                self.id_to_index[vid] = start_idx + i
            
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            
            logger.info(f"✓ Upserted. Total vectors: {self.index.ntotal}")
            return valid_ids
            
        except Exception as e:
            logger.error(f"❌ Upsert error: {e}")
            raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")

    def hybrid_search(self, query: str, language: str, top_k: int = 5, threshold: float = MIN_SCORE_THRESHOLD) -> List[Dict]:
        """
        Hybrid search: Semantic + Keyword + Re-ranking (if available)
        """
        try:
            logger.info(f"🔍 [Hybrid] Query='{query[:50]}...', Lang={language}, K={top_k}, Threshold={threshold}")
            
            if self.index.ntotal == 0:
                logger.warning("⚠ Empty index")
                return []
            
            # Step 1: Generate embedding
            query_embedding = self.generate_embedding(query)
            query_vec = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vec)
            
            # Step 2: Vector search (get more for re-ranking)
            retrieve_k = min(top_k * 3, self.index.ntotal)
            distances, indices = self.index.search(query_vec, retrieve_k)
            
            # Step 3: Hybrid scoring
            results = []
            query_terms = set(query.lower().split())
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                content = doc.get('content', '')
                doc_lang = doc.get('language', 'vi')
                
                # Semantic score
                semantic_score = float(dist)
                
                # Keyword score (simple BM25-like)
                content_terms = set(content.lower().split())
                common = query_terms.intersection(content_terms)
                keyword_score = len(common) / max(len(query_terms), 1)
                
                # Language bonus
                lang_bonus = 0.15 if doc_lang == language else 0
                
                # Hybrid score
                hybrid_score = (0.7 * semantic_score) + (0.3 * keyword_score) + lang_bonus
                
                if hybrid_score >= threshold:
                    results.append({
                        'vector_id': doc.get('vector_id', f"content_{idx}"),
                        'content': content,
                        'score': hybrid_score,
                        'semantic_score': semantic_score,
                        'keyword_score': keyword_score,
                        'metadata': doc,
                        'language_match': doc_lang == language
                    })
            
            # Step 4: Re-ranking with cross-encoder (if available)
            if cross_encoder and len(results) > 1:
                try:
                    pairs = [[query, r['content'][:512]] for r in results]  # Limit length to avoid OOM
                    rerank_scores = cross_encoder.predict(pairs)
                    
                    for i, score in enumerate(rerank_scores):
                        results[i]['rerank_score'] = float(score)
                        # Combine: 60% hybrid + 40% rerank
                        results[i]['final_score'] = (0.5 * results[i]['score']) + (0.4 * float(score))
                    
                    results.sort(key=lambda x: x.get('final_score', x['score']), reverse=True)
                    logger.info(f"✓ Re-ranked {len(results)} results")
                except Exception as e:
                    logger.warning(f"⚠ Re-ranking failed: {e}, using hybrid scores")
                    results.sort(key=lambda x: x['score'], reverse=True)
            else:
                results.sort(key=lambda x: x['score'], reverse=True)
            
            final_results = results[:top_k]
            logger.info(f"✓ [Hybrid] Returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return []

    def vector_search(self, vector: List[float], language: str, top_k: int = 5, threshold: float = MIN_SCORE_THRESHOLD) -> List[Dict]:
        """Direct vector search (legacy support)"""
        try:
            if self.index.ntotal == 0:
                return []
            
            if len(vector) != self.dimension:
                raise ValueError(f"Invalid dimension: {len(vector)} (expected {self.dimension})")
            
            query_vec = np.array([vector], dtype=np.float32)
            faiss.normalize_L2(query_vec)
            
            distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= len(self.documents):
                    continue
                
                score = float(dist)
                if score < threshold:
                    continue
                
                doc = self.documents[idx]
                doc_lang = doc.get('language', 'vi')
                lang_bonus = 0.2 if doc_lang == language else 0
                
                results.append({
                    'vector_id': doc.get('vector_id', f"content_{idx}"),
                    'content': doc.get('content', ''),
                    'score': score + lang_bonus,
                    'metadata': doc,
                    'language_match': doc_lang == language
                })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"✓ [Vector] Returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> List[bool]:
        """Delete vectors (rebuild approach with regeneration fallback)"""
        try:
            logger.info(f"🗑️ Deleting {len(ids)} vectors")
            
            delete_mask = [False] * len(self.documents)
            for i, doc in enumerate(self.documents):
                vid = doc.get('vector_id', f"content_{i}")
                if vid in ids:
                    delete_mask[i] = True
            
            # Collect kept docs and regenerate vectors for them
            kept_docs = [doc for i, doc in enumerate(self.documents) if not delete_mask[i]]
            results = [vid in ids for vid in [doc.get('vector_id', f"content_{i}") for i, doc in enumerate(self.documents)]]
            
            if len(kept_docs) == 0:
                # Clear all
                self.clear_index()
                logger.info("✓ Cleared entire index")
                return results[:len(ids)]  # Partial match
            
            # Regenerate embeddings for kept docs (expensive, but accurate)
            new_embeddings = []
            for doc in kept_docs:
                try:
                    content = doc.get('content', '')
                    if content:
                        emb = self.generate_embedding(content)
                        new_embeddings.append(emb)
                    else:
                        logger.warning(f"⚠ Skip regen for doc without content: {doc.get('vector_id')}")
                        new_embeddings.append(np.zeros(self.dimension, dtype=np.float32))  # Fallback zero vec
                except Exception as e:
                    logger.warning(f"⚠ Regen failed for {doc.get('vector_id')}: {e}")
                    new_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
            
            # Rebuild index
            embeddings_array = np.array(new_embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings_array)
            self.documents = kept_docs
            self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
            
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            
            logger.info(f"✓ Deleted. Remaining: {len(self.documents)} vectors")
            return results
            
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    def clear_index(self) -> bool:
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.id_to_index = {}
            
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            
            logger.info("✓ Index cleared completely")
            return True
        except Exception as e:
            logger.error(f"❌ Clear error: {e}")
            raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# ==================== FASTAPI APP ====================
app = FastAPI(title="Enhanced FAISS RAG Service", version="2.0")
faiss_service = FAISSService()
chunker = SmartChunker()

@app.post("/chunk_text")
async def chunk_text(request: ChunkRequest):
    """Smart text chunking"""
    try:
        chunks = chunker.chunk_text(
            request.text,
            request.language,
            request.max_chunk_length,
            request.min_chunk_length
        )
        return {
            "success": True,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"❌ Chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

@app.post("/generate_embedding")
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = faiss_service.generate_embedding(request.text)
        return {"success": True, "embedding": embedding}
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upsert")
async def upsert_vectors(vectors: List[VectorData]):
    try:
        ids = faiss_service.upsert_vectors(vectors)
        return {"success": True, "ids": ids}
    except Exception as e:
        logger.error(f"❌ Upsert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_vectors(request: DeleteRequest):
    try:
        results = faiss_service.delete_vectors(request.ids)
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_vectors(request: SearchRequest):
    """Hybrid search (recommended)"""
    try:
        results = faiss_service.hybrid_search(
            request.query,
            request.language,
            request.top_k,
            request.threshold
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector_search")
async def vector_search_vectors(request: VectorSearchRequest):
    """Legacy vector search"""
    try:
        results = faiss_service.vector_search(
            request.vector,
            request.language,
            request.top_k,
            request.threshold
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_index")
async def clear_index():
    try:
        faiss_service.clear_index()
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Clear index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_vectors": faiss_service.index.ntotal,
        "total_metadata": len(faiss_service.documents),
        "cross_encoder_available": cross_encoder is not None,
        "langdetect_available": LANGDETECT_AVAILABLE,
        "embedding_model": EMBEDDING_MODEL,
        "min_threshold": MIN_SCORE_THRESHOLD
    }

@app.get("/")
async def root():
    return {
        "service": "Enhanced FAISS RAG Service",
        "version": "2.0",
        "features": [
            "Smart chunking (language-aware)",
            "Hybrid search (semantic + keyword)",
            "Cross-encoder re-ranking" if cross_encoder else "Basic hybrid search",
            "Multi-language support (vi, en, ja, zh, ko)",
            "Configurable via .env (model, threshold)"
        ]
    }