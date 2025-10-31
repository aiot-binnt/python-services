from fastapi import FastAPI, HTTPException, Path
import faiss
import numpy as np
import os
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
from openai import OpenAI, APIError, BadRequestError, RateLimitError
from dotenv import load_dotenv
import math
import re
import shutil

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
    max_chunk_length: int = 500
    min_chunk_length: int = 200
    overlap_ratio: float = 0.1


# ==================== EMBEDDING FUNCTION (GLOBAL, NO BOT_ID NEEDED) ====================
def generate_embedding(text: str) -> List[float]:
    try:
        if not text or not text.strip():
            raise ValueError("Empty text")


        text = text.encode('utf-8', errors='ignore').decode('utf-8')  
        if len(text) > 8000:
            text = text[:8000] + " [truncated]"
            logger.warning("Text truncated to 8000 chars for embedding")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,  
            input=text
        )
        embedding = response.data[0].embedding

        logger.info(f"✓ Generated embedding for text preview: {text[:50]}...")
        return embedding

    except BadRequestError as e:  
        logger.error(f"❌ BadRequest in embedding: {str(e)} | Text preview: {text[:100]}...")
        raise HTTPException(status_code=400, detail=f"Invalid text for embedding: {str(e)}")
    except RateLimitError as e: 
        logger.error(f"❌ Rate limit in embedding: {str(e)}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    except APIError as e:  
        logger.error(f"❌ OpenAI API error in embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Unexpected error in embedding: {type(e).__name__}: {str(e)} | Text preview: {text[:100]}...")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

# ==================== SMART CHUNKER ====================
class SmartChunker:
    """Improved text chunking with SENTENCE-BASED overlap (no more cut-off words!)"""
    
    LANGUAGE_RULES = {
        'vi': {
            'sentence_delimiters': r'(?<=[.!?])\s+',
            'paragraph_delimiters': r'\n{2,}',
            'cjk_delims': None
        },
        'en': {
            'sentence_delimiters': r'(?<=[.!?])\s+',
            'paragraph_delimiters': r'\n{2,}',
            'cjk_delims': None
        },
        'ja': {
            'sentence_delimiters': r'(?<=[。！？])\s*',
            'paragraph_delimiters': r'\n{1,}',
            'cjk_delims': r'[ \t\n\r\0\x0B。！？、；：（）「」『』]'
        },
        'zh': {
            'sentence_delimiters': r'(?<=[。！？])\s*',
            'paragraph_delimiters': r'\n{1,}',
            'cjk_delims': r'[ \t\n\r\0\x0B。！？、；：（）「」『』]'
        },
        'ko': {
            'sentence_delimiters': r'(?<=[.!?])\s*',
            'paragraph_delimiters': r'\n{1,}',
            'cjk_delims': r'[ \t\n\r\0\x0B。！？、；：（）「」『』]'
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
        """Enhanced: Detect multi-line tables/lists/headers"""
        text = text.strip()
        if re.match(r'^\s*[-•○●]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return True
        if re.search(r'\|\s*\S', text) or text.count('\t') > 1:
            return True
        if re.match(r'^#+\s+', text) or re.match(r'^[A-Z][A-Z\s]+:?$', text):
            return True
        return False
    
    @classmethod
    def _join_sentences(cls, sentences: List[str], is_char_based: bool) -> str:
        """Join sentences with appropriate separator"""
        if not sentences:
            return ''
        separator = '' if is_char_based else ' '
        return separator.join(sentences)
    
    @classmethod
    def _build_overlap(cls, prev_sentences: List[str], max_length: int, overlap_ratio: float, 
                      language: str, is_char_based: bool) -> str:
        """
        Build overlap from COMPLETE SENTENCES (last N sentences that fit in overlap window)
        
        KEY IMPROVEMENT: Instead of cutting text at overlap_length characters,
        we take the last N complete sentences whose total length <= overlap_ratio * max_length
        
        Example:
        - prev_sentences = ["Sentence 1.", "Sentence 2 is long.", "Sentence 3."]
        - overlap_ratio = 0.2, max_length = 500 => target = 100 words
        - Result: Take "Sentence 3." (if < 100 words) or "Sentence 2 is long. Sentence 3."
        """
        if not prev_sentences:
            return ''
        
        target_overlap_length = int(max_length * overlap_ratio)
        overlap_sentences = []
        overlap_length = 0
        
        # Iterate from end to beginning
        for sent in reversed(prev_sentences):
            sent_len = cls.calculate_length(sent, language)
            
            # If adding this sentence still fits => add it
            if overlap_length + sent_len <= target_overlap_length:
                overlap_sentences.insert(0, sent)  # Insert at beginning to preserve order
                overlap_length += sent_len
            else:
                # If no sentences yet and this one is smaller than target => still take it
                if not overlap_sentences and sent_len < target_overlap_length:
                    overlap_sentences.insert(0, sent)
                break
        
        return cls._join_sentences(overlap_sentences, is_char_based)
    
    @classmethod
    def chunk_text(cls, text: str, language: str, max_length: int = 500, min_length: int = 20, overlap_ratio: float = 0.1) -> List[str]:
        if language == 'mixed':
            language = 'vi'
        
        rules = cls.LANGUAGE_RULES.get(language, cls.LANGUAGE_RULES['en'])
        sentence_delims = rules['sentence_delimiters']
        paragraph_delims = rules['paragraph_delimiters']
        cjk_delims = rules['cjk_delims']
        is_char_based = cls.is_char_based(language)
        
        paragraphs = re.split(paragraph_delims, text.strip(), flags=re.UNICODE | re.MULTILINE)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        prev_sentences = []  # KEY CHANGE: Store list of sentences instead of text chunk
        overlap_count = 0
        
        for para in paragraphs:
            if not para:
                continue
            
            # === HANDLE SPECIAL STRUCTURES ===
            if cls.is_special_structure(para):
                para_len = cls.calculate_length(para, language)
                if para_len >= min_length:
                    # Add sentence-based overlap
                    if prev_sentences:
                        overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                        if overlap_text:
                            para = overlap_text + '\n' + para
                            overlap_count += 1
                    chunks.append(para)
                    prev_sentences = [para]  # Treat whole structure as one sentence
                continue
            
            # === SMALL PARAGRAPH (fits in one chunk) ===
            para_len = cls.calculate_length(para, language)
            if para_len <= max_length:
                if para_len >= min_length:
                    # Add sentence-based overlap
                    if prev_sentences:
                        overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                        if overlap_text:
                            para = overlap_text + (' ' if not is_char_based else '') + para
                            overlap_count += 1
                    chunks.append(para)
                    # Split paragraph into sentences for next overlap
                    para_sentences = re.split(sentence_delims, para, flags=re.UNICODE)
                    prev_sentences = [s.strip() for s in para_sentences if s.strip()]
                continue
            
            # === LARGE PARAGRAPH => Split into multiple chunks ===
            sentences = re.split(sentence_delims, para, flags=re.UNICODE)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            current_chunk_sentences = []  # List of sentences for current chunk
            current_len = 0
            
            for sent in sentences:
                sent_len = cls.calculate_length(sent, language)
                
                # If adding this sentence still fits
                if current_len + sent_len <= max_length:
                    current_chunk_sentences.append(sent)
                    current_len += sent_len
                else:
                    # Chunk is full => create new chunk
                    if current_chunk_sentences and current_len >= min_length:
                        chunk_text = cls._join_sentences(current_chunk_sentences, is_char_based)
                        
                        # Add sentence-based overlap from previous chunk
                        if prev_sentences:
                            overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                            if overlap_text:
                                chunk_text = overlap_text + (' ' if not is_char_based else '') + chunk_text
                                overlap_count += 1
                        
                        chunks.append(chunk_text)
                        prev_sentences = current_chunk_sentences[:]  # Save sentences for next overlap
                    
                    # Start new chunk with current sentence
                    current_chunk_sentences = [sent]
                    current_len = sent_len
            
            # Handle last chunk of paragraph
            if current_chunk_sentences and current_len >= min_length:
                chunk_text = cls._join_sentences(current_chunk_sentences, is_char_based)
                
                if prev_sentences:
                    overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                    if overlap_text:
                        chunk_text = overlap_text + (' ' if not is_char_based else '') + chunk_text
                        overlap_count += 1
                
                chunks.append(chunk_text)
                prev_sentences = current_chunk_sentences[:]
            
            # === FALLBACK FOR CJK (if sentence splitting fails) ===
            if is_char_based and cjk_delims and len(sentences) <= 1 and para_len > max_length:
                rough_words = re.split(cjk_delims, para)
                current_chunk_words = []
                current_len = 0
                
                for word in rough_words:
                    if word.strip():
                        word_len = len(word.strip())
                        if current_len + word_len <= max_length:
                            current_chunk_words.append(word.strip())
                            current_len += word_len
                        else:
                            if current_chunk_words and current_len >= min_length:
                                chunk_text = ' '.join(current_chunk_words)
                                
                                if prev_sentences:
                                    overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                                    if overlap_text:
                                        chunk_text = overlap_text + chunk_text
                                        overlap_count += 1
                                
                                chunks.append(chunk_text)
                                prev_sentences = [chunk_text]
                            
                            current_chunk_words = [word.strip()]
                            current_len = word_len
                
                if current_chunk_words and current_len >= min_length:
                    chunk_text = ' '.join(current_chunk_words)
                    
                    if prev_sentences:
                        overlap_text = cls._build_overlap(prev_sentences, max_length, overlap_ratio, language, is_char_based)
                        if overlap_text:
                            chunk_text = overlap_text + chunk_text
                            overlap_count += 1
                    
                    chunks.append(chunk_text)
                    prev_sentences = [chunk_text]
        
        # Cleanup
        chunks = [c.strip() for c in chunks if cls.calculate_length(c, language) >= min_length]
        chunks = list(dict.fromkeys(chunks))  # Dedup order-preserving
        
        avg_size = sum(cls.calculate_length(c, language) for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"✓ Chunked: {len(chunks)} chunks (lang: {language}, avg: {avg_size:.1f}, overlaps: {overlap_count})")
        return chunks
    

# ==================== FAISS SERVICE ====================
class FAISSService:
    def __init__(self, bot_id: str, index_path: Optional[str] = None):
        self.dimension = 1536
        self.bot_id = bot_id 
        if index_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "faiss_index")
            bot_dir = os.path.join(base_path, bot_id) 
            index_path = os.path.join(bot_dir, "index")
            self.metadata_path = os.path.join(bot_dir, "metadata.json")  
        self.index_path = index_path
        self.documents = []
        self.id_to_index = {}
        logger.info(f"🔧 Initializing FAISS for bot_id: {bot_id} at {index_path}")
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
                    logger.info(f"✓ Loaded {len(self.documents)} documents metadata for bot {self.bot_id}")
                    
                    self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
                    
                    if self.index.ntotal != len(self.documents):
                        logger.warning(f"⚠ Index sync issue for bot {self.bot_id}: {self.index.ntotal} vs {len(self.documents)}")
                        self._sync_index_and_metadata()
                else:
                    self.documents = [{} for _ in range(self.index.ntotal)]
                    self.id_to_index = {f"content_{i}": i for i in range(self.index.ntotal)}
                    self._save_metadata()
            else:
                logger.info(f"✓ Creating new FAISS index for bot {self.bot_id} with Inner Product (cosine similarity)")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.documents = []
                self.id_to_index = {}
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS for bot {self.bot_id}: {e}")
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
            logger.info(f"✓ Saved metadata to {self.metadata_path} for bot {self.bot_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata for bot {self.bot_id}: {e}")

    def upsert_vectors(self, vectors: List[VectorData]) -> List[str]:
        try:
            logger.info(f"📥 Upserting {len(vectors)} vectors for bot {self.bot_id}")
            
            valid_vectors = []
            valid_metadatas = []
            valid_ids = []
            
            for v in vectors:
                if len(v.values) == self.dimension:
                    # Normalize for cosine similarity
                    normalized = np.array([v.values], dtype=np.float32)
                    faiss.normalize_L2(normalized)
                    valid_vectors.append(normalized[0])
                    # Enforce bot_id in metadata
                    metadata = {**v.metadata}
                    if 'bot_id' not in metadata:
                        metadata['bot_id'] = self.bot_id
                    valid_metadatas.append({**metadata, 'vector_id': v.id})
                    valid_ids.append(v.id)
                else:
                    logger.warning(f"⚠ Skip {v.id} for bot {self.bot_id}: dimension {len(v.values)} (expected {self.dimension})")
            
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
            
            logger.info(f"✓ Upserted for bot {self.bot_id}. Total vectors: {self.index.ntotal}")
            return valid_ids
            
        except Exception as e:
            logger.error(f"❌ Upsert error for bot {self.bot_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")

    def hybrid_search(self, query: str, language: str, top_k: int = 5, threshold: float = MIN_SCORE_THRESHOLD) -> List[Dict]:
        """
        Hybrid search: Semantic + Keyword + Re-ranking (if available)
        """
        try:
            logger.info(f"🔍 [Hybrid] Query='{query[:50]}...', Lang={language}, K={top_k}, Threshold={threshold}, Bot={self.bot_id}")
            
            if self.index.ntotal == 0:
                logger.warning(f"⚠ Empty index for bot {self.bot_id}")
                return []
            
            # Step 1: Generate embedding (use global function)
            query_embedding = generate_embedding(query)  
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
                if doc.get('bot_id') != self.bot_id:
                    continue
                
                content = doc.get('content', '')
                doc_languages = doc.get('languages', [language])  
                
                # Semantic score
                semantic_score = float(dist)
                
                # Keyword score (simple BM25-like)
                content_terms = set(content.lower().split())
                common = query_terms.intersection(content_terms)
                keyword_score = len(common) / max(len(query_terms), 1)
                
                lang_bonus = 0.15 if language in doc_languages else 0.05  
                
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
                        'language_match': language in doc_languages
                    })
            
            # Step 4: Re-ranking with cross-encoder (if available)
            if cross_encoder and len(results) > 1:
                try:
                    pairs = [[query, r['content'][:512]] for r in results]  
                    rerank_scores = cross_encoder.predict(pairs)
                    
                    for i, score in enumerate(rerank_scores):
                        results[i]['rerank_score'] = float(score)
                        results[i]['final_score'] = (0.5 * results[i]['score']) + (0.4 * float(score))
                    
                    results.sort(key=lambda x: x.get('final_score', x['score']), reverse=True)
                    logger.info(f"✓ Re-ranked {len(results)} results for bot {self.bot_id}")
                except Exception as e:
                    logger.warning(f"⚠ Re-ranking failed for bot {self.bot_id}: {e}, using hybrid scores")
                    results.sort(key=lambda x: x['score'], reverse=True)
            else:
                results.sort(key=lambda x: x['score'], reverse=True)
            
            final_results = results[:top_k]
            logger.info(f"✓ [Hybrid] Returned {len(final_results)} results for bot {self.bot_id}")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error for bot {self.bot_id}: {e}")
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
                
                doc = self.documents[idx]
                if doc.get('bot_id') != self.bot_id:
                    continue
                
                score = float(dist)
                if score < threshold:
                    continue
                
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
            logger.info(f"✓ [Vector] Returned {len(results)} results for bot {self.bot_id}")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Vector search error for bot {self.bot_id}: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> List[bool]:
        """Delete vectors (rebuild approach with regeneration fallback)"""
        try:
            logger.info(f"🗑️ Deleting {len(ids)} vectors for bot {self.bot_id}")
            
            delete_mask = [False] * len(self.documents)
            for i, doc in enumerate(self.documents):
                vid = doc.get('vector_id', f"content_{i}")
                if vid in ids:
                    delete_mask[i] = True
            
            # Collect kept docs and regenerate vectors for them
            kept_docs = [doc for i, doc in enumerate(self.documents) if not delete_mask[i]]
            results = [vid in ids for vid in [doc.get('vector_id', f"content_{i}") for i, doc in enumerate(self.documents)]]
            
            if len(kept_docs) == 0:
                # Clear all for this bot
                self.clear_index()
                logger.info(f"✓ Cleared entire index for bot {self.bot_id}")
                return results[:len(ids)]  # Partial match
            
            new_embeddings = []
            for doc in kept_docs:
                try:
                    content = doc.get('content', '')
                    if content:
                        emb = generate_embedding(content) 
                        new_embeddings.append(emb)
                    else:
                        logger.warning(f"⚠ Skip regen for doc without content in bot {self.bot_id}: {doc.get('vector_id')}")
                        new_embeddings.append(np.zeros(self.dimension, dtype=np.float32)) 
                except Exception as e:
                    logger.warning(f"⚠ Regen failed for {doc.get('vector_id')} in bot {self.bot_id}: {e}")
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
            
            logger.info(f"✓ Deleted for bot {self.bot_id}. Remaining: {len(self.documents)} vectors")
            return results
            
        except Exception as e:
            logger.error(f"❌ Delete error for bot {self.bot_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    def clear_index(self) -> bool:
        try:
            bot_dir = os.path.dirname(self.index_path)
            if os.path.exists(bot_dir):
                shutil.rmtree(bot_dir)
                logger.info(f"✓ Deleted bot folder: {bot_dir}")
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.id_to_index = {}
            
            logger.info(f"✓ Index cleared completely for bot {self.bot_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Clear error for bot {self.bot_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

# ==================== FASTAPI APP ====================
app = FastAPI(title="Enhanced FAISS RAG Service", version="2.0")

@app.post("/generate_embedding")
async def generate_embedding_endpoint(request: EmbeddingRequest):
    try:
        embedding = generate_embedding(request.text)
        return {"success": True, "embedding": embedding}
    except HTTPException:
        raise  # Re-raise FastAPI exceptions
    except Exception as e:
        logger.error(f"❌ Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal embedding error")

@app.post("/upsert/{bot_id}")
async def upsert_vectors(bot_id: str, vectors: List[VectorData]): 
    try:
        faiss_service = FAISSService(bot_id)
        ids = faiss_service.upsert_vectors(vectors)
        return {"success": True, "ids": ids}
    except Exception as e:
        logger.error(f"❌ Upsert failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete/{bot_id}")
async def delete_vectors(bot_id: str, request: DeleteRequest):  
    try:
        faiss_service = FAISSService(bot_id)
        results = faiss_service.delete_vectors(request.ids)
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Delete failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/{bot_id}")
async def search_vectors(bot_id: str, request: SearchRequest): 
    try:
        faiss_service = FAISSService(bot_id)
        results = faiss_service.hybrid_search(
            request.query,
            request.language,
            request.top_k,
            request.threshold
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Search failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector_search/{bot_id}")
async def vector_search_vectors(bot_id: str, request: VectorSearchRequest):  
    try:
        faiss_service = FAISSService(bot_id)
        results = faiss_service.vector_search(
            request.vector,
            request.language,
            request.top_k,
            request.threshold
        )
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Vector search failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chunk")
async def chunk_text_endpoint(request: ChunkRequest):
    try:
        chunks = SmartChunker.chunk_text(
            request.text,
            request.language,
            request.max_chunk_length,
            request.min_chunk_length,
            request.overlap_ratio
        )
        logger.info(f"✓ Chunked via API: {len(chunks)} chunks for lang {request.language}")
        return {"success": True, "chunks": chunks}
    except Exception as e:
        logger.error(f"❌ Chunking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/clear_index/{bot_id}")
async def clear_index(bot_id: str): 
    try:
        faiss_service = FAISSService(bot_id)
        success = faiss_service.clear_index()
        return {"success": success}
    except Exception as e:
        logger.error(f"❌ Clear index failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/{bot_id}")
async def health_check(bot_id: str = Path(..., description="Bot ID")):
    try:
        faiss_service = FAISSService(bot_id)
        return {
            "status": "healthy",
            "bot_id": bot_id,
            "total_vectors": faiss_service.index.ntotal,
            "total_metadata": len(faiss_service.documents),
            "cross_encoder_available": cross_encoder is not None,
            "langdetect_available": LANGDETECT_AVAILABLE,
            "embedding_model": EMBEDDING_MODEL,
            "min_threshold": MIN_SCORE_THRESHOLD
        }
    except Exception as e:
        logger.error(f"❌ Health check failed for bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Enhanced FAISS RAG Service",
        "version": "2.0",
        "features": [
            "Hybrid search (semantic + keyword)",
            "Cross-encoder re-ranking" if cross_encoder else "Basic hybrid search",
            "Multi-language support (vi, en, ja, zh, ko)",
            "Configurable via .env (model, threshold)",
            "Per-bot isolation (faiss_index/{bot_id}/)"
        ]
    }