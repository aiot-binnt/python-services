from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import os
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
except Exception as e:
    raise Exception(f"Failed to configure Gemini API: {e}")

class VectorData(BaseModel):
    id: str
    values: List[float]
    metadata: Dict

class SearchRequest(BaseModel):
    query: str
    language: str
    top_k: int = 5 
    threshold: float = 0.6

class VectorSearchRequest(BaseModel):
    vector: List[float]
    language: str
    top_k: int = 5
    threshold: float = 0.6

class DeleteRequest(BaseModel):
    ids: List[str]

class EmbeddingRequest(BaseModel):
    text: str

class FAISSService:
    # def __init__(self, index_path="F:/ThucTap/chatbot-agent/python_services/faiss_index"):
    #     self.dimension = 768
    #     self.index_path = index_path
    #     self.metadata_path = index_path + "_metadata.json"
    def __init__(self, index_path=None):
        self.dimension = 768
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        self.index_path = index_path
        self.metadata_path = index_path + "_metadata.json"
        self.documents = []
        self.id_to_index = {}
        
        logger.info(f"Initializing FAISS with index_path: {index_path}")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self._load_index_and_metadata()

    def _load_index_and_metadata(self):
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
                
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        self.documents = json.load(f)
                    logger.info(f"Loaded {len(self.documents)} documents metadata")
                    
                    # Xây dựng ánh xạ id_to_index
                    self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
                    logger.info(f"Built id_to_index mapping: {self.id_to_index}")
                    
                    if self.index.ntotal != len(self.documents):
                        logger.warning(f"Index and metadata out of sync: {self.index.ntotal} vectors vs {len(self.documents)} metadata entries")
                        self._sync_index_and_metadata()
                else:
                    logger.info("No metadata file found, creating empty documents list")
                    self.documents = [{} for _ in range(self.index.ntotal)]
                    self.id_to_index = {f"content_{i}": i for i in range(self.index.ntotal)}
                    self._save_metadata()
            else:
                logger.info(f"No existing FAISS index found at {self.index_path}")
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                self.documents = []
                self.id_to_index = {}
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata: {e}")
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            self.documents = []
            self.id_to_index = {}

    def _sync_index_and_metadata(self):
        if self.index.ntotal < len(self.documents):
            self.documents = self.documents[:self.index.ntotal]
            self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
        else:
            self.documents.extend([{} for _ in range(self.index.ntotal - len(self.documents))])
            self.id_to_index = {f"content_{i}": i for i in range(self.index.ntotal)}
        self._save_metadata()

    def _save_metadata(self):
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        try:
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided for embedding")
            
            text = text.replace('\x00', '')
            text = ' '.join(text.split())
            
            result = genai.embed_content(model="models/text-embedding-004", content=text)
            embedding = result['embedding']
            
            if len(embedding) != self.dimension:
                raise ValueError(f"Invalid embedding dimension: expected {self.dimension}, got {len(embedding)}")
            
            logger.info(f"Generated embedding for text: {text[:50]}... (length: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

    def upsert_vectors(self, vectors: List[VectorData]) -> List[str]:
        try:
            logger.info(f"Received {len(vectors)} vectors for upsert")
            
            valid_vectors = []
            valid_metadatas = []
            valid_ids = []
            valid_id_nums = []
            
            for v in vectors:
                if len(v.values) == self.dimension:
                    valid_vectors.append(v.values)
                    valid_metadatas.append({**v.metadata, 'vector_id': v.id})
                    valid_ids.append(v.id)
                    try:
                        id_num = int(v.id.replace('content_', ''))
                        valid_id_nums.append(id_num)
                    except ValueError:
                        logger.warning(f"Invalid vector ID format: {v.id}, skipping")
                        continue
                else:
                    logger.warning(f"Skipping vector {v.id} with invalid dimension: {len(v.values)}")
            
            if not valid_vectors:
                raise ValueError("No valid vectors to upsert")
            
            embeddings = np.array(valid_vectors, dtype=np.float32)
            ids = np.array(valid_id_nums, dtype=np.int64)
            logger.info(f"Valid embeddings shape: {embeddings.shape}, IDs: {ids}")
            
            self.index.add_with_ids(embeddings, ids)
            self.documents.extend(valid_metadatas)
            self.id_to_index.update({v.id: len(self.documents) - len(valid_ids) + i for i, v in enumerate(vectors) if v.id in valid_ids})
            
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            
            logger.info(f"Vectors upserted successfully. Total vectors: {self.index.ntotal}, id_to_index: {self.id_to_index}")
            return valid_ids
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to upsert vectors: {str(e)}")

    def search(self, query: str, language: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict]:
        try:
            logger.info(f"[Search] Query='{query[:30]}...', Language={language}, TopK={top_k}, Threshold={threshold}")
            
            if self.index.ntotal == 0:
                logger.warning("[Search] Empty FAISS index")
                return []
            
            query_embedding = self.generate_embedding(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            logger.info(f"Index has {self.index.ntotal} vectors")
            
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            logger.info(f"Raw search results - distances: {distances[0].tolist()}, indices: {indices[0].tolist()}")
            
            results = []
            debug_info = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                debug_entry = {'index': int(idx), 'distance': float(dist)}
                
                vector_id = f"content_{idx}"
                if vector_id not in self.id_to_index:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Vector ID {vector_id} not in id_to_index'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Search] Vector ID {vector_id} not found in id_to_index, skipping")
                    continue
                
                doc_idx = self.id_to_index[vector_id]
                if doc_idx >= len(self.documents):
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Invalid document index {doc_idx} for vector_id {vector_id}'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Search] Invalid document index {doc_idx} for vector_id {vector_id}, skipping")
                    continue
                
                score = max(0, 1 - (dist / 2))
                debug_entry['score'] = score
                
                if score < threshold:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Low score: {score:.4f} < {threshold}'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Search] Skipped vector_id {vector_id} due to low score: {score:.4f}")
                    continue
                
                doc_metadata = self.documents[doc_idx]
                doc_language = doc_metadata.get('language', 'vi')
                language_bonus = 0.2 if doc_language == language else 0
                final_score = score + language_bonus
                
                query_words = set(query.lower().split())
                content_words = set(doc_metadata.get('content', '').lower().split())
                common_words = query_words.intersection(content_words)
                keyword_score = len(common_words) / max(len(query_words), 1)
                
                debug_entry['keyword_score'] = keyword_score
                debug_entry['common_words'] = list(common_words)
                debug_entry['content'] = doc_metadata.get('content', '')[:50] + '...'
                debug_entry['language_match'] = doc_language == language
                debug_entry['vector_id'] = vector_id
                
                if keyword_score < 0.1:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Low keyword overlap: {keyword_score:.2f} < 0.1'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Skip] Low keyword overlap for '{doc_metadata.get('content', '')[:30]}...' (Keyword score: {keyword_score:.2f})")
                    continue
                
                debug_entry['status'] = 'included'
                debug_entry['final_score'] = final_score
                debug_info.append(debug_entry)
                
                logger.debug(f"[Result] vector_id={vector_id}, score={score:.4f}, final_score={final_score:.4f}, lang_match={doc_language == language}, keyword_score={keyword_score:.2f}")
                
                result = {
                    'vector_id': vector_id,
                    'content': doc_metadata.get('content', ''),
                    'score': final_score,
                    'metadata': doc_metadata,
                    'original_score': score,
                    'language_match': doc_language == language,
                    'keyword_score': keyword_score
                }
                results.append(result)
            
            logger.info(f"[Search] Debug info: {json.dumps(debug_info, ensure_ascii=False)}")
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"[Search] Returned {len(results)} results after filtering")
            return results[:top_k]
            
        except Exception as e:
            logger.exception(f"[Search] Error: {e}")
            return []

    def vector_search(self, vector: List[float], language: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict]:
        try:
            logger.info(f"Vector search with vector length: {len(vector)} (language: {language}, threshold: {threshold})")
            
            if self.index.ntotal == 0:
                logger.info("No vectors in index")
                return []
            
            if len(vector) != self.dimension:
                raise ValueError(f"Invalid vector dimension: expected {self.dimension}, got {len(vector)}")
            
            query_embedding = np.array([vector], dtype=np.float32)
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            logger.info(f"Raw vector search results - distances: {distances[0].tolist()}, indices: {indices[0].tolist()}")
            
            results = []
            debug_info = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                debug_entry = {'index': int(idx), 'distance': float(dist)}
                
                vector_id = f"content_{idx}"
                if vector_id not in self.id_to_index:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Vector ID {vector_id} not in id_to_index'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Vector Search] Vector ID {vector_id} not found in id_to_index, skipping")
                    continue
                
                doc_idx = self.id_to_index[vector_id]
                if doc_idx >= len(self.documents):
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Invalid document index {doc_idx} for vector_id {vector_id}'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Vector Search] Invalid document index {doc_idx} for vector_id {vector_id}, skipping")
                    continue
                
                score = max(0, 1 - (dist / 2))
                debug_entry['score'] = score
                
                if score < threshold:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Low score: {score:.4f} < {threshold}'
                    debug_info.append(debug_entry)
                    logger.debug(f"[Vector Search] Skipped vector_id {vector_id} due to low score: {score:.4f}")
                    continue
                
                doc_metadata = self.documents[doc_idx]
                doc_language = doc_metadata.get('language', 'vi')
                language_bonus = 0.2 if doc_language == language else 0
                final_score = score + language_bonus
                
                debug_entry['status'] = 'included'
                debug_entry['final_score'] = final_score
                debug_entry['content'] = doc_metadata.get('content', '')[:50] + '...'
                debug_entry['language_match'] = doc_language == language
                debug_entry['vector_id'] = vector_id
                debug_info.append(debug_entry)
                
                result = {
                    'vector_id': vector_id,
                    'content': doc_metadata.get('content', ''),
                    'score': final_score,
                    'metadata': doc_metadata,
                    'language_match': doc_language == language
                }
                results.append(result)
            
            logger.info(f"[Vector Search] Debug info: {json.dumps(debug_info, ensure_ascii=False)}")
            results.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Vector search returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching vectors by vector: {e}")
            return []

    def debug_search(self, query: str, language: str, top_k: int = 5) -> List[Dict]:
        try:
            logger.info(f"[Debug Search] Query='{query[:30]}...', Language={language}, TopK={top_k}")
            
            if self.index.ntotal == 0:
                logger.warning("[Debug Search] Empty FAISS index")
                return []
            
            query_embedding = self.generate_embedding(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            logger.info(f"Index has {self.index.ntotal} vectors")
            
            distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            logger.info(f"Raw debug search results - distances: {distances[0].tolist()}, indices: {indices[0].tolist()}")
            
            debug_info = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                debug_entry = {'index': int(idx), 'distance': float(dist)}
                
                vector_id = f"content_{idx}"
                if vector_id not in self.id_to_index:
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Vector ID {vector_id} not in id_to_index'
                    debug_info.append(debug_entry)
                    continue
                
                doc_idx = self.id_to_index[vector_id]
                if doc_idx >= len(self.documents):
                    debug_entry['status'] = 'skipped'
                    debug_entry['reason'] = f'Invalid document index {doc_idx} for vector_id {vector_id}'
                    debug_info.append(debug_entry)
                    continue
                
                score = max(0, 1 - (dist / 2))
                debug_entry['score'] = score
                
                doc_metadata = self.documents[doc_idx]
                doc_language = doc_metadata.get('language', 'vi')
                language_bonus = 0.2 if doc_language == language else 0
                final_score = score + language_bonus
                
                query_words = set(query.lower().split())
                content_words = set(doc_metadata.get('content', '').lower().split())
                common_words = query_words.intersection(content_words)
                keyword_score = len(common_words) / max(len(query_words), 1)
                
                debug_entry['keyword_score'] = keyword_score
                debug_entry['common_words'] = list(common_words)
                debug_entry['content'] = doc_metadata.get('content', '')[:50] + '...'
                debug_entry['language_match'] = doc_language == language
                debug_entry['final_score'] = final_score
                debug_entry['vector_id'] = vector_id
                debug_entry['metadata'] = doc_metadata
                debug_info.append(debug_entry)
            
            logger.info(f"[Debug Search] Debug info: {json.dumps(debug_info, ensure_ascii=False)}")
            return debug_info
            
        except Exception as e:
            logger.exception(f"[Debug Search] Error: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> List[bool]:
        try:
            logger.info(f"Deleting vectors: {ids}")
            
            id_nums = []
            valid_ids = []
            for vector_id in ids:
                try:
                    id_num = int(vector_id.replace('content_', ''))
                    id_nums.append(id_num)
                    valid_ids.append(vector_id)
                except ValueError:
                    logger.warning(f"Invalid vector ID format: {vector_id}, skipping")
                    continue
            
            if not id_nums:
                logger.info("No valid IDs to delete")
                return [False] * len(ids)
            
            id_selector = np.array(id_nums, dtype=np.int64)
            removed = self.index.remove_ids(id_selector)
            logger.info(f"Removed {removed} vectors from index")
            
            new_documents = []
            results = [False] * len(ids)
            for i, doc in enumerate(self.documents):
                vector_id = doc.get('vector_id', f"content_{i}")
                if vector_id not in valid_ids:
                    new_documents.append(doc)
                else:
                    idx = valid_ids.index(vector_id)
                    results[idx] = True
                    logger.debug(f"Marked vector {vector_id} for deletion")
            
            self.documents = new_documents
            self.id_to_index = {doc.get('vector_id', f"content_{i}"): i for i, doc in enumerate(self.documents)}
            
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
            
            logger.info(f"Vectors deleted successfully. Remaining: {self.index.ntotal}, id_to_index: {self.id_to_index}")
            return results
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {str(e)}")

    def clear_index(self):
        try:
            logger.info("Clearing FAISS index and metadata")
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            self.documents = []
            self.id_to_index = {}
            
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                logger.info(f"Deleted FAISS index file: {self.index_path}")
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
                logger.info(f"Deleted metadata file: {self.metadata_path}")
                
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")

# FastAPI app
app = FastAPI()
faiss_service = FAISSService()

@app.post("/generate_embedding")
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = faiss_service.generate_embedding(request.text)
        return {"success": True, "embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

@app.post("/upsert")
async def upsert_vectors(vectors: List[VectorData]):
    ids = faiss_service.upsert_vectors(vectors)
    if not ids:
        raise HTTPException(status_code=500, detail="Failed to upsert vectors")
    return {"success": True, "ids": ids}

@app.post("/delete")
async def delete_vectors(request: DeleteRequest):
    results = faiss_service.delete_vectors(request.ids)
    if not results:
        raise HTTPException(status_code=500, detail="No vectors processed")
    return {"success": True, "results": results}

@app.post("/search")
async def search_vectors(request: SearchRequest):
    results = faiss_service.search(request.query, request.language, request.top_k, request.threshold)
    return {"success": True, "results": results}

@app.post("/vector_search")
async def vector_search_vectors(request: VectorSearchRequest):
    results = faiss_service.vector_search(request.vector, request.language, request.top_k, request.threshold)
    return {"success": True, "results": results}

@app.post("/debug_search")
async def debug_search_vectors(request: SearchRequest):
    results = faiss_service.debug_search(request.query, request.language, request.top_k)
    return {"success": True, "results": results}

@app.post("/clear_index")
async def clear_index():
    try:
        faiss_service.clear_index()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_vectors": faiss_service.index.ntotal,
        "total_metadata": len(faiss_service.documents),
        "id_to_index": faiss_service.id_to_index
    }

@app.get("/list_vectors")
async def list_vectors():
    try:
        vectors = [
            {"vector_id": doc.get("vector_id", f"content_{i}"), "metadata": doc}
            for i, doc in enumerate(faiss_service.documents)
        ]
        return {"success": True, "vectors": vectors, "id_to_index": faiss_service.id_to_index}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list vectors: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)