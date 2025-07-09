import numpy as np
import yaml
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import warnings
from rag.load_models import setup_logger

warnings.filterwarnings("ignore", message="Local mode is not recommended for collections with more than 20,000 points")
warnings.filterwarnings("ignore", message="Collection .* contains .* points")


class DocumentIndexer:
    def __init__(self, config_path: str = "rag/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("Indexer")
        
        os.makedirs(self.config['qdrant_path'], exist_ok=True)
        self.client = QdrantClient(path=self.config['qdrant_path'])
        
        self.embedding_dimension = None
    
    def _detect_embedding_dimension(self, documents: List[Dict[str, Any]]) -> int:
        for doc in documents:
            if doc.get('embedding') is not None:
                emb = doc['embedding']
                if isinstance(emb, np.ndarray):
                    return emb.shape[-1]
                elif isinstance(emb, list):
                    return len(emb)
        
        raise ValueError("No embeddings found in documents")
    
    def _create_collection_with_dimension(self, vector_size: int):
        collection_name = self.config['collection_name']
        
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.logger.info(f"Created collection '{collection_name}' (dim: {vector_size})")
    
    def _document_exists(self, content_hash: str) -> bool:
        collection_name = self.config['collection_name']
        
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "content_hash",
                            "match": {"value": content_hash}
                        }
                    ]
                },
                limit=1
            )
            return len(results[0]) > 0
        except Exception:
            return False
    
    def _prepare_embedding(self, document: Dict[str, Any]) -> np.ndarray:
        emb = document['embedding']
        
        if isinstance(emb, list):
            return np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            return emb.astype(np.float32)
        elif hasattr(emb, 'numpy'):
            return emb.numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported embedding type: {type(emb)}")
    
    def index_documents(self, documents: List[Dict[str, Any]], force_reindex: bool = False):
        if not documents:
            self.logger.warning("No documents to index")
            return
        
        collection_name = self.config['collection_name']
        
        self.embedding_dimension = self._detect_embedding_dimension(documents)
        self._create_collection_with_dimension(self.embedding_dimension)
        
        if not force_reindex:
            try:
                count_result = self.client.count(collection_name)
                if count_result.count > 0:
                    self.logger.info(f"Collection already contains {count_result.count} documents - skipping indexing")
                    return
            except Exception:
                pass
        
        # Count by type
        type_counts = {}
        for doc in documents:
            doc_type = doc.get('doc_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        type_info = ", ".join([f"{doc_type}: {count}" for doc_type, count in type_counts.items()])
        self.logger.info(f"Indexing {len(documents)} documents ({type_info})")
        
        points = []
        for i, doc in enumerate(documents):
            try:
                embedding = self._prepare_embedding(doc)
                
                if embedding.shape[-1] != self.embedding_dimension:
                    continue
                
                payload = {
                    'doc_id': doc['id'],
                    'text': doc['text'],
                    'content_hash': doc['content_hash'],
                    'doc_type': doc.get('doc_type', 'unknown'),
                    'metadata': doc['metadata']
                }
                
                point = PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
                
            except Exception as e:
                continue
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=collection_name, points=batch)
            except Exception as e:
                self.logger.error(f"Error indexing batch: {e}")
        
        self.logger.info(f"Successfully indexed {len(points)} documents")
    
    def get_collection_info(self) -> Dict[str, Any]:
        collection_name = self.config['collection_name']
        
        try:
            info = self.client.get_collection(collection_name)
            count_result = self.client.count(collection_name)
            
            # Get type breakdown
            type_counts = {}
            try:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    limit=100000,
                    with_payload=True
                )
                
                for point in scroll_result[0]:
                    doc_type = point.payload.get('doc_type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    
            except Exception:
                type_counts = {"error": "Could not retrieve type breakdown"}
            
            return {
                'collection_name': collection_name,
                'status': info.status,
                'vectors_count': count_result.count,
                'embedding_dimension': self.embedding_dimension,
                'type_breakdown': type_counts
            }
        except Exception as e:
            return {'error': str(e)}