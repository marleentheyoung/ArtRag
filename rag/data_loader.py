import pickle
import yaml
import hashlib
import os
from typing import Dict, List, Any, Optional, Iterator
import numpy as np
from load_models import setup_logger


class DataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("DataLoader")
        
        self.artwork_path = self.config.get('artwork_dataset_path', './data/real.pkl')
        self.artist_path = self.config.get('artist_dataset_path', './data/real_artist.pkl')
    
    def yield_pickle(self, file_path: str) -> Iterator[Dict[str, Any]]:
        with open(file_path, "rb") as f:
            while True:
                try:
                    doc = pickle.load(f)
                    yield doc
                except EOFError:
                    break
                except Exception as e:
                    self.logger.warning(f"Error reading pickle object: {e}")
                    break
    
    def load_data(self, max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading datasets from artwork: {self.artwork_path}, artist: {self.artist_path}")
        
        documents = []
        loaded_count = 0
        
        if os.path.exists(self.artwork_path):
            for raw_doc in self.yield_pickle(self.artwork_path):
                if max_documents and loaded_count >= max_documents:
                    break
                
                processed_doc = self._process_document(raw_doc, loaded_count, doc_type='artwork')
                if processed_doc:
                    documents.append(processed_doc)
                    loaded_count += 1
        
        if os.path.exists(self.artist_path):
            for raw_doc in self.yield_pickle(self.artist_path):
                if max_documents and loaded_count >= max_documents:
                    break
                
                processed_doc = self._process_document(raw_doc, loaded_count, doc_type='artist')
                if processed_doc:
                    documents.append(processed_doc)
                    loaded_count += 1
        
        self._log_summary(documents)
        return documents
    
    def _process_document(self, raw_doc: Dict[str, Any], idx: int, doc_type: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_doc, dict):
            return None
        
        text = raw_doc.get('text', '').strip()
        if not text:
            return None
        
        embedding = self._extract_embedding(raw_doc)
        
        # Get the original ID and use it as the document ID
        original_id = raw_doc.get('id')
        
        if original_id:
            doc_id = str(original_id)
        else:
            doc_id = f'{doc_type}_{idx}'
        
        doc = {
            'id': doc_id,
            'text': text,
            'embedding': embedding,
            'content_hash': hashlib.md5(text.encode('utf-8')).hexdigest(),
            'doc_type': doc_type,
            'metadata': self._extract_metadata(raw_doc, idx, doc_type, original_id)
        }
        
        return doc
    
    def _extract_embedding(self, doc_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        emb = doc_dict.get('embeddings')
        if emb is None:
            return None
        
        if isinstance(emb, list):
            return np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            return emb.astype(np.float32)
        elif hasattr(emb, 'numpy'):
            return emb.numpy().astype(np.float32)
        
        return None
    
    def _extract_metadata(self, doc_dict: Dict[str, Any], idx: int, doc_type: str, original_id: Any = None) -> Dict[str, Any]:
        # Only exclude text and embeddings, but preserve all other metadata including ID
        excluded_fields = {'text', 'embeddings'}
        metadata = {
            'original_index': idx, 
            'doc_type': doc_type
        }
        
        # Add the original ID as picture_id for artwork documents
        if original_id is not None:
            if doc_type == 'artwork':
                # Clean up picture_id to stop at .jpg
                picture_id = str(original_id)
                if '.jpg' in picture_id:
                    picture_id = picture_id.split('.jpg')[0] + '.jpg'
                metadata['picture_id'] = picture_id
            metadata['original_id'] = str(original_id)
        
        for key, value in doc_dict.items():
            if key not in excluded_fields and value is not None:
                if isinstance(value, np.ndarray):
                    metadata[key] = value.tolist()
                elif hasattr(value, 'numpy'):
                    metadata[key] = value.numpy().tolist()
                else:
                    metadata[key] = value
        
        return metadata
    
    def _log_summary(self, documents: List[Dict[str, Any]]) -> None:
        total_docs = len(documents)
        self.logger.info(f"Loaded {total_docs} documents")
        
        if total_docs == 0:
            return
        
        type_counts = {}
        for doc in documents:
            doc_type = doc.get('doc_type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        for doc_type, count in type_counts.items():
            self.logger.info(f"  {doc_type}: {count} documents")
        
        with_embeddings = sum(1 for doc in documents if doc['embedding'] is not None)
        coverage = (with_embeddings / total_docs) * 100
        self.logger.info(f"Embedding coverage: {with_embeddings}/{total_docs} ({coverage:.0f}%)")
        
        if documents:
            sample_metadata_keys = set()
            for doc in documents[:5]:  # Check first 5 documents
                if doc.get('metadata'):
                    sample_metadata_keys.update(doc['metadata'].keys())
            
            if sample_metadata_keys:
                self.logger.info(f"Available metadata fields: {sorted(sample_metadata_keys)}")
        
        if with_embeddings > 0:
            for doc in documents:
                if doc['embedding'] is not None:
                    dim = doc['embedding'].shape[-1]
                    self.logger.info(f"Embedding dimension: {dim}")
                    break
    
    def has_precomputed_embeddings(self) -> bool:
        try:
            for file_path in [self.artwork_path, self.artist_path]:
                if not os.path.exists(file_path):
                    continue
                    
                count = 0
                for raw_doc in self.yield_pickle(file_path):
                    if count >= 5:
                        break
                    if isinstance(raw_doc, dict) and raw_doc.get('embeddings') is not None:
                        return True
                    count += 1
            return False
        except Exception:
            return False

    
    def get_document_by_picture_id(self, picture_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a document by its picture_id (original ID for artworks)
        """
        documents = self.load_data()
        for doc in documents:
            if doc.get('metadata', {}).get('picture_id') == picture_id:
                return doc
        return None
    
    def get_all_picture_ids(self) -> List[str]:
        """
        Get all picture IDs from artwork documents
        """
        documents = self.load_data()
        picture_ids = []
        for doc in documents:
            if doc.get('doc_type') == 'artwork':
                picture_id = doc.get('metadata', {}).get('picture_id')
                if picture_id:
                    picture_ids.append(picture_id)
        return picture_ids
