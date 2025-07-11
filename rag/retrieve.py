import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rag.load_models import setup_logger


class DocumentRetriever:
    def __init__(self, config_path: str = "config.yaml", model=None, qdrant_client=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("Retriever")
        
        self.model = model
        self.client = qdrant_client if qdrant_client is not None else QdrantClient(path=self.config['qdrant_path'])
        
        self.clusterer = None
        self.use_clustering = self.config.get('use_clustering', False)
        
        if self.use_clustering:
            try:
                from clustering import DocumentClusterer
                self.clusterer = DocumentClusterer(config_path)
                self.clusterer.load_clusters()
                self.logger.info("Loaded existing clusters for retrieval")
            except Exception as e:
                self.logger.warning(f"No existing clusters found - clustering disabled for retrieval: {e}")
                self.use_clustering = False
    
    def _get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, prompt_name="query", convert_to_tensor=False)
        return embedding
    
    def _build_filter(self, doc_types: Optional[List[str]] = None, cluster_doc_ids: Optional[List[str]] = None) -> Optional[Filter]:
        conditions = []
        
        if doc_types:
            if len(doc_types) == 1:
                conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_types[0])))
            else:
                type_conditions = [
                    FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
                    for doc_type in doc_types
                ]
                conditions.append({"should": type_conditions})
        
        if cluster_doc_ids:
            cluster_conditions = [
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                for doc_id in cluster_doc_ids
            ]
            conditions.append({"should": cluster_conditions})
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return Filter(must=[conditions[0]]) if isinstance(conditions[0], dict) and "should" not in conditions[0] else Filter(**conditions[0])
        else:
            return Filter(must=conditions)
    
    def _extract_document_info(self, result) -> Dict[str, Any]:
        doc = {
            'text': result.payload['text'],
            'metadata': result.payload['metadata'],
            'doc_id': result.payload['doc_id'],
            'doc_type': result.payload['doc_type'],
            'score': result.score
        }
        
        if result.payload['doc_type'] == 'artwork':
            picture_id = result.payload['metadata'].get('picture_id')
            # print(f"DEBUG: Artwork doc_id={result.payload['doc_id']}, picture_id from metadata={picture_id}")
            if picture_id:
                doc['picture_id'] = picture_id
        
        return doc
    
    def retrieve_clustered(self, query: str, top_k: int = None, top_clusters: int = None, 
                          doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.config['retrieval_top_k']
        if top_clusters is None:
            top_clusters = self.config.get('top_clusters_for_query', 3)
        
        collection_name = self.config['collection_name']
        
        query_embedding = self._get_embedding(query)
        
        relevant_clusters = self.clusterer.get_query_cluster(query_embedding, top_clusters)
        relevant_doc_ids = self.clusterer.get_documents_from_clusters(relevant_clusters)
        
        self.logger.info(f"Query mapped to clusters {relevant_clusters} with {len(relevant_doc_ids)} documents")
        
        query_filter = self._build_filter(doc_types=doc_types, cluster_doc_ids=relevant_doc_ids)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k
        )
        
        retrieved_docs = []
        for result in results:
            doc = self._extract_document_info(result)
            doc['cluster_id'] = self.clusterer.doc_to_cluster.get(result.payload['doc_id'], -1)
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def retrieve_standard(self, query: str, top_k: int = None, 
                         doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.config['retrieval_top_k']
        
        collection_name = self.config['collection_name']
        
        query_embedding = self._get_embedding(query)
        
        query_filter = self._build_filter(doc_types=doc_types)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=query_filter,
            limit=top_k
        )
        
        retrieved_docs = []
        for result in results:
            doc = self._extract_document_info(result)
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def retrieve(self, query: str, top_k: int = None, 
                doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.use_clustering and self.clusterer is not None:
            return self.retrieve_clustered(query, top_k, doc_types=doc_types)
        else:
            return self.retrieve_standard(query, top_k, doc_types=doc_types)
    
    def retrieve_artworks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k, doc_types=['artwork'])
    
    def retrieve_artists(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        return self.retrieve(query, top_k, doc_types=['artist'])
    
    def compare_retrieval_methods(self, query: str, top_k: int = 5, 
                                doc_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        standard_results = self.retrieve_standard(query, top_k, doc_types=doc_types)
        
        if self.use_clustering and self.clusterer is not None:
            clustered_results = self.retrieve_clustered(query, top_k, doc_types=doc_types)
        else:
            clustered_results = []
        
        return {
            "standard": standard_results,
            "clustered": clustered_results
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        collection_name = self.config['collection_name']
        
        try:
            count_result = self.client.count(collection_name)
            
            # Get type breakdown
            type_counts = {}
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=100000,
                with_payload=True
            )
            
            for point in scroll_result[0]:
                doc_type = point.payload.get('doc_type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                'total_documents': count_result.count,
                'type_breakdown': type_counts,
                'clustering_enabled': self.use_clustering
            }
        except Exception as e:
            return {'error': str(e)}
