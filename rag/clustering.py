import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
import pickle
import os
from collections import defaultdict
from rag.load_models import setup_logger


class DocumentClusterer:
    def __init__(self, config_path: str = "rag/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger = setup_logger("Clusterer")
        
        self.n_clusters = self.config.get('n_clusters', 8)
        self.cluster_model = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.doc_to_cluster = {}
        self.cluster_to_docs = defaultdict(list)
        
        self.cache_dir = "./clustering_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cluster_cache_file = os.path.join(self.cache_dir, "clusters.pkl")
    
    def fit_clusters(self, documents: List[Dict[str, Any]], 
                    n_clusters: Optional[int] = None,
                    force_recompute: bool = False) -> Dict[str, Any]:
        
        if not force_recompute and os.path.exists(self.cluster_cache_file):
            try:
                self.load_clusters()
                self.logger.info(f"Loaded cached clusters ({self.n_clusters} clusters)")
                return self.get_cluster_info()
            except Exception as e:
                self.logger.warning(f"Failed to load cached clusters: {e}")
        
        if n_clusters is not None:
            self.n_clusters = n_clusters
        
        self.logger.info(f"Clustering {len(documents)} documents into {self.n_clusters} clusters...")
        
        embeddings = []
        doc_ids = []
        
        for doc in documents:
            if doc.get('embedding') is not None:
                embeddings.append(doc['embedding'])
                doc_ids.append(doc['id'])
        
        if not embeddings:
            self.logger.warning("No embeddings found - clustering disabled")
            return {"error": "No embeddings available"}
        
        embeddings = np.array(embeddings)
        self.logger.info(f"Embedding matrix shape: {embeddings.shape}")
        
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.cluster_model.fit_predict(embeddings)
        self.cluster_centers = self.cluster_model.cluster_centers_
        
        self.doc_to_cluster = {}
        self.cluster_to_docs = defaultdict(list)
        
        for doc_id, label in zip(doc_ids, self.cluster_labels):
            self.doc_to_cluster[doc_id] = int(label)
            self.cluster_to_docs[int(label)].append(doc_id)
        
        cluster_sizes = [len(docs) for docs in self.cluster_to_docs.values()]
        
        self.logger.info(f"Clustering complete:")
        self.logger.info(f"  Clusters: {self.n_clusters}")
        self.logger.info(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        self.save_clusters()
        
        return self.get_cluster_info()
    
    def get_query_cluster(self, query_embedding: np.ndarray, top_clusters: int = 3) -> List[int]:
        if self.cluster_centers is None:
            raise ValueError("Clusters not fitted yet")
        
        distances = np.linalg.norm(self.cluster_centers - query_embedding, axis=1)
        top_cluster_indices = np.argsort(distances)[:top_clusters]
        
        return top_cluster_indices.tolist()
    
    def get_documents_from_clusters(self, cluster_ids: List[int]) -> List[str]:
        doc_ids = []
        for cluster_id in cluster_ids:
            doc_ids.extend(self.cluster_to_docs[cluster_id])
        return doc_ids
    
    def get_cluster_info(self) -> Dict[str, Any]:
        if self.cluster_labels is None:
            return {"error": "No clusters fitted"}
        
        cluster_sizes = {i: len(docs) for i, docs in self.cluster_to_docs.items()}
        
        return {
            "n_clusters": self.n_clusters,
            "total_documents": len(self.doc_to_cluster),
            "cluster_sizes": cluster_sizes
        }
    
    def save_clusters(self):
        cache_data = {
            "n_clusters": self.n_clusters,
            "cluster_labels": self.cluster_labels,
            "cluster_centers": self.cluster_centers,
            "doc_to_cluster": self.doc_to_cluster,
            "cluster_to_docs": dict(self.cluster_to_docs)
        }
        
        with open(self.cluster_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.logger.info(f"Saved clusters to {self.cluster_cache_file}")
    
    def load_clusters(self):
        with open(self.cluster_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.n_clusters = cache_data["n_clusters"]
        self.cluster_labels = cache_data["cluster_labels"]
        self.cluster_centers = cache_data["cluster_centers"]
        self.doc_to_cluster = cache_data["doc_to_cluster"]
        self.cluster_to_docs = defaultdict(list, cache_data["cluster_to_docs"])
    
    def analyze_clusters(self, documents: List[Dict[str, Any]], sample_size: int = 3) -> Dict[str, Any]:
        if not self.cluster_to_docs:
            return {"error": "No clusters available"}
        
        analysis = {}
        doc_lookup = {doc['id']: doc for doc in documents}
        
        for cluster_id, doc_ids in self.cluster_to_docs.items():
            sample_docs = doc_ids[:sample_size]
            
            type_counts = defaultdict(int)
            for doc_id in doc_ids:
                if doc_id in doc_lookup:
                    doc_type = doc_lookup[doc_id].get('doc_type', 'unknown')
                    type_counts[doc_type] += 1
            
            cluster_info = {
                "size": len(doc_ids),
                "type_distribution": dict(type_counts),
                "sample_documents": []
            }
            
            for doc_id in sample_docs:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    cluster_info["sample_documents"].append({
                        "id": doc_id,
                        "doc_type": doc.get('doc_type', 'unknown'),
                        "text_preview": doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                    })
            
            analysis[f"cluster_{cluster_id}"] = cluster_info
        
        return analysis
    
    def get_cluster_type_distribution(self, documents: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        if not self.cluster_to_docs:
            return {"error": "No clusters available"}
        
        doc_lookup = {doc['id']: doc for doc in documents}
        cluster_distributions = {}
        
        for cluster_id, doc_ids in self.cluster_to_docs.items():
            type_counts = defaultdict(int)
            for doc_id in doc_ids:
                if doc_id in doc_lookup:
                    doc_type = doc_lookup[doc_id].get('doc_type', 'unknown')
                    type_counts[doc_type] += 1
            
            cluster_distributions[f"cluster_{cluster_id}"] = dict(type_counts)
        
        return cluster_distributions