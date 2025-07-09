import os
import sys
import yaml
import time
from contextlib import contextmanager

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from data_loader import DataLoader
from index import DocumentIndexer
from retrieve import DocumentRetriever
from generate import ResponseGenerator
from clustering import DocumentClusterer
from load_models import ModelLoader, setup_logger


@contextmanager
def timer(description: str, logger=None):
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    message = f"{description}: {elapsed:.2f}s"
    if logger:
        logger.info(message)
    else:
        print(f"[TIMING] {message}")


class RAGPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = setup_logger("RAGPipeline")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        with timer("Loading embedding model", self.logger):
            self.model_loader = ModelLoader(config_path)
            self.embedding_tokenizer, self.embedding_model = self.model_loader.load_embedding_model()
        
        with timer("Loading generator model", self.logger):
            self.generator_tokenizer, self.generator_model = self.model_loader.load_generator_model()
        
        self.data_loader = DataLoader(config_path)
        self.indexer = DocumentIndexer(config_path)
        self.retriever = DocumentRetriever(config_path, self.embedding_model, self.indexer.client)
        self.generator = ResponseGenerator(config_path, self.generator_tokenizer, self.generator_model)
        self.clusterer = DocumentClusterer(config_path)
        
        self.logger.info(f"Pipeline ready (device: {self.model_loader.get_device()})")

    def build_index(self, force_reindex: bool = False):
        with timer("Loading data", self.logger):
            documents = self.data_loader.load_data()
        
        with timer("Indexing documents", self.logger):
            self.indexer.index_documents(documents, force_reindex=force_reindex)
        
        if self.config.get('use_clustering', True):
            with timer("Clustering documents", self.logger):
                force_recompute = force_reindex  # Also recompute clusters if forcing reindex
                cluster_info = self.clusterer.fit_clusters(documents, force_recompute=force_recompute)
                self.logger.info(f"Clustering: {cluster_info['n_clusters']} clusters, {cluster_info['total_documents']} documents")
        else:
            self.logger.info("Clustering disabled")

    def query(self, user_query: str, doc_types: list = None) -> tuple:
        self.logger.info(f"Processing query: {user_query}")
        
        with timer("Document retrieval", self.logger):
            retrieved_docs = self.retriever.retrieve(user_query, doc_types=doc_types)
        
        with timer("Response generation", self.logger):
            response = self.generator.generate(user_query, retrieved_docs)
        
        return response, retrieved_docs
    
    def query_with_citations(self, user_query: str, doc_types: list = None) -> dict:
        self.logger.info(f"Processing query with citations: {user_query}")
        
        with timer("Document retrieval", self.logger):
            retrieved_docs = self.retriever.retrieve(user_query, doc_types=doc_types)
        
        with timer("Response generation with citations", self.logger):
            result = self.generator.generate_with_citations(user_query, retrieved_docs)
        
        return result
    
    def query_artworks(self, user_query: str) -> tuple:
        return self.query(user_query, doc_types=['artwork'])
    
    def query_artists(self, user_query: str) -> tuple:
        return self.query(user_query, doc_types=['artist'])
    
    def compare_retrieval_methods(self, user_query: str, doc_types: list = None):
        with timer("Retrieval comparison", self.logger):
            results = self.retriever.compare_retrieval_methods(user_query, doc_types=doc_types)
        
        print(f"\n=== RETRIEVAL COMPARISON ===")
        print(f"Query: {user_query}")
        if doc_types:
            print(f"Document types: {', '.join(doc_types)}")
        
        print(f"\n--- Standard Retrieval ---")
        for i, doc in enumerate(results['standard'], 1):
            doc_type = doc.get('doc_type', 'unknown')
            print(f"{i}. Score: {doc['score']:.3f} | Type: {doc_type} | {doc['text'][:80]}...")
        
        print(f"\n--- Clustered Retrieval ---")
        for i, doc in enumerate(results['clustered'], 1):
            doc_type = doc.get('doc_type', 'unknown')
            cluster_id = doc.get('cluster_id', '?')
            print(f"{i}. Score: {doc['score']:.3f} | Type: {doc_type} | Cluster: {cluster_id} | {doc['text'][:80]}...")
    
    def analyze_clusters(self):
        with timer("Cluster analysis", self.logger):
            documents = self.data_loader.load_data()
            analysis = self.clusterer.analyze_clusters(documents)
        
        print(f"\n=== CLUSTER ANALYSIS ===")
        for cluster_id, info in analysis.items():
            type_dist = info.get('type_distribution', {})
            type_info = ", ".join([f"{k}: {v}" for k, v in type_dist.items()])
            print(f"\n{cluster_id} (size: {info['size']}, types: {type_info}):")
            for doc in info['sample_documents']:
                doc_type = doc.get('doc_type', 'unknown')
                print(f"  - [{doc_type}] {doc['text_preview']}")
    
    def get_stats(self):
        with timer("Getting system stats", self.logger):
            index_info = self.indexer.get_collection_info()
            retrieval_stats = self.retriever.get_retrieval_stats()
            cluster_distribution = {}
            if self.config.get('use_clustering', True):
                try:
                    documents = self.data_loader.load_data()
                    cluster_distribution = self.clusterer.get_cluster_type_distribution(documents)
                except Exception as e:
                    cluster_distribution = {"error": str(e)}
        
        print(f"\n=== SYSTEM STATS ===")
        print(f"Collection: {index_info.get('collection_name', 'N/A')}")
        print(f"Total documents: {index_info.get('vectors_count', 0)}")
        print(f"Embedding dimension: {index_info.get('embedding_dimension', 'N/A')}")
        
        if 'type_breakdown' in index_info:
            print(f"Document types: {index_info['type_breakdown']}")
        
        if retrieval_stats.get('clustering_enabled', False):
            print(f"Clustering: enabled")
            if cluster_distribution and 'error' not in cluster_distribution:
                print(f"Cluster type distribution available")
        else:
            print(f"Clustering: disabled")
    
    def run_interactive(self):
        print("RAG Pipeline Interactive Mode")
        print("Commands:")
        print("  'quit' - Exit")
        print("  'stats' - Show system statistics")
        print("  'clusters' - Analyze clusters")
        print("  'compare <query>' - Compare retrieval methods")
        print("  'artworks <query>' - Search only artworks")
        print("  'artists <query>' - Search only artists")
        print("  'citations <query>' - Get response with detailed citations")
        print("-" * 60)
        
        while True:
            user_input = input("\nEnter your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                self.get_stats()
                continue
            
            if user_input.lower() == 'clusters':
                self.analyze_clusters()
                continue
            
            if user_input.lower().startswith('compare '):
                query = user_input[8:]
                self.compare_retrieval_methods(query)
                continue
            
            if user_input.lower().startswith('artworks '):
                query = user_input[9:]
                try:
                    with timer(f"Full query processing", self.logger):
                        response, retrieved_docs = self.query_artworks(query)
                    self._print_response(query, response, retrieved_docs, "artworks")
                except Exception as e:
                    print(f"Error processing artwork query: {e}")
                continue
            
            if user_input.lower().startswith('artists '):
                query = user_input[8:]
                try:
                    with timer(f"Full query processing", self.logger):
                        response, retrieved_docs = self.query_artists(query)
                    self._print_response(query, response, retrieved_docs, "artists")
                except Exception as e:
                    print(f"Error processing artist query: {e}")
                continue
            
            if user_input.lower().startswith('citations '):
                query = user_input[10:]
                try:
                    with timer(f"Full query processing with citations", self.logger):
                        result = self.query_with_citations(query)
                    self._print_citations_response(query, result)
                except Exception as e:
                    print(f"Error processing citations query: {e}")
                continue
            
            if not user_input:
                continue
            
            try:
                with timer(f"Full query processing", self.logger):
                    response, retrieved_docs = self.query(user_input)
                self._print_response(user_input, response, retrieved_docs)
                
            except Exception as e:
                print(f"Error processing query: {e}")
    
    def _print_response(self, query: str, response: str, retrieved_docs: list, query_type: str = "all"):
        print(f"\nQuery ({query_type}): {query}")
        print(f"\nAnswer: {response}")
        print(f"\nRetrieved documents ({len(retrieved_docs)}):")
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc.get('doc_type', 'unknown')
            cluster_info = f" | Cluster: {doc['cluster_id']}" if 'cluster_id' in doc else ""
            print(f"{i}. [{doc_type}] Score: {doc['score']:.3f}{cluster_info} | {doc['text'][:100]}...")
    
    def _print_citations_response(self, query: str, result: dict):
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {result['response']}")
        
        source_summary = result['source_summary']
        print(f"\nSource Summary:")
        print(f"  Total sources: {source_summary['total_sources']}")
        print(f"  Type breakdown: {source_summary['type_breakdown']}")
        
        print(f"\nDetailed Citations:")
        for citation in result['citations']:
            cluster_info = f" | Cluster: {citation['cluster_id']}" if 'cluster_id' in citation else ""
            print(f"{citation['index']}. [{citation['doc_type']}] Score: {citation['score']:.3f}{cluster_info}")
            print(f"   {citation['text_preview']}")


def main():
    logger = setup_logger("Main")
    
    use_clustering = True
    force_reindex = False
    
    if '--no-clustering' in sys.argv:
        use_clustering = False
        sys.argv.remove('--no-clustering')
    
    if '--force-reindex' in sys.argv:
        force_reindex = True
        sys.argv.remove('--force-reindex')
        logger.info("Force reindex enabled")
    
    if use_clustering:
        logger.info("Starting RAG Pipeline with Clustering")
    else:
        logger.info("Starting RAG Pipeline without Clustering")
    
    with timer("Total pipeline initialization", logger):
        pipeline = RAGPipeline()
        
        if not use_clustering:
            pipeline.retriever.use_clustering = False
        
        pipeline.build_index(force_reindex=force_reindex)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            pipeline.run_interactive()
        elif sys.argv[1] == 'stats':
            pipeline.get_stats()
        elif sys.argv[1] == 'clusters':
            pipeline.analyze_clusters()
        elif sys.argv[1] == 'compare':
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Tell me about Julie Mehretu's artwork"
            pipeline.compare_retrieval_methods(query)
        elif sys.argv[1] == 'artworks':
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "abstract paintings"
            with timer(f"Full artwork query processing", logger):
                response, retrieved_docs = pipeline.query_artworks(query)
            pipeline._print_response(query, response, retrieved_docs, "artworks")
        elif sys.argv[1] == 'artists':
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "contemporary painters"
            with timer(f"Full artist query processing", logger):
                response, retrieved_docs = pipeline.query_artists(query)
            pipeline._print_response(query, response, retrieved_docs, "artists")
        elif sys.argv[1] == 'citations':
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Tell me about Julie Mehretu's artwork"
            with timer(f"Full citations query processing", logger):
                result = pipeline.query_with_citations(query)
            pipeline._print_citations_response(query, result)
        else:
            query = " ".join(sys.argv[1:])
            with timer(f"Full query processing", logger):
                response, retrieved_docs = pipeline.query(query)
            pipeline._print_response(query, response, retrieved_docs)
    else:
        test_query = "How did Monet depict flowers?"
        with timer(f"Full test query processing", logger):
            response, retrieved_docs = pipeline.query(test_query)
        pipeline._print_response(test_query, response, retrieved_docs)


if __name__ == "__main__":
    main()