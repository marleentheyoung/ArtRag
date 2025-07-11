import os
import sys
import yaml
import time
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Tuple, Dict, List, Any, Optional, Callable

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from rag.data_loader import DataLoader
from rag.index import DocumentIndexer
from rag.retrieve import DocumentRetriever
from rag.generate import ResponseGenerator
from rag.clustering import DocumentClusterer
from rag.load_models import ModelLoader, setup_logger


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


class QueryHandler:

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.logger = pipeline.logger
    
    def execute_query(self, query_func: Callable, query: str, query_type: str, *args, **kwargs) -> bool:
        try:
            with timer(f"Full {query_type} query processing", self.logger):
                result = query_func(query, *args, **kwargs)
            
            if isinstance(result, tuple):
                # Standard query returns (response, retrieved_docs)
                response, retrieved_docs = result
                self.pipeline._print_response(query, response, retrieved_docs, query_type)
            elif isinstance(result, dict):
                # Citations query returns dict
                self.pipeline._print_citations_response(query, result)
            else:
                print(f"Unexpected result type from {query_type} query")
            
            return True
            
        except Exception as e:
            print(f"Error processing {query_type} query: {e}")
            return False


class RAGPipeline:
    def __init__(self, config_path: str = "rag/config.yaml"):
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
        
        self.log_file = "query_logs.json"
        self._initialize_log_file()
        self.query_handler = QueryHandler(self)
        self.logger.info(f"Pipeline ready (device: {self.model_loader.get_device()})")

    def _initialize_log_file(self):
        """Initialize the JSON log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def _log_query_result(self, query: str, response: str, retrieved_docs: list, 
                         latencies: dict, query_type: str = "all", doc_types: list = None):
        try:
            retrieved_docs_data = []
            for i, doc in enumerate(retrieved_docs, 1):
                doc_data = {
                    "index": i,
                    "doc_type": doc.get('doc_type', 'unknown'),
                    "score": doc.get('score', 0.0),
                    "text": doc['text']
                }
                
                if doc.get('doc_type') == 'artwork' and doc.get('picture_id'):
                    doc_data['picture_id'] = doc['picture_id']
                
                if 'cluster_id' in doc:
                    doc_data['cluster_id'] = doc['cluster_id']
                
                retrieved_docs_data.append(doc_data)
            
            log_entry = {
                "query": query,
                "query_type": query_type,
                "doc_types_filter": doc_types,
                "response": response,
                "retrieved_documents": retrieved_docs_data,
                "top_k": len(retrieved_docs),
                "latencies": latencies,
                "total_time": sum(latencies.values())
            }
            
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            logs.append(log_entry)
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            self.logger.info(f"Logged query result to {self.log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to log query result: {e}")

    def build_index(self, force_reindex: bool = False):
        with timer("Loading data", self.logger):
            documents = self.data_loader.load_data()
        
        with timer("Indexing documents", self.logger):
            self.indexer.index_documents(documents, force_reindex=force_reindex)
        
        if self.config.get('use_clustering', True):
            with timer("Clustering documents", self.logger):
                force_recompute = force_reindex  # recompute clusters if forcing reindex
                cluster_info = self.clusterer.fit_clusters(documents, force_recompute=force_recompute)
                self.logger.info(f"Clustering: {cluster_info['n_clusters']} clusters, {cluster_info['total_documents']} documents")
        else:
            self.logger.info("Clustering disabled")

    def _execute_query_with_logging(self, query_func: Callable, query: str, query_type: str, 
                                   doc_types: Optional[List[str]] = None) -> Tuple[Any, List[Dict[str, Any]]]:
        self.logger.info(f"Processing {query_type} query: {query}")
        latencies = {}
        
        with timer("Document retrieval", self.logger):
            start_time = time.time()
            retrieved_docs = self.retriever.retrieve(query, doc_types=doc_types)
            latencies["document_retrieval"] = time.time() - start_time
        
        with timer("Response generation", self.logger):
            start_time = time.time()
            response = query_func(query, retrieved_docs)
            latencies["response_generation"] = time.time() - start_time
        
        if isinstance(response, dict):
            # Citations response
            self._log_query_result(query, response['response'], retrieved_docs, latencies, query_type, doc_types)
        else:
            # Standard response
            self._log_query_result(query, response, retrieved_docs, latencies, query_type, doc_types)
        
        return response, retrieved_docs

    def query(self, user_query: str, doc_types: list = None) -> tuple:
        return self._execute_query_with_logging(
            self.generator.generate, user_query, "all", doc_types
        )
    
    def query_with_citations(self, user_query: str, doc_types: list = None) -> dict:
        response, retrieved_docs = self._execute_query_with_logging(
            self.generator.generate_with_citations, user_query, "citations", doc_types
        )
        return response
    
    def query_artworks(self, user_query: str) -> tuple:
        return self._execute_query_with_logging(
            self.generator.generate, user_query, "artworks", ['artwork']
        )
    
    def query_artists(self, user_query: str) -> tuple:
        return self._execute_query_with_logging(
            self.generator.generate, user_query, "artists", ['artist']
        )
    
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
            picture_info = f" | Picture ID: {doc['picture_id']}" if doc.get('picture_id') else ""
            print(f"{i}. Score: {doc['score']:.3f} | Type: {doc_type}{picture_info} | {doc['text'][:80]}...")
        
        print(f"\n--- Clustered Retrieval ---")
        for i, doc in enumerate(results['clustered'], 1):
            doc_type = doc.get('doc_type', 'unknown')
            cluster_id = doc.get('cluster_id', '?')
            picture_info = f" | Picture ID: {doc['picture_id']}" if doc.get('picture_id') else ""
            print(f"{i}. Score: {doc['score']:.3f} | Type: {doc_type} | Cluster: {cluster_id}{picture_info} | {doc['text'][:80]}...")
    
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
    
    def get_log_summary(self):
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                print("No queries logged yet.")
                return
            
            print(f"\n=== QUERY LOG SUMMARY ===")
            print(f"Total queries logged: {len(logs)}")
            
            total_retrieval = sum(log['latencies'].get('document_retrieval', 0) for log in logs)
            total_generation = sum(log['latencies'].get('response_generation', 0) for log in logs)
            
            print(f"Average retrieval time: {total_retrieval/len(logs):.2f}s")
            print(f"Average generation time: {total_generation/len(logs):.2f}s")
            type_counts = {}
            for log in logs:
                query_type = log.get('query_type', 'unknown')
                type_counts[query_type] = type_counts.get(query_type, 0) + 1
            
            print(f"Query type breakdown: {type_counts}")
            print(f"\nRecent queries:")
            for log in logs[-5:]:
                query = log['query'][:50] + "..." if len(log['query']) > 50 else log['query']
                total_time = log['total_time']
                print(f"  '{query}' ({total_time:.2f}s)")
        
        except Exception as e:
            print(f"Error reading log summary: {e}")

    def _handle_command_query(self, command: str, query: str) -> bool:
        command_map = {
            'compare': self.compare_retrieval_methods,
            'artworks': self.query_handler.execute_query,
            'artists': self.query_handler.execute_query,
            'citations': self.query_handler.execute_query
        }
        
        if command not in command_map:
            return False
        
        if command == 'compare':
            command_map[command](query)
        elif command == 'artworks':
            return command_map[command](self.query_artworks, query, "artworks")
        elif command == 'artists':
            return command_map[command](self.query_artists, query, "artists")
        elif command == 'citations':
            return command_map[command](self.query_with_citations, query, "citations")
        
        return True

    def run_interactive(self):
        print("RAG Pipeline Interactive Mode")
        print("Commands:")
        print("  'quit' - Exit")
        print("  'stats' - Show system statistics")
        print("  'logs' - Show query log summary")
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
            
            if not user_input:
                continue
            
            simple_commands = {
                'stats': self.get_stats,
                'logs': self.get_log_summary,
                'clusters': self.analyze_clusters
            }
            
            if user_input.lower() in simple_commands:
                simple_commands[user_input.lower()]()
                continue
            
            command_prefixes = ['compare ', 'artworks ', 'artists ', 'citations ']
            command_handled = False
            
            for prefix in command_prefixes:
                if user_input.lower().startswith(prefix):
                    command = prefix.strip()
                    query = user_input[len(prefix):]
                    if self._handle_command_query(command, query):
                        command_handled = True
                        break
            
            if command_handled:
                continue
            
            if not self.query_handler.execute_query(self.query, user_input, "all"):
                print("Query execution failed. Please try again.")
    
    def _print_response(self, query: str, response: str, retrieved_docs: list, query_type: str = "all"):
        print(f"\nQuery ({query_type}): {query}")
        print(f"\nAnswer: {response}")
        
        print(f"\nRetrieved documents ({len(retrieved_docs)}):")
        for i, doc in enumerate(retrieved_docs, 1):
            doc_type = doc.get('doc_type', 'unknown')
            cluster_info = f" | Cluster: {doc['cluster_id']}" if 'cluster_id' in doc else ""
            picture_info = f" | Picture ID: {doc['picture_id']}" if doc.get('picture_id') else ""
            print(f"{i}. [{doc_type}] Score: {doc['score']:.3f}{cluster_info}{picture_info} | {doc['text'][:100]}...")
    
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
            picture_info = f" | Picture ID: {citation['picture_id']}" if citation.get('picture_id') else ""
            print(f"{citation['index']}. [{citation['doc_type']}] Score: {citation['score']:.3f}{cluster_info}{picture_info}")
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
        command = sys.argv[1]
        no_query_commands = {
            'interactive': pipeline.run_interactive,
            'stats': pipeline.get_stats,
            'logs': pipeline.get_log_summary,
            'clusters': pipeline.analyze_clusters
        }
        
        if command in no_query_commands:
            no_query_commands[command]()
            return
        
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        query_commands = {
            'compare': (pipeline.compare_retrieval_methods, "Tell me about Julie Mehretu's artwork"),
            'artworks': (pipeline.query_artworks, "abstract paintings"),
            'artists': (pipeline.query_artists, "contemporary painters"),
            'citations': (pipeline.query_with_citations, "Tell me about Julie Mehretu's artwork")
        }
        
        if command in query_commands:
            func, default_query = query_commands[command]
            query = query or default_query
            
            with timer(f"Full {command} query processing", logger):
                result = func(query)
            
            if command == 'compare':
                # compare_retrieval_methods handles its own output
                pass
            elif command == 'citations':
                pipeline._print_citations_response(query, result)
            else:
                response, retrieved_docs = result
                pipeline._print_response(query, response, retrieved_docs, command)
        else:
            # regular query
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
