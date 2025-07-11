import os
# Set environment variables for better performance
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import streamlit as st
import pandas as pd
import sys
import time
import requests
import hashlib
import re
from PIL import Image
from io import BytesIO
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
from duckduckgo_search import DDGS
import random
# Import your RAG pipeline components
try:
    from rag.data_loader import DataLoader
    from rag.index import DocumentIndexer
    from rag.retrieve import DocumentRetriever
    from rag.generate import ResponseGenerator
    from rag.clustering import DocumentClusterer
    from rag.load_models import ModelLoader, setup_logger
    RAG_AVAILABLE = True
except ImportError as e:
    st.warning(f"RAG components not available: {e}")
    RAG_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="ArtRag",
    page_icon="🎨",
    layout="wide"
)

@contextmanager
def timer(description: str):
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    return elapsed

class StreamlitRAGPipeline:
    """Streamlit-optimized version of RAGPipeline"""
    
    def __init__(self, config_path: str = "rag/config.yaml"):
        self.config_path = config_path
        self.logger = setup_logger("StreamlitRAGPipeline")
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components (will be loaded lazily)
        self.model_loader = None
        self.embedding_model = None
        self.generator_model = None
        self.data_loader = None
        self.indexer = None
        self.retriever = None
        self.generator = None
        self.clusterer = None
        self.initialized = False
    
    def initialize_models(self):
        """Initialize all models - called when first needed"""
        if self.initialized:
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load models
            status_text.text("Loading embedding model...")
            progress_bar.progress(20)
            self.model_loader = ModelLoader(self.config_path)
            self.embedding_tokenizer, self.embedding_model = self.model_loader.load_embedding_model()
            
            status_text.text("Loading generator model...")
            progress_bar.progress(40)
            self.generator_tokenizer, self.generator_model = self.model_loader.load_generator_model()
            
            status_text.text("Initializing components...")
            progress_bar.progress(60)
            
            # Initialize other components
            self.data_loader = DataLoader(self.config_path)
            self.indexer = DocumentIndexer(self.config_path)
            self.retriever = DocumentRetriever(self.config_path, self.embedding_model, self.indexer.client)
            self.generator = ResponseGenerator(self.config_path, self.generator_tokenizer, self.generator_model)
            self.clusterer = DocumentClusterer(self.config_path)
            
            progress_bar.progress(80)
            status_text.text("Building index...")
            
            # Build index if needed
            self.build_index()
            
            progress_bar.progress(100)
            status_text.text("Ready!")
            time.sleep(0.5)
            
            self.initialized = True
            
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            raise
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def build_index(self, force_reindex: bool = False):
        """Build the document index"""
        documents = self.data_loader.load_data()
        self.indexer.index_documents(documents, force_reindex=force_reindex)
        
        if self.config.get('use_clustering', True):
            cluster_info = self.clusterer.fit_clusters(documents, force_recompute=force_reindex)
            self.logger.info(f"Clustering: {cluster_info.get('n_clusters', 0)} clusters, {cluster_info.get('total_documents', 0)} documents")
    
    def query(self, user_query: str, doc_types: list = None):
        """Process a query through the RAG pipeline"""
        if not self.initialized:
            self.initialize_models()
        
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(user_query, doc_types=doc_types)
        
        # Generate response
        response = self.generator.generate(user_query, retrieved_docs)
        
        return response, retrieved_docs
    
    def query_with_citations(self, user_query: str, doc_types: list = None):
        """Process a query with detailed citations"""
        if not self.initialized:
            self.initialize_models()
        
        retrieved_docs = self.retriever.retrieve(user_query, doc_types=doc_types)
        result = self.generator.generate_with_citations(user_query, retrieved_docs)
        
        return result

@st.cache_resource
def init_main_rag_system():
    """Initialize the main RAG pipeline"""
    if not RAG_AVAILABLE:
        return None
    
    try:
        return StreamlitRAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize main RAG system: {e}")
        return None

import urllib.parse

def search_wikimedia_image(query):
    """Search Wikimedia Commons for an image related to the query - FIXED method."""
    try:
        # Clean the query for better Wikimedia search
        clean_query = query.replace(' painting', '').replace(' artwork', '').replace('.jpg', '').strip()
        clean_query = re.sub(r'\d+', '', clean_query)  # Remove all digits
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()  # Clean multiple spaces
        print(f"Cleaned query: {clean_query}")
        
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"File:{clean_query}",
            "srnamespace": 6,  # File namespace
            "srlimit": 3,  # Get top 3 results
            "srprop": "title"
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('query', {}).get('search'):
            # Try each result until we find a working image
            for result in data['query']['search']:
                file_title = result['title']
                # Remove 'File:' prefix
                filename = file_title.replace('File:', '')
                
                # URL encode the filename properly
                encoded_filename = urllib.parse.quote(filename)
                image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_filename}"
                
                # Test if the image exists and is accessible - USE GET WITH REDIRECTS
                try:
                    img_response = requests.get(
                        image_url, 
                        timeout=10, 
                        allow_redirects=True,
                        stream=True  # Don't download full content
                    )
                    
                    if img_response.status_code == 200:
                        content_type = img_response.headers.get('content-type', '').lower()
                        if content_type.startswith('image/'):
                            final_url = img_response.url  # Get final redirected URL
                            img_response.close()
                            return final_url
                    
                    img_response.close()
                except:
                    continue
                    
    except Exception as e:
        print(f"Wikimedia search failed: {e}")
    
    return None

def search_image_duckduckgo(query, max_results=1, max_retries=2):
    """Search DuckDuckGo for images - SLOWER fallback method."""
    for attempt in range(max_retries):
        try:
            # Shorter delays for better UX
            delay = random.uniform(1, 3) + (attempt * 1)
            time.sleep(delay)
            
            with DDGS() as ddgs:
                results = ddgs.images(query, max_results=max_results)
                results = list(results)
                return results[0]['image'] if results else None
                
        except Exception as e:
            print(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    return None

@st.cache_data(ttl=3600)  # Cache images for 1 hour
def get_artwork_image(query):
    """Get image with Wikimedia first, DuckDuckGo as fallback - CACHED."""
    # Method 1: Try Wikimedia Commons first (fast and reliable)
    wikimedia_url = search_wikimedia_image(query)
    if wikimedia_url:
        return wikimedia_url, "Wikimedia Commons"
    
    # Method 2: Fallback to DuckDuckGo (slower but more comprehensive)
    ddg_url = search_image_duckduckgo(query)
    if ddg_url:
        return ddg_url, "DuckDuckGo"
    
    return None, "No source"

def extract_query(artwork_data):
    """Build a better search query from structured artwork data."""
    doc_id = artwork_data.get('doc_id', '')
    
    # Remove file extensions and -None suffix
    query = doc_id.replace('.jpg-None', '').replace('.png-None', '')
    query = query.replace('-scaled', '')
    
    # Replace hyphens with spaces for better search
    query = query.replace('-', ' ')
    
    # Clean up common patterns
    query = query.replace('  ', ' ').strip()
    
    return query

def extract_picture_ids_from_docs(retrieved_docs: List[Dict]) -> List[Tuple[str, str]]:
    """Extract picture_id values and search for images - returns (url, source) tuples."""
    image_data = []
    
    for doc in retrieved_docs:
        # Try to get picture_id first
        picture_id = None
        if 'picture_id' in doc:
            picture_id = doc['picture_id']
        elif 'metadata' in doc and 'picture_id' in doc['metadata']:
            picture_id = doc['metadata']['picture_id']
        
        # If no picture_id, use doc_id
        if not picture_id:
            picture_id = doc.get('doc_id', '')
        
        # Create search query and get image
        if picture_id:
            query = extract_query({'doc_id': picture_id})
            # Get image with source info
            image_url, source = get_artwork_image(query)
            if image_url:
                image_data.append((image_url, source))
    
    return image_data

def display_relevant_images(retrieved_docs: List[Dict], max_images: int = 3):
    """Display the most relevant images with source attribution."""
    with st.spinner("🔍 Searching for artwork images..."):
        # Extract image URLs from documents
        image_data = extract_picture_ids_from_docs(retrieved_docs)
        
        if not image_data:
            st.info("🖼️ No images found for the retrieved documents.")
            return
        
        # Take only the top N most relevant images
        top_image_data = image_data[:max_images]
        
        st.subheader(f"🖼️ Most Relevant Images ({len(top_image_data)})")
        
        # Display images in columns
        if len(top_image_data) == 1:
            cols = st.columns(1)
        elif len(top_image_data) == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        for i, (image_url, source) in enumerate(top_image_data):
            with cols[i % len(cols)]:
                try:
                    st.image(image_url, caption=f"Image {i+1}", use_container_width=True)
                    
                    # Display metadata
                    if i < len(retrieved_docs):
                        doc_info = retrieved_docs[i]
                        st.caption(f"⭐ Score: {doc_info.get('score', 0):.3f}")
                        if 'doc_type' in doc_info:
                            st.caption(f"📂 Type: {doc_info['doc_type']}")
                    
                    # Display source
                    if source == "Wikimedia Commons":
                        st.caption("🏛️ Source: Wikimedia Commons")
                    elif source == "DuckDuckGo":
                        st.caption("🔍 Source: DuckDuckGo")
                        
                except Exception as e:
                    st.error(f"❌ Could not load image {i+1}: {e}")

# Response display functions
def display_rag_response(response, retrieved_docs, query_type="all"):
    """Display RAG response with retrieved documents"""
    st.subheader("🤖 AI Response")
    st.write(response)
    
    if retrieved_docs:
        st.subheader(f"📚 Retrieved Documents ({len(retrieved_docs)})")
        
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"Document {i} - {doc.get('doc_type', 'unknown')} (Score: {doc['score']:.3f})"):
                cluster_info = f" | Cluster: {doc['cluster_id']}" if 'cluster_id' in doc else ""
                st.write(f"**Type:** {doc.get('doc_type', 'unknown')}")
                st.write(f"**Score:** {doc['score']:.3f}{cluster_info}")
                st.write(f"**Content:** {doc['text']}")

def display_rag_response_with_images(response: str, retrieved_docs: List[Dict], query_type: str = "all", show_images: bool = True):
    """Enhanced display function that includes images"""
    # Display AI response
    st.subheader("🤖 AI Response")
    st.write(response)
    
    # Display relevant images if enabled
    if show_images:
        display_relevant_images(retrieved_docs, max_images=3)
    
    # Display retrieved documents
    if retrieved_docs:
        st.subheader(f"📚 Retrieved Documents ({len(retrieved_docs)})")
        
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"Document {i} - {doc.get('doc_type', 'unknown')} (Score: {doc['score']:.3f})"):
                cluster_info = f" | Cluster: {doc['cluster_id']}" if 'cluster_id' in doc else ""
                st.write(f"**Type:** {doc.get('doc_type', 'unknown')}")
                st.write(f"**Score:** {doc['score']:.3f}{cluster_info}")
                
                # Display picture_id if available
                if 'picture_id' in doc:
                    st.write(f"**Picture ID:** {doc['picture_id']}")
                elif 'metadata' in doc and 'picture_id' in doc['metadata']:
                    st.write(f"**Picture ID:** {doc['metadata']['picture_id']}")
                
                st.write(f"**Content:** {doc['text']}")

def display_citations_response_with_images(result: Dict, show_images: bool = True):
    """Enhanced citations display with images"""
    # Display AI response
    st.subheader("🤖 AI Response with Citations")
    st.write(result['response'])
    
    # Display relevant images if enabled
    if show_images and 'citations' in result:
        display_relevant_images(result['citations'], max_images=3)
    
    # Display source summary
    source_summary = result['source_summary']
    st.subheader("📊 Source Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Sources", source_summary['total_sources'])
    with col2:
        st.write("**Type Breakdown:**")
        for doc_type, count in source_summary['type_breakdown'].items():
            st.write(f"- {doc_type}: {count}")
    
    # Display detailed citations
    st.subheader("📚 Detailed Citations")
    for citation in result['citations']:
        with st.expander(f"Citation {citation['index']} - {citation['doc_type']} (Score: {citation['score']:.3f})"):
            cluster_info = f" | Cluster: {citation['cluster_id']}" if 'cluster_id' in citation else ""
            st.write(f"**Type:** {citation['doc_type']}")
            st.write(f"**Score:** {citation['score']:.3f}{cluster_info}")
            
            # Display picture_id if available
            if 'picture_id' in citation:
                st.write(f"**Picture ID:** {citation['picture_id']}")
            elif 'metadata' in citation and 'picture_id' in citation['metadata']:
                st.write(f"**Picture ID:** {citation['metadata']['picture_id']}")
            
            st.write(f"**Content:** {citation['text_preview']}")

def main():
    # Initialize main RAG system
    main_rag_system = init_main_rag_system()
    
    # Banner with Van Gogh's Starry Night
    st.markdown(
        """
        <div style="width: 100%; height: 300px; overflow: hidden; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg" 
                 style="width: 100%; height: 100%; object-fit: cover; object-position: center;">
        </div>
        
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #2E8B57; font-size: 3.5em; margin-bottom: 15px; font-weight: bold;">
                🎨 ARTVisory: Advanced Art Information System
            </h1>
            <p style="font-size: 1.4em; color: #666;">
                Welcome to ArtRag. Discover art through our comprehensive RAG pipeline!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar with elegant logo
    # Sidebar with elegant logo
    with st.sidebar:        
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                st.image("/Users/marleendejonge/Downloads/ArtRag-main/ARTvisory.png", width=120)
            except:
                # Fallback to emoji if image file is not found
                st.markdown('<div style="font-size: 3em; margin-bottom: 10px; text-align: center;">🎨</div>', unsafe_allow_html=True)
        
        # Add the text content with dark blue title
        st.markdown(
            """
            <div style="color: #1e3a8a; font-size: 1.2em; font-weight: bold; letter-spacing: 1px; text-align: center; margin-top: -10px;">
                ARTVisory
            </div>
            <div style="color: #1e3a8a; font-size: 0.9em; margin-top: 5px; text-align: center; padding-bottom: 10px;">
                AI Art Explorer
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.header("🔧 System Options")
        
        query_type = st.selectbox(
            "Select which type of documents to query:",
            ["All Documents", "Artworks Only", "Artists Only", "With Citations"],
            help="Filter the types of documents to retrieve"
        )
        
        # Image display options
        st.subheader("🖼️ Image Options")
        show_relevant_images = st.checkbox("Show Relevant Images", value=True, help="Display the 3 most relevant images")
        
        # Cache management
        st.subheader("⚡ Performance")
        if st.button("Clear Image Cache"):
            st.cache_data.clear()
            st.success("Image cache cleared!")
    
    # Main query interface
    st.subheader("💬 Ask about art")
    
    # Use form for better performance
    with st.form("query_form", clear_on_submit=False):
        user_query = st.text_input("Enter your question:", placeholder="e.g., Tell me about Van Gogh's sunflowers")
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)
    
    if submitted and user_query:
        if main_rag_system:
            # Use main RAG pipeline with enhanced image display
            with st.spinner("Processing query..."):
                try:
                    if query_type == "All Documents":
                        response, retrieved_docs = main_rag_system.query(user_query)
                        display_rag_response_with_images(response, retrieved_docs,
                                                       show_images=show_relevant_images)
                    elif query_type == "Artworks Only":
                        response, retrieved_docs = main_rag_system.query(user_query, doc_types=['artwork'])
                        display_rag_response_with_images(response, retrieved_docs, "artworks",
                                                       show_images=show_relevant_images)
                    elif query_type == "Artists Only":
                        response, retrieved_docs = main_rag_system.query(user_query, doc_types=['artist'])
                        display_rag_response_with_images(response, retrieved_docs, "artists",
                                                       show_images=show_relevant_images)
                    elif query_type == "With Citations":
                        result = main_rag_system.query_with_citations(user_query)
                        display_citations_response_with_images(result,
                                                             show_images=show_relevant_images)
                        
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.exception(e)
        else:
            st.warning("RAG Pipeline is not available. Please check the system initialization.")
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🟢 System Status")
    st.sidebar.write(f"Main RAG Pipeline: {'✅ Ready' if main_rag_system else '❌ Error'}")
    st.sidebar.write(f"RAG Components: {'✅ Available' if RAG_AVAILABLE else '❌ Missing'}")

if __name__ == "__main__":
    main()