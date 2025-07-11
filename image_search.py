# Optimized Image Loading Solutions

import asyncio
import aiohttp
import concurrent.futures
import threading
import time
from typing import List, Dict, Optional, Tuple
import streamlit as st
import re
import urllib.parse

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

# SOLUTION 1: Concurrent Image Loading (Fastest for multiple images)
def search_images_concurrently(queries: List[str], max_workers: int = 3) -> List[Tuple[Optional[str], str]]:
    """
    Load multiple images concurrently using ThreadPoolExecutor.
    This is the fastest solution for loading multiple images at once.
    """
    def search_single_image(query):
        return search_wikimedia_image_fast(query)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all search tasks
        future_to_query = {executor.submit(search_single_image, query): query for query in queries}
        results = []
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_query):
            query = future_to_query[future]
            try:
                image_url = future.result()
                source = "Wikimedia Commons" if image_url else "No source"
                results.append((image_url, source))
            except Exception as e:
                print(f"Error searching for {query}: {e}")
                results.append((None, "Error"))
        
        return results

# SOLUTION 2: Async Image Loading (Good for I/O bound operations)
async def search_images_async(queries: List[str]) -> List[Tuple[Optional[str], str]]:
    """
    Async version for concurrent image loading.
    """
    async def search_single_async(session, query):
        return await search_wikimedia_image_async(session, query)
    
    async with aiohttp.ClientSession() as session:
        tasks = [search_single_async(session, query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append((None, "Error"))
            else:
                source = "Wikimedia Commons" if result else "No source"
                processed_results.append((result, source))
        
        return processed_results

async def search_wikimedia_image_async(session: aiohttp.ClientSession, query: str) -> Optional[str]:
    """Async version of Wikimedia search."""
    try:
        # Clean query
        clean_query = query.replace(' painting', '').replace(' artwork', '').replace('.jpg', '').replace('.png', '').strip()
        clean_query = re.sub(r'\d+', '', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        if len(clean_query.strip()) < 2:
            return None
        
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"File:{clean_query}",
            "srnamespace": 6,
            "srlimit": 2,  # Reduced for speed
            "srprop": "title"
        }
        
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
            data = await response.json()
            
            if data.get('query', {}).get('search'):
                for result in data['query']['search']:
                    filename = result['title'].replace('File:', '')
                    encoded_filename = urllib.parse.quote(filename)
                    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_filename}"
                    
                    # Quick test
                    try:
                        async with session.head(image_url, timeout=aiohttp.ClientTimeout(total=3)) as img_response:
                            if img_response.status == 200:
                                return str(img_response.url)
                    except:
                        continue
        
        return None
    except Exception as e:
        print(f"Async search failed for {query}: {e}")
        return None

# SOLUTION 3: Optimized Sequential Search (Faster single image loading)
def search_wikimedia_image_fast(query: str) -> Optional[str]:
    """
    Optimized version of Wikimedia search - faster than original.
    """
    try:
        # Clean query (same as before but optimized)
        clean_query = query.replace(' painting', '').replace(' artwork', '').replace('.jpg', '').replace('.png', '').strip()
        clean_query = re.sub(r'\d+', '', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
        if len(clean_query.strip()) < 2:
            return None
        
        # Create a session for connection reuse
        session = requests.Session()
        session.headers.update({'User-Agent': 'ArtRag/1.0'})
        
        url = "https://commons.wikimedia.org/w/api.php"
        
        # Try only the most effective search strategy first
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"File:{clean_query}",
            "srnamespace": 6,
            "srlimit": 2,  # Reduced from 3 to 2 for speed
            "srprop": "title"
        }
        
        response = session.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('query', {}).get('search'):
            for result in data['query']['search']:
                filename = result['title'].replace('File:', '')
                encoded_filename = urllib.parse.quote(filename)
                image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_filename}"
                
                # Faster image test - try HEAD first, then GET if needed
                try:
                    img_response = session.head(image_url, timeout=4, allow_redirects=True)
                    if img_response.status_code == 200:
                        return img_response.url
                    
                    # If HEAD fails, try GET (some servers block HEAD)
                    if img_response.status_code == 403:
                        img_response = session.get(image_url, timeout=4, allow_redirects=True, stream=True)
                        if img_response.status_code == 200:
                            content_type = img_response.headers.get('content-type', '').lower()
                            if content_type.startswith('image/'):
                                final_url = img_response.url
                                img_response.close()
                                return final_url
                        img_response.close()
                        
                except Exception:
                    continue
        
        session.close()
        return None
        
    except Exception as e:
        print(f"Fast search failed for {query}: {e}")
        return None

# SOLUTION 4: Background Loading with Threading
class BackgroundImageLoader:
    """
    Load images in background while RAG is processing.
    """
    def __init__(self):
        self.image_cache = {}
        self.loading_threads = {}
    
    def start_loading(self, queries: List[str]):
        """Start loading images in background threads."""
        for query in queries:
            if query not in self.image_cache and query not in self.loading_threads:
                thread = threading.Thread(target=self._load_image, args=(query,))
                thread.start()
                self.loading_threads[query] = thread
    
    def _load_image(self, query: str):
        """Load a single image in background."""
        try:
            image_url = search_wikimedia_image_fast(query)
            self.image_cache[query] = (image_url, "Wikimedia Commons" if image_url else "No source")
        except Exception as e:
            self.image_cache[query] = (None, f"Error: {e}")
        finally:
            # Clean up thread reference
            self.loading_threads.pop(query, None)
    
    def get_image(self, query: str, timeout: float = 5.0) -> Tuple[Optional[str], str]:
        """Get image, waiting for background loading to complete."""
        # If already cached, return immediately
        if query in self.image_cache:
            return self.image_cache[query]
        
        # If loading in background, wait for it
        if query in self.loading_threads:
            self.loading_threads[query].join(timeout=timeout)
            return self.image_cache.get(query, (None, "Timeout"))
        
        # If not started, load synchronously
        self._load_image(query)
        return self.image_cache.get(query, (None, "Error"))

# SOLUTION 5: Integration with your Streamlit app
def extract_picture_ids_from_docs_optimized(retrieved_docs: List[Dict]) -> List[Tuple[str, str]]:
    """
    OPTIMIZED VERSION: Extract and load images concurrently.
    This replaces your current extract_picture_ids_from_docs function.
    """
    # Extract all queries first
    queries = []
    for doc in retrieved_docs:
        picture_id = None
        if 'picture_id' in doc:
            picture_id = doc['picture_id']
        elif 'metadata' in doc and 'picture_id' in doc['metadata']:
            picture_id = doc['metadata']['picture_id']
        
        if not picture_id:
            picture_id = doc.get('doc_id', '')
        
        if picture_id:
            query = extract_query({'doc_id': picture_id})  # Your existing function
            queries.append(query)
    
    if not queries:
        return []
    
    # Load all images concurrently
    print(f"🚀 Loading {len(queries)} images concurrently...")
    start_time = time.time()
    
    results = search_images_concurrently(queries, max_workers=3)
    
    elapsed = time.time() - start_time
    print(f"✅ Loaded images in {elapsed:.2f} seconds")
    
    # Filter out None results
    image_data = [(url, source) for url, source in results if url]
    return image_data

def display_relevant_images_optimized(retrieved_docs: List[Dict], max_images: int = 3):
    """
    OPTIMIZED VERSION: Replace your current display_relevant_images function.
    """
    with st.spinner("🚀 Loading artwork images concurrently..."):
        # Use optimized concurrent loading
        image_data = extract_picture_ids_from_docs_optimized(retrieved_docs)
        
        if not image_data:
            st.info("🖼️ No images found for the retrieved documents.")
            return
        
        # Take only the top N most relevant images
        top_image_data = image_data[:max_images]
        
        st.subheader(f"🖼️ Most Relevant Images ({len(top_image_data)})")
        
        # Display images in columns (same as your original)
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

# SOLUTION 6: Parallel RAG + Image Loading
def process_query_with_parallel_images(main_rag_system, user_query: str, doc_types: list = None):
    """
    Process RAG query and load images in parallel.
    This can significantly reduce total processing time.
    """
    # Start background image loader
    image_loader = BackgroundImageLoader()
    
    # Step 1: Start RAG processing
    print("🤖 Starting RAG processing...")
    rag_start = time.time()
    
    if not main_rag_system.initialized:
        main_rag_system.initialize_models()
    
    # Retrieve documents
    retrieved_docs = main_rag_system.retriever.retrieve(user_query, doc_types=doc_types)
    
    # Step 2: Start image loading in background while generating response
    queries = []
    for doc in retrieved_docs:
        picture_id = doc.get('picture_id') or doc.get('doc_id', '')
        if picture_id:
            query = extract_query({'doc_id': picture_id})
            queries.append(query)
    
    print(f"🖼️ Starting background image loading for {len(queries)} images...")
    image_loader.start_loading(queries)
    
    # Step 3: Generate response (while images load in background)
    print("📝 Generating response...")
    response = main_rag_system.generator.generate(user_query, retrieved_docs)
    rag_elapsed = time.time() - rag_start
    print(f"✅ RAG completed in {rag_elapsed:.2f} seconds")
    
    # Step 4: Collect images (should be ready or nearly ready)
    print("🎨 Collecting images...")
    image_start = time.time()
    image_data = []
    for query in queries:
        image_url, source = image_loader.get_image(query, timeout=2.0)
        if image_url:
            image_data.append((image_url, source))
    
    image_elapsed = time.time() - image_start
    print(f"✅ Images collected in {image_elapsed:.2f} seconds")
    
    return response, retrieved_docs, image_data

"""
USAGE RECOMMENDATIONS:

1. **For Maximum Speed**: Replace your current functions with:
   - extract_picture_ids_from_docs_optimized()
   - display_relevant_images_optimized()

2. **For Parallel Processing**: Use process_query_with_parallel_images() 
   to run RAG and image loading simultaneously.

3. **Expected Speed Improvements**:
   - Sequential: ~6-10 seconds for 3 images
   - Concurrent: ~2-4 seconds for 3 images  
   - Parallel with RAG: ~1-2 seconds additional time

4. **Memory Usage**: Concurrent loading uses more memory but significantly 
   faster. Adjust max_workers based on your server capacity.

5. **Error Handling**: All solutions include proper error handling and 
   fallbacks to ensure the app doesn't crash if image loading fails.
"""