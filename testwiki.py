#!/usr/bin/env python3
"""
Fixed Wikimedia Commons Image Search Tester
Tests image search with proper redirect handling
"""

import requests
import time
from typing import Optional, Tuple

def test_wikimedia_search_fixed(query: str) -> Tuple[Optional[str], str]:
    """Test Wikimedia Commons search with redirect handling."""
    print(f"\n🔍 Testing Wikimedia for: '{query}'")
    
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        
        # Try different search approaches
        search_approaches = [
            {"srsearch": query, "approach": "direct"},
            {"srsearch": f"File:{query}", "approach": "with File:"},
            {"srsearch": f'"{query}"', "approach": "exact phrase"},
        ]
        
        for approach in search_approaches:
            approach_name = approach.pop("approach")
            print(f"  📝 Trying {approach_name} search...")
            
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srnamespace": 6,  # File namespace
                "srlimit": 5,
                "srprop": "title|snippet",
                **approach
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()  # Raise exception for bad status codes
                data = response.json()
            except requests.RequestException as e:
                print(f"  ❌ API request failed: {e}")
                continue
            except ValueError as e:
                print(f"  ❌ JSON decode failed: {e}")
                continue

            if data.get('query', {}).get('search'):
                print(f"  ✅ Found {len(data['query']['search'])} results with {approach_name}")
                
                for i, result in enumerate(data['query']['search'][:3]):
                    file_title = result['title']
                    filename = file_title.replace('File:', '')
                    print(f"    {i+1}. {filename}")
                    
                    # Test if image is accessible - FOLLOW REDIRECTS
                    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}"
                    try:
                        # Use GET instead of HEAD, and follow redirects
                        print(f"    🔗 Testing: {image_url}")
                        img_response = requests.get(
                            image_url, 
                            timeout=15, 
                            allow_redirects=True,
                            stream=True  # Don't download full content initially
                        )
                        
                        if img_response.status_code == 200:
                            # Check if it's actually an image
                            content_type = img_response.headers.get('content-type', '').lower()
                            if content_type.startswith('image/'):
                                final_url = img_response.url  # Get the final redirected URL
                                
                                # Get content length if available
                                content_length = img_response.headers.get('content-length')
                                if content_length:
                                    size_info = f"{int(content_length):,} bytes"
                                else:
                                    size_info = "unknown size"
                                
                                print(f"    ✅ ACCESSIBLE: {final_url}")
                                print(f"    📏 Size: {size_info}")
                                print(f"    🖼️ Content-Type: {content_type}")
                                
                                # Close the stream
                                img_response.close()
                                
                                return final_url, f"Wikimedia ({approach_name})"
                            else:
                                print(f"    ❌ Not an image (content-type: {content_type})")
                        elif img_response.status_code == 301 or img_response.status_code == 302:
                            print(f"    🔄 Redirect detected ({img_response.status_code})")
                            redirect_url = img_response.headers.get('location', 'Unknown')
                            print(f"    📍 Redirect to: {redirect_url}")
                        else:
                            print(f"    ❌ Not accessible (status: {img_response.status_code})")
                            
                        # Always close the response
                        img_response.close()
                        
                    except requests.RequestException as e:
                        print(f"    ❌ Request error: {e}")
                    except Exception as e:
                        print(f"    ❌ Unexpected error: {e}")
            else:
                print(f"  ❌ No results with {approach_name}")
        
        print("  ❌ No accessible images found on Wikimedia")
        return None, "Wikimedia (no accessible results)"
        
    except Exception as e:
        print(f"  ❌ Wikimedia search failed: {e}")
        return None, f"Wikimedia (error: {e})"

def quick_test_single_query(query: str):
    """Quick test for a single query with fixed redirect handling."""
    print(f"\n🎯 QUICK TEST: '{query}'")
    print("-" * 50)
    
    # Test Wikimedia with redirect handling
    wiki_url, wiki_source = test_wikimedia_search_fixed(query)
    
    if wiki_url:
        print(f"\n✅ SUCCESS! Wikimedia found: {wiki_url}")
        
        # Test if we can actually load the image (just headers)
        try:
            print("🔍 Verifying image accessibility...")
            response = requests.head(wiki_url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = response.headers.get('content-length', 'unknown')
                print(f"✅ Image confirmed working")
                print(f"   📄 Content-Type: {content_type}")
                print(f"   📏 Content-Length: {content_length} bytes")
            else:
                print(f"❌ Image URL not working: {response.status_code}")
        except Exception as e:
            print(f"❌ Error verifying image: {e}")
    else:
        print(f"\n❌ No Wikimedia results: {wiki_source}")

def test_your_problematic_query():
    """Test the specific query that was failing."""
    problematic_query = "Fifteen Sunflowers in a Vase VAN GOGH Vincent 1888 2"
    
    print("🧪 TESTING YOUR PROBLEMATIC QUERY")
    print("=" * 60)
    
    # Test original query
    quick_test_single_query(problematic_query)
    
    # Test simplified versions
    simplified_queries = [
        "Van Gogh Sunflowers",
        "Sunflowers Van Gogh", 
        "Vincent van Gogh Sunflowers",
        "Vase with Fifteen Sunflowers",
        "Van Gogh Vase Sunflowers"
    ]
    
    print(f"\n🔄 TESTING SIMPLIFIED VERSIONS")
    print("=" * 60)
    
    for simplified in simplified_queries:
        quick_test_single_query(simplified)
        time.sleep(1)  # Small delay between tests

def interactive_test():
    """Interactive testing mode."""
    print("\n🎯 INTERACTIVE TEST MODE")
    print("Enter queries to test (type 'quit' to exit)")
    
    while True:
        query = input("\n🎨 Enter artwork query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        quick_test_single_query(query)

# Test the problematic queries
if __name__ == "__main__":
    print("🎨 FIXED WIKIMEDIA COMMONS IMAGE SEARCH TESTER")
    print("=" * 60)
    
    # Choose mode
    print("Choose test mode:")
    print("1. Test your problematic query + simplified versions")
    print("2. Test famous artworks")
    print("3. Interactive testing")
    print("4. Single query test")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_your_problematic_query()
        
    elif choice == "2":
        famous_queries = [
            "Van Gogh Sunflowers",
            "Starry Night", 
            "Mona Lisa",
            "Water Lilies Monet",
            "Girl with Pearl Earring",
            "The Scream",
            "Guernica Picasso"
        ]
        
        print("\n🖼️ TESTING FAMOUS ARTWORKS")
        print("=" * 60)
        
        for query in famous_queries:
            quick_test_single_query(query)
            print("\n" + "="*60)
            time.sleep(1)  # Small delay between tests
            
    elif choice == "3":
        interactive_test()
        
    elif choice == "4":
        query = input("Enter query to test: ").strip()
        if query:
            quick_test_single_query(query)
    else:
        print("Invalid choice. Running problematic query test...")
        test_your_problematic_query()