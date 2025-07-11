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

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

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
                        img_response = requests.get(image_url, timeout=10, allow_redirects=True)
                        
                        if img_response.status_code == 200:
                            # Check if it's actually an image
                            content_type = img_response.headers.get('content-type', '')
                            if content_type.startswith('image/'):
                                final_url = img_response.url  # Get the final redirected URL
                                print(f"    ✅ ACCESSIBLE: {final_url}")
                                print(f"    📏 Size: {len(img_response.content)} bytes")
                                return final_url, f"Wikimedia ({approach_name})"
                            else:
                                print(f"    ❌ Not an image (content-type: {content_type})")
                        else:
                            print(f"    ❌ Not accessible (status: {img_response.status_code})")
                    except Exception as e:
                        print(f"    ❌ Error testing: {e}")
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
        
        # Test if we can actually load the image
        try:
            response = requests.get(wiki_url, stream=True, timeout=10)
            if response.status_code == 200:
                size = len(response.content)
                print(f"✅ Image confirmed working - Size: {size:,} bytes")
            else:
                print(f"❌ Image URL not working: {response.status_code}")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
    else:
        print(f"\n❌ No Wikimedia results: {wiki_source}")

# Test the problematic queries
if __name__ == "__main__":
    test_queries = [
        "Van Gogh Sunflowers",
        "Starry Night", 
        "Mona Lisa",
        "Water Lilies Monet"
    ]
    
    for query in test_queries:
        quick_test_single_query(query)
        print("\n" + "="*60)