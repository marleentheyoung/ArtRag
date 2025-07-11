import requests
from duckduckgo_search import DDGS

def extract_query(artwork_data):
    """
    Builds a search query from structured artwork data.
    """
    return artwork_data.get('doc_id', '')

from duckduckgo_search import DDGS

def search_wikimedia_image(query):
    """Search Wikimedia Commons for an image related to the query."""
    url = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,  # 6 = File namespace (images/media)
        "srlimit": 1       # Only top result
    }

    response = requests.get(url, params=params)
    data = response.json()

    if not data['query']['search']:
        return None

    # Extract the file title (e.g., "File:Van Gogh - Still Life Vase with Five Sunflowers.jpg")
    file_title = data['query']['search'][0]['title']

    # Construct the full image URL
    # Wikimedia image URLs follow this format:
    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{file_title.split(':', 1)[1]}"
    
    return image_url

def search_image(ddgs, query, max_results=1):
    results = ddgs.images(query, max_results=max_results)
    results = list(results)
    return results[0]['image'] if results else None

def main():
    artwork_data = {'text': '“Still Life Vase with Five Sunflowers” is a notable artwork created by the renowned artist Vincent van Gogh in 1888, during his stay in Arles, Bouches-du-Rhône, France. Executed with oil on canvas, this piece belongs to the Post-Impressionism movement and represents the genre of flower painting. Unfortunately, this distinguished artwork has been destroyed. The artwork features five sunflowers arranged in a vase, capturing a moment of natural beauty amidst the vibrancy of life. The sunflowers, displaying various stages of bloom and decay, are depicted with van Gogh’s characteristic bold and expressive brushstrokes. The color palette is rich, highlighting the contrast between the vivid yellow petals and the dark background, evoking both the brilliance and fleeting nature of life.', 'metadata': {'original_index': 16084, 'doc_type': 'artwork', 'picture_id': 'still-life-vase-with-five-sunflowers-vincent-van-gogh-1888-arles-bouches-du-rhone-france.jpg', 'original_id': 'still-life-vase-with-five-sunflowers-vincent-van-gogh-1888-arles-bouches-du-rhone-france.jpg-None', 'id': 'still-life-vase-with-five-sunflowers-vincent-van-gogh-1888-arles-bouches-du-rhone-france.jpg-None'}, 'doc_id': 'still-life-vase-with-five-sunflowers-vincent-van-gogh-1888-arles-bouches-du-rhone-france.jpg-None', 'doc_type': 'artwork', 'score': 0.658087590929098, 'picture_id': 'still-life-vase-with-five-sunflowers-vincent-van-gogh-1888-arles-bouches-du-rhone-france.jpg'}

    query = extract_query(artwork_data)
    print(f"Search query: {query}")
    
    with DDGS() as ddgs:
        image_url = search_image(ddgs, query)
        print(f"Found image URL: {image_url}" if image_url else "No image found.")
        # print(search_wikimedia_image(query))
        print(query)
    if image_url:
        print(f"Found image URL: {image_url}")
    else:
        print("No image found.")

if __name__ == "__main__":
    main()
