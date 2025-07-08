def load_data():
    """Load your JSON data"""
    # Replace with your actual data loading logic
    # For now, using the sample from your document
    def get_sample_artwork_data():
    sample_data = [
        {
            "file": "starry_night.jpg",
            "artist": "Vincent van Gogh",
            "title": "The Starry Night",
            "year": 1889,
            "description": "A swirling night sky over a quiet town, painted during Van Gogh's stay at the asylum in Saint-Rémy-de-Provence.",
            "tags": ["Post-Impressionism", "Landscape", "Night"],
            "metadata": {
                "medium": "Oil on canvas",
                "dimensions": "73.7 cm × 92.1 cm",
                "location": "Museum of Modern Art, New York"
            },
            "date_added": "2024-06-12"
        },
        {
            "file": "mona_lisa.jpg",
            "artist": "Leonardo da Vinci",
            "title": "Mona Lisa",
            "year": 1503,
            "description": "A portrait of a woman with an enigmatic expression, widely regarded as a masterpiece of the Italian Renaissance.",
            "tags": ["Renaissance", "Portrait"],
            "metadata": {
                "medium": "Oil on poplar panel",
                "dimensions": "77 cm × 53 cm",
                "location": "Louvre Museum, Paris"
            },
            "date_added": "2024-06-12"
        }
    ]
    return sample_data

def filter_results(results, selected_companies=None, sentiment_filter="All", year_range=None):
    """Filter the results based on company, sentiment, and year range."""
    if not results:
        return []
    
    filtered = results.copy()
    
    # Filter by company
    if selected_companies:
        filtered = [r for r in filtered if r.get('ticker') in selected_companies]
    
    # Filter by sentiment
    if sentiment_filter and sentiment_filter != "All":
        filtered = [r for r in filtered if r.get('climate_sentiment') == sentiment_filter]
    
    # Filter by year range
    if year_range:
        filtered = [
            r for r in filtered
            if r.get('year') and str(r['year']).isdigit() and 
            year_range[0] <= int(r['year']) <= year_range[1]
        ]
    
    return filtered