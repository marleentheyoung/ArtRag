import pandas as pd
import requests
import json
import re
from typing import Optional, Dict, Any


class ArtistWikiRAG:
    def __init__(self, csv_file_path: str, ollama_url: str = "http://localhost:11434"):
        self.csv_file_path = csv_file_path
        self.ollama_url = ollama_url
        self.artist_df = None
        self.load_artist_data()

    def load_artist_data(self):
        try:
            self.artist_df = pd.read_csv(self.csv_file_path)
            self.artist_df['n.artist_name'] = self.artist_df['n.firstname'] + ' ' + self.artist_df['n.lastname']
            print(f"Loaded {len(self.artist_df)} artists from CSV")
        except FileNotFoundError:
            print(f"CSV file not found: {self.csv_file_path}")
            print("Please ensure your CSV has columns: 'n.firstname', 'n.lastname', 'n.wikidata'")

    def call_ollama_llm(self, query: str, model: str = "llama3:instruct") -> Optional[str]:
        prompt = f"""
        Respond ONLY with the artist name this query is about. 
        If no specific artist is mentioned, return "NO_ARTIST".

        Example: Tell me about Monet.
        Response: Claude Monet

        Query: "{query}"

        Artist name:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                artist_name = result['response'].strip()

                if artist_name.upper() == "NO_ARTIST":
                    return None

                artist_name = re.sub(r'^["\']|["\']$', '', artist_name)
                return artist_name
            else:
                print(f"Error calling Ollama: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None
        
    # def call_ollama_llm(self, query: str, model: str = "llama3:instruct") -> Optional[str]:
    #     prompt = f"""
    #     Respond ONLY with the painting name of the relevant painting in this query. 
    #     If no painting is relevant artist is mentioned, return "NO_ARTIST".

    #     Example: Tell me about Monet.
    #     Response: Claude Monet

    #     Query: "{query}"

    #     Artist name:"""

    #     try:
    #         response = requests.post(
    #             f"{self.ollama_url}/api/generate",
    #             json={

    def get_wiki_id_from_csv(self, artist_name: str) -> Optional[str]:
        if self.artist_df is None:
            return None

        exact_match = self.artist_df[self.artist_df['n.artist_name'].str.lower() == artist_name.lower()]

        if not exact_match.empty:
            return exact_match.iloc[0]['n.wikidata']

        partial_match = self.artist_df[self.artist_df['n.artist_name'].str.lower().str.contains(artist_name.lower(), na=False)]

        if not partial_match.empty:
            return partial_match.iloc[0]['n.wikidata']

        return None

    def get_wikipedia_content(self, wikidata: str) -> Optional[Dict[str, Any]]:
        try:
            # Step 1: Resolve Wikidata ID to Wikipedia title
            wikidata_api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata}.json"
            wikidata_response = requests.get(wikidata_api_url, timeout=10)
            if wikidata_response.status_code != 200:
                print(f"Error resolving Wikidata ID {wikidata}: {wikidata_response.status_code}")
                return None

            wikidata_data = wikidata_response.json()
            entities = wikidata_data.get('entities', {})
            entity = entities.get(wikidata, {})

            sitelinks = entity.get('sitelinks', {})
            enwiki = sitelinks.get('enwiki')
            if not enwiki:
                print(f"No English Wikipedia page found for Wikidata ID {wikidata}")
                return None

            wikipedia_title = enwiki.get('title')
            print(f"Resolved Wikidata ID {wikidata} to Wikipedia title '{wikipedia_title}'")

            # Step 2: Use the MediaWiki API to get the full text in plain format
            parse_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": wikipedia_title,
                "format": "json",
                "prop": "text"
            }
            parse_response = requests.get(parse_url, params=params, timeout=10)

            if parse_response.status_code != 200:
                print(f"MediaWiki parse API error: {parse_response.status_code}")
                return None

            parse_data = parse_response.json()
            html_content = parse_data.get('parse', {}).get('text', {}).get('*', '')

            # Optionally strip HTML tags to get plain text (basic example)
            text_content = re.sub(r'<[^>]+>', '', html_content)

            return {
                'title': wikipedia_title,
                'summary': '',  # Could still get the summary if desired
                'full_content': text_content,
                'url': f"https://en.wikipedia.org/wiki/{wikipedia_title.replace(' ', '_')}",
                'wikidata': wikidata
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching Wikipedia content: {e}")
            return None


    def process_with_rag(self, wiki_content: Dict[str, Any], original_query: str) -> str:
        content = wiki_content['full_content']
        query_keywords = re.findall(r'\b\w+\b', original_query.lower())
        sentences = content.split('.')
        relevant_sentences = [s.strip() for s in sentences if any(k in s.lower() for k in query_keywords)]
        context = '. '.join(relevant_sentences[:5])

        rag_prompt = f"""
        Based on the following context about {wiki_content['title']}, answer this question: {original_query}

        Context: {context}

        Answer:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": rag_prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return f"Error generating RAG response: {response.status_code}"

        except requests.exceptions.RequestException as e:
            return f"Error connecting to LLM for RAG: {e}"

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        results = {
            'query': query,
            'artist_name': None,
            'wikidata': None,
            'wiki_content': None,
            'rag_response': None,
            'error': None
        }

        try:
            print("Step 1: Extracting artist name from query...")
            artist_name = self.call_ollama_llm(query)
            # painting_name = self.call_ollama_llm_painting(query)
            results['artist_name'] = artist_name

            if not artist_name:
                results['error'] = "No artist name found in query"
                return results

            print(f"Found artist: {artist_name}")

            print("Step 2: Looking up Wikipedia ID...")
            wikidata = self.get_wiki_id_from_csv(artist_name)

            results['wikidata'] = wikidata

            if not wikidata:
                results['error'] = f"No Wikipedia ID found for artist: {artist_name}"
                return results

            print(f"Found Wikipedia ID: {wikidata}")

            print("Step 3: Fetching Wikipedia content...")
            wiki_content = self.get_wikipedia_content(wikidata)
            results['wiki_content'] = wiki_content

            if not wiki_content:
                results['error'] = "Failed to fetch Wikipedia content"
                return results

            print(f"Retrieved content for: {wiki_content['title']}")

            print("Step 4: Processing with RAG...")
            rag_response = self.process_with_rag(wiki_content, query)
            results['rag_response'] = rag_response

            print("Pipeline completed successfully!")

        except Exception as e:
            results['error'] = f"Pipeline error: {str(e)}"

        return results


# Example usage
if __name__ == "__main__":
    query = "Tell me about the early life of Picasso"

    rag_system = ArtistWikiRAG(
        csv_file_path="data/artists1000_cleaned.csv",
        ollama_url="http://localhost:11434"
    )

    results = rag_system.run_pipeline(query)

    print("\n" + "=" * 50)
    print("PIPELINE RESULTS")
    print("=" * 50)
    print(f"Query: {results['query']}")
    print(f"Artist Name: {results['artist_name']}")
    print(f"Wiki ID: {results['wikidata']}")

    if results['error']:
        print(f"Error: {results['error']}")
    elif results['rag_response']:
        print(f"\nRAG Response:\n{results['rag_response']}")

    if results['wiki_content']:
        with open(f"{results['artist_name']}_wiki_content.txt", "w", encoding="utf-8") as f:
            f.write(results['wiki_content']['full_content'])
        print(f"\nFull Wikipedia content saved to {results['artist_name']}_wiki_content.txt")
