import streamlit as st
import pandas as pd
from helpers import load_data
from wiki_api import ArtistWikiRAG

# Page config
st.set_page_config(
    page_title="ArtRag",
    page_icon="🎨",
    layout="wide"
)

# Initialize the ArtistWikiRAG system once, outside main() to cache
@st.cache_resource
def init_rag_system():
    return ArtistWikiRAG(
        csv_file_path="data/artists1000_cleaned.csv",
        ollama_url="http://localhost:11434"
    )

def main():
    rag_system = init_rag_system()

    # Banner image
    st.markdown(
        """
        <div style="position:relative; overflow:hidden; border-radius:10px; margin-bottom:10px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                 style="width:100%; object-fit: cover; max-height: 250px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Welcome text
    st.markdown(
        """
        <div style='background-color:#f0f0f5;padding:20px;border-radius:10px;margin-top:10px'>
            <h1 style='color:#4B0082;'>🎨 ArtRag: Ask About Art</h1>
            <p>Welcome to <b>ArtRag</b>. Ask about an artist and we'll fetch Wikipedia info for you!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar (optional)
    with st.sidebar:
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                data = load_data()
                df = pd.DataFrame(data)
                st.session_state.data = df
                st.success("Sample data loaded!")

    # Show sample data (optional)
    if st.session_state.get("data") is not None:
        st.subheader("🖼️ Sample Artworks")
        st.dataframe(st.session_state.data)

    # User query input
    st.subheader("💬 Ask about an artist")
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Searching Wikipedia..."):
            results = rag_system.run_pipeline(user_query)

        if results['error']:
            st.error(results['error'])
        else:
            st.success(f"Here's what we found about {results['artist_name']}:")

            st.write(f"**Artist Name:** {results['artist_name']}")
            st.write(f"**Wikipedia Page:** {results['wiki_content']['url']}")
            st.write(f"**Summary:** {results['wiki_content']['summary']}")
            
            with st.expander("See Full Wikipedia Content"):
                st.text(results['wiki_content']['full_content'])
            
            st.subheader("🤖 AI Response")
            st.write(results['rag_response'])

    if not st.session_state.get("data"):
        st.info("You can optionally load sample data using the sidebar.")

if __name__ == "__main__":
    main()
