import streamlit as st
import pandas as pd
from helpers import load_data

# Page config
st.set_page_config(
    page_title="ArtRag",
    page_icon="🎨",
    layout="wide"
)

def main():
    # Banner image with horizontal crop and limited height using HTML
    st.markdown(
        """
        <div style="position:relative; overflow:hidden; border-radius:10px; margin-bottom:10px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
                 style="width:100%; object-fit: cover; max-height: 250px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Welcome text in a colored box
    st.markdown(
        """
        <div style='background-color:#f0f0f5;padding:20px;border-radius:10px;margin-top:10px'>
            <h1 style='color:#4B0082;'>🎨 ArtRag: Ask About Art</h1>
            <p style='font-size:16px'>
                Welcome to <b>ArtRag</b>, your playful tool for exploring artworks and artists.<br>
                Load a collection of sample artworks or a saved embedding index, and then ask your questions.<br>
                Whether you're curious about a painting, an artist, or an art movement, we'll try to find the best answer for you.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar buttons
    with st.sidebar:
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                data = load_data()
                df = pd.DataFrame(data)
                st.session_state.data = df
                st.session_state.index_loaded = False
                st.success("Sample data loaded!")

        if st.button("Load Embedding Index"):
            with st.spinner("Loading embedding index..."):
                # TODO: Replace this with your actual index loading logic
                st.session_state.index_loaded = True
                st.success("Embedding index loaded!")

    # Show sample data if loaded
    if st.session_state.get("data") is not None:
        st.subheader("🖼️ Sample Artworks")
        st.dataframe(st.session_state.data)

    # User query input
    st.subheader("💬 Ask about the artworks or artists")
    user_query = st.text_input("Enter your question:")

    if user_query:
        st.write(f"You asked: '{user_query}'")
        # Placeholder: implement actual query logic here
        st.info("Search functionality coming soon!")

    if not st.session_state.get("data") and not st.session_state.get("index_loaded"):
        st.info("Please load sample data or an embedding index using the sidebar.")

if __name__ == "__main__":
    main()
