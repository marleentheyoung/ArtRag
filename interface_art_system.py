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
    st.title("🎨 ArtRag Sample Data Viewer")
    st.write("Load and view sample artwork data.")

    # Sidebar button to load sample data
    with st.sidebar:
        if st.button("Load Sample Data"):
            with st.spinner("Loading data..."):
                data = load_data()
                # Convert data to a DataFrame for display
                df = pd.DataFrame(data)
                st.session_state.data = df
                st.success("Data loaded!")

    # Show data if loaded
    if "data" in st.session_state:
        st.subheader("Sample Artworks")
        st.dataframe(st.session_state.data)
    else:
        st.info("Please load data using the button in the sidebar.")

if __name__ == "__main__":
    main()
