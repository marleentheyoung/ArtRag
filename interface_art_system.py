import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from green_investment_rag import ArtRAG  # Import your main class
from helpers import load_data, filter_results

# Page config
st.set_page_config(
    page_title="Green Investment Analyzer",
    page_icon="🌱",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def main():
    st.title("🌱 Green Investment Analyzer")
    st.subheader("Extract climate investment insights from earnings calls")
    
    # Sidebar for data loading
    with st.sidebar:
        # Load prebuilt index
        if st.button("Load Prebuilt Index"):
            with st.spinner("Loading prebuilt index..."):
                rag = GreenInvestmentRAG()
                rag.load_snippets('climate_snippets.json')
                rag.load_index('climate_index.faiss')
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
                st.success("Loaded index and snippets from disk.")

        # Build and save index (only when needed)
        if st.button("Build and Save Embedding Index"):
            if not st.session_state.data_loaded:
                st.warning("Please load data first!")
            else:
                with st.spinner("Building and saving embedding index..."):
                    rag = st.session_state.rag_system
                    rag.build_embedding_index()
                    rag.save_index('climate_index.faiss')
                    rag.save_snippets('climate_snippets.json')
                    st.success("Embedding index built and saved.")

        st.header("📊 Data Management")
        
        if st.button("Load Sample Data"):
            with st.spinner("Loading data and building index..."):
                data = load_data()
                rag = GreenInvestmentRAG()
                rag.load_earnings_data(data)
                rag.build_embedding_index()
                
                st.session_state.rag_system = rag
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(rag.snippets)} snippets!")
        
        # File upload option
        uploaded_files = st.file_uploader("Upload JSON files", type=['json'], accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                try:
                    combined_data = []
                    for file in uploaded_files:
                        file_data = json.load(file)
                        
                        # If each file is a list of documents
                        if isinstance(file_data, list):
                            combined_data.extend(file_data)
                        # If each file is a single document
                        else:
                            combined_data.append(file_data)
                    
                    rag = GreenInvestmentRAG()
                    rag.load_earnings_data(combined_data)
                    rag.build_embedding_index()
                    
                    st.session_state.rag_system = rag
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(rag.snippets)} snippets from {len(uploaded_files)} file(s)!")
                
                except Exception as e:
                    st.error(f"Error loading files: {e}")

    if not st.session_state.data_loaded:
        st.warning("Please load data first using the sidebar.")
        return
    
    rag = st.session_state.rag_system
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📈 Analytics", "🏢 Company Compare", "📊 Trends"])
    
    with tab1:
        st.header("Investment Search")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_type = st.radio("Search Type", ["Category", "Custom Query", "Semantic Search"])
            
            if search_type == "Category":
                category = st.selectbox("Select Investment Category", 
                                      list(rag.investment_categories.keys()))
            elif search_type == "Semantic Search":
                query = st.text_input("Enter your semantic query")
                if st.button("Semantic Search") and query:
                    results = rag.query_embedding_index(query, top_k=50)

                    # # Optionally apply company/sentiment/year filters (reuse filter_results())
                    # filtered_results = filter_results(
                    #     results,
                    #     selected_companies=selected_companies if selected_companies else None,
                    #     sentiment_filter=sentiment_filter,
                    #     year_range=year_range
                    # )
                    display_results(results[:20])

            else:
                query = st.text_input("Enter your search query")
        
        with col2:
            st.subheader("Filters")
            
            # Company filter
            all_companies = list(set([s.ticker for s in rag.snippets]))
            selected_companies = st.multiselect("Filter by Company", all_companies)
            
            # Sentiment filter
            sentiment_filter = st.selectbox("Filter by Sentiment", 
                                          ["All", "opportunity", "neutral", "risk"])
            
            # Year range
            years = [int(s.year) for s in rag.snippets if s.year and str(s.year).isdigit()]
            if years:
                year_range = st.slider("Year Range", 
                                     min_value=min(years), 
                                     max_value=max(years),
                                     value=(min(years), max(years)))
            else:
                year_range = None
        
        # Search button and results (moved outside columns)
        if search_type == "Category":
            if st.button("Search by Category"):
                results = rag.search_by_category(category, top_k=50)  # Get more results before filtering
                # Apply filters
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                display_results(filtered_results[:10])  # Show top 10 after filtering
        else:
            if st.button("Search") and query:
                results = rag.search_by_query(query, top_k=50)  # Get more results before filtering
                # Apply filters
                filtered_results = filter_results(
                    results, 
                    selected_companies=selected_companies if selected_companies else None,
                    sentiment_filter=sentiment_filter,
                    year_range=year_range
                )
                display_results(filtered_results[:10])  # Show top 10 after filtering
    
    with tab2:
        st.header("Investment Analytics")
        
        # Category overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Analysis")
            selected_category = st.selectbox("Analyze Category", 
                                           list(rag.investment_categories.keys()),
                                           key="analytics_category")
            
            if st.button("Generate Summary"):
                summary = rag.get_investment_summary(selected_category)
                
                st.metric("Total Mentions", summary['total_mentions'])
                st.metric("Companies Mentioned", summary['companies_mentioned'])
                
                # Company breakdown chart
                if summary['company_breakdown']:
                    df = pd.DataFrame(list(summary['company_breakdown'].items()), 
                                    columns=['Company', 'Mentions'])
                    fig = px.bar(df, x='Company', y='Mentions', 
                               title=f"{selected_category.title()} Mentions by Company")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            
            # Overall sentiment analysis
            all_sentiments = [s.climate_sentiment for s in rag.snippets if s.climate_sentiment]
            if all_sentiments:
                sentiment_counts = pd.Series(all_sentiments).value_counts()
                
                fig = px.pie(values=sentiment_counts.values, 
                            names=sentiment_counts.index,
                            title="Overall Climate Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
    
    with tab3:
        st.header("Company Comparison")
        
        # Select companies to compare
        companies = st.multiselect("Select Companies to Compare", 
                                 list(set([s.ticker for s in rag.snippets])))
        
        if len(companies) >= 2:
            category_comp = st.selectbox("Compare by Category", 
                                       list(rag.investment_categories.keys()),
                                       key="compare_category")
            
            if st.button("Compare Companies"):
                comparison_data = []
                for company in companies:
                    results = rag.search_by_category(category_comp, top_k=50)
                    company_results = rag.filter_by_company(results, [company])
                    
                    comparison_data.append({
                        'Company': company,
                        'Mentions': len(company_results),
                        'Avg Score': sum(r['score'] for r in company_results) / len(company_results) if company_results else 0
                    })
                
                df = pd.DataFrame(comparison_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(df, x='Company', y='Mentions', 
                               title=f"{category_comp.title()} Mentions Comparison")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df, x='Company', y='Avg Score', 
                               title="Average Relevance Score")
                    st.plotly_chart(fig, use_container_width=True)
        elif companies:
            st.info("Please select at least 2 companies to compare.")
    
    with tab4:
        st.header("Investment Trends")
        
        # Time series analysis
        trend_category = st.selectbox("Select Category for Trend Analysis", 
                                    list(rag.investment_categories.keys()),
                                    key="trend_category")
            
            
        if st.button("Generate Trend Analysis"):
            results = rag.search_by_category(trend_category, top_k=50000)
            
            if results:
                # Create time series data
                trend_data = []
                for result in results:
                    # Convert year to int for proper sorting
                    year = int(result['year']) if result['year'] and str(result['year']).isdigit() else None
                    if year:
                        trend_data.append({
                            'Date': result['date'],
                            'Year': year,
                            'Quarter': result['quarter'],
                            'Score': result['score'],
                            'Company': result['ticker']
                        })
                
                df = pd.DataFrame(trend_data)
                
                if not df.empty:
                    # Aggregate by year-quarter
                    df['Year-Quarter'] = df['Year'].astype(str) + '-' + df['Quarter']
                    quarterly_mentions = df.groupby('Year-Quarter').size().reset_index(name='Mentions')
                    
                    # Sort by year-quarter for proper ordering
                    quarterly_mentions = quarterly_mentions.sort_values('Year-Quarter')
                    
                    fig = px.line(quarterly_mentions, x='Year-Quarter', y='Mentions',
                                title=f"{trend_category.title()} Investment Mentions Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Company trend comparison
                    company_trends = df.groupby(['Year-Quarter', 'Company']).size().reset_index(name='Mentions')
                    company_trends = company_trends.sort_values('Year-Quarter')
                    
                    fig2 = px.line(company_trends, x='Year-Quarter', y='Mentions', 
                                 color='Company', 
                                 title=f"{trend_category.title()} Mentions by Company")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No trend data available for the selected category.")
            else:
                st.info("No results found for the selected category.")

def display_results(results):
    """Display search results in a nice format"""
    if not results:
        st.warning("No results found.")
        return
    
    st.subheader(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        with st.expander(f"Result {i+1}: {result['company']} ({result['score']:.3f})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Speaker:** {result['speaker']} ({result['profession']})")
            
            with col2:
                st.metric("Relevance Score", f"{result['score']:.3f}")
                st.write(f"**Date:** {result['date']}")
                st.write(f"**Quarter:** {result['quarter']} {result['year']}")
                if result['climate_sentiment']:
                    sentiment_color = {
                        'opportunity': '🟢',
                        'neutral': '🟡', 
                        'risk': '🔴'
                    }
                    st.write(f"**Sentiment:** {sentiment_color.get(result['climate_sentiment'], '')} {result['climate_sentiment']}")

if __name__ == "__main__":
    main()