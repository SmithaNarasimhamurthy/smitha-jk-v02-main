# Streamlit dashboard for textile supplier risk identification results
# This interactive dashboard provides comprehensive exploration of risk analysis results
# including supplier profiles, temporal trends, and detailed article analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('../')
sys.path.append('../src')

# Page configuration
st.set_page_config(
    page_title="Textile Supplier Risk Analysis Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff8800;
        font-weight: bold;
    }
    .risk-low {
        color: #44ff44;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class RiskDashboard:
    """Main dashboard class for textile supplier risk analysis"""
    
    def __init__(self):
        self.risk_categories = [
            'geopolitical_regulatory',
            'agricultural_environmental', 
            'financial_operational',
            'supply_chain_logistics',
            'market_competitive'
        ]
        
        self.load_data()
    
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Load main results
            if os.path.exists('results/article_classifications.csv'):
                self.articles_data = pd.read_csv('results/article_classifications.csv')
                self.articles_data['published_date'] = pd.to_datetime(self.articles_data['published_date'])
            else:
                self.articles_data = None
            
            # Load supplier analysis
            if os.path.exists('results/supplier_risk_analysis.csv'):
                self.supplier_analysis = pd.read_csv('results/supplier_risk_analysis.csv', index_col=0)
            else:
                self.supplier_analysis = None
            
            # Load temporal trends
            if os.path.exists('results/temporal_risk_trends.csv'):
                self.temporal_trends = pd.read_csv('results/temporal_risk_trends.csv', index_col=0)
            else:
                self.temporal_trends = None
            
            # Load summaries
            self.load_json_summaries()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please run the main pipeline first to generate results.")
    
    def load_json_summaries(self):
        """Load JSON summary files"""
        summary_files = {
            'data_summary': 'results/data_summary.json',
            'risk_category_summary': 'results/risk_category_summary.json',
            'training_summary': 'results/training_summary.json',
            'pipeline_summary': 'results/pipeline_summary.json'
        }
        
        for key, filepath in summary_files.items():
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        setattr(self, key, json.load(f))
                else:
                    setattr(self, key, {})
            except Exception:
                setattr(self, key, {})
    
    def render_header(self):
        """Render the main dashboard header"""
        st.markdown('<h1 class="main-header">🏭 Textile Supplier Risk Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the comprehensive risk analysis dashboard for textile dye suppliers. 
        This interactive tool provides insights into potential risks across 10 major suppliers 
        based on news article analysis from 2023-2024.
        """)
        
        # Display key metrics if data is available
        if hasattr(self, 'pipeline_summary') and self.pipeline_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Articles", 
                         self.pipeline_summary.get('total_articles_processed', 'N/A'))
            
            with col2:
                st.metric("Suppliers Identified", 
                         self.pipeline_summary.get('unique_suppliers_identified', 'N/A'))
            
            with col3:
                st.metric("Models Trained", 
                         self.pipeline_summary.get('models_trained', 'N/A'))
            
            with col4:
                if 'execution_time' in self.pipeline_summary:
                    st.metric("Pipeline Runtime", 
                             self.pipeline_summary['execution_time'])
    
    def render_overview_page(self):
        """Render the overview page"""
        st.header("📊 Risk Analysis Overview")
        
        if self.articles_data is None:
            st.warning("No data available. Please run the main pipeline first.")
            return
        
        # Risk distribution by supplier
        st.subheader("Risk Distribution by Supplier")
        
        supplier_risk_totals = self.articles_data.groupby('supplier')[self.risk_categories].mean()
        supplier_risk_totals['total_risk'] = supplier_risk_totals.sum(axis=1)
        supplier_risk_totals = supplier_risk_totals.sort_values('total_risk', ascending=True)
        
        fig_bar = px.bar(
            x=supplier_risk_totals['total_risk'],
            y=supplier_risk_totals.index,
            orientation='h',
            title="Total Risk Score by Supplier",
            labels={'x': 'Total Risk Score', 'y': 'Supplier'}
        )
        fig_bar.update_layout(height=600)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Risk category heatmap
        st.subheader("Risk Category Heatmap")
        
        risk_matrix = self.articles_data.groupby('supplier')[self.risk_categories].mean().fillna(0)
        
        fig_heatmap = px.imshow(
            risk_matrix.values,
            x=[cat.replace('_', ' ').title() for cat in self.risk_categories],
            y=risk_matrix.index,
            aspect="auto",
            color_continuous_scale="RdYlBu_r",
            title="Risk Intensity by Supplier and Category"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Recent high-risk articles
        st.subheader("Recent High-Risk Articles")
        
        recent_articles = self.articles_data.nlargest(10, 'total_risk_score')[
            ['title', 'supplier', 'published_date', 'total_risk_score', 'source']
        ]
        st.dataframe(recent_articles, use_container_width=True)
    
    def render_supplier_analysis_page(self):
        """Render the supplier analysis page"""
        st.header("🏭 Supplier Risk Profiles")
        
        if self.articles_data is None:
            st.warning("No data available. Please run the main pipeline first.")
            return
        
        # Supplier selection
        suppliers = self.articles_data['supplier'].dropna().unique()
        selected_suppliers = st.multiselect(
            "Select suppliers to analyze:",
            suppliers,
            default=suppliers[:3] if len(suppliers) >= 3 else suppliers
        )
        
        if not selected_suppliers:
            st.info("Please select at least one supplier to analyze.")
            return
        
        # Radar chart comparison
        st.subheader("Risk Profile Comparison")
        
        fig_radar = go.Figure()
        
        for supplier in selected_suppliers:
            supplier_data = self.articles_data[self.articles_data['supplier'] == supplier]
            risk_scores = supplier_data[self.risk_categories].mean()
            
            categories = [cat.replace('_', ' ').title() for cat in self.risk_categories]
            values = risk_scores.tolist()
            values += [values[0]]  # Close the radar chart
            categories += [categories[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=supplier,
                line=dict(width=2)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(self.articles_data[self.risk_categories].max())]
                )
            ),
            showlegend=True,
            title="Supplier Risk Profile Comparison",
            height=600
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed supplier metrics
        st.subheader("Detailed Supplier Metrics")
        
        for supplier in selected_suppliers:
            with st.expander(f"📈 {supplier} - Detailed Analysis"):
                supplier_data = self.articles_data[self.articles_data['supplier'] == supplier]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Articles", len(supplier_data))
                    st.metric("Avg Risk Score", f"{supplier_data['total_risk_score'].mean():.3f}")
                
                with col2:
                    positive_risk_pct = (supplier_data['risk_direction'] == 'positive').mean() * 100
                    st.metric("Increased Risk %", f"{positive_risk_pct:.1f}%")
                    st.metric("Latest Article", supplier_data['published_date'].max().strftime('%Y-%m-%d'))
                
                with col3:
                    dominant_risk = supplier_data[self.risk_categories].mean().idxmax()
                    st.metric("Dominant Risk", dominant_risk.replace('_', ' ').title())
                    st.metric("Risk Rank", f"#{supplier_data['risk_rank'].min():.0f}")
                
                # Risk breakdown chart
                risk_breakdown = supplier_data[self.risk_categories].mean()
                fig_breakdown = px.bar(
                    x=[cat.replace('_', ' ').title() for cat in risk_breakdown.index],
                    y=risk_breakdown.values,
                    title=f"Risk Breakdown - {supplier}"
                )
                st.plotly_chart(fig_breakdown, use_container_width=True)
    
    def render_temporal_analysis_page(self):
        """Render the temporal analysis page"""
        st.header("📅 Temporal Risk Analysis")
        
        if self.articles_data is None:
            st.warning("No data available. Please run the main pipeline first.")
            return
        
        # Time period selection
        st.subheader("Risk Trends Over Time")
        
        # Prepare monthly data
        monthly_data = self.articles_data.copy()
        monthly_data['month'] = monthly_data['published_date'].dt.to_period('M')
        monthly_risk = monthly_data.groupby('month')[self.risk_categories + ['total_risk_score']].mean()
        
        # Time series chart
        fig_time = go.Figure()
        
        for category in self.risk_categories:
            fig_time.add_trace(go.Scatter(
                x=monthly_risk.index.astype(str),
                y=monthly_risk[category],
                mode='lines+markers',
                name=category.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        fig_time.update_layout(
            title="Risk Category Trends Over Time",
            xaxis_title="Month",
            yaxis_title="Average Risk Score",
            height=500
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Risk events timeline
        st.subheader("High-Risk Events Timeline")
        
        high_risk_threshold = self.articles_data['total_risk_score'].quantile(0.9)
        high_risk_events = self.articles_data[
            self.articles_data['total_risk_score'] > high_risk_threshold
        ].sort_values('published_date')
        
        if len(high_risk_events) > 0:
            fig_timeline = px.scatter(
                high_risk_events,
                x='published_date',
                y='total_risk_score',
                color='supplier',
                size='total_risk_score',
                hover_data=['title'],
                title="High-Risk Events Timeline"
            )
            fig_timeline.update_layout(height=500)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show recent events table
            st.subheader("Recent High-Risk Events")
            recent_events = high_risk_events.head(10)[
                ['published_date', 'title', 'supplier', 'total_risk_score']
            ]
            st.dataframe(recent_events, use_container_width=True)
    
    def render_risk_categories_page(self):
        """Render the risk categories analysis page"""
        st.header("🏷️ Risk Categories Analysis")
        
        if self.articles_data is None:
            st.warning("No data available. Please run the main pipeline first.")
            return
        
        # Category selection
        selected_category = st.selectbox(
            "Select risk category to analyze:",
            self.risk_categories,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Category overview
        col1, col2, col3 = st.columns(3)
        
        category_data = self.articles_data[selected_category]
        
        with col1:
            st.metric("Average Score", f"{category_data.mean():.3f}")
        
        with col2:
            st.metric("High Risk Articles", 
                     len(category_data[category_data > category_data.quantile(0.8)]))
        
        with col3:
            st.metric("Max Score", f"{category_data.max():.3f}")
        
        # Distribution chart
        st.subheader(f"Distribution of {selected_category.replace('_', ' ').title()} Risk Scores")
        
        fig_dist = px.histogram(
            self.articles_data,
            x=selected_category,
            nbins=30,
            title=f"Distribution of {selected_category.replace('_', ' ').title()} Risk Scores"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Top suppliers for this category
        st.subheader(f"Top Suppliers by {selected_category.replace('_', ' ').title()} Risk")
        
        top_suppliers = self.articles_data.groupby('supplier')[selected_category].agg([
            'mean', 'count', 'std'
        ]).round(3).sort_values('mean', ascending=False).head(10)
        
        st.dataframe(top_suppliers, use_container_width=True)
        
        # Articles with highest risk in this category
        st.subheader(f"Highest Risk Articles - {selected_category.replace('_', ' ').title()}")
        
        top_articles = self.articles_data.nlargest(10, selected_category)[
            ['title', 'supplier', 'published_date', selected_category, 'source']
        ]
        st.dataframe(top_articles, use_container_width=True)
    
    def render_search_analysis_page(self):
        """Render the search and analysis page"""
        st.header("🔍 Search & Analysis")
        
        if self.articles_data is None:
            st.warning("No data available. Please run the main pipeline first.")
            return
        
        # Search functionality
        st.subheader("Article Search")
        
        search_term = st.text_input("Search articles by title or content:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            supplier_filter = st.multiselect(
                "Filter by supplier:",
                self.articles_data['supplier'].dropna().unique()
            )
        
        with col2:
            risk_threshold = st.slider(
                "Minimum risk score:",
                min_value=0.0,
                max_value=float(self.articles_data['total_risk_score'].max()),
                value=0.0,
                step=0.1
            )
        
        # Apply filters
        filtered_data = self.articles_data.copy()
        
        if search_term:
            filtered_data = filtered_data[
                filtered_data['title'].str.contains(search_term, case=False, na=False)
            ]
        
        if supplier_filter:
            filtered_data = filtered_data[
                filtered_data['supplier'].isin(supplier_filter)
            ]
        
        filtered_data = filtered_data[
            filtered_data['total_risk_score'] >= risk_threshold
        ]
        
        st.info(f"Found {len(filtered_data)} articles matching your criteria")
        
        if len(filtered_data) > 0:
            # Display results
            display_columns = [
                'title', 'supplier', 'published_date', 'total_risk_score', 
                'risk_direction', 'source'
            ]
            
            st.dataframe(
                filtered_data[display_columns].sort_values('total_risk_score', ascending=False),
                use_container_width=True
            )
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download filtered results as CSV",
                data=csv,
                file_name=f"risk_analysis_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def render_about_page(self):
        """Render the about page"""
        st.header("ℹ️ About This Analysis")
        
        st.markdown("""
        ## Textile Supplier Risk Identification System
        
        This comprehensive risk analysis system analyzes news articles related to 10 major textile dye suppliers 
        to identify potential supply chain risks across five key categories.
        
        ### Risk Categories
        
        1. **Geopolitical and Regulatory Risks**
           - Trade wars, tariffs, and sanctions
           - Regulatory changes and compliance issues
           - Political instability and policy changes
        
        2. **Agricultural and Environmental Risks**
           - Climate change impacts and natural disasters
           - Environmental regulations and sustainability issues
           - Water scarcity and energy concerns
        
        3. **Financial and Operational Risks**
           - Financial instability and bankruptcy risks
           - Labor strikes and operational disruptions
           - Production capacity and manufacturing issues
        
        4. **Supply Chain and Logistics Risks**
           - Transportation and shipping disruptions
           - Inventory management challenges
           - Fuel price fluctuations and logistics costs
        
        5. **Market and Competitive Risks**
           - Market demand fluctuations
           - Competitive pressures and price volatility
           - Consumer behavior changes and fashion trends
        
        ### Target Suppliers
        
        The analysis focuses on these 10 major textile suppliers:
        
        1. Welspun Living Limited
        2. Teejay Lanka PLC
        3. Arvind Limited
        4. Caleres, Inc.
        5. Interloop Limited
        6. Kitex Garments Limited
        7. ThredUp Inc.
        8. G-III Apparel Group, Ltd.
        9. Mint Velvet
        10. White Stuff Limited
        
        ### Methodology
        
        The system employs advanced NLP and machine learning techniques including:
        - Text preprocessing and feature extraction
        - Sentiment analysis and entity recognition
        - Multi-label classification for risk categories
        - Temporal trend analysis and pattern recognition
        - Interactive visualization and reporting
        
        ### Data Sources
        
        Analysis is based on news articles from 2023-2024 covering these suppliers, 
        sourced from reputable news outlets and industry publications.
        """)
        
        # Technical details if summaries are available
        if hasattr(self, 'training_summary') and self.training_summary:
            st.subheader("Technical Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Training Samples: {self.training_summary.get('training_samples', 'N/A')}")
                st.info(f"Feature Count: {self.training_summary.get('features_count', 'N/A')}")
            
            with col2:
                if self.training_summary.get('risk_category_results'):
                    best_model = max(
                        self.training_summary['risk_category_results'].items(),
                        key=lambda x: x[1].get('overall_f1', 0)
                    )
                    st.info(f"Best Model: {best_model[0]}")
                    st.info(f"F1 Score: {best_model[1].get('overall_f1', 'N/A'):.3f}")

def main():
    """Main function to run the Streamlit dashboard"""
    
    # Initialize dashboard
    dashboard = RiskDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "Overview",
            "Supplier Analysis", 
            "Temporal Analysis",
            "Risk Categories",
            "Search & Analysis",
            "About"
        ]
    )
    
    # Render selected page
    if page == "Overview":
        dashboard.render_overview_page()
    elif page == "Supplier Analysis":
        dashboard.render_supplier_analysis_page()
    elif page == "Temporal Analysis":
        dashboard.render_temporal_analysis_page()
    elif page == "Risk Categories":
        dashboard.render_risk_categories_page()
    elif page == "Search & Analysis":
        dashboard.render_search_analysis_page()
    elif page == "About":
        dashboard.render_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Textile Supplier Risk Analysis**")
    st.sidebar.markdown("Data Science Solution for Supply Chain Risk Management")

if __name__ == "__main__":
    main() 