# Visualization-only Streamlit dashboard for textile supplier risk analysis results
# This application loads pre-generated results and presents them through interactive visualizations
# No data processing or model training is performed - purely visualization focused

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Risk Analysis Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #48cab2, #2dd4bf);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E86AB 0%, #A23B72 100%);
    }
</style>
""", unsafe_allow_html=True)

class VisualizationDashboard:
    """Comprehensive visualization dashboard for risk analysis results"""
    
    def __init__(self):
        self.risk_categories = [
            'geopolitical_regulatory',
            'agricultural_environmental', 
            'financial_operational',
            'supply_chain_logistics',
            'market_competitive'
        ]
        
        self.risk_colors = {
            'geopolitical_regulatory': '#FF6B6B',
            'agricultural_environmental': '#4ECDC4',
            'financial_operational': '#45B7D1',
            'supply_chain_logistics': '#96CEB4',
            'market_competitive': '#FFEAA7'
        }
        
        self.suppliers = [
            "Welspun Living Limited", "Teejay Lanka PLC", "Arvind Limited",
            "Caleres, Inc.", "Interloop Limited", "Kitex Garments Limited",
            "ThredUp Inc.", "G-III Apparel Group, Ltd.", "Mint Velvet", "White Stuff Limited"
        ]
        
        self.load_all_data()
    
    def load_all_data(self):
        """Load all available result files"""
        try:
            # Load main article classifications
            self.articles_data = self._safe_load_csv('results/article_classifications.csv')
            if self.articles_data is not None:
                self.articles_data['published_date'] = pd.to_datetime(self.articles_data['published_date'])
            
            # Load supplier risk analysis
            self.supplier_analysis = self._safe_load_csv('results/supplier_risk_analysis.csv', index_col=0)
            
            # Load temporal trends
            self.temporal_trends = self._safe_load_csv('results/temporal_risk_trends.csv', index_col=0)
            
            # Load predictions
            self.predictions = self._safe_load_csv('results/predictions.csv')
            
            # Load processed articles
            self.processed_articles = self._safe_load_csv('data/processed_articles.csv')
            
            # Load JSON summaries
            self.data_summary = self._safe_load_json('results/data_summary.json')
            self.risk_category_summary = self._safe_load_json('results/risk_category_summary.json')
            self.training_summary = self._safe_load_json('results/training_summary.json')
            self.pipeline_summary = self._safe_load_json('results/pipeline_summary.json')
            
            # Data validation and preparation
            self._prepare_data()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure the pipeline has been run and results are available in the ../results/ directory")
    
    def _safe_load_csv(self, filepath: str, **kwargs) -> Optional[pd.DataFrame]:
        """Safely load CSV file"""
        try:
            if os.path.exists(filepath):
                return pd.read_csv(filepath, **kwargs)
        except Exception as e:
            st.warning(f"Could not load {filepath}: {e}")
        return None
    
    def _safe_load_json(self, filepath: str) -> Dict:
        """Safely load JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"Could not load {filepath}: {e}")
        return {}
    
    def _prepare_data(self):
        """Prepare and validate loaded data"""
        # Use articles_data as primary dataset if available
        if self.articles_data is not None:
            self.main_data = self.articles_data
        elif self.predictions is not None:
            self.main_data = self.predictions
        elif self.processed_articles is not None:
            self.main_data = self.processed_articles
        else:
            self.main_data = None
            return
        
        # Ensure risk categories exist
        for category in self.risk_categories:
            if category not in self.main_data.columns:
                self.main_data[category] = 0
        
        # Calculate total risk score if not exists
        if 'total_risk_score' not in self.main_data.columns:
            self.main_data['total_risk_score'] = self.main_data[self.risk_categories].sum(axis=1)
        
        # Add risk level classification
        self.main_data['risk_level'] = pd.cut(
            self.main_data['total_risk_score'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
    
    def render_header(self):
        """Render the main dashboard header"""
        st.markdown('<h1 class="main-header">📊 Risk Analysis Visualization Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Interactive exploration of textile supplier risk analysis results
        </div>
        """, unsafe_allow_html=True)
        
        if self.pipeline_summary:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{self.pipeline_summary.get('total_articles_processed', 'N/A')}</h3>
                    <p>Total Articles</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{self.pipeline_summary.get('unique_suppliers_identified', 'N/A')}</h3>
                    <p>Suppliers</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(self.risk_categories)}</h3>
                    <p>Risk Categories</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if self.main_data is not None:
                    avg_risk = self.main_data['total_risk_score'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{avg_risk:.3f}</h3>
                        <p>Avg Risk Score</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col5:
                if self.main_data is not None:
                    high_risk_count = len(self.main_data[self.main_data['risk_level'] == 'High'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{high_risk_count}</h3>
                        <p>High Risk Articles</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_executive_summary(self):
        """Render executive summary page"""
        st.markdown('<h2 class="section-header">📈 Executive Summary</h2>', unsafe_allow_html=True)
        
        if self.main_data is None:
            st.error("No data available for visualization")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Risk distribution overview
            st.subheader("🎯 Risk Distribution Overview")
            
            # Create risk summary by supplier
            supplier_risks = self.main_data.groupby('supplier')[self.risk_categories].mean()
            supplier_risks['total_risk'] = supplier_risks.sum(axis=1)
            supplier_risks = supplier_risks.sort_values('total_risk', ascending=False)
            
            fig_overview = go.Figure()
            
            # Add bars for each risk category
            for i, category in enumerate(self.risk_categories):
                fig_overview.add_trace(go.Bar(
                    name=category.replace('_', ' ').title(),
                    x=supplier_risks.index,
                    y=supplier_risks[category],
                    marker_color=self.risk_colors[category]
                ))
            
            fig_overview.update_layout(
                title="Risk Categories by Supplier",
                barmode='stack',
                height=500,
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_overview, use_container_width=True)
        
        with col2:
            # Top risk insights
            st.subheader("🔴 Top Risk Insights")
            
            # Highest risk supplier
            if not supplier_risks.empty:
                highest_risk_supplier = supplier_risks.index[0]
                highest_risk_score = supplier_risks.iloc[0]['total_risk']
                
                st.markdown(f"""
                <div class="risk-high">
                    Highest Risk Supplier:<br>
                    <strong>{highest_risk_supplier}</strong><br>
                    Risk Score: {highest_risk_score:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                # Dominant risk category
                dominant_category = supplier_risks.iloc[0][self.risk_categories].idxmax()
                st.markdown(f"""
                <div class="risk-medium">
                    Dominant Risk Category:<br>
                    <strong>{dominant_category.replace('_', ' ').title()}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Recent high-risk articles
                recent_high_risk = self.main_data[
                    self.main_data['total_risk_score'] > self.main_data['total_risk_score'].quantile(0.9)
                ].nlargest(3, 'total_risk_score')
                
                st.markdown("**Recent High-Risk Articles:**")
                for _, article in recent_high_risk.iterrows():
                    st.markdown(f"• {article['title'][:60]}...")
        
        # Risk trend over time
        if 'published_date' in self.main_data.columns:
            st.subheader("📅 Risk Trends Over Time")
            
            monthly_data = self.main_data.copy()
            monthly_data['month'] = monthly_data['published_date'].dt.to_period('M')
            monthly_trends = monthly_data.groupby('month')[self.risk_categories + ['total_risk_score']].mean()
            
            fig_trends = go.Figure()
            
            fig_trends.add_trace(go.Scatter(
                x=monthly_trends.index.astype(str),
                y=monthly_trends['total_risk_score'],
                mode='lines+markers',
                name='Total Risk Score',
                line=dict(width=3, color='#FF6B6B'),
                marker=dict(size=8)
            ))
            
            fig_trends.update_layout(
                title="Total Risk Score Trend Over Time",
                xaxis_title="Month",
                yaxis_title="Average Risk Score",
                height=400
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
    
    def render_supplier_deep_dive(self):
        """Render supplier deep dive analysis"""
        st.markdown('<h2 class="section-header">🏭 Supplier Deep Dive</h2>', unsafe_allow_html=True)
        
        if self.main_data is None:
            st.error("No data available for visualization")
            return
        
        # Supplier selection
        available_suppliers = self.main_data['supplier'].dropna().unique()
        selected_supplier = st.selectbox(
            "🔍 Select Supplier for Deep Analysis:",
            available_suppliers,
            index=0 if len(available_suppliers) > 0 else None
        )
        
        if selected_supplier:
            supplier_data = self.main_data[self.main_data['supplier'] == selected_supplier]
            
            # Supplier overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📰 Total Articles", len(supplier_data))
            
            with col2:
                avg_risk = supplier_data['total_risk_score'].mean()
                st.metric("⚠️ Average Risk Score", f"{avg_risk:.3f}")
            
            with col3:
                high_risk_pct = (supplier_data['total_risk_score'] > supplier_data['total_risk_score'].median()).mean() * 100
                st.metric("🔴 High Risk Articles %", f"{high_risk_pct:.1f}%")
            
            with col4:
                if 'published_date' in supplier_data.columns:
                    latest_date = supplier_data['published_date'].max()
                    st.metric("📅 Latest Article", latest_date.strftime('%Y-%m-%d'))
            
            # Risk profile visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart for risk profile
                st.subheader("🎯 Risk Profile")
                
                risk_scores = supplier_data[self.risk_categories].mean()
                
                fig_radar = go.Figure()
                
                categories = [cat.replace('_', ' ').title() for cat in self.risk_categories]
                values = risk_scores.tolist()
                values += [values[0]]  # Close the radar chart
                categories += [categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_supplier,
                    line=dict(color='#FF6B6B', width=3),
                    fillcolor='rgba(255, 107, 107, 0.3)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(values[:-1]) * 1.2]
                        )
                    ),
                    title=f"Risk Profile - {selected_supplier}",
                    height=400
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Risk distribution by category
                st.subheader("📊 Risk Distribution")
                
                fig_dist = go.Figure(data=[go.Bar(
                    x=[cat.replace('_', ' ').title() for cat in self.risk_categories],
                    y=risk_scores.values,
                    marker_color=[self.risk_colors[cat] for cat in self.risk_categories],
                    text=[f"{val:.3f}" for val in risk_scores.values],
                    textposition='auto',
                )])
                
                fig_dist.update_layout(
                    title="Risk Scores by Category",
                    xaxis_title="Risk Categories",
                    yaxis_title="Average Risk Score",
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Temporal analysis for supplier
            if 'published_date' in supplier_data.columns:
                st.subheader("📈 Temporal Risk Analysis")
                
                supplier_temporal = supplier_data.copy()
                supplier_temporal['month'] = supplier_temporal['published_date'].dt.to_period('M')
                monthly_supplier = supplier_temporal.groupby('month')[self.risk_categories].mean()
                
                fig_temporal = go.Figure()
                
                for category in self.risk_categories:
                    fig_temporal.add_trace(go.Scatter(
                        x=monthly_supplier.index.astype(str),
                        y=monthly_supplier[category],
                        mode='lines+markers',
                        name=category.replace('_', ' ').title(),
                        line=dict(color=self.risk_colors[category], width=2)
                    ))
                
                fig_temporal.update_layout(
                    title=f"Risk Evolution Over Time - {selected_supplier}",
                    xaxis_title="Month",
                    yaxis_title="Risk Score",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Article-level analysis
            st.subheader("📝 Article Analysis")
            
            # Top risk articles for this supplier
            top_articles = supplier_data.nlargest(5, 'total_risk_score')[
                ['title', 'published_date', 'total_risk_score', 'source']
            ]
            
            st.markdown("**🔴 Highest Risk Articles:**")
            for _, article in top_articles.iterrows():
                risk_level = "🔴" if article['total_risk_score'] > 0.5 else "🟡" if article['total_risk_score'] > 0.2 else "🟢"
                st.markdown(f"{risk_level} **{article['title']}** (Score: {article['total_risk_score']:.3f})")
                st.markdown(f"   📅 {article['published_date'].strftime('%Y-%m-%d')} | 📰 {article['source']}")
    
    def render_risk_category_analysis(self):
        """Render risk category analysis"""
        st.markdown('<h2 class="section-header">🏷️ Risk Category Analysis</h2>', unsafe_allow_html=True)
        
        if self.main_data is None:
            st.error("No data available for visualization")
            return
        
        # Category selection
        selected_category = st.selectbox(
            "🎯 Select Risk Category:",
            self.risk_categories,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        category_data = self.main_data[selected_category]
        
        # Category overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Average Score", f"{category_data.mean():.3f}")
        
        with col2:
            st.metric("📈 Max Score", f"{category_data.max():.3f}")
        
        with col3:
            high_risk_count = len(category_data[category_data > category_data.quantile(0.8)])
            st.metric("🔴 High Risk Articles", high_risk_count)
        
        with col4:
            std_dev = category_data.std()
            st.metric("📏 Std Deviation", f"{std_dev:.3f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution histogram
            st.subheader("📊 Score Distribution")
            
            fig_hist = px.histogram(
                self.main_data,
                x=selected_category,
                nbins=30,
                title=f"Distribution of {selected_category.replace('_', ' ').title()} Scores",
                color_discrete_sequence=[self.risk_colors[selected_category]]
            )
            
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by supplier
            st.subheader("📦 Scores by Supplier")
            
            fig_box = px.box(
                self.main_data.dropna(subset=['supplier']),
                x='supplier',
                y=selected_category,
                title=f"{selected_category.replace('_', ' ').title()} by Supplier",
                color_discrete_sequence=[self.risk_colors[selected_category]]
            )
            
            fig_box.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Top performers analysis
        st.subheader("🏆 Category Leaders")
        
        supplier_category_scores = self.main_data.groupby('supplier')[selected_category].agg([
            'mean', 'count', 'std', 'max'
        ]).round(3).sort_values('mean', ascending=False)
        
        # Enhanced table display
        supplier_category_scores.columns = ['Average', 'Article Count', 'Std Dev', 'Max Score']
        supplier_category_scores['Risk Level'] = pd.cut(
            supplier_category_scores['Average'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        st.dataframe(
            supplier_category_scores.style.background_gradient(subset=['Average'], cmap='Reds'),
            use_container_width=True
        )
        
        # Time series for category
        if 'published_date' in self.main_data.columns:
            st.subheader("📈 Category Trend Over Time")
            
            category_temporal = self.main_data.copy()
            category_temporal['month'] = category_temporal['published_date'].dt.to_period('M')
            monthly_category = category_temporal.groupby('month')[selected_category].agg(['mean', 'count'])
            
            fig_category_trend = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Average Risk Score', 'Article Count'],
                vertical_spacing=0.1
            )
            
            fig_category_trend.add_trace(
                go.Scatter(
                    x=monthly_category.index.astype(str),
                    y=monthly_category['mean'],
                    mode='lines+markers',
                    name='Average Score',
                    line=dict(color=self.risk_colors[selected_category], width=3)
                ),
                row=1, col=1
            )
            
            fig_category_trend.add_trace(
                go.Bar(
                    x=monthly_category.index.astype(str),
                    y=monthly_category['count'],
                    name='Article Count',
                    marker_color=self.risk_colors[selected_category],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig_category_trend.update_layout(
                height=500,
                title_text=f"{selected_category.replace('_', ' ').title()} Analysis Over Time"
            )
            
            st.plotly_chart(fig_category_trend, use_container_width=True)
    
    def render_comparative_analysis(self):
        """Render comparative analysis"""
        st.markdown('<h2 class="section-header">⚖️ Comparative Analysis</h2>', unsafe_allow_html=True)
        
        if self.main_data is None:
            st.error("No data available for visualization")
            return
        
        # Multi-supplier comparison
        st.subheader("🏭 Multi-Supplier Comparison")
        
        available_suppliers = self.main_data['supplier'].dropna().unique()
        selected_suppliers = st.multiselect(
            "Select suppliers to compare:",
            available_suppliers,
            default=available_suppliers[:4] if len(available_suppliers) >= 4 else available_suppliers
        )
        
        if selected_suppliers:
            # Comparative radar chart
            fig_comparison = go.Figure()
            
            for supplier in selected_suppliers:
                supplier_data = self.main_data[self.main_data['supplier'] == supplier]
                risk_scores = supplier_data[self.risk_categories].mean()
                
                categories = [cat.replace('_', ' ').title() for cat in self.risk_categories]
                values = risk_scores.tolist()
                values += [values[0]]  # Close the radar chart
                categories += [categories[0]]
                
                fig_comparison.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=supplier,
                    line=dict(width=2),
                    fillcolor=f'rgba({np.random.randint(100, 255)}, {np.random.randint(100, 255)}, {np.random.randint(100, 255)}, 0.1)'
                ))
            
            fig_comparison.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, self.main_data[self.risk_categories].max().max() * 1.1]
                    )
                ),
                title="Supplier Risk Profile Comparison",
                height=600
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Side-by-side metrics
            st.subheader("📊 Side-by-Side Metrics")
            
            comparison_metrics = []
            for supplier in selected_suppliers:
                supplier_data = self.main_data[self.main_data['supplier'] == supplier]
                metrics = {
                    'Supplier': supplier,
                    'Total Articles': len(supplier_data),
                    'Avg Risk Score': supplier_data['total_risk_score'].mean(),
                    'High Risk %': (supplier_data['total_risk_score'] > supplier_data['total_risk_score'].median()).mean() * 100,
                    'Dominant Risk': supplier_data[self.risk_categories].mean().idxmax().replace('_', ' ').title()
                }
                comparison_metrics.append(metrics)
            
            comparison_df = pd.DataFrame(comparison_metrics)
            st.dataframe(
                comparison_df.style.highlight_max(subset=['Avg Risk Score'], color='lightcoral')
                .highlight_min(subset=['Avg Risk Score'], color='lightgreen'),
                use_container_width=True
            )
        
        # Risk correlation analysis
        st.subheader("🔗 Risk Category Correlations")
        
        correlation_matrix = self.main_data[self.risk_categories].corr()
        
        fig_corr = px.imshow(
            correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale='RdBu',
            aspect="auto",
            title="Risk Category Correlation Matrix"
        )
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Risk ranking
        st.subheader("🏆 Overall Risk Ranking")
        
        supplier_rankings = self.main_data.groupby('supplier').agg({
            'total_risk_score': ['mean', 'count', 'std'],
            **{cat: 'mean' for cat in self.risk_categories}
        }).round(3)
        
        supplier_rankings.columns = ['_'.join(col).strip() if col[1] else col[0] for col in supplier_rankings.columns.values]
        supplier_rankings = supplier_rankings.sort_values('total_risk_score_mean', ascending=False)
        supplier_rankings['Rank'] = range(1, len(supplier_rankings) + 1)
        
        # Reorder columns for better display
        display_cols = ['Rank', 'total_risk_score_mean', 'total_risk_score_count'] + [f'{cat}_mean' for cat in self.risk_categories]
        
        st.dataframe(
            supplier_rankings[display_cols].style.background_gradient(subset=['total_risk_score_mean'], cmap='Reds'),
            use_container_width=True
        )
    
    def render_advanced_analytics(self):
        """Render advanced analytics and insights"""
        st.markdown('<h2 class="section-header">🧠 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        if self.main_data is None:
            st.error("No data available for visualization")
            return
        
        # Risk clustering analysis
        st.subheader("🎯 Risk Pattern Analysis")
        
        # Create scatter plot matrix for risk categories
        fig_scatter = px.scatter_matrix(
            self.main_data[self.risk_categories + ['total_risk_score']],
            dimensions=self.risk_categories,
            color='total_risk_score',
            title="Risk Category Relationships",
            color_continuous_scale='Viridis'
        )
        
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Sentiment vs Risk Analysis (if available)
        if 'sentiment_polarity' in self.main_data.columns:
            st.subheader("😊 Sentiment vs Risk Analysis")
            
            fig_sentiment = px.scatter(
                self.main_data.dropna(subset=['sentiment_polarity']),
                x='sentiment_polarity',
                y='total_risk_score',
                color='supplier',
                size='text_length' if 'text_length' in self.main_data.columns else None,
                hover_data=['title'],
                title="Sentiment Polarity vs Risk Score"
            )
            
            fig_sentiment.update_layout(height=500)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Risk evolution patterns
        if 'published_date' in self.main_data.columns:
            st.subheader("📈 Risk Evolution Patterns")
            
            # Weekly risk heatmap
            weekly_data = self.main_data.copy()
            weekly_data['week'] = weekly_data['published_date'].dt.isocalendar().week
            weekly_data['year'] = weekly_data['published_date'].dt.year
            weekly_risks = weekly_data.groupby(['year', 'week'])[self.risk_categories].mean()
            
            # Create heatmap for recent weeks
            recent_weeks = weekly_risks.tail(20)
            if len(recent_weeks) > 0:
                fig_heatmap = px.imshow(
                    recent_weeks.T.values,
                    x=[f"{row[0]}-W{row[1]}" for row in recent_weeks.index],
                    y=[cat.replace('_', ' ').title() for cat in self.risk_categories],
                    color_continuous_scale='Reds',
                    title="Weekly Risk Evolution (Recent 20 Weeks)"
                )
                
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Statistical insights
        st.subheader("📊 Statistical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution by quartiles
            st.markdown("**Risk Score Quartiles:**")
            quartiles = self.main_data['total_risk_score'].quantile([0.25, 0.5, 0.75, 1.0])
            
            for i, (q, val) in enumerate(quartiles.items()):
                quartile_name = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'Q4 (100%)'][i]
                st.markdown(f"• {quartile_name}: {val:.3f}")
        
        with col2:
            # Top risk factors
            st.markdown("**Most Volatile Risk Categories:**")
            risk_volatility = self.main_data[self.risk_categories].std().sort_values(ascending=False)
            
            for cat, vol in risk_volatility.items():
                st.markdown(f"• {cat.replace('_', ' ').title()}: {vol:.3f}")
        
        # Outlier detection
        st.subheader("🎯 Outlier Detection")
        
        # Identify articles with unusually high risk scores
        risk_threshold = self.main_data['total_risk_score'].quantile(0.95)
        outliers = self.main_data[self.main_data['total_risk_score'] > risk_threshold]
        
        if len(outliers) > 0:
            st.markdown(f"**Found {len(outliers)} high-risk outliers (top 5% risk scores):**")
            
            outlier_display = outliers.nlargest(10, 'total_risk_score')[
                ['title', 'supplier', 'total_risk_score', 'published_date']
            ]
            
            for _, article in outlier_display.iterrows():
                st.markdown(f"🚨 **{article['title'][:80]}...** (Score: {article['total_risk_score']:.3f}) - {article['supplier']}")

def main():
    """Main function to run the visualization dashboard"""
    
    # Initialize dashboard
    dashboard = VisualizationDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Check if data is available
    if dashboard.main_data is None:
        st.error("❌ No data available for visualization")
        st.info("""
        📋 **To use this dashboard:**
        1. Run the main risk analysis pipeline: `python main.py`
        2. Ensure results are generated in the `../results/` directory
        3. Refresh this page
        """)
        return
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Dashboard Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis View:",
        [
            "📈 Executive Summary",
            "🏭 Supplier Deep Dive", 
            "🏷️ Risk Category Analysis",
            "⚖️ Comparative Analysis",
            "🧠 Advanced Analytics"
        ]
    )
    
    # Data overview in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Overview")
    if dashboard.main_data is not None:
        st.sidebar.markdown(f"**Articles:** {len(dashboard.main_data):,}")
        st.sidebar.markdown(f"**Suppliers:** {dashboard.main_data['supplier'].nunique()}")
        st.sidebar.markdown(f"**Avg Risk Score:** {dashboard.main_data['total_risk_score'].mean():.3f}")
        
        # Risk level distribution
        if 'risk_level' in dashboard.main_data.columns:
            risk_dist = dashboard.main_data['risk_level'].value_counts()
            st.sidebar.markdown("**Risk Levels:**")
            for level, count in risk_dist.items():
                st.sidebar.markdown(f"• {level}: {count}")
    
    # Render selected page
    if page == "📈 Executive Summary":
        dashboard.render_executive_summary()
    elif page == "🏭 Supplier Deep Dive":
        dashboard.render_supplier_deep_dive()
    elif page == "🏷️ Risk Category Analysis":
        dashboard.render_risk_category_analysis()
    elif page == "⚖️ Comparative Analysis":
        dashboard.render_comparative_analysis()
    elif page == "🧠 Advanced Analytics":
        dashboard.render_advanced_analytics()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Risk Analysis Dashboard**")
    st.sidebar.markdown("*Visualization-Only Interface*")
    st.sidebar.markdown("No data processing performed")

if __name__ == "__main__":
    main() 