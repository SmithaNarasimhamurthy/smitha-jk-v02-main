# Visualization module for textile supplier risk analysis
# This module creates comprehensive visualizations including risk profiles,
# temporal trends, geographic analysis, and interactive dashboards

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from wordcloud import WordCloud
import warnings
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style preferences
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RiskVisualizationEngine:
    """
    Comprehensive visualization engine for supplier risk analysis
    """
    
    def __init__(self):
        """Initialize the visualization engine"""
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
            "Welspun Living Limited",
            "Teejay Lanka PLC", 
            "Arvind Limited",
            "Caleres, Inc.",
            "Interloop Limited",
            "Kitex Garments Limited",
            "ThredUp Inc.",
            "G-III Apparel Group, Ltd.",
            "Mint Velvet",
            "White Stuff Limited"
        ]
    
    def create_supplier_risk_heatmap(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create a heatmap showing risk levels across suppliers and categories
        
        Args:
            df (pd.DataFrame): Processed risk data
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly heatmap figure
        """
        # Aggregate risk scores by supplier
        supplier_risks = df.groupby('supplier')[self.risk_categories].mean().fillna(0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=supplier_risks.values,
            x=[cat.replace('_', ' ').title() for cat in self.risk_categories],
            y=supplier_risks.index,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            colorbar=dict(title="Risk Score")
        ))
        
        fig.update_layout(
            title="Supplier Risk Heatmap by Category",
            xaxis_title="Risk Categories",
            yaxis_title="Suppliers",
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_risk_trend_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create temporal trend analysis of risks
        
        Args:
            df (pd.DataFrame): Processed risk data with dates
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly line chart figure
        """
        # Prepare data
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['month_year'] = df['published_date'].dt.to_period('M').astype(str)
        
        # Aggregate by month
        monthly_risks = df.groupby('month_year')[self.risk_categories].mean().reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=len(self.risk_categories), 
            cols=1,
            subplot_titles=[cat.replace('_', ' ').title() for cat in self.risk_categories],
            vertical_spacing=0.05
        )
        
        for i, category in enumerate(self.risk_categories):
            fig.add_trace(
                go.Scatter(
                    x=monthly_risks['month_year'],
                    y=monthly_risks[category],
                    mode='lines+markers',
                    name=category.replace('_', ' ').title(),
                    line=dict(color=self.risk_colors[category]),
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Risk Trends Over Time by Category",
            height=1200,
            width=1000
        )
        
        fig.update_xaxes(title_text="Time Period", row=len(self.risk_categories), col=1)
        fig.update_yaxes(title_text="Risk Score")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_supplier_comparison_radar(self, df: pd.DataFrame, suppliers: List[str] = None, 
                                       save_path: str = None) -> go.Figure:
        """
        Create radar chart comparing risk profiles of selected suppliers
        
        Args:
            df (pd.DataFrame): Processed risk data
            suppliers (List[str]): List of suppliers to compare
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly radar chart figure
        """
        if suppliers is None:
            suppliers = df['supplier'].value_counts().head(5).index.tolist()
        
        # Aggregate risk scores by supplier
        supplier_risks = df.groupby('supplier')[self.risk_categories].mean().fillna(0)
        
        fig = go.Figure()
        
        for supplier in suppliers:
            if supplier in supplier_risks.index:
                values = supplier_risks.loc[supplier].tolist()
                values += [values[0]]  # Close the radar chart
                
                categories = [cat.replace('_', ' ').title() for cat in self.risk_categories]
                categories += [categories[0]]  # Close the radar chart
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=supplier,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(supplier_risks.max())]
                )
            ),
            title="Supplier Risk Profile Comparison",
            showlegend=True,
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_risk_distribution_charts(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create distribution charts for risk categories
        
        Args:
            df (pd.DataFrame): Processed risk data
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly subplot figure with distributions
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[cat.replace('_', ' ').title() for cat in self.risk_categories],
            specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "xy"}]]
        )
        
        for i, category in enumerate(self.risk_categories):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[category],
                    name=category.replace('_', ' ').title(),
                    marker_color=self.risk_colors[category],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Risk Score Distributions by Category",
            height=800,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_sentiment_vs_risk_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create scatter plot showing relationship between sentiment and risk
        
        Args:
            df (pd.DataFrame): Processed risk data with sentiment
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly scatter plot figure
        """
        if 'sentiment_polarity' not in df.columns:
            logger.warning("Sentiment data not available")
            return None
        
        # Calculate total risk score
        df['total_risk'] = df[self.risk_categories].sum(axis=1)
        
        fig = px.scatter(
            df,
            x='sentiment_polarity',
            y='total_risk',
            color='risk_direction',
            size='text_length',
            hover_data=['supplier', 'title'],
            title="Sentiment vs Risk Analysis",
            labels={
                'sentiment_polarity': 'Sentiment Polarity',
                'total_risk': 'Total Risk Score',
                'text_length': 'Article Length'
            }
        )
        
        fig.update_layout(height=600, width=800)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_wordcloud_by_risk_category(self, df: pd.DataFrame, category: str, 
                                        save_path: str = None) -> None:
        """
        Create word cloud for specific risk category
        
        Args:
            df (pd.DataFrame): Processed risk data
            category (str): Risk category to analyze
            save_path (str): Optional path to save the figure
        """
        # Filter articles with high risk in this category
        high_risk_articles = df[df[category] > df[category].quantile(0.75)]
        
        if len(high_risk_articles) == 0:
            logger.warning(f"No high-risk articles found for category: {category}")
            return
        
        # Combine all text
        text = ' '.join(high_risk_articles['processed_text'].dropna())
        
        if not text:
            logger.warning(f"No text data available for category: {category}")
            return
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {category.replace("_", " ").title()} Risk Articles', 
                 fontsize=16, pad=20)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
    
    def create_temporal_risk_heatmap(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create heatmap showing risk evolution over time
        
        Args:
            df (pd.DataFrame): Processed risk data with dates
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly heatmap figure
        """
        # Prepare temporal data
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['week'] = df['published_date'].dt.isocalendar().week
        df['year'] = df['published_date'].dt.year
        
        # Aggregate by week and category
        weekly_risks = df.groupby(['year', 'week'])[self.risk_categories].mean().reset_index()
        weekly_risks['year_week'] = weekly_risks['year'].astype(str) + '-W' + weekly_risks['week'].astype(str).str.zfill(2)
        
        # Reshape for heatmap
        heatmap_data = weekly_risks.set_index('year_week')[self.risk_categories]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.T.values,
            x=heatmap_data.index,
            y=[cat.replace('_', ' ').title() for cat in self.risk_categories],
            colorscale='RdYlBu_r',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Risk Evolution Over Time (Weekly)",
            xaxis_title="Time Period",
            yaxis_title="Risk Categories",
            height=500,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_supplier_network_analysis(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create network-style visualization showing supplier relationships and shared risks
        
        Args:
            df (pd.DataFrame): Processed risk data
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly network figure
        """
        # Calculate risk correlations between suppliers
        supplier_risks = df.groupby('supplier')[self.risk_categories].mean().fillna(0)
        correlation_matrix = supplier_risks.T.corr()
        
        # Create network edges for highly correlated suppliers
        edges = []
        for i, supplier1 in enumerate(correlation_matrix.index):
            for j, supplier2 in enumerate(correlation_matrix.columns):
                if i < j and correlation_matrix.iloc[i, j] > 0.5:  # Threshold for connection
                    edges.append((supplier1, supplier2, correlation_matrix.iloc[i, j]))
        
        # Generate positions for suppliers (circular layout)
        import math
        n_suppliers = len(supplier_risks.index)
        positions = {}
        for i, supplier in enumerate(supplier_risks.index):
            angle = 2 * math.pi * i / n_suppliers
            positions[supplier] = (math.cos(angle), math.sin(angle))
        
        # Create network visualization
        fig = go.Figure()
        
        # Add edges
        for supplier1, supplier2, correlation in edges:
            x0, y0 = positions[supplier1]
            x1, y1 = positions[supplier2]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=correlation*5, color='gray'),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add nodes
        x_vals, y_vals, texts = [], [], []
        for supplier in supplier_risks.index:
            x, y = positions[supplier]
            x_vals.append(x)
            y_vals.append(y)
            texts.append(supplier)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(width=2, color='darkblue')),
            text=texts,
            textposition='top center',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Supplier Risk Correlation Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_risk_summary_dashboard(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """
        Create comprehensive summary dashboard
        
        Args:
            df (pd.DataFrame): Processed risk data
            save_path (str): Optional path to save the figure
            
        Returns:
            go.Figure: Plotly dashboard figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Risk Distribution by Supplier",
                "Risk Trends Over Time", 
                "Risk Direction Split",
                "Top Risk Articles by Category"
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "table"}]]
        )
        
        # 1. Risk distribution by supplier
        supplier_total_risk = df.groupby('supplier')[self.risk_categories].sum().sum(axis=1).sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=supplier_total_risk.index,
                y=supplier_total_risk.values,
                name="Total Risk",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Risk trends (simplified)
        df['published_date'] = pd.to_datetime(df['published_date'])
        monthly_risk = df.groupby(df['published_date'].dt.to_period('M'))[self.risk_categories].sum().sum(axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=monthly_risk.index.astype(str),
                y=monthly_risk.values,
                mode='lines+markers',
                name="Monthly Risk",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Risk direction pie chart
        direction_counts = df['risk_direction'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=direction_counts.index,
                values=direction_counts.values,
                name="Risk Direction",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Top risk articles table
        top_articles = df.nlargest(5, [col for col in self.risk_categories])[['title', 'supplier'] + self.risk_categories]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Title', 'Supplier'] + [cat.replace('_', ' ').title() for cat in self.risk_categories]),
                cells=dict(values=[top_articles[col] for col in top_articles.columns])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Risk Analysis Summary Dashboard",
            height=800,
            width=1400
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def generate_all_visualizations(self, df: pd.DataFrame, output_dir: str = "visualizations/"):
        """
        Generate all visualizations and save to specified directory
        
        Args:
            df (pd.DataFrame): Processed risk data
            output_dir (str): Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Generating comprehensive risk visualizations...")
        
        # Generate all visualizations
        visualizations = [
            ("supplier_risk_heatmap.html", self.create_supplier_risk_heatmap),
            ("risk_trends.html", self.create_risk_trend_analysis),
            ("supplier_comparison_radar.html", self.create_supplier_comparison_radar),
            ("risk_distributions.html", self.create_risk_distribution_charts),
            ("sentiment_vs_risk.html", self.create_sentiment_vs_risk_analysis),
            ("temporal_risk_heatmap.html", self.create_temporal_risk_heatmap),
            ("supplier_network.html", self.create_supplier_network_analysis),
            ("risk_summary_dashboard.html", self.create_risk_summary_dashboard)
        ]
        
        for filename, func in visualizations:
            try:
                save_path = os.path.join(output_dir, filename)
                func(df, save_path)
                logger.info(f"Generated: {filename}")
            except Exception as e:
                logger.error(f"Error generating {filename}: {e}")
        
        # Generate word clouds for each risk category
        for category in self.risk_categories:
            try:
                save_path = os.path.join(output_dir, f"wordcloud_{category}.png")
                self.create_wordcloud_by_risk_category(df, category, save_path)
                logger.info(f"Generated: wordcloud_{category}.png")
            except Exception as e:
                logger.error(f"Error generating wordcloud for {category}: {e}")
        
        logger.info(f"All visualizations generated and saved to {output_dir}")

def main():
    """Main function to demonstrate visualization capabilities"""
    print("Risk Visualization Engine loaded successfully!")
    print("Use this module to generate comprehensive risk analysis visualizations.")

if __name__ == "__main__":
    main() 