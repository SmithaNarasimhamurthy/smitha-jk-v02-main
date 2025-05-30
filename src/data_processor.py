# Data processing module for textile supplier risk identification
# This module handles loading news article data, preprocessing text content,
# and preparing data for machine learning model training and analysis

import json
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from typing import Dict, List, Tuple, Optional
import time
from urllib.parse import urlparse
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplierDataProcessor:
    """
    Main class for processing supplier news data and extracting relevant information
    for risk analysis
    """
    
    def __init__(self, json_file_path: str):
        """
        Initialize the data processor
        
        Args:
            json_file_path (str): Path to the JSON file containing news articles
        """
        self.json_file_path = json_file_path
        self.data = None
        self.processed_data = None
        
        # Initialize NLTK components
        self._setup_nltk()
        
        # Define supplier names for mapping
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
        
        # Risk keywords for initial classification
        self.risk_keywords = {
            'geopolitical_regulatory': [
                'tariff', 'trade war', 'sanction', 'regulation', 'policy', 'government',
                'tax', 'compliance', 'legal', 'lawsuit', 'political', 'election',
                'brexit', 'import', 'export', 'customs', 'border'
            ],
            'agricultural_environmental': [
                'drought', 'flood', 'climate', 'weather', 'environmental', 'sustainability',
                'carbon', 'emission', 'pollution', 'water', 'energy', 'renewable',
                'organic', 'eco-friendly', 'green', 'natural disaster'
            ],
            'financial_operational': [
                'bankruptcy', 'debt', 'profit', 'loss', 'revenue', 'financial',
                'strike', 'labor', 'worker', 'union', 'production', 'manufacturing',
                'factory', 'closure', 'layoff', 'hiring', 'investment'
            ],
            'supply_chain_logistics': [
                'supply chain', 'logistics', 'transportation', 'shipping', 'delivery',
                'warehouse', 'inventory', 'shortage', 'delay', 'disruption',
                'fuel', 'oil price', 'logistics cost', 'container'
            ],
            'market_competitive': [
                'competition', 'competitor', 'market share', 'price', 'demand',
                'consumer', 'retail', 'sales', 'growth', 'decline', 'trend',
                'fashion', 'style', 'brand', 'acquisition', 'merger'
            ]
        }
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error setting up NLTK: {e}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load news article data from JSON file
        
        Returns:
            pd.DataFrame: Loaded and initially processed data
        """
        try:
            logger.info("Loading data from JSON file...")
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            self.data = pd.DataFrame(data)
            logger.info(f"Loaded {len(self.data)} articles")
            
            # Basic data cleaning
            self.data = self._initial_cleaning()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _initial_cleaning(self) -> pd.DataFrame:
        """
        Perform initial data cleaning and preprocessing
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Remove duplicates based on title and link
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['title', 'link'], keep='first')
        logger.info(f"Removed {initial_count - len(self.data)} duplicate articles")
        
        # Convert datetime column
        if 'published_datetime_utc' in self.data.columns:
            self.data['published_datetime_utc'] = pd.to_datetime(
                self.data['published_datetime_utc'], errors='coerce'
            )
        
        # Handle missing values
        text_columns = ['title', 'Full_Article']
        for col in text_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna('')
        
        return self.data
    
    def extract_article_content(self, url: str, timeout: int = 10) -> str:
        """
        Extract article content from URL
        
        Args:
            url (str): Article URL
            timeout (int): Request timeout in seconds
            
        Returns:
            str: Extracted article content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def identify_supplier(self, text: str) -> Optional[str]:
        """
        Identify which supplier the article is about
        
        Args:
            text (str): Article text
            
        Returns:
            Optional[str]: Identified supplier name
        """
        text_lower = text.lower()
        
        for supplier in self.suppliers:
            # Check for exact matches and variations
            supplier_variations = [
                supplier.lower(),
                supplier.lower().replace(' limited', ''),
                supplier.lower().replace(' plc', ''),
                supplier.lower().replace(' inc.', ''),
                supplier.lower().replace(' ltd.', ''),
                supplier.lower().replace(',', ''),
            ]
            
            for variation in supplier_variations:
                if variation in text_lower:
                    return supplier
        
        return None
    
    def classify_risk_category(self, text: str) -> Dict[str, float]:
        """
        Classify text into risk categories based on keywords
        
        Args:
            text (str): Article text
            
        Returns:
            Dict[str, float]: Risk category scores
        """
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.risk_keywords.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword.lower())
            
            # Normalize score
            category_scores[category] = score / len(keywords) if keywords else 0
        
        return category_scores
    
    def determine_risk_direction(self, text: str) -> str:
        """
        Determine if the risk is positive (increased risk) or negative (decreased risk)
        
        Args:
            text (str): Article text
            
        Returns:
            str: 'positive' for increased risk, 'negative' for decreased risk
        """
        positive_indicators = [
            'crisis', 'problem', 'issue', 'concern', 'worry', 'threat', 'risk',
            'decline', 'fall', 'drop', 'decrease', 'loss', 'bankruptcy',
            'closure', 'shutdown', 'strike', 'protest', 'conflict'
        ]
        
        negative_indicators = [
            'growth', 'increase', 'rise', 'improvement', 'success', 'profit',
            'expansion', 'investment', 'agreement', 'partnership', 'recovery',
            'solution', 'resolve', 'stable', 'strong', 'positive'
        ]
        
        text_lower = text.lower()
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        return 'positive' if positive_score > negative_score else 'negative'
    
    def process_articles(self, extract_missing_content: bool = False) -> pd.DataFrame:
        """
        Process all articles in the dataset
        
        Args:
            extract_missing_content (bool): Whether to extract content from URLs for missing articles
            
        Returns:
            pd.DataFrame: Processed articles with risk classifications
        """
        if self.data is None:
            self.load_data()
        
        logger.info("Processing articles...")
        
        processed_articles = []
        
        for idx, row in self.data.iterrows():
            try:
                # Get article content
                content = row.get('Full_Article', '') or ''
                title = row.get('title', '') or ''
                
                # If content is missing and extraction is enabled, try to extract from URL
                if not content and extract_missing_content and row.get('link'):
                    content = self.extract_article_content(row['link'])
                    time.sleep(1)  # Be respectful to servers
                
                # Combine title and content for analysis
                full_text = f"{title} {content}"
                
                # Preprocess text
                processed_text = self.preprocess_text(full_text)
                
                if not processed_text:
                    continue
                
                # Identify supplier
                supplier = self.identify_supplier(full_text)
                
                # Classify risk categories
                risk_scores = self.classify_risk_category(processed_text)
                
                # Determine risk direction
                risk_direction = self.determine_risk_direction(processed_text)
                
                # Create processed article record
                processed_article = {
                    'article_id': idx,
                    'title': title,
                    'link': row.get('link', ''),
                    'published_date': row.get('published_datetime_utc'),
                    'source': row.get('source_name', ''),
                    'supplier': supplier,
                    'processed_text': processed_text,
                    'risk_direction': risk_direction,
                    'text_length': len(processed_text),
                    **risk_scores
                }
                
                processed_articles.append(processed_article)
                
                if idx % 100 == 0:
                    logger.info(f"Processed {idx} articles...")
                    
            except Exception as e:
                logger.error(f"Error processing article {idx}: {e}")
                continue
        
        self.processed_data = pd.DataFrame(processed_articles)
        logger.info(f"Successfully processed {len(self.processed_data)} articles")
        
        return self.processed_data
    
    def save_processed_data(self, output_path: str = "data/processed_articles.csv"):
        """
        Save processed data to CSV file
        
        Args:
            output_path (str): Output file path
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        else:
            logger.warning("No processed data to save")

def main():
    """Main function to run the data processing pipeline"""
    processor = SupplierDataProcessor('suppliers_news.json')
    
    # Load and process data
    processor.load_data()
    processed_data = processor.process_articles(extract_missing_content=False)
    
    # Save processed data
    processor.save_processed_data()
    
    # Print summary statistics
    print("\n=== Processing Summary ===")
    print(f"Total articles processed: {len(processed_data)}")
    print(f"Articles with identified suppliers: {processed_data['supplier'].notna().sum()}")
    print(f"Date range: {processed_data['published_date'].min()} to {processed_data['published_date'].max()}")
    
    print("\n=== Supplier Distribution ===")
    print(processed_data['supplier'].value_counts())
    
    print("\n=== Risk Direction Distribution ===")
    print(processed_data['risk_direction'].value_counts())

if __name__ == "__main__":
    main() 