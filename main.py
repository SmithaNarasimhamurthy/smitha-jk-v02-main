# Main pipeline script for textile supplier risk identification
# This script orchestrates the complete data science pipeline from data loading
# to model training, prediction, and comprehensive results generation

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings

# Add src to Python path
sys.path.append('src')

# Import our modules
from data_processor import SupplierDataProcessor
from risk_classifier import RiskClassifier
from visualizations import RiskVisualizationEngine

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskAnalysisPipeline:
    """
    Main pipeline for textile supplier risk identification and analysis
    """
    
    def __init__(self, json_file_path: str = "suppliers_news.json"):
        """
        Initialize the risk analysis pipeline
        
        Args:
            json_file_path (str): Path to the input JSON file
        """
        self.json_file_path = json_file_path
        self.data_processor = SupplierDataProcessor(json_file_path)
        self.risk_classifier = RiskClassifier()
        self.visualizer = RiskVisualizationEngine()
        
        self.raw_data = None
        self.processed_data = None
        self.predictions = None
        self.training_results = None
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary output directories"""
        directories = ['data', 'results', 'models', 'visualizations', 'dashboard']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("Output directories created/verified")
    
    def run_data_processing(self, extract_missing_content: bool = False) -> pd.DataFrame:
        """
        Run the data processing pipeline
        
        Args:
            extract_missing_content (bool): Whether to extract content from URLs
            
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info("=== Starting Data Processing Phase ===")
        
        # Load and process data
        self.raw_data = self.data_processor.load_data()
        logger.info(f"Loaded {len(self.raw_data)} raw articles")
        
        # Process articles
        self.processed_data = self.data_processor.process_articles(
            extract_missing_content=extract_missing_content
        )
        logger.info(f"Processed {len(self.processed_data)} articles")
        
        # Save processed data
        self.data_processor.save_processed_data("data/processed_articles.csv")
        
        # Generate summary statistics
        self._generate_data_summary()
        
        return self.processed_data
    
    def run_model_training(self) -> dict:
        """
        Run the machine learning model training pipeline
        
        Returns:
            dict: Training results and metrics
        """
        logger.info("=== Starting Model Training Phase ===")
        
        if self.processed_data is None:
            raise ValueError("Data must be processed before training models")
        
        # Filter data with identified suppliers for training
        training_data = self.processed_data[self.processed_data['supplier'].notna()].copy()
        logger.info(f"Training on {len(training_data)} articles with identified suppliers")
        
        # Train models
        self.training_results = self.risk_classifier.train(training_data)
        
        # Save trained models
        self.risk_classifier.save_models("models/")
        
        # Generate training summary
        self._generate_training_summary()
        
        return self.training_results
    
    def run_predictions(self) -> pd.DataFrame:
        """
        Run predictions on the processed data
        
        Returns:
            pd.DataFrame: Data with predictions
        """
        logger.info("=== Starting Prediction Phase ===")
        
        if not self.risk_classifier.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Make predictions on all processed data
        self.predictions = self.risk_classifier.predict(self.processed_data)
        
        # Save predictions
        self.predictions.to_csv("results/predictions.csv", index=False)
        logger.info("Predictions saved to results/predictions.csv")
        
        return self.predictions
    
    def generate_results(self) -> None:
        """Generate comprehensive analysis results"""
        logger.info("=== Starting Results Generation Phase ===")
        
        if self.predictions is None:
            raise ValueError("Predictions must be generated before creating results")
        
        # Generate supplier risk analysis
        self._generate_supplier_risk_analysis()
        
        # Generate temporal risk trends
        self._generate_temporal_analysis()
        
        # Generate risk category summary
        self._generate_risk_category_summary()
        
        # Generate article classifications
        self._generate_article_classifications()
        
        logger.info("Results generation completed")
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations"""
        logger.info("=== Starting Visualization Generation Phase ===")
        
        if self.predictions is None:
            raise ValueError("Predictions must be generated before creating visualizations")
        
        # Generate all visualizations
        self.visualizer.generate_all_visualizations(
            self.predictions, 
            output_dir="visualizations/"
        )
        
        logger.info("Visualization generation completed")
    
    def _generate_data_summary(self) -> None:
        """Generate data processing summary"""
        summary = {
            'total_articles': len(self.processed_data),
            'articles_with_suppliers': self.processed_data['supplier'].notna().sum(),
            'unique_suppliers': self.processed_data['supplier'].nunique(),
            'date_range': {
                'start': self.processed_data['published_date'].min(),
                'end': self.processed_data['published_date'].max()
            },
            'risk_direction_distribution': self.processed_data['risk_direction'].value_counts().to_dict(),
            'supplier_distribution': self.processed_data['supplier'].value_counts().to_dict()
        }
        
        # Save summary
        import json
        with open('results/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Data summary saved to results/data_summary.json")
    
    def _generate_training_summary(self) -> None:
        """Generate model training summary"""
        if self.training_results:
            training_summary = {
                'training_samples': self.training_results.get('training_samples', 0),
                'features_count': self.training_results.get('features_count', 0),
                'risk_category_results': {},
                'risk_direction_results': {}
            }
            
            # Extract risk category results
            if 'risk_category_results' in self.training_results:
                for model_name, results in self.training_results['risk_category_results'].items():
                    training_summary['risk_category_results'][model_name] = {
                        'overall_f1': results.get('overall_f1', 0),
                        'category_metrics': results.get('category_metrics', {})
                    }
            
            # Extract risk direction results
            if 'risk_direction_results' in self.training_results:
                for model_name, results in self.training_results['risk_direction_results'].items():
                    training_summary['risk_direction_results'][model_name] = {
                        'f1_score': results.get('f1_score', 0)
                    }
            
            # Save summary
            import json
            with open('results/training_summary.json', 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
            
            logger.info("Training summary saved to results/training_summary.json")
    
    def _generate_supplier_risk_analysis(self) -> None:
        """Generate detailed supplier risk analysis"""
        # Aggregate risk scores by supplier
        supplier_analysis = self.predictions.groupby('supplier').agg({
            'geopolitical_regulatory': ['mean', 'std', 'count'],
            'agricultural_environmental': ['mean', 'std', 'count'],
            'financial_operational': ['mean', 'std', 'count'],
            'supply_chain_logistics': ['mean', 'std', 'count'],
            'market_competitive': ['mean', 'std', 'count'],
            'risk_direction': lambda x: (x == 'positive').sum() / len(x),
            'published_date': ['min', 'max']
        }).round(4)
        
        # Flatten column names
        supplier_analysis.columns = [f"{col[0]}_{col[1]}" for col in supplier_analysis.columns]
        
        # Calculate overall risk score
        risk_columns = [col for col in supplier_analysis.columns if col.endswith('_mean')]
        supplier_analysis['overall_risk_score'] = supplier_analysis[risk_columns].sum(axis=1)
        
        # Rank suppliers by risk
        supplier_analysis['risk_rank'] = supplier_analysis['overall_risk_score'].rank(ascending=False)
        
        # Save supplier risk analysis
        supplier_analysis.to_csv('results/supplier_risk_analysis.csv')
        logger.info("Supplier risk analysis saved to results/supplier_risk_analysis.csv")
    
    def _generate_temporal_analysis(self) -> None:
        """Generate temporal risk trends analysis"""
        # Prepare temporal data
        temporal_data = self.predictions.copy()
        temporal_data['published_date'] = pd.to_datetime(temporal_data['published_date'])
        temporal_data['year_month'] = temporal_data['published_date'].dt.to_period('M')
        
        # Aggregate by month
        temporal_trends = temporal_data.groupby('year_month').agg({
            'geopolitical_regulatory': 'mean',
            'agricultural_environmental': 'mean',
            'financial_operational': 'mean',
            'supply_chain_logistics': 'mean',
            'market_competitive': 'mean',
            'article_id': 'count'
        }).round(4)
        
        temporal_trends.columns = list(temporal_trends.columns[:-1]) + ['article_count']
        temporal_trends['total_risk'] = temporal_trends[self.visualizer.risk_categories].sum(axis=1)
        
        # Save temporal trends
        temporal_trends.to_csv('results/temporal_risk_trends.csv')
        logger.info("Temporal risk trends saved to results/temporal_risk_trends.csv")
    
    def _generate_risk_category_summary(self) -> None:
        """Generate risk category summary"""
        risk_summary = {}
        
        for category in self.visualizer.risk_categories:
            category_data = self.predictions[category]
            risk_summary[category] = {
                'mean_score': category_data.mean(),
                'std_score': category_data.std(),
                'max_score': category_data.max(),
                'min_score': category_data.min(),
                'articles_above_threshold': (category_data > 0.1).sum(),
                'top_suppliers': self.predictions.groupby('supplier')[category].mean().nlargest(3).to_dict()
            }
        
        # Save risk category summary
        import json
        with open('results/risk_category_summary.json', 'w') as f:
            json.dump(risk_summary, f, indent=2, default=str)
        
        logger.info("Risk category summary saved to results/risk_category_summary.json")
    
    def _generate_article_classifications(self) -> None:
        """Generate detailed article classifications"""
        # Select relevant columns for article classifications
        classification_columns = [
            'article_id', 'title', 'link', 'published_date', 'source', 
            'supplier', 'risk_direction'
        ] + self.visualizer.risk_categories
        
        article_classifications = self.predictions[classification_columns].copy()
        
        # Add predicted columns if available
        predicted_columns = [col for col in self.predictions.columns if col.startswith('predicted_')]
        if predicted_columns:
            article_classifications = pd.concat([
                article_classifications,
                self.predictions[predicted_columns]
            ], axis=1)
        
        # Calculate total risk score
        article_classifications['total_risk_score'] = article_classifications[self.visualizer.risk_categories].sum(axis=1)
        
        # Rank articles by risk
        article_classifications['risk_rank'] = article_classifications['total_risk_score'].rank(ascending=False)
        
        # Save article classifications
        article_classifications.to_csv('results/article_classifications.csv', index=False)
        logger.info("Article classifications saved to results/article_classifications.csv")
    
    def run_complete_pipeline(self, extract_missing_content: bool = False) -> dict:
        """
        Run the complete risk analysis pipeline
        
        Args:
            extract_missing_content (bool): Whether to extract content from URLs
            
        Returns:
            dict: Complete pipeline results
        """
        logger.info("=== STARTING COMPLETE RISK ANALYSIS PIPELINE ===")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Processing
            processed_data = self.run_data_processing(extract_missing_content)
            
            # Step 2: Model Training
            training_results = self.run_model_training()
            
            # Step 3: Predictions
            predictions = self.run_predictions()
            
            # Step 4: Results Generation
            self.generate_results()
            
            # Step 5: Visualizations
            self.generate_visualizations()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            # Final summary
            pipeline_summary = {
                'execution_time': str(execution_time),
                'total_articles_processed': len(processed_data),
                'articles_with_predictions': len(predictions),
                'unique_suppliers_identified': predictions['supplier'].nunique(),
                'models_trained': len(training_results.get('risk_category_results', {})),
                'visualizations_generated': True,
                'results_generated': True
            }
            
            # Save pipeline summary
            import json
            with open('results/pipeline_summary.json', 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total execution time: {execution_time}")
            logger.info(f"Results available in: results/ directory")
            logger.info(f"Visualizations available in: visualizations/ directory")
            
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise

def main():
    """Main function to run the risk analysis pipeline"""
    print("🚀 Starting Textile Supplier Risk Identification Pipeline")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = RiskAnalysisPipeline("suppliers_news.json")
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(extract_missing_content=False)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {results['total_articles_processed']} articles")
        print(f"🏭 Identified {results['unique_suppliers_identified']} suppliers")
        print(f"⏱️  Execution time: {results['execution_time']}")
        print("\n📁 Check the following directories for results:")
        print("   • results/ - Analysis results and summaries")
        print("   • visualizations/ - Interactive charts and graphs")
        print("   • models/ - Trained machine learning models")
        print("\n🌐 Run the Streamlit dashboard to explore results interactively:")
        print("   streamlit run dashboard/app.py")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Check the risk_analysis.log file for detailed error information")

if __name__ == "__main__":
    main() 