# Machine learning models for risk classification in textile supplier analysis
# This module implements various ML approaches for classifying news articles into
# risk categories and determining risk directions (positive/negative impact)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from textblob import TextBlob
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskClassifier:
    """
    Advanced machine learning classifier for textile supplier risk analysis
    """
    
    def __init__(self):
        """Initialize the risk classifier"""
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.label_encoders = {}
        self.is_trained = False
        
        # Risk categories
        self.risk_categories = [
            'geopolitical_regulatory',
            'agricultural_environmental', 
            'financial_operational',
            'supply_chain_logistics',
            'market_competitive'
        ]
        
        # Initialize model configurations
        self._setup_models()
    
    def _setup_models(self):
        """Setup different model configurations"""
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'name': 'Random Forest'
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'name': 'XGBoost'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    multi_class='ovr'
                ),
                'name': 'Logistic Regression'
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                ),
                'name': 'LightGBM'
            }
        }
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        features_df = df.copy()
        
        # Text length features
        features_df['title_length'] = features_df['title'].str.len()
        features_df['text_word_count'] = features_df['processed_text'].str.split().str.len()
        
        # Sentiment analysis features
        features_df['sentiment_polarity'] = features_df['processed_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if x else 0
        )
        features_df['sentiment_subjectivity'] = features_df['processed_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity if x else 0
        )
        
        # Temporal features
        if 'published_date' in features_df.columns:
            features_df['published_date'] = pd.to_datetime(features_df['published_date'])
            features_df['year'] = features_df['published_date'].dt.year
            features_df['month'] = features_df['published_date'].dt.month
            features_df['day_of_week'] = features_df['published_date'].dt.dayofweek
            features_df['quarter'] = features_df['published_date'].dt.quarter
        
        # Source credibility (based on known reliable sources)
        reliable_sources = [
            'Reuters', 'Bloomberg', 'Financial Times', 'Wall Street Journal',
            'BBC', 'CNN', 'Associated Press', 'The Guardian'
        ]
        features_df['is_reliable_source'] = features_df['source'].isin(reliable_sources).astype(int)
        
        # Supplier encoding
        if 'supplier' in features_df.columns:
            supplier_encoder = LabelEncoder()
            features_df['supplier_encoded'] = supplier_encoder.fit_transform(
                features_df['supplier'].fillna('Unknown')
            )
            self.label_encoders['supplier'] = supplier_encoder
        
        return features_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for machine learning models
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Features, risk labels, direction labels
        """
        # Extract additional features
        features_df = self.extract_features(df)
        
        # Prepare text features using TF-IDF
        if 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            
            text_features = self.vectorizers['tfidf'].fit_transform(
                features_df['processed_text'].fillna('')
            )
        else:
            text_features = self.vectorizers['tfidf'].transform(
                features_df['processed_text'].fillna('')
            )
        
        # Prepare numerical features
        numerical_columns = [
            'title_length', 'text_word_count', 'sentiment_polarity',
            'sentiment_subjectivity', 'is_reliable_source'
        ]
        
        # Add temporal features if available
        if 'year' in features_df.columns:
            numerical_columns.extend(['year', 'month', 'day_of_week', 'quarter'])
        
        # Add supplier encoding if available
        if 'supplier_encoded' in features_df.columns:
            numerical_columns.append('supplier_encoded')
        
        numerical_features = features_df[numerical_columns].fillna(0).values
        
        # Scale numerical features
        if 'numerical' not in self.scalers:
            self.scalers['numerical'] = StandardScaler()
            numerical_features_scaled = self.scalers['numerical'].fit_transform(numerical_features)
        else:
            numerical_features_scaled = self.scalers['numerical'].transform(numerical_features)
        
        # Combine text and numerical features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([
            text_features,
            csr_matrix(numerical_features_scaled)
        ])
        
        # Prepare risk category labels (multi-label)
        risk_labels = features_df[self.risk_categories].values
        
        # Prepare risk direction labels
        direction_encoder = LabelEncoder()
        direction_labels = direction_encoder.fit_transform(features_df['risk_direction'])
        self.label_encoders['risk_direction'] = direction_encoder
        
        return combined_features, risk_labels, direction_labels
    
    def create_binary_risk_labels(self, df: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
        """
        Create binary risk labels based on risk scores
        
        Args:
            df (pd.DataFrame): Input dataframe with risk scores
            threshold (float): Threshold for binary classification
            
        Returns:
            np.ndarray: Binary risk labels
        """
        risk_scores = df[self.risk_categories].values
        return (risk_scores > threshold).astype(int)
    
    def train_risk_category_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train models for risk category classification
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Risk category labels
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info("Training risk category classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
        )
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {config['name']} for risk categories...")
                
                # Use MultiOutputClassifier for multi-label classification
                model = MultiOutputClassifier(config['model'])
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics for each risk category
                category_metrics = {}
                for i, category in enumerate(self.risk_categories):
                    f1 = f1_score(y_test[:, i], y_pred[:, i])
                    category_metrics[category] = f1
                
                # Overall F1 score (macro average)
                overall_f1 = np.mean(list(category_metrics.values()))
                
                results[model_name] = {
                    'model': model,
                    'category_metrics': category_metrics,
                    'overall_f1': overall_f1,
                    'predictions': y_pred,
                    'true_labels': y_test
                }
                
                logger.info(f"{config['name']} - Overall F1: {overall_f1:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {config['name']}: {e}")
                continue
        
        # Select best model based on overall F1 score
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['overall_f1'])
            self.models['risk_category'] = results[best_model_name]['model']
            logger.info(f"Best risk category model: {best_model_name}")
        
        return results
    
    def train_risk_direction_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train models for risk direction classification
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Risk direction labels
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        logger.info("Training risk direction classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {config['name']} for risk direction...")
                
                model = config['model']
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results[model_name] = {
                    'model': model,
                    'f1_score': f1,
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'classification_report': classification_report(
                        y_test, y_pred, 
                        target_names=self.label_encoders['risk_direction'].classes_
                    )
                }
                
                logger.info(f"{config['name']} - F1 Score: {f1:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {config['name']}: {e}")
                continue
        
        # Select best model based on F1 score
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
            self.models['risk_direction'] = results[best_model_name]['model']
            logger.info(f"Best risk direction model: {best_model_name}")
        
        return results
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models on the provided dataset
        
        Args:
            df (pd.DataFrame): Training dataset
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info("Starting model training...")
        
        # Prepare training data
        X, y_risk, y_direction = self.prepare_training_data(df)
        
        # Convert risk scores to binary labels
        y_risk_binary = self.create_binary_risk_labels(df)
        
        # Train risk category models
        risk_category_results = self.train_risk_category_models(X, y_risk_binary)
        
        # Train risk direction models
        risk_direction_results = self.train_risk_direction_models(X, y_direction)
        
        self.is_trained = True
        
        results = {
            'risk_category_results': risk_category_results,
            'risk_direction_results': risk_direction_results,
            'training_samples': len(df),
            'features_count': X.shape[1]
        }
        
        logger.info("Model training completed successfully!")
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Predictions dataframe
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        X, _, _ = self.prepare_training_data(df)
        
        predictions_df = df.copy()
        
        # Predict risk categories
        if 'risk_category' in self.models:
            risk_predictions = self.models['risk_category'].predict(X)
            
            for i, category in enumerate(self.risk_categories):
                predictions_df[f'predicted_{category}'] = risk_predictions[:, i]
        
        # Predict risk directions
        if 'risk_direction' in self.models:
            direction_predictions = self.models['risk_direction'].predict(X)
            direction_labels = self.label_encoders['risk_direction'].inverse_transform(direction_predictions)
            predictions_df['predicted_risk_direction'] = direction_labels
        
        return predictions_df
    
    def save_models(self, model_dir: str = "models/"):
        """
        Save trained models and preprocessors
        
        Args:
            model_dir (str): Directory to save models
        """
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{model_name}_model.pkl")
        
        # Save vectorizers
        for vec_name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f"{model_dir}/{vec_name}_vectorizer.pkl")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/{scaler_name}_scaler.pkl")
        
        # Save label encoders
        for encoder_name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{model_dir}/{encoder_name}_encoder.pkl")
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "models/"):
        """
        Load trained models and preprocessors
        
        Args:
            model_dir (str): Directory containing saved models
        """
        import os
        
        try:
            # Load models
            for model_file in os.listdir(model_dir):
                if model_file.endswith('_model.pkl'):
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(f"{model_dir}/{model_file}")
                
                elif model_file.endswith('_vectorizer.pkl'):
                    vec_name = model_file.replace('_vectorizer.pkl', '')
                    self.vectorizers[vec_name] = joblib.load(f"{model_dir}/{model_file}")
                
                elif model_file.endswith('_scaler.pkl'):
                    scaler_name = model_file.replace('_scaler.pkl', '')
                    self.scalers[scaler_name] = joblib.load(f"{model_dir}/{model_file}")
                
                elif model_file.endswith('_encoder.pkl'):
                    encoder_name = model_file.replace('_encoder.pkl', '')
                    self.label_encoders[encoder_name] = joblib.load(f"{model_dir}/{model_file}")
            
            self.is_trained = True
            logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")

def main():
    """Main function to demonstrate the risk classifier"""
    # This would typically load processed data and train models
    print("Risk Classifier module loaded successfully!")
    print("Use this module to train and deploy risk classification models.")

if __name__ == "__main__":
    main() 