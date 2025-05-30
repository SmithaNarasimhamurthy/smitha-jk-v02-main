# 🏭 Textile Supplier Risk Identification System

A comprehensive data science solution for identifying and analyzing supply chain risks in textile dye suppliers using advanced NLP and machine learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Results & Outputs](#results--outputs)
- [Dashboard](#dashboard)
- [Technical Details](#technical-details)
- [Contributing](#contributing)

## 🎯 Overview

This project analyzes news articles related to 10 major textile dye suppliers to identify potential supply chain risks across five key categories:

1. **Geopolitical and Regulatory Risks**
2. **Agricultural and Environmental Risks**
3. **Financial and Operational Risks**
4. **Supply Chain and Logistics Risks**
5. **Market and Competitive Risks**

### Target Suppliers

- Welspun Living Limited
- Teejay Lanka PLC
- Arvind Limited
- Caleres, Inc.
- Interloop Limited
- Kitex Garments Limited
- ThredUp Inc.
- G-III Apparel Group, Ltd.
- Mint Velvet
- White Stuff Limited

## ✨ Features

### Core Functionality
- **Advanced NLP Pipeline**: Text preprocessing, sentiment analysis, and feature extraction
- **Machine Learning Models**: Multi-label classification for risk categories
- **Risk Direction Analysis**: Determines if risks are increasing or decreasing
- **Temporal Analysis**: Tracks risk evolution over time (2023-2024)
- **Supplier Profiling**: Individual risk profiles for each supplier

### Visualizations
- Interactive risk heatmaps
- Temporal trend analysis
- Supplier comparison radar charts
- Risk distribution charts
- Word clouds for risk categories
- Network analysis of supplier relationships

### Dashboard
- **Streamlit Web Interface**: Interactive exploration of results
- **Multi-page Navigation**: Overview, supplier analysis, temporal trends, risk categories
- **Search & Filter**: Advanced filtering and search capabilities
- **Export Functionality**: Download filtered results

## 📁 Project Structure

```
smitha-second-approach/
├── APPROACH.md                    # Detailed methodology documentation
├── ASSESSMENT.md                  # Project requirements
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── main.py                        # Main pipeline script
├── suppliers_news.json           # Input data (news articles)
│
├── src/                          # Source code modules
│   ├── data_processor.py         # Data loading and preprocessing
│   ├── risk_classifier.py        # Machine learning models
│   └── visualizations.py         # Visualization engine
│
├── data/                         # Processed data files
│   └── processed_articles.csv    # Cleaned and processed articles
│
├── models/                       # Trained ML models
│   ├── risk_category_model.pkl   # Risk classification model
│   ├── risk_direction_model.pkl  # Risk direction model
│   └── *.pkl                     # Vectorizers, scalers, encoders
│
├── results/                      # Analysis results
│   ├── supplier_risk_analysis.csv # Supplier risk profiles
│   ├── temporal_risk_trends.csv   # Time-series risk data
│   ├── article_classifications.csv # Article-level results
│   └── *.json                     # Summary statistics
│
├── visualizations/               # Generated charts and graphs
│   ├── supplier_risk_heatmap.html
│   ├── risk_trends.html
│   └── *.html, *.png             # Interactive and static visualizations
│
├── dashboard/                    # Streamlit dashboard
│   └── app.py                    # Dashboard application
│
└── tests/                        # Unit tests (for future development)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd smitha-second-approach
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data (First Run Only)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## ⚡ Quick Start

### Run the Complete Pipeline
```bash
python main.py
```

This will:
1. ✅ Load and process the news articles
2. ✅ Train machine learning models
3. ✅ Generate risk predictions
4. ✅ Create comprehensive results
5. ✅ Generate interactive visualizations

# After running this, manually copy both data and results directory and paste it inside dashboard directory

### Launch the First Dashboard
```bash
streamlit run dashboard/app.py
```

### Launch the Second Dashboard
```bash
streamlit run dashboard/app-visual.py
```

Access the dashboard at: `http://localhost:8501`

## 📖 Usage Guide

### 1. Data Processing
```python
from src.data_processor import SupplierDataProcessor

# Initialize processor
processor = SupplierDataProcessor('suppliers_news.json')

# Load and process data
raw_data = processor.load_data()
processed_data = processor.process_articles()

# Save results
processor.save_processed_data('data/processed_articles.csv')
```

### 2. Model Training
```python
from src.risk_classifier import RiskClassifier

# Initialize classifier
classifier = RiskClassifier()

# Train models
training_results = classifier.train(processed_data)

# Save models
classifier.save_models('models/')
```

### 3. Generate Visualizations
```python
from src.visualizations import RiskVisualizationEngine

# Initialize visualizer
visualizer = RiskVisualizationEngine()

# Generate all visualizations
visualizer.generate_all_visualizations(predictions, 'visualizations/')
```

### 4. Custom Analysis
```python
# Load results for custom analysis
import pandas as pd

# Load article classifications
articles = pd.read_csv('results/article_classifications.csv')

# Load supplier analysis
suppliers = pd.read_csv('results/supplier_risk_analysis.csv')

# Perform custom analysis
high_risk_suppliers = suppliers[suppliers['overall_risk_score'] > threshold]
```

## 📊 Results & Outputs

### Generated Files

#### Data Files
- `data/processed_articles.csv` - Cleaned and processed articles with risk scores
- `results/article_classifications.csv` - Individual article risk classifications
- `results/supplier_risk_analysis.csv` - Supplier-level risk aggregations
- `results/temporal_risk_trends.csv` - Monthly risk trend data

#### Summary Files (JSON)
- `results/data_summary.json` - Data processing statistics
- `results/training_summary.json` - Model training metrics
- `results/risk_category_summary.json` - Risk category analysis
- `results/pipeline_summary.json` - Overall pipeline results

#### Visualizations
- `visualizations/supplier_risk_heatmap.html` - Interactive risk heatmap
- `visualizations/risk_trends.html` - Temporal trend analysis
- `visualizations/supplier_comparison_radar.html` - Radar chart comparisons
- `visualizations/wordcloud_*.png` - Word clouds for each risk category

### Key Metrics
- **Risk Scores**: 0-1 scale for each risk category
- **Risk Direction**: Positive (increased risk) or Negative (decreased risk)
- **Risk Rank**: Ranking of suppliers by overall risk
- **Temporal Trends**: Monthly aggregated risk scores

## 🌐 Dashboard

The Streamlit dashboard provides interactive exploration with the following pages:

### 📊 Overview
- Summary statistics and key metrics
- Risk distribution by supplier
- Risk category heatmap
- Recent high-risk articles

### 🏭 Supplier Analysis
- Individual supplier risk profiles
- Radar chart comparisons
- Detailed metrics and breakdowns
- Risk ranking and trends

### 📅 Temporal Analysis
- Risk trends over time
- High-risk events timeline
- Monthly aggregations
- Seasonal patterns

### 🏷️ Risk Categories
- Category-specific analysis
- Distribution charts
- Top suppliers by category
- Highest risk articles

### 🔍 Search & Analysis
- Advanced search functionality
- Multi-criteria filtering
- Export capabilities
- Custom analysis tools

### ℹ️ About
- Methodology documentation
- Technical details
- Data sources
- Model performance metrics

## 🔧 Technical Details

### Architecture
- **Data Processing**: Pandas, NLTK, BeautifulSoup
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **NLP**: TextBlob, TF-IDF, Sentiment Analysis
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Models**: Multi-label classification, ensemble methods

### Model Performance
- **Risk Category Classification**: Multi-output classifier with F1-score optimization
- **Risk Direction**: Binary classification with sentiment features
- **Feature Engineering**: TF-IDF, sentiment, temporal, and metadata features
- **Evaluation**: Cross-validation with temporal splits

### Scalability
- **Modular Design**: Easy to add new suppliers or risk categories
- **Configurable Parameters**: Adjustable thresholds and model settings
- **Extensible Framework**: Plugin architecture for new features

## 🎯 Use Cases

### Supply Chain Management
- **Risk Monitoring**: Early warning system for supply chain disruptions
- **Supplier Assessment**: Data-driven supplier risk evaluation
- **Strategic Planning**: Long-term risk mitigation strategies

### Business Intelligence
- **Competitive Analysis**: Monitor competitor risks and opportunities
- **Market Intelligence**: Track industry trends and patterns
- **Investment Decisions**: Risk-informed investment strategies

### Research & Development
- **Academic Research**: Framework for supply chain risk analysis
- **Methodology Development**: NLP applications in business intelligence
- **Data Science**: End-to-end ML pipeline example

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd smitha-second-approach

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Include tests and documentation
5. Submit a pull request

### Areas for Contribution
- Additional risk categories
- Enhanced NLP models
- Real-time data integration
- Performance optimizations
- UI/UX improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions, suggestions, or support:
- Create an issue on GitHub
- Contact the development team
- Join our community discussions

## 🙏 Acknowledgments

- News data sources and publishers
- Open-source libraries and frameworks
- Textile industry domain experts
- Data science community contributions

---

**Built with ❤️ for supply chain risk management** 