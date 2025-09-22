# MLB Pitcher-Hitter Matchup Analysis

A comprehensive machine learning pipeline for analyzing and predicting MLB pitcher-batter matchup outcomes using Statcast data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data Structure](#data-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project creates a sophisticated analysis framework for MLB pitcher-hitter matchups by:

- Processing raw Statcast data into structured, ML-ready datasets
- Creating comprehensive player profiles with advanced metrics
- Building interaction features that capture matchup dynamics
- Training machine learning models to predict various outcomes

The system is designed to answer questions like:
- "How will Mike Trout perform against Jacob deGrom's slider?"
- "What's the probability of a strikeout in this at-bat?"
- "Which pitcher matchup gives our team the best advantage?"

## 🚀 Features

### Data Processing
- **Multi-tier data structure** for different analysis needs
- **Comprehensive player profiles** with 50+ advanced metrics
- **Interaction feature engineering** (pitcher tendencies × batter weaknesses)
- **Situational context** (count, runners, pressure situations)

### Advanced Metrics
- **Pitcher Profiles**: Velocity, movement, command, stuff scores
- **Batter Profiles**: Contact quality, discipline, power, batted ball tendencies
- **Matchup Analytics**: Platoon splits, pitch-type specific performance
- **Contextual Stats**: Count-specific, situational, and leverage-based metrics

### Machine Learning
- **Multiple modeling approaches** for different prediction tasks
- **Feature interaction modeling** for true matchup analysis
- **Ensemble methods** combining different data granularities
- **Real-time prediction capabilities** (planned)

## 📊 Data Structure

The project uses a three-tier data architecture:

### Tier 1: Pitch-Level Data
```
pitch_level_enhanced.parquet
```
- Every single pitch with full context
- Game situation, count, runners, pressure
- Perfect for sequence analysis and deep learning

### Tier 2: Player-Specific Performance
```
pitcher_by_pitch_type.parquet    # Pitcher stats by pitch type
batter_by_pitch_type.parquet     # Batter stats by pitch type
batter_vs_handedness.parquet     # Platoon splits
```
- Detailed breakdowns by pitch type and situation
- Ideal for specific matchup analysis

### Tier 3: Overall Player Summaries
```
pitcher_overall_summary.parquet  # Comprehensive pitcher profiles
batter_overall_summary.parquet   # Comprehensive batter profiles
```
- High-level player quality metrics
- Usage-weighted statistics
- Quick lookup and comparison tables

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/jhbradee/Pitcher-Hitter-Analysis.git
cd pitcher-hitter-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed
```

### Dependencies
```txt
pandas>=1.5.0
numpy>=1.23.0
pybaseball>=2.2.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📖 Usage

### 1. Data Collection and Processing

```python
# Run the data transformation pipeline
python transform_data.py

# This will create:
# - Enhanced pitch-level dataset
# - Pitcher profiles (overall + by pitch type)
# - Batter profiles (overall + by pitch type + platoon splits)
```

### 2. Machine Learning Pipeline ⚠️ *Work in Progress*

```python
# Train models for matchup prediction
python ml_pipeline.py

# This will:
# - Create interaction features
# - Train multiple model types
# - Evaluate performance
# - Generate feature importance reports
```

### 3. Analysis Examples

```python
import pandas as pd

# Load processed data
pitcher_profiles = pd.read_parquet('data/processed/pitcher_overall_summary.parquet')
batter_profiles = pd.read_parquet('data/processed/batter_overall_summary.parquet')

# Analyze a specific matchup
def analyze_matchup(pitcher_name, batter_name):
    pitcher_stats = pitcher_profiles[pitcher_profiles['formatted_name'].str.contains(pitcher_name)]
    batter_stats = batter_profiles[batter_profiles['formatted_name'].str.contains(batter_name)]
    
    # Your analysis code here
    return matchup_analysis
```

## 🗂 Project Structure

```
mlb-pitcher-hitter-analysis/
│
├── data/
│   ├── raw/                    # Raw Statcast parquet files
│   └── processed/              # Processed, ML-ready datasets
│
├── src/
│   ├── pitcher_profile_creation.py         # Pitcher data processing pipeline
│   ├── batter_profile_creation.py         # Pitcher data processing pipeline
│   ├── ml_pipeline.py                      # ML training pipeline (WIP)
│   └── utils/                              # Helper functions
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── tests/
│   └── test_data_processing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 Data Processing Pipeline

### Step 1: Raw Data Loading
```python
# Loads all parquet files from data/raw/
# Combines multiple seasons into single dataset
# Handles missing data and basic cleaning
```

### Step 2: Feature Engineering
```python
# Creates derived metrics:
# - Strike zone indicators
# - Contact quality measures  
# - Situational context
# - Pressure indicators
```

### Step 3: Player Profiling
```python
# Pitcher profiles:
# - Velocity, movement, spin rate by pitch type
# - Command and stuff metrics
# - Count-specific tendencies
# - Platoon splits

# Batter profiles:
# - Contact quality and discipline
# - Batted ball tendencies
# - Performance vs different pitch types
# - Situational hitting
```

### Step 4: Interaction Features
```python
# Creates matchup-specific features:
# - Pitcher velocity × Batter contact ability
# - Command × Plate discipline
# - Stuff × Swing decisions
# - Power suppression × Power generation
```

## 🤖 Machine Learning Models

### Current Models ⚠️ *Work in Progress*

#### At-Bat Outcome Prediction
- **Target**: Hit probability, strikeout probability, walk probability
- **Features**: Overall pitcher/batter profiles + interactions + situation
- **Models**: Random Forest, Gradient Boosting
- **Use Case**: General matchup evaluation

#### Pitch-Specific Outcome Prediction
- **Target**: Contact quality (exit velocity, hard contact probability)
- **Features**: Pitch-type specific profiles + pitch characteristics
- **Models**: Gradient Boosting Regressor
- **Use Case**: "What happens if pitcher throws a slider here?"

#### Planned Enhancements
- [ ] Pitch sequence modeling with LSTM
- [ ] Multi-task learning (predict multiple outcomes simultaneously)
- [ ] Hierarchical models (pitch type → outcome given type)
- [ ] Real-time inference API
- [ ] Model retraining pipeline

### Key Innovation: Interaction Effects

Traditional analysis looks at players in isolation. This project creates features that capture true matchup dynamics:

```python
# Examples of interaction features:
velocity_vs_contact = pitcher_velocity × (1 - batter_contact_rate)
stuff_vs_discipline = pitcher_stuff_score × (1 - batter_discipline_score)
command_vs_chase = pitcher_command × batter_chase_rate
```

## 📈 Performance Metrics

### Model Evaluation
- **Classification**: Accuracy, Precision, Recall, ROC-AUC
- **Regression**: R², RMSE, Mean Absolute Error
- **Calibration**: Reliability diagrams for probability predictions
- **Feature Importance**: SHAP values for interpretability

### Validation Strategy
- **Time-based splits**: Train on earlier seasons, test on recent
- **Cross-validation**: 5-fold CV within seasons
- **Hold-out testing**: Reserve latest month for final evaluation

## 🚧 Current Status

### ✅ Completed
- [x] Raw data processing pipeline
- [x] Comprehensive pitcher profiling
- [x] Comprehensive batter profiling
- [x] Three-tier data architecture
- [x] Feature engineering framework
- [x] Interaction feature creation

### 🔄 In Progress
- [ ] Machine learning pipeline completion
- [ ] Model training and evaluation
- [ ] Performance benchmarking
- [ ] Documentation and examples

### 📋 Planned
- [ ] Real-time prediction API
- [ ] Interactive dashboard
- [ ] Advanced sequence modeling
- [ ] Park factor integration
- [ ] Injury/fatigue modeling

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
```

### Areas for Contribution
- Model improvements and new algorithms
- Feature engineering ideas
- Performance optimization
- Documentation and examples
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLB & Statcast** for providing comprehensive baseball data
- **pybaseball** for excellent Python interface to baseball data
- **scikit-learn** and the broader Python ML ecosystem
- The **sabermetrics community** for advanced baseball analytics research

## 📞 Contact

- **Author**: Jessica Bradee
- **Email**: jhbradee@gmail.com
- **Project Link**: https://github.com/yourusername/mlb-pitcher-hitter-analysis

---

## 📊 Data Sources

This project uses publicly available MLB Statcast data accessed through the `pybaseball` library. All data is used in accordance with MLB's data usage policies.

**Note**: This project is for educational and research purposes. Any commercial use should comply with appropriate data licensing agreements.
