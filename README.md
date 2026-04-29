# Music Recommendation System

A comparative study of machine learning models for music recommendation using two approaches:
- **LightGBM**: A gradient boosting model with engineered features
- **Neural Collaborative Filtering (NCF)**: A PyTorch-based deep learning model

## Project Structure

```
├── main.py                     # Main pipeline orchestrator
├── config.py                   # Configuration and global settings
├── data_ingestion.py          # Data loading and cleaning
├── features.py                # Feature engineering
├── embedding_model.py         # Embedding utilities
├── evaluate.py                # Model evaluation and plotting
├── models/
│   ├── lgbm_model.py         # LightGBM model implementation
│   └── ncf_model.py          # PyTorch NCF model implementation
├── data/
│   ├── music_info.csv        # Song metadata
│   └── user_listening_history.csv  # User-song interactions
└── plots/                     # Output directory for evaluation plots
```

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load and clean the music interaction data
2. Apply negative sampling to create labeled examples
3. Split data into train/validation/test sets
4. Train the LightGBM model with feature engineering
5. Train the PyTorch NCF model
6. Generate ROC curves comparing both models

## Pipeline Overview

### Dataset used

[Kaggle link](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm?select=User+Listening+History.csv)

### Data Processing
- **Load & Clean**: Remove sparse users/songs, handle missing values
- **Negative Sampling**: Generate negative examples with configurable ratio
- **Train/Val/Test Split**: Random split by user to prevent data leakage

### LightGBM Pathway
- Feature engineering (statistical, temporal, collaborative features)
- sklearn pipeline with preprocessing (imputation, scaling, target encoding)
- Hyperparameter tuning with Optuna (20 trials)

### PyTorch NCF Pathway
- User and item embedding layers
- Multi-layer perceptron for interaction modeling
- Hyperparameter tuning with Optuna (10 trials)

### Evaluation
- ROC curves with AUC scores
- Individual and comparative visualizations
- Saved to `plots/` directory

## Configuration

Edit `config.py` to modify:
- `SEED`: Random seed for reproducibility (default: 42)
- `TARGET_INTERACTIONS`: Target number of interactions to sample
- `NEGATIVE_RATIO`: Ratio of negative to positive samples (default: 4)

## Output Files

- `lgbm_roc.png`: LightGBM ROC curve
- `ncf_roc.png`: PyTorch NCF ROC curve
- `model_comparison_roc.png`: Side-by-side comparison

## Dependencies

See `requirements.txt` for full list. Key packages:
- `pandas`, `numpy`: Data manipulation
- `polars`: Fast multi-threaded operations
- `scikit-learn`: Preprocessing and metrics
- `lightgbm`: Gradient boosting
- `torch`: Deep learning framework
- `optuna`: Hyperparameter optimization
- `matplotlib`: Visualization
