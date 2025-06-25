# MLOps Group 8 - Spotify Music Popularity Prediction

[![CI](https://github.com/adriansotomora/mlops_group8/actions/workflows/ci.yml/badge.svg)](https://github.com/adriansotomora/mlops_group8/actions/workflows/ci.yml)

This repository provides a modular, **production-quality** MLOps pipeline for Spotify music popularity prediction, built as part of an academic project in MLOps fundamentals. The codebase is designed to bridge the gap between research prototypes (Jupyter notebooks) and scalable, maintainable machine learning systems in production.

---

## 🚦 Project Status

**Phase 1: Modularization, Testing, and Best Practices**
- Jupyter notebook translated into well-documented, test-driven Python modules
- End-to-end pipeline: data ingestion, validation, preprocessing, feature engineering, model training, evaluation, and batch inference
- Robust configuration via `config.yaml` and reproducibility through explicit artifact management
- Extensive unit testing with pytest (89 total tests)
- Strict adherence to software engineering and MLOps best practices

**Phase 2: Hydra, MLflow, and W&B Integration**
- All pipeline steps now execute as MLflow runs
- Dynamic configuration managed with Hydra
- Metrics and artifacts automatically logged to Weights & Biases
- Automated training pipeline with CI/CD workflows

**Phase 3: CI/CD and FastAPI Serving**
- GitHub Actions workflow with automated testing and linting
- FastAPI application (`app/main.py`) exposes prediction and health endpoints
- Includes `/predict_batch` route for validating and scoring multiple records
- Docker containerization for scalable deployment

---

## 📁 Repository Structure

```text
.
├── README.md
├── config.yaml                  # Central pipeline configuration
├── environment.yml              # Reproducible conda environment
├── conda.yaml                   # MLflow conda environment
├── dvc.yaml                     # Data Version Control pipeline
├── Dockerfile                   # Container deployment
├── data/
│   ├── raw/                     # Original Spotify songs dataset
│   ├── raw_holdout/             # Holdout data for inference testing
│   ├── processed/               # Intermediate processed datasets
│   └── predictions/             # Model prediction outputs
├── models/                      # Serialized models and preprocessing artifacts
├── logs/                        # Pipeline execution logs and validation reports
├── app/
│   └── main.py                  # FastAPI application for model serving
├── .github/workflows/           # GitHub Actions CI/CD pipeline
├── src/
│   ├── data_load/               # Data ingestion utilities
│   ├── data_validation/         # Schema and data quality validation
│   ├── preprocess/              # Data preprocessing and scaling
│   ├── features/                # Feature engineering (polynomial, genre parsing)
│   ├── model/                   # Model training with stepwise selection
│   ├── evaluation/              # Model evaluation and metrics calculation
│   ├── inference/               # Batch inference with prediction intervals
│   └── main.py                  # Pipeline orchestrator
└── tests/                       # Comprehensive unit and integration tests
```

Artifact paths in `config.yaml` such as `data/processed`, `models/`, and `logs/` are resolved relative to this project root. Generated metrics, preprocessing pipelines, and trained models are versioned and stored under these directories.

---

## 🔬 Problem Description

The pipeline predicts **Spotify track popularity** (0-100 scale) based on audio features and metadata. It applies rigorous validation and modular design suitable for research, teaching, and real-world deployment scenarios.

### Data Dictionary

| Feature           | Description                                                    | Type      | Range/Values           |
|-------------------|----------------------------------------------------------------|-----------|------------------------|
| track_popularity  | Target variable: Spotify popularity score                     | int       | 0-100                  |
| danceability      | How suitable a track is for dancing                           | float     | 0.0-1.0                |
| energy            | Perceptual measure of intensity and power                      | float     | 0.0-1.0                |
| loudness          | Overall loudness of track in decibels                         | float     | min: -60.0             |
| speechiness       | Presence of spoken words in a track                           | float     | 0.0-1.0                |
| acousticness      | Confidence measure of whether track is acoustic               | float     | 0.0-1.0                |
| liveness          | Detects presence of audience in recording                      | float     | 0.0-1.0                |
| valence           | Musical positiveness conveyed by track                        | float     | 0.0-1.0                |
| tempo             | Overall estimated tempo in beats per minute                   | float     | min: 0.0               |
| duration_ms       | Track duration in milliseconds                                | float     | min: 0.0               |
| artist_popularity | Popularity of the artist                                       | int       | 0-100                  |
| key               | Key the track is in (using standard Pitch Class notation)     | float     | 0.0-11.0               |
| mode              | Major (1) or minor (0) modality                               | float     | 0.0-1.0                |
| artist_genres     | List of genres associated with the artist                     | str       | Genre strings          |

*Engineered features include genre dummy variables and polynomial combinations of audio features.*

---

## 🛠️ Pipeline Modules

### 0. Pipeline Orchestration (`src/main.py`)
- Single entry point orchestrating the entire MLOps workflow
- Supports configurable pipeline stages via Hydra configuration
- MLflow project execution with parameter passing and artifact tracking
- Robust logging and error handling for production environments

### 1. Data Loading (`src/data_load/data_loader.py`)
- Loads Spotify songs data from CSV with configurable parameters
- Environment variable management for secure configurations
- Comprehensive error handling and logging

### 2. Data Validation (`src/data_validation/data_validator.py`)
- Schema validation: column types, ranges, missing values, data quality checks
- Configurable strictness: `raise` or `warn` on validation errors
- Detailed validation reports (JSON) with HTML summaries logged to W&B
- Ensures data integrity throughout the pipeline

### 3. Data Preprocessing (`src/preprocess/preprocessing.py`)
- Raw data cleaning and outlier removal using IQR method
- Train/test/holdout splitting with stratification
- Feature scaling (MinMax/Standard) with sklearn pipelines
- Creates inference holdout set for unbiased evaluation

### 4. Feature Engineering (`src/features/features.py`)
- Genre parsing from artist metadata into dummy variables
- Polynomial feature generation for audio features interaction
- Configurable feature selection and exclusion rules
- Feature importance tracking and documentation

### 5. Model Training (`src/model/model.py`)
- Linear regression with stepwise feature selection using statsmodels
- Automated feature selection based on statistical significance
- Model registry supporting multiple algorithms (easily extensible)
- Hyperparameter tracking and model versioning

### 6. Model Evaluation (`src/evaluation/evaluator.py`)
- Comprehensive regression metrics: RMSE, MAE, R², Adjusted R²
- Residual analysis and prediction interval calculations
- Model performance visualization and reporting
- Statistical validation of model assumptions

### 7. Batch Inference (`src/inference/inferencer.py`)
- Loads preprocessing and model artifacts for new data prediction
- Applies identical transformations to ensure consistency
- Generates predictions with 95% confidence intervals
- Exports results with original data for analysis

### 8. Unit Testing (`tests/`)
- Comprehensive pytest suite covering all modules (89 total tests)
- Mock data generation for isolated testing
- CI/CD integration with coverage requirements

---

## ⚙️ Configuration and Reproducibility

- **config.yaml**: Centralized configuration for all pipeline parameters
- **environment.yml**: Reproducible Conda environment with exact versions
- **MLflow Projects**: Reproducible pipeline execution with parameter tracking
- **W&B Artifacts**: Versioned data, models, and experiment artifacts
- **Hydra**: Dynamic configuration overrides and environment management

---

## 🚀 Quickstart

**Environment setup:**
```bash
# Clone repository
git clone https://github.com/adriansotomora/mlops_group8.git
cd mlops_group8

# Create and activate conda environment
conda env create -f environment.yml
conda activate group8_full

# Set up environment variables (optional - values also configured in config.yaml)
# Create .env file with your Weights & Biases credentials:
# WANDB_API_KEY=your_wandb_api_key
# Alternatively, update config.yaml with your WANDB_ENTITY

# Authenticate with Weights & Biases
wandb login
```

**Run end-to-end pipeline:**
```bash
# Run complete pipeline using MLflow (recommended)
mlflow run . -P steps=all

# Run via Python with Hydra
python main.py main.steps=all

# Run specific stages
mlflow run . -P steps="preprocess,features,model"

# Run with hyperparameter overrides using MLflow
mlflow run . -e main_with_override -P steps=all -P hydra_options="model.linear_regression.stepwise.threshold_in=0.01"
```

All steps automatically log metrics and artifacts to W&B. The pipeline creates a holdout dataset during preprocessing for unbiased inference testing.

**Run inference:**
```bash
# Batch inference on new data
python src/main.py main.steps=inference

# Direct inference script
python -m src.inference.inferencer --input data/raw_holdout/inference_holdout_data.csv --output data/predictions/results.csv
```

**Serve model via FastAPI:**
```bash
# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "artist_popularity": 63,
       "genre": "rock",
       "danceability": 0.735,
       "energy": 0.578,
       "liveness": 0.0985,
       "loudness": -11.84,
       "speechiness": 0.0596,
       "tempo": 75.0,
       "valence": 0.471,
       "duration_ms": 207959.0,
       "acousticness": 0.514,
       "key": 1.0
     }'
```

**Run tests:**
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run linting
flake8 .
```

---

## 🐳 Docker Deployment

Before building, set environment variables in `.env`:

```bash
WANDB_PROJECT=mlops_group8
WANDB_ENTITY=your-entity
WANDB_API_KEY=your-api-key
```

Build and run Docker container:
```bash
# Build image
docker build -t spotify-popularity-api .

# Run container
docker run --env-file .env -p 8000:8000 spotify-popularity-api
```

The server automatically detects the `PORT` environment variable, making it compatible with cloud platforms like Render, Heroku, or AWS ECS.

---

## 📈 Next Steps

- **Model Enhancement**: Implement additional algorithms (Random Forest, XGBoost) based on current model registry
- **Model Monitoring**: Enhance logging and metrics tracking for production deployments
- **Automated Deployment**: Extend CI/CD pipeline for automatic model deployment
- **Feature Store**: Centralized feature management and versioning
- **Performance Optimization**: Pipeline efficiency improvements and caching

---

## 📚 Academic and Teaching Notes

- **Course**: MBD-EN2024ELECTIVOS-MBDMCSBT_38E88_467445 - MLOPS: MACHINE LEARNING OPERATIONS
- **Institution**: IE University
- **Best Practices**: Demonstrates academic excellence in MLOps engineering, suitable for production deployment
- **Extensibility**: Configuration-driven architecture enables easy extension for advanced MLOps topics
- **Assessment**: Comprehensive test suite and documentation support peer evaluation and grading

### Learning Objectives Achieved:
1. **Pipeline Modularization**: Jupyter notebook → Production-ready modules
2. **MLOps Toolchain**: MLflow, W&B, Hydra integration
3. **CI/CD Implementation**: Automated testing and deployment
4. **Model Serving**: RESTful API with FastAPI
5. **Reproducibility**: Environment and experiment management

---

## 👩‍💻 Authors and Acknowledgments

**Group 8 Members:**
- ADRIAN SOTO MORA
- MANUEL EDUARDO BONNELLY SANCHEZ  
- SERGIO LEBED WRIGHT
- YEABSIRA SELESHI
- HIROMITSU FUJIYAMA

**Course Information:**
- **Institution**: IE University
- **Program**: Master in Big Data & Analytics
- **Course**: MLOPS: Machine Learning Operations
- **Academic Year**: 2024-2025

**Acknowledgments:**
- Course instructors for MLOps fundamentals and best practices
- Spotify for providing rich audio feature datasets
- Open-source MLOps community for tools and methodologies

---

## 📜 License

This project is developed for academic and educational purposes as part of the IE University MLOps course. 

**Academic Use License:**
- ✅ Educational and research use permitted
- ✅ Code sharing within academic community encouraged  
- ✅ Modification and extension for learning purposes allowed
- ❌ Commercial use prohibited without explicit permission
- ❌ Distribution outside academic context requires attribution

For questions, collaborations, or commercial licensing inquiries, please contact the authors or course instructors.

---

**Project Repository**: [https://github.com/adriansotomora/mlops_group8](https://github.com/adriansotomora/mlops_group8)

**Experiment Tracking**: [Weights & Biases Project](https://wandb.ai/manuelbonnelly-ie-university/mlops_group8)
