# MLOps Group 8 - Spotify Music Popularity Prediction

## Project Overview
This MLOps project predicts the popularity of Spotify songs using machine learning. A linear regression model estimates `track_popularity` from audio features such as danceability, energy, and tempo. The pipeline includes data validation, preprocessing, feature engineering, model training, evaluation, and inference. The repository provides production-ready code structure and a comprehensive test suite.

## Directory Tree
```
mlops_group8/
├── src/
│   ├── data_load/
│   ├── data_validation/
│   ├── preprocess/
│   ├── features/
│   ├── model/
│   ├── evaluation/
│   ├── inference/
│   └── main.py
├── tests/
│   ├── mock_data/
│   └── test_*.py
├── data/
│   ├── raw/
│   ├── raw_holdout/
│   ├── processed/
│   └── predictions/
├── models/
├── notebooks/
├── environment.yml
└── config.yaml
```

## Environment Setup
Python 3.10 is used, and dependencies are managed with `environment.yml`. Set up the environment with:

```bash
conda env create -f environment.yml
conda activate mlops_group8
```

Main libraries: pandas, scikit-learn, numpy, pytest, PyYAML, matplotlib, seaborn. All packages required for ML, data processing, testing, and visualization are included.

## Usage
The main entry point is `src/main.py`. Run the full pipeline with the config file:

```bash
python -m src.main --config config.yaml
```

You can also run specific stages: `--stage data` (data processing only), `--stage train` (model training only). The pipeline automatically executes data loading, validation, preprocessing, feature engineering, model training, and evaluation in sequence.

## Testing
### Running Tests

It is recommended to use the following commands to run tests, as this avoids Python path issues that may occur with some project structures:

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=html
```

Each module (data loader, preprocessing, feature engineering, model, evaluator, inferencer) has corresponding test files. Mock data is used to verify the pipeline, and code coverage reports can be generated.

## Artifact Locations
- **Models**: `models/linear_regression.pkl`, `models/preprocessing_pipeline.pkl`
- **Metrics**: `models/metrics.json`
- **Selected Features**: `models/linear_regression_selected_features.json`
- **Processed Data**: `data/processed/`
- **Logs**: `logs/pipeline_main.log`
- **Predictions**: `data/predictions/`

All artifact locations can be customized in the config file.

## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a pull request

Please use black (formatter) and flake8 (linter) to maintain code quality. Add tests for new features and ensure all existing tests pass.