MLOps Group 8 - Spotify Music Popularity Prediction🎵 Project OverviewThis MLOps project aims to predict the popularity of songs on Spotify using various audio features. A linear regression model is trained to estimate track_popularity based on attributes like danceability, energy, tempo, and engineered genre or polynomial features.The project follows MLOps best practices, featuring a modular pipeline that includes:Initial raw data loading (via data_loader.py).Data validation (placeholder, structure exists for data_validator.py).Preprocessing, including a split to create a holdout set for inference testing (preprocessing.py).Configurable feature engineering (features.py).Model training (currently Linear Regression with Stepwise Selection using statsmodels - in model.py).Model evaluation (using evaluator.py).Batch inference on new or holdout data, including prediction intervals (inferencer.py).The entire pipeline is orchestrated by src/main.py, and the repository is structured for reproducibility and maintainability, complete with a comprehensive test suite and configuration-driven components.🛠️ Tech StackPython Version: 3.10Core Libraries:Pandas: Data manipulation and analysis.NumPy: Numerical operations.Scikit-learn: Preprocessing (scaling), model splitting.Statsmodels: Linear regression modeling and statistical analysis.Joblib: Saving/loading scikit-learn objects (e.g., scalers).PyYAML: Configuration management.Testing:Pytest: Test framework.Pytest-Cov: Test coverage.Environment Management: CondaLinting/Formatting: Flake8, Black (recommended, included in environment.yml)📁 Project Structuremlops_group8/
├── config.yaml                # Main configuration file for all pipeline stages
├── environment.yml            # Conda environment specification
├── README.md                  # This file
├── data/
│   ├── raw/                   # Original, immutable data (e.g., Songs_2025.csv)
│   ├── raw_holdout/           # Raw data split off for inference testing (e.g., inference_holdout_data.csv)
│   ├── processed/             # Intermediate and final processed datasets
│   │   ├── Songs_2025_processed.csv # Output of preprocessing.py (main data)
│   │   └── features.csv             # Output of features.py (engineered features for model)
│   └── predictions/           # Output of inference runs (e.g., holdout_predictions_with_intervals.csv)
├── logs/                      # Log files for pipeline runs
│   ├── main_orchestrator.log  # Log for the src/main.py orchestrator
│   ├── preprocessing.log      # Specific logs for preprocessing.py
│   ├── features.log           # Specific logs for features.py
│   ├── model.log              # Specific logs for model.py
│   └── inference.log          # Specific logs for inferencer.py
├── models/                    # Saved model artifacts
│   ├── preprocessing_pipeline.pkl # Scaler from preprocessing
│   ├── linear_regression.pkl      # Trained model
│   ├── linear_regression_selected_features.json # Features used by the model
│   └── metrics.json               # Evaluation metrics
├── notebooks/                 # Jupyter notebooks for exploration and analysis (if any)
├── src/                       # Source code for the pipeline
│   ├── __init__.py
│   ├── data_load/
│   │   ├── __init__.py
│   │   └── data_loader.py     # Loads initial raw data
│   ├── data_validation/       # Data validation scripts (currently placeholder)
│   │   ├── __init__.py
│   │   └── data_validator.py  # Example, actual implementation may vary
│   ├── preprocess/
│   │   ├── __init__.py
│   │   └── preprocessing.py   # Preprocesses data, creates holdout, saves scaler
│   ├── features/
│   │   ├── __init__.py
│   │   └── features.py        # Performs feature engineering
│   ├── model/
│   │   ├── __init__.py
│   │   └── model.py           # Trains and evaluates the model
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py       # Calculates regression metrics
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inferencer.py      # Runs batch inference with prediction intervals
│   └── main.py                # Main pipeline orchestrator script (PREFERRED ENTRY POINT)
└── tests/                     # Test suite
    ├── __init__.py
    ├── mock_data/
    │   └── mock_data_spotify.csv # Mock data for testing various components
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_model.py
    ├── test_evaluator.py
    └── test_inferencer.py
(Note: The main.py in the project root is considered deprecated in favor of src/main.py for better module organization).🚀 Getting Started1. Environment SetupThis project uses Python 3.10. Dependencies are managed with Conda and listed in environment.yml.To set up the Conda environment:conda env create -f environment.yml
conda activate mlops_group8
2. ConfigurationThe pipeline behavior is controlled by config.yaml. Before running, review and customize this file as needed. It defines:Paths for input data (raw, processed), output artifacts (models, scalers, predictions), and logs.Parameters for data splitting (test size, holdout size for inference).Preprocessing steps: columns to drop, outlier removal settings, scaling method and columns.Feature engineering settings: genre parsing, polynomial feature creation, columns to exclude.Model parameters: active model type, stepwise selection thresholds for linear regression.Data validation schema (if data_validator.py is fully implemented).⚙️ Pipeline UsageThe primary entry point for the pipeline is src/main.py. It orchestrates the execution of different stages based on command-line arguments. Always run commands from the project root directory.Training PipelineThe training pipeline involves data preprocessing, feature engineering, model training, and evaluation.Run the Entire Training Pipeline:This executes all training stages sequentially. A holdout dataset for inference testing is also created during the preprocessing stage.python -m src.main --stage all_training 
(The --config config.yaml argument is used by default if config.yaml is in the root.)Run Individual Training Stages:Preprocessing: Loads raw data via data_loader.py, creates the inference holdout set, preprocesses the main dataset, and saves the processed data and scaler.python -m src.main --stage preprocess
Feature Engineering: Loads the output of the preprocessing stage, engineers new features, and saves the final feature set. (Requires preprocessing to be completed).python -m src.main --stage features
Model Training & Evaluation: Loads engineered features, trains the model, evaluates it using evaluator.py, and saves model artifacts. (Requires preprocessing and feature engineering to be completed).python -m src.main --stage model
Inference PipelineAfter a model has been trained and its artifacts (model, scaler, selected features list) are saved, you can run inference on new, unseen data or the holdout set using inferencer.py, orchestrated by main.py.Running Inference via main.py:The inference stage requires specifying an input data file and an output path for predictions.python -m src.main --stage inference --input_file <path_to_input_data.csv> --output_file <path_to_save_predictions.csv>
Example (using the holdout data created by preprocessing.py):(Assumes data_source.inference_holdout_path in config.yaml is, e.g., data/raw_holdout/inference_holdout_data.csv)python -m src.main --stage inference \
                   --input_file data/raw_holdout/inference_holdout_data.csv \
                   --output_file data/predictions/holdout_predictions_with_intervals.csv
The inference process will apply all necessary data transformations using the saved scaler and feature configurations before generating predictions and prediction intervals. The output CSV will include the original data along with prediction, prediction_pi_lower, and prediction_pi_upper columns.✅ TestingA comprehensive test suite is located in the tests/ directory. Each core module has corresponding test files (e.g., test_preprocessing.py, test_inferencer.py).Running All Tests:From the project root directory, execute:python -m pytest -v
This command discovers and runs all tests with verbose output.Running Tests with Coverage:To generate a code coverage report (ensure pytest-cov is installed):python -m pytest -v --cov=src --cov-report=term-missing --cov-report=html
This shows a terminal summary and creates a detailed HTML report in the htmlcov/ directory.📦 Artifact LocationsKey artifacts generated and used by the pipeline are stored in the following default locations (customizable in config.yaml):Data:Raw data: data/raw/Songs_2025.csv (or as per data_source.raw_path)Raw holdout data (for inference): data/raw_holdout/inference_holdout_data.csv (or as per data_source.inference_holdout_path)Preprocessed data (output of preprocessing.py): data/processed/Songs_2025_processed.csv (or as per data_source.processed_path)Engineered features (output of features.py): data/processed/features.csv (or as per artifacts.engineered_features_filename)Feature list (output by features.py): data/processed/feature_list.txt (or as per artifacts.feature_list_filename)Predictions (output of inferencer.py): data/predictions/ (path specified via CLI)Models & Related Artifacts:Preprocessing scaler: models/preprocessing_pipeline.pkl (or as per artifacts.preprocessing_pipeline)Trained model: models/linear_regression.pkl (or as per model.linear_regression.save_path)Selected features list (for model): models/linear_regression_selected_features.json (or as per model.linear_regression.selected_features_path)Evaluation metrics: models/metrics.json (or as per artifacts.metrics_path)Logs: logs/ (e.g., logs/main_orchestrator.log, logs/preprocessing.log, etc., as defined in logging.log_file in config.yaml or per script defaults)🤝 ContributingContributions are welcome! Please follow these general guidelines:Fork the repository.Create your feature branch: git checkout -b feature/your-amazing-featureCommit your changes: git commit -am 'Add some amazing feature'Push to the branch: git push origin feature/your-amazing-featureOpen a Pull Request against the main branch for review.Please adhere to code quality standards by using black for code formatting and flake8 for linting. Ensure that new features are accompanied by relevant tests and that all existing tests pass.