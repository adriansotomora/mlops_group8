import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GenreParser(BaseEstimator, TransformerMixin):
    """Custom transformer to parse genres into binary columns."""
    def __init__(self):
        self.genres_map = {
            'pop': r'(?:pop)', 'rock': r'(?:rock)',
            'electronic': r'(?:house|edm|electro|techno|progressive)',
            'latin': r'(?:latin|reggaeton|salsa|bachata)',
            'hip-hop': r'(?:hip hop|rap|urban|trap)', 'indie': r'(?:indie)',
            'jazz': r'(?:jazz)', 'r&b': r'(?:r&b|soul)',
            'metal': r'(?:metal|punk|hardcore)',
            'classic': r'(?:classical|orchestra|symphony)', 'country': r'(?:country)'
        }
        self.genre_columns_ = []

    def fit(self, X, y=None):
        # No fitting necessary, but store column names for transform
        self.genre_columns_ = [f"genre_{g.replace('&', 'and').replace('-', '_')}" for g in self.genres_map]
        return self

    def transform(self, X):
        X_ = X.copy()
        if 'artist_genres' not in X_.columns:
            # Add all genre columns as zeros if missing
            for col in self.genre_columns_:
                X_[col] = 0
            return X_
        for genre, pattern in self.genres_map.items():
            colname = f"genre_{genre.replace('&', 'and').replace('-', '_')}"
            X_[colname] = X_['artist_genres'].astype(str).str.contains(pattern, case=False, regex=True, na=False).astype(int)
        return X_

# You can add more custom transformers as needed (e.g., for dropping columns, etc.)

def build_feature_pipeline(config):
    # No feature engineering: passthrough pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    # Optionally, you can keep genre parsing or imputation if needed, otherwise just passthrough
    pipeline = Pipeline([
        ("passthrough", "passthrough")
    ])
    return pipeline