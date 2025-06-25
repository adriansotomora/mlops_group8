"""
Tests for src/features/pipeline.py module
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import sys
from pathlib import Path

# Add project root to path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import GenreParser, build_feature_pipeline


class TestGenreParser:
    """Test cases for GenreParser transformer class."""

    def test_init(self):
        """Test GenreParser initialization."""
        parser = GenreParser()
        
        # Check that genres_map is properly initialized
        assert hasattr(parser, 'genres_map')
        assert isinstance(parser.genres_map, dict)
        assert len(parser.genres_map) == 11  # Should have 11 genre categories
        
        # Check specific genre patterns
        assert 'pop' in parser.genres_map
        assert 'rock' in parser.genres_map
        assert 'hip-hop' in parser.genres_map
        assert 'r&b' in parser.genres_map
        
        # Check that genre_columns_ is initialized as empty list
        assert hasattr(parser, 'genre_columns_')
        assert parser.genre_columns_ == []

    def test_fit(self):
        """Test GenreParser fit method."""
        parser = GenreParser()
        df = pd.DataFrame({'artist_genres': ['pop rock', 'jazz', 'electronic']})
        
        # Fit should return self
        result = parser.fit(df)
        assert result is parser
        
        # Check that genre_columns_ is populated after fit
        assert len(parser.genre_columns_) == 11
        assert 'genre_pop' in parser.genre_columns_
        assert 'genre_rock' in parser.genre_columns_
        assert 'genre_hip_hop' in parser.genre_columns_  # hip-hop -> hip_hop
        assert 'genre_randb' in parser.genre_columns_  # r&b -> randb

    def test_fit_with_none_y(self):
        """Test GenreParser fit method with y=None."""
        parser = GenreParser()
        df = pd.DataFrame({'artist_genres': ['pop']})
        
        result = parser.fit(df, y=None)
        assert result is parser
        assert len(parser.genre_columns_) == 11

    def test_transform_with_artist_genres(self):
        """Test GenreParser transform with artist_genres column present."""
        parser = GenreParser()
        df = pd.DataFrame({
            'artist_genres': [
                'pop rock',
                'hip hop jazz',
                'electronic house',
                'latin reggaeton',
                'metal punk',
                'classical symphony',
                'country',
                'indie rock',
                'r&b soul',
                'unknown genre'
            ],
            'other_column': range(10)
        })
        
        parser.fit(df)
        result = parser.transform(df)
        
        # Check that original columns are preserved
        assert 'artist_genres' in result.columns
        assert 'other_column' in result.columns
        
        # Check that all genre columns are added
        for genre_col in parser.genre_columns_:
            assert genre_col in result.columns
        
        # Check specific genre detection
        assert result.loc[0, 'genre_pop'] == 1  # 'pop rock' contains pop
        assert result.loc[0, 'genre_rock'] == 1  # 'pop rock' contains rock
        assert result.loc[1, 'genre_hip_hop'] == 1  # 'hip hop jazz' contains hip hop
        assert result.loc[1, 'genre_jazz'] == 1  # 'hip hop jazz' contains jazz
        assert result.loc[2, 'genre_electronic'] == 1  # 'electronic house' contains electronic
        assert result.loc[3, 'genre_latin'] == 1  # 'latin reggaeton' contains latin
        assert result.loc[4, 'genre_metal'] == 1  # 'metal punk' contains metal
        assert result.loc[5, 'genre_classic'] == 1  # 'classical symphony' contains classical
        assert result.loc[6, 'genre_country'] == 1  # 'country' contains country
        assert result.loc[8, 'genre_randb'] == 1  # 'r&b soul' contains r&b
        
        # Check that unknown genre doesn't match any patterns
        assert result.loc[9, 'genre_pop'] == 0
        assert result.loc[9, 'genre_rock'] == 0

    def test_transform_missing_artist_genres(self):
        """Test GenreParser transform when artist_genres column is missing."""
        parser = GenreParser()
        df = pd.DataFrame({
            'other_column': [1, 2, 3],
            'another_column': ['a', 'b', 'c']
        })
        
        parser.fit(df)  # This will still create genre_columns_
        result = parser.transform(df)
        
        # Check that original columns are preserved
        assert 'other_column' in result.columns
        assert 'another_column' in result.columns
        
        # Check that all genre columns are added with zeros
        for genre_col in parser.genre_columns_:
            assert genre_col in result.columns
            assert (result[genre_col] == 0).all()

    def test_transform_case_insensitive(self):
        """Test that genre detection is case insensitive."""
        parser = GenreParser()
        df = pd.DataFrame({
            'artist_genres': [
                'POP ROCK',
                'Hip Hop',
                'ELECTRONIC',
                'Jazz',
                'METAL'
            ]
        })
        
        parser.fit(df)
        result = parser.transform(df)
        
        # Check case insensitive detection
        assert result.loc[0, 'genre_pop'] == 1
        assert result.loc[0, 'genre_rock'] == 1
        assert result.loc[1, 'genre_hip_hop'] == 1
        assert result.loc[2, 'genre_electronic'] == 1
        assert result.loc[3, 'genre_jazz'] == 1
        assert result.loc[4, 'genre_metal'] == 1

    def test_transform_with_na_values(self):
        """Test GenreParser transform with NaN values in artist_genres."""
        parser = GenreParser()
        df = pd.DataFrame({
            'artist_genres': ['pop rock', np.nan, 'jazz', None, 'electronic']
        })
        
        parser.fit(df)
        result = parser.transform(df)
        
        # Check that NaN values are handled (should be 0 for all genres)
        assert result.loc[1, 'genre_pop'] == 0  # NaN row
        assert result.loc[3, 'genre_jazz'] == 0  # None row
        
        # Check that valid values still work
        assert result.loc[0, 'genre_pop'] == 1
        assert result.loc[2, 'genre_jazz'] == 1
        assert result.loc[4, 'genre_electronic'] == 1

    def test_transform_empty_dataframe(self):
        """Test GenreParser transform with empty DataFrame."""
        parser = GenreParser()
        df = pd.DataFrame({'artist_genres': []})
        
        parser.fit(df)
        result = parser.transform(df)
        
        # Should return empty DataFrame with genre columns
        assert len(result) == 0
        for genre_col in parser.genre_columns_:
            assert genre_col in result.columns

    def test_complex_genre_patterns(self):
        """Test complex genre pattern matching."""
        parser = GenreParser()
        df = pd.DataFrame({
            'artist_genres': [
                'progressive house electronic',  # Should match electronic
                'trap hip hop urban',           # Should match hip-hop  
                'bachata latin dance',          # Should match latin
                'hardcore metal punk',          # Should match metal
                'orchestra classical music',    # Should match classic
                'soul r&b rhythm',              # Should match r&b
                'edm electronic dance',         # Should match electronic
                'salsa latin music'             # Should match latin
            ]
        })
        
        parser.fit(df)
        result = parser.transform(df)
        
        # Check complex pattern matching
        assert result.loc[0, 'genre_electronic'] == 1  # progressive house
        assert result.loc[1, 'genre_hip_hop'] == 1     # trap
        assert result.loc[2, 'genre_latin'] == 1       # bachata
        assert result.loc[3, 'genre_metal'] == 1       # hardcore
        assert result.loc[4, 'genre_classic'] == 1     # orchestra
        assert result.loc[5, 'genre_randb'] == 1      # soul
        assert result.loc[6, 'genre_electronic'] == 1  # edm
        assert result.loc[7, 'genre_latin'] == 1       # salsa


class TestBuildFeaturePipeline:
    """Test cases for build_feature_pipeline function."""

    def test_build_feature_pipeline_returns_pipeline(self):
        """Test that build_feature_pipeline returns a sklearn Pipeline."""
        config = {}
        pipeline = build_feature_pipeline(config)
        
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "passthrough"
        assert pipeline.steps[0][1] == "passthrough"

    def test_build_feature_pipeline_with_config(self):
        """Test build_feature_pipeline with various config inputs."""
        configs = [
            {},
            {"features": {}},
            {"features": {"some_setting": True}},
            None
        ]
        
        for config in configs:
            pipeline = build_feature_pipeline(config)
            assert isinstance(pipeline, Pipeline)
            # Should always return passthrough pipeline regardless of config

    def test_pipeline_fit_transform(self):
        """Test that the built pipeline can fit and transform data."""
        config = {}
        pipeline = build_feature_pipeline(config)
        
        # Test with sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [10, 20, 30, 40, 50]
        })
        
        # Fit and transform should work (passthrough)
        pipeline.fit(df)
        result = pipeline.transform(df)
        
        # Should return the same DataFrame (passthrough behavior)
        pd.testing.assert_frame_equal(result, df)

    def test_pipeline_integration_with_genre_parser(self):
        """Test how the pipeline would work if GenreParser was included."""
        # This tests the potential integration even though current pipeline is passthrough
        parser = GenreParser()
        df = pd.DataFrame({
            'artist_genres': ['pop rock', 'jazz electronic'],
            'other_feature': [1, 2]
        })
        
        # Test that GenreParser could be used in a pipeline context
        parser.fit(df)
        transformed = parser.transform(df)
        
        # Verify it produces expected output that could feed into downstream pipeline
        assert 'genre_pop' in transformed.columns
        assert 'genre_rock' in transformed.columns
        assert 'genre_jazz' in transformed.columns
        assert 'genre_electronic' in transformed.columns
        assert 'other_feature' in transformed.columns
