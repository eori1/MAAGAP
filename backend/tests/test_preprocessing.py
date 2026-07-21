"""Unit tests for MAAGAP preprocessor pipeline."""

import os
import pytest
import numpy as np
import pandas as pd

from maagap.synthetic_generator import SyntheticDataGenerator
from maagap.preprocessing_pipeline import MAAGAPPreprocessor
from maagap.config import MODELS_DIR


@pytest.fixture(scope="module")
def sample_data(tmp_path_factory):
    gen = SyntheticDataGenerator(seed=42)
    out_dir = tmp_path_factory.mktemp("test_preprocessing_data")
    df_proj, df_qtr = gen.generate_synthetic_dataset(n_projects=50, output_dir=str(out_dir))
    return df_proj, df_qtr


def test_preprocessor_fit_transform(sample_data):
    df_proj, df_qtr = sample_data
    preprocessor = MAAGAPPreprocessor()
    
    X_static, X_temporal, static_cols, temporal_cols = preprocessor.fit_transform(df_proj, df_qtr)
    
    assert X_static.shape[0] == 50
    assert X_static.shape[1] > 20
    assert X_temporal.shape == (50, 4, len(temporal_cols))
    assert preprocessor.is_fitted


def test_preprocessor_save_load(sample_data, tmp_path):
    df_proj, df_qtr = sample_data
    preprocessor = MAAGAPPreprocessor()
    preprocessor.fit_transform(df_proj, df_qtr)
    
    save_path = str(tmp_path / "test_pipeline.pkl")
    preprocessor.save(save_path)
    
    assert os.path.exists(save_path)
    
    loaded = MAAGAPPreprocessor.load(save_path)
    assert loaded.is_fitted
    
    X_test_static = loaded.transform_static(df_proj.head(10))
    assert X_test_static.shape == (10, len(loaded.engineer.static_feature_names))
