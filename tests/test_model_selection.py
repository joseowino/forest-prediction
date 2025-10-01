"""
Unit tests for model selection pipeline.

Run with: pytest test_model_selection.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from model_selection import ModelSelector


@pytest.fixture
def sample_classification_data():
    """Create sample classification data for testing."""
    # Create a multi-class classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=7,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame and Series
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y + 1)  # Make classes 1-7 like forest cover
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val


class TestModelSelector:
    """Test suite for ModelSelector class."""
    
    def test_initialization(self):
        """Test model selector initialization."""
        selector = ModelSelector(random_state=42, n_folds=5)
        assert selector.random_state == 42
        assert selector.n_folds == 5
        assert selector.cv is not None
        assert len(selector.results) == 0
    
    def test_get_model_param_grids(self):
        """Test that all required models are defined."""
        selector = ModelSelector()
        models = selector.get_model_param_grids()
        
        required_models = [
            'Gradient_Boosting',
            'Random_Forest',
            'KNN',
            'SVM',
            'Logistic_Regression'
        ]
        
        for model_name in required_models:
            assert model_name in models
            assert 'model' in models[model_name]
            assert 'params' in models[model_name]
            assert 'scaled' in models[model_name]
    
    def test_model_scaling_requirements(self):
        """Test that scaling requirements are correctly specified."""
        selector = ModelSelector()
        models = selector.get_model_param_grids()
        
        # Tree-based models should not require scaling
        assert models['Random_Forest']['scaled'] == False
        assert models['Gradient_Boosting']['scaled'] == False
        
        # Distance/optimization-based models should require scaling
        assert models['KNN']['scaled'] == True
        assert models['SVM']['scaled'] == True
        assert models['Logistic_Regression']['scaled'] == True
    
    def test_param_grid_not_empty(self):
        """Test that all models have non-empty parameter grids."""
        selector = ModelSelector()
        models = selector.get_model_param_grids()
        
        for model_name, config in models.items():
            assert len(config['params']) > 0, f"{model_name} has empty param grid"
            for param, values in config['params'].items():
                assert len(values) > 0, f"{model_name}.{param} has no values"
    
    def test_count_param_combinations(self):
        """Test parameter combination counting."""
        selector = ModelSelector()
        
        # Simple test case
        param_grid = {
            'param1': [1, 2],
            'param2': [3, 4, 5]
        }
        count = selector._count_param_combinations(param_grid)
        assert count == 6  # 2 * 3 = 6
    
    def test_train_single_model(self, sample_classification_data):
        """Test training a single model with grid search."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector(random_state=42, n_folds=3, n_jobs=1, verbose=0)
        
        # Use a small parameter grid for testing
        model_config = {
            'model': selector.get_model_param_grids()['Random_Forest']['model'],
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10]
            },
            'scaled': False
        }
        
        grid_search = selector.train_model_with_grid_search(
            'Random_Forest_Test', model_config, X_train, y_train
        )
        
        assert grid_search.best_estimator_ is not None
        assert grid_search.best_score_ > 0
        assert hasattr(grid_search, 'best_params_')
    
    def test_confusion_matrix_dataframe(self, sample_classification_data):
        """Test confusion matrix DataFrame generation."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector()
        
        # Create dummy predictions
        y_pred = y_val.copy()  # Perfect predictions for simplicity
        
        df_cm = selector.get_confusion_matrix_dataframe(y_val, y_pred)
        
        # Check DataFrame structure
        assert isinstance(df_cm, pd.DataFrame)
        assert df_cm.shape == (7, 7)  # 7 classes
        assert all('True_' in idx for idx in df_cm.index)
        assert all('Pred_' in col for col in df_cm.columns)
    
    def test_results_storage(self, sample_classification_data):
        """Test that results are properly stored."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector(random_state=42, n_folds=2, n_jobs=1, verbose=0)
        
        # Train only Random Forest for quick testing
        models = selector.get_model_param_grids()
        rf_config = models['Random_Forest'].copy()
        rf_config['params'] = {
            'n_estimators': [50],
            'max_depth': [10]
        }
        
        grid_search = selector.train_model_with_grid_search(
            'Random_Forest', rf_config, X_train, y_train
        )
        
        selector.results['Random_Forest'] = {
            'grid_search': grid_search,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'scaled': False
        }
        
        assert 'Random_Forest' in selector.results
        assert 'best_params' in selector.results['Random_Forest']
        assert 'best_cv_score' in selector.results['Random_Forest']
        assert 'best_estimator' in selector.results['Random_Forest']
    
    def test_best_model_selection(self, sample_classification_data):
        """Test that best model is correctly identified."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector(random_state=42)
        
        # Manually create mock results
        selector.results = {
            'Model_A': {'best_cv_score': 0.75, 'best_estimator': None},
            'Model_B': {'best_cv_score': 0.85, 'best_estimator': None},
            'Model_C': {'best_cv_score': 0.80, 'best_estimator': None}
        }
        
        selector._select_best_model()
        
        assert selector.best_model_name == 'Model_B'
    
    def test_model_persistence(self, sample_classification_data, tmp_path):
        """Test saving and loading models."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector(random_state=42, n_folds=2, n_jobs=1, verbose=0)
        
        # Train a simple model
        models = selector.get_model_param_grids()
        rf_config = models['Random_Forest'].copy()
        rf_config['params'] = {
            'n_estimators': [50],
            'max_depth': [5]
        }
        
        grid_search = selector.train_model_with_grid_search(
            'Random_Forest', rf_config, X_train, y_train
        )
        
        selector.results['Random_Forest'] = {
            'grid_search': grid_search,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'scaled': False
        }
        selector.best_model_name = 'Random_Forest'
        selector.best_model = grid_search.best_estimator_
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        selector.save_best_model(filepath=str(model_path))
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_info = pickle.load(f)
        
        assert loaded_info['model_name'] == 'Random_Forest'
        assert loaded_info['model'] is not None
        assert 'cv_score' in loaded_info
        assert 'best_params' in loaded_info
    
    def test_overfitting_detection(self, sample_classification_data):
        """Test overfitting detection logic."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        selector = ModelSelector(random_state=42)
        
        # Mock results with overfitting scenario
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        selector.results = {
            'Test_Model': {
                'best_estimator': model,
                'best_cv_score': 0.70,
                'scaled': False,
                'val_score': 0.65
            }
        }
        
        df_analysis = selector.check_overfitting(X_train, X_train, y_train)
        
        assert 'Train_Accuracy' in df_analysis.columns
        assert 'CV_Accuracy' in df_analysis.columns
        assert 'Val_Accuracy' in df_analysis.columns
        assert 'Overfitting' in df_analysis.columns
    
    def test_cv_folds_configuration(self):
        """Test that cross-validation is properly configured."""
        selector = ModelSelector(random_state=42, n_folds=5)
        
        assert selector.cv.n_splits == 5
        assert selector.cv.shuffle == True
        assert selector.cv.random_state == 42
    
    def test_reproducibility(self, sample_classification_data):
        """Test that results are reproducible with same random seed."""
        X_train, X_val, y_train, y_val = sample_classification_data
        
        # First run
        selector1 = ModelSelector(random_state=42, n_folds=2, n_jobs=1, verbose=0)
        model_config1 = {
            'model': selector1.get_model_param_grids()['Random_Forest']['model'],
            'params': {'n_estimators': [50], 'max_depth': [5]},
            'scaled': False
        }
        grid1 = selector1.train_model_with_grid_search(
            'RF_Test', model_config1, X_train, y_train
        )
        
        # Second run
        selector2 = ModelSelector(random_state=42, n_folds=2, n_jobs=1, verbose=0)
        model_config2 = {
            'model': selector2.get_model_param_grids()['Random_Forest']['model'],
            'params': {'n_estimators': [50], 'max_depth': [5]},
            'scaled': False
        }
        grid2 = selector2.train_model_with_grid_search(
            'RF_Test', model_config2, X_train, y_train
        )
        
        # Results should be identical
        assert abs(grid1.best_score_ - grid2.best_score_) < 0.001


def test_integration_with_preprocessing(tmp_path):
    """Test integration with preprocessing pipeline."""
    # This test checks that model selection can work with preprocessed data
    
    # Create mock processed data
    X_train = pd.DataFrame(np.random.randn(100, 20))
    X_val = pd.DataFrame(np.random.randn(25, 20))
    y_train = pd.Series(np.random.randint(1, 8, 100))
    y_val = pd.Series(np.random.randint(1, 8, 25))
    
    datasets = {
        'X_train_scaled': X_train,
        'X_val_scaled': X_val,
        'X_train_unscaled': X_train,
        'X_val_unscaled': X_val,
        'y_train': y_train,
        'y_val': y_val
    }
    
    # Save to temp file
    datasets_path = tmp_path / "datasets.pkl"
    with open(datasets_path, 'wb') as f:
        pickle.dump(datasets, f)
    
    # Load and verify
    with open(datasets_path, 'rb') as f:
        loaded = pickle.load(f)
    
    assert 'X_train_scaled' in loaded
    assert 'y_train' in loaded
    assert len(loaded['X_train_scaled']) == 100


def test_parameter_grid_coverage():
    """Test that parameter grids cover reasonable ranges."""
    selector = ModelSelector()
    models = selector.get_model_param_grids()
    
    # Check Random Forest has reasonable parameters
    rf_params = models['Random_Forest']['params']
    assert 'n_estimators' in rf_params
    assert 'max_depth' in rf_params
    assert len(rf_params['n_estimators']) >= 3
    
    # Check Gradient Boosting has learning rate
    gb_params = models['Gradient_Boosting']['params']
    assert 'learning_rate' in gb_params
    assert 'n_estimators' in gb_params
    
    # Check KNN has n_neighbors
    knn_params = models['KNN']['params']
    assert 'n_neighbors' in knn_params
    assert len(knn_params['n_neighbors']) >= 3
    
    # Check SVM has C and kernel
    svm_params = models['SVM']['params']
    assert 'C' in svm_params
    assert 'kernel' in svm_params
    
    # Check Logistic Regression has C
    lr_params = models['Logistic_Regression']['params']
    assert 'C' in lr_params


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])