"""
Forest Cover Type Prediction - Model Selection and Hyperparameter Tuning
========================================================================

This module implements comprehensive model selection using cross-validation
and grid search across multiple machine learning algorithms.

Author: Developer 2
Date: 2025-09-30
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
from datetime import datetime

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ModelSelector:
    """
    Comprehensive model selection framework for forest cover classification.
    
    Implements grid search with cross-validation across multiple algorithms
    and provides tools for model evaluation and visualization.
    """
    
    def __init__(self, random_state=42, n_folds=5, n_jobs=-1, verbose=2):
        """
        Initialize the model selector.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        n_folds : int, default=5
            Number of folds for cross-validation
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all cores)
        verbose : int, default=2
            Verbosity level for grid search
        """
        self.random_state = random_state
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_model_param_grids(self):
        """
        Define models and their hyperparameter grids.
        
        Returns:
        --------
        dict
            Dictionary with model configurations
        """
        models = {
            'Gradient_Boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 1.0]
                },
                'scaled': False  # Tree-based, doesn't need scaling
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'scaled': False  # Tree-based, doesn't need scaling
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski'],
                    'p': [1, 2]
                },
                'scaled': True  # Distance-based, needs scaling
            },
            'SVM': {
                'model': SVC(random_state=self.random_state),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'degree': [2, 3]  # For poly kernel
                },
                'scaled': True  # Needs scaling
            },
            'Logistic_Regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='multinomial'
                ),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'saga']
                },
                'scaled': True  # Needs scaling
            }
        }
        
        return models
    
    def train_model_with_grid_search(self, model_name, model_config, X_train, y_train):
        """
        Perform grid search for a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model_config : dict
            Model configuration with 'model' and 'params'
        X_train : pd.DataFrame or np.array
            Training features
        y_train : pd.Series or np.array
            Training labels
            
        Returns:
        --------
        GridSearchCV
            Fitted grid search object
        """
        print(f"\n{'='*70}")
        print(f"Training {model_name}")
        print(f"{'='*70}")
        print(f"Parameter grid size: {self._count_param_combinations(model_config['params'])}")
        print(f"Total fits: {self._count_param_combinations(model_config['params']) * self.n_folds}")
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=self.cv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ {model_name} training complete!")
        print(f"Time elapsed: {elapsed_time/60:.2f} minutes")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search
    
    def train_all_models(self, X_train_scaled, X_train_unscaled, y_train):
        """
        Train all models with grid search.
        
        Parameters:
        -----------
        X_train_scaled : pd.DataFrame
            Scaled training features
        X_train_unscaled : pd.DataFrame
            Unscaled training features
        y_train : pd.Series
            Training labels
        """
        print("\n" + "="*70)
        print("STARTING MODEL SELECTION WITH GRID SEARCH")
        print("="*70)
        print(f"Cross-validation folds: {self.n_folds}")
        print(f"Parallel jobs: {self.n_jobs}")
        print(f"Training samples: {len(y_train)}")
        
        models = self.get_model_param_grids()
        
        # Create progress bar for models
        model_progress = tqdm(models.items(), desc="Training Models", 
                            unit="model", position=0, leave=True)
        
        for model_name, model_config in model_progress:
            # Update progress bar description
            model_progress.set_description(f"Training {model_name}")
            
            # Use scaled or unscaled data based on model requirements
            X_train = X_train_scaled if model_config['scaled'] else X_train_unscaled
            
            # Perform grid search
            grid_search = self.train_model_with_grid_search(
                model_name, model_config, X_train, y_train
            )
            
            # Store results
            self.results[model_name] = {
                'grid_search': grid_search,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_results': pd.DataFrame(grid_search.cv_results_),
                'best_estimator': grid_search.best_estimator_,
                'scaled': model_config['scaled']
            }
        
        # Close progress bar
        model_progress.close()
        
        # Select best overall model
        self._select_best_model()
        
    def _select_best_model(self):
        """Select the best model based on cross-validation scores."""
        best_score = -1
        best_name = None
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        for model_name, result in self.results.items():
            cv_score = result['best_cv_score']
            print(f"{model_name:25s} CV Score: {cv_score:.4f}")
            
            if cv_score > best_score:
                best_score = cv_score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['best_estimator']
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Best CV Score: {best_score:.4f}")
        print(f"{'='*70}")
    
    def evaluate_on_validation(self, X_val_scaled, X_val_unscaled, y_val):
        """
        Evaluate all models on validation set.
        
        Parameters:
        -----------
        X_val_scaled : pd.DataFrame
            Scaled validation features
        X_val_unscaled : pd.DataFrame
            Unscaled validation features
        y_val : pd.Series
            Validation labels
            
        Returns:
        --------
        dict
            Validation scores for all models
        """
        print("\n" + "="*70)
        print("VALIDATION SET EVALUATION")
        print("="*70)
        
        val_scores = {}
        
        # Create progress bar for evaluation
        eval_progress = tqdm(self.results.items(), desc="Evaluating Models", 
                           unit="model", leave=True)
        
        for model_name, result in eval_progress:
            eval_progress.set_description(f"Evaluating {model_name}")
            
            X_val = X_val_scaled if result['scaled'] else X_val_unscaled
            model = result['best_estimator']
            
            # Get predictions
            y_pred = model.predict(X_val)
            val_score = accuracy_score(y_val, y_pred)
            val_scores[model_name] = val_score
            
            # Store predictions
            result['val_predictions'] = y_pred
            result['val_score'] = val_score
            
            print(f"{model_name:25s} Validation Score: {val_score:.4f}")
        
        eval_progress.close()
        return val_scores
    
    def check_overfitting(self, X_train_scaled, X_train_unscaled, y_train):
        """
        Check for overfitting by comparing train and validation scores.
        
        Parameters:
        -----------
        X_train_scaled : pd.DataFrame
            Scaled training features
        X_train_unscaled : pd.DataFrame
            Unscaled training features
        y_train : pd.Series
            Training labels
            
        Returns:
        --------
        pd.DataFrame
            Overfitting analysis results
        """
        print("\n" + "="*70)
        print("OVERFITTING ANALYSIS")
        print("="*70)
        
        analysis = []
        
        # Create progress bar for overfitting check
        overfit_progress = tqdm(self.results.items(), desc="Checking Overfitting", 
                              unit="model", leave=True)
        
        for model_name, result in overfit_progress:
            overfit_progress.set_description(f"Analyzing {model_name}")
            
            X_train = X_train_scaled if result['scaled'] else X_train_unscaled
            model = result['best_estimator']
            
            # Train score
            y_train_pred = model.predict(X_train)
            train_score = accuracy_score(y_train, y_train_pred)
            
            # CV score
            cv_score = result['best_cv_score']
            
            # Validation score
            val_score = result.get('val_score', 0)
            
            # Overfitting indicators
            train_cv_gap = train_score - cv_score
            train_val_gap = train_score - val_score
            
            analysis.append({
                'Model': model_name,
                'Train_Accuracy': train_score,
                'CV_Accuracy': cv_score,
                'Val_Accuracy': val_score,
                'Train_CV_Gap': train_cv_gap,
                'Train_Val_Gap': train_val_gap,
                'Overfitting': 'Yes' if train_score > 0.98 else 'No'
            })
            
            result['train_score'] = train_score
        
        overfit_progress.close()
        
        df_analysis = pd.DataFrame(analysis)
        print(df_analysis.to_string(index=False))
        
        # Check constraint
        print("\n" + "="*70)
        print("CONSTRAINT CHECK: Train Accuracy < 0.98")
        print("="*70)
        for _, row in df_analysis.iterrows():
            status = "✓ PASS" if row['Train_Accuracy'] < 0.98 else "✗ FAIL"
            print(f"{row['Model']:25s} {row['Train_Accuracy']:.4f} {status}")
        
        return df_analysis
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(1, 8), yticklabels=range(1, 8),
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def get_confusion_matrix_dataframe(self, y_true, y_pred):
        """
        Get confusion matrix as a DataFrame.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        pd.DataFrame
            Confusion matrix with labeled indices and columns
        """
        cm = confusion_matrix(y_true, y_pred)
        
        df_cm = pd.DataFrame(
            cm,
            index=[f'True_{i}' for i in range(1, 8)],
            columns=[f'Pred_{i}' for i in range(1, 8)]
        )
        
        return df_cm
    
    def plot_learning_curve(self, X_scaled, X_unscaled, y, model_name=None, save_path=None):
        """
        Plot learning curve for a model.
        
        Parameters:
        -----------
        X_scaled : pd.DataFrame
            Scaled features
        X_unscaled : pd.DataFrame
            Unscaled features
        y : pd.Series
            Labels
        model_name : str, optional
            Name of the model (uses best model if None)
        save_path : str, optional
            Path to save the plot
        """
        if model_name is None:
            model_name = self.best_model_name
        
        result = self.results[model_name]
        model = result['best_estimator']
        X = X_scaled if result['scaled'] else X_unscaled
        
        print(f"\nGenerating learning curve for {model_name}...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Use tqdm to show progress for learning curve calculation
        print("Computing learning curve (this may take a few minutes)...")
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes_abs, val_mean, 'o-', color='g', label='Cross-validation score')
        
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color='r')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                         alpha=0.1, color='g')
        
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning curve saved to {save_path}")
        
        plt.show()
        print("✓ Learning curve generated")
    
    def save_best_model(self, filepath='results/best_model.pkl'):
        """
        Save the best model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_info = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'best_params': self.results[self.best_model_name]['best_params'],
            'cv_score': self.results[self.best_model_name]['best_cv_score'],
            'scaled': self.results[self.best_model_name]['scaled'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"\n✓ Best model saved to {filepath}")
        print(f"  Model: {self.best_model_name}")
        print(f"  CV Score: {model_info['cv_score']:.4f}")
        print(f"  Requires scaling: {model_info['scaled']}")
    
    def save_results_summary(self, filepath='results/model_selection_summary.csv'):
        """
        Save summary of all models to CSV.
        
        Parameters:
        -----------
        filepath : str
            Path to save the summary
        """
        summary_data = []
        
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Best_CV_Score': result['best_cv_score'],
                'Train_Score': result.get('train_score', None),
                'Val_Score': result.get('val_score', None),
                'Requires_Scaling': result['scaled'],
                'Best_Params': str(result['best_params'])
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('Best_CV_Score', ascending=False)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_summary.to_csv(filepath, index=False)
        
        print(f"\n✓ Results summary saved to {filepath}")
    
    def _count_param_combinations(self, param_grid):
        """Count total number of parameter combinations."""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count


def main():
    """
    Main execution function for model selection pipeline.
    """
    print("="*70)
    print("FOREST COVER TYPE CLASSIFICATION - MODEL SELECTION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load processed data
    print("\nLoading processed datasets...")
    with open('data/processed/datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    # Make writable copies to avoid read-only array issues
    X_train_scaled = datasets['X_train_scaled'].copy()
    X_val_scaled = datasets['X_val_scaled'].copy()
    X_train_unscaled = datasets['X_train_unscaled'].copy()
    X_val_unscaled = datasets['X_val_unscaled'].copy()
    y_train = datasets['y_train'].copy()
    y_val = datasets['y_val'].copy()
    
    print(f"✓ Data loaded successfully")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Features: {X_train_scaled.shape[1]}")
    
    # Initialize model selector
    selector = ModelSelector(random_state=42, n_folds=5, n_jobs=-1, verbose=1)
    
    # Train all models with grid search
    selector.train_all_models(X_train_scaled, X_train_unscaled, y_train)
    
    # Evaluate on validation set
    selector.evaluate_on_validation(X_val_scaled, X_val_unscaled, y_val)
    
    # Check for overfitting
    df_overfitting = selector.check_overfitting(X_train_scaled, X_train_unscaled, y_train)
    
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Generate confusion matrix for best model
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS FOR BEST MODEL")
    print("="*70)
    
    best_model_name = selector.best_model_name
    best_result = selector.results[best_model_name]
    y_val_pred = best_result['val_predictions']
    
    # Confusion matrix plot
    selector.plot_confusion_matrix(
        y_val, y_val_pred, best_model_name,
        save_path='results/plots/confusion_matrix.png'
    )
    
    # Confusion matrix DataFrame
    df_cm = selector.get_confusion_matrix_dataframe(y_val, y_val_pred)
    df_cm.to_csv('results/confusion_matrix.csv')
    print(f"✓ Confusion matrix DataFrame saved to results/confusion_matrix.csv")
    
    # Learning curve
    X_combined_scaled = pd.concat([X_train_scaled, X_val_scaled])
    X_combined_unscaled = pd.concat([X_train_unscaled, X_val_unscaled])
    y_combined = pd.concat([y_train, y_val])
    
    selector.plot_learning_curve(
        X_combined_scaled, X_combined_unscaled, y_combined,
        model_name=best_model_name,
        save_path='results/plots/learning_curve.png'
    )
    
    # Save best model
    selector.save_best_model()
    
    # Save results summary
    selector.save_results_summary()
    
    # Final summary
    print("\n" + "="*70)
    print("MODEL SELECTION COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest Model: {best_model_name}")
    print(f"CV Score: {best_result['best_cv_score']:.4f}")
    print(f"Validation Score: {best_result['val_score']:.4f}")
    print(f"Train Score: {best_result['train_score']:.4f}")
    print(f"\nBest Parameters:")
    for param, value in best_result['best_params'].items():
        print(f"  {param}: {value}")
    
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print("✓ results/best_model.pkl")
    print("✓ results/model_selection_summary.csv")
    print("✓ results/confusion_matrix.csv")
    print("✓ results/plots/confusion_matrix.png")
    print("✓ results/plots/learning_curve.png")


if __name__ == "__main__":
    main()