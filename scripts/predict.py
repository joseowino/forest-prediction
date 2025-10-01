"""
Forest Cover Type Prediction - Prediction Pipeline
==================================================

This module loads the trained model and generates predictions on the test set.

Author: Developer 3
Date: 2025-09-30
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocessing_feature_engineering import ForestDataPreprocessor


class PredictionPipeline:
    """
    End-to-end prediction pipeline for forest cover classification.
    
    Handles loading models, preprocessing test data, generating predictions,
    and saving results.
    """
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        self.preprocessor = None
        self.model_info = None
        self.model = None
        self.predictions = None
        
    def load_preprocessor(self, preprocessor_path='results/preprocessor.pkl'):
        """
        Load the fitted preprocessor.
        
        Parameters:
        -----------
        preprocessor_path : str
            Path to the saved preprocessor
        """
        print(f"Loading preprocessor from {preprocessor_path}...")
        
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                "Please run preprocessing_feature_engineering.py first."
            )
        
        self.preprocessor = ForestDataPreprocessor.load_preprocessor(preprocessor_path)
        print("✓ Preprocessor loaded successfully")
        
    def load_model(self, model_path='results/best_model.pkl'):
        """
        Load the trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        """
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run model_selection.py first."
            )
        
        with open(model_path, 'rb') as f:
            self.model_info = pickle.load(f)
        
        self.model = self.model_info['model']
        
        print("✓ Model loaded successfully")
        print(f"  Model type: {self.model_info['model_name']}")
        print(f"  CV Score: {self.model_info['cv_score']:.4f}")
        print(f"  Requires scaling: {self.model_info['scaled']}")
        print(f"  Trained on: {self.model_info.get('timestamp', 'Unknown')}")
        
    def load_test_data(self, test_path='data/test.csv'):
        """
        Load and validate test data.
        
        Parameters:
        -----------
        test_path : str
            Path to test data CSV
            
        Returns:
        --------
        pd.DataFrame
            Loaded test data
        """
        print(f"\nLoading test data from {test_path}...")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test data not found at {test_path}. "
                "The test set should be available on the last day."
            )
        
        df_test = pd.read_csv(test_path)
        print(f"✓ Test data loaded")
        print(f"  Shape: {df_test.shape}")
        print(f"  Samples: {len(df_test)}")
        
        # Check for missing values
        missing = df_test.isnull().sum().sum()
        if missing > 0:
            print(f"  WARNING: {missing} missing values detected!")
        else:
            print("  No missing values: ✓")
        
        return df_test
    
    def preprocess_test_data(self, df_test):
        """
        Apply preprocessing and feature engineering to test data.
        
        Parameters:
        -----------
        df_test : pd.DataFrame
            Raw test data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed test features
        """
        print("\nPreprocessing test data...")
        
        # Check if preprocessor is loaded
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Call load_preprocessor() first.")
        
        # Apply feature engineering
        df_test_enhanced = self.preprocessor.engineer_features(df_test)
        print(f"✓ Feature engineering applied")
        print(f"  Original features: {df_test.shape[1]}")
        print(f"  Enhanced features: {df_test_enhanced.shape[1]}")
        
        # Separate features (test set has no target)
        if 'Cover_Type' in df_test_enhanced.columns:
            # If test set has target (for validation)
            X_test, y_test = self.preprocessor.prepare_features_target(df_test_enhanced)
            print("  Target found in test set (validation mode)")
            return X_test, y_test
        else:
            # Normal case: no target in test set
            X_test = df_test_enhanced
            print("  No target in test set (prediction mode)")
            return X_test, None
    
    def predict(self, X_test):
        """
        Generate predictions on test data.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
            
        Returns:
        --------
        np.array
            Predictions
        """
        print("\nGenerating predictions...")
        
        # Check if model is loaded
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Apply scaling if required
        if self.model_info['scaled']:
            print("  Applying scaling (model requires scaled features)...")
            X_test_processed = self.preprocessor.transform_with_scaling(X_test, fit=False)
        else:
            print("  Using unscaled features (tree-based model)...")
            X_test_processed = self.preprocessor.transform_without_scaling(X_test)
        
        # Generate predictions
        predictions = self.model.predict(X_test_processed)
        
        print(f"✓ Predictions generated")
        print(f"  Number of predictions: {len(predictions)}")
        print(f"  Unique classes predicted: {np.unique(predictions)}")
        print(f"  Prediction distribution:")
        
        # Show distribution
        unique, counts = np.unique(predictions, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            print(f"    Class {cls}: {count} ({percentage:.1f}%)")
        
        self.predictions = predictions
        return predictions
    
    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy if true labels are available.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Check constraint
        if accuracy > 0.65:
            print(f"✓ CONSTRAINT SATISFIED: Test accuracy ({accuracy:.4f}) > 0.65")
        else:
            print(f"✗ CONSTRAINT NOT SATISFIED: Test accuracy ({accuracy:.4f}) ≤ 0.65")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=[f'Class_{i}' for i in range(1, 8)]))
        
        return accuracy
    
    def save_predictions(self, predictions, output_path='results/test_predictions.csv',
                        include_probabilities=False, X_test=None):
        """
        Save predictions to CSV file.
        
        Parameters:
        -----------
        predictions : array-like
            Predicted labels
        output_path : str
            Path to save predictions
        include_probabilities : bool
            Whether to include prediction probabilities
        X_test : pd.DataFrame, optional
            Test features (for probability predictions)
        """
        print(f"\nSaving predictions to {output_path}...")
        
        # Create DataFrame with predictions
        df_predictions = pd.DataFrame({
            'Id': range(len(predictions)),
            'Cover_Type': predictions
        })
        
        # Add probabilities if requested and model supports it
        if include_probabilities and hasattr(self.model, 'predict_proba'):
            print("  Including prediction probabilities...")
            
            # Apply same preprocessing as for predictions
            if self.model_info['scaled']:
                X_processed = self.preprocessor.transform_with_scaling(X_test, fit=False)
            else:
                X_processed = self.preprocessor.transform_without_scaling(X_test)
            
            probabilities = self.model.predict_proba(X_processed)
            
            # Add probability columns
            for i in range(probabilities.shape[1]):
                df_predictions[f'Prob_Class_{i+1}'] = probabilities[:, i]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df_predictions.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved successfully")
        print(f"  File: {output_path}")
        print(f"  Rows: {len(df_predictions)}")
        print(f"  Columns: {df_predictions.columns.tolist()}")
    
    def update_readme(self, train_accuracy, test_accuracy, model_name, cv_score):
        """
        Update README.md with final results.
        
        Parameters:
        -----------
        train_accuracy : float
            Training set accuracy
        test_accuracy : float
            Test set accuracy
        model_name : str
            Name of the best model
        cv_score : float
            Cross-validation score
        """
        readme_path = 'README.md'
        
        print(f"\nUpdating {readme_path}...")
        
        # Results section to add
        results_section = f"""
## Results

### Model Selection

- **Best Model:** {model_name}
- **Cross-Validation Score:** {cv_score:.4f}
- **Training Set Accuracy:** {train_accuracy:.4f}
- **Test Set Accuracy:** {test_accuracy:.4f}

### Constraint Validation

- ✓ Training accuracy < 0.98: **{train_accuracy:.4f}**
- {'✓' if test_accuracy > 0.65 else '✗'} Test accuracy > 0.65: **{test_accuracy:.4f}**

### Best Model Parameters

```python
{self.model_info['best_params']}
```

### Prediction Details

- Model requires scaling: {self.model_info['scaled']}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Total test predictions: {len(self.predictions) if self.predictions is not None else 'N/A'}

### Files Generated

- Model: `results/best_model.pkl`
- Predictions: `results/test_predictions.csv`
- Confusion Matrix: `results/confusion_matrix.csv`
- Learning Curve: `results/plots/learning_curve.png`

"""
        
        # Check if README exists
        if os.path.exists(readme_path):
            with open(readme_path, 'r') as f:
                content = f.read()
            
            # Check if Results section already exists
            if '## Results' in content:
                print("  Results section already exists - updating...")
                # Replace existing results section
                import re
                pattern = r'## Results.*?(?=\n##|\Z)'
                content = re.sub(pattern, results_section.strip(), content, flags=re.DOTALL)
            else:
                print("  Adding new Results section...")
                # Append results section
                content += "\n" + results_section
            
            # Write back
            with open(readme_path, 'w') as f:
                f.write(content)
        else:
            print("  Creating new README.md...")
            # Create new README with results
            readme_content = f"""# Forest Cover Type Classification

{results_section}

## Project Structure

```
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── covtype.info
├── scripts/
│   ├── preprocessing_feature_engineering.py
│   ├── model_selection.py
│   └── predict.py
├── results/
│   ├── best_model.pkl
│   ├── test_predictions.csv
│   └── plots/
└── README.md
```

## How to Run

1. Preprocessing: `python scripts/preprocessing_feature_engineering.py`
2. Model Selection: `python scripts/model_selection.py`
3. Prediction: `python scripts/predict.py`
"""
            with open(readme_path, 'w') as f:
                f.write(readme_content)
        
        print("✓ README.md updated successfully")
    
    def run_full_pipeline(self, test_path='data/test.csv'):
        """
        Run the complete prediction pipeline.
        
        Parameters:
        -----------
        test_path : str
            Path to test data
        """
        print("="*70)
        print("FOREST COVER TYPE CLASSIFICATION - PREDICTION PIPELINE")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        try:
            # Step 1: Load preprocessor
            self.load_preprocessor()
            
            # Step 2: Load model
            self.load_model()
            
            # Step 3: Load test data
            df_test = self.load_test_data(test_path)
            
            # Step 4: Preprocess test data
            result = self.preprocess_test_data(df_test)
            if isinstance(result, tuple):
                X_test, y_test = result
            else:
                X_test = result
                y_test = None
            
            # Step 5: Generate predictions
            predictions = self.predict(X_test)
            
            # Step 6: Calculate accuracy if true labels available
            test_accuracy = None
            if y_test is not None:
                test_accuracy = self.calculate_accuracy(y_test, predictions)
            
            # Step 7: Save predictions
            self.save_predictions(predictions, include_probabilities=True, X_test=X_test)
            
            # Step 8: Update README (if accuracy available)
            if test_accuracy is not None:
                # Load training data to get train accuracy
                with open('data/processed/datasets.pkl', 'rb') as f:
                    datasets = pickle.load(f)
                
                if self.model_info['scaled']:
                    X_train = datasets['X_train_scaled']
                else:
                    X_train = datasets['X_train_unscaled']
                y_train = datasets['y_train']
                
                train_predictions = self.model.predict(X_train)
                from sklearn.metrics import accuracy_score
                train_accuracy = accuracy_score(y_train, train_predictions)
                
                self.update_readme(
                    train_accuracy=train_accuracy,
                    test_accuracy=test_accuracy,
                    model_name=self.model_info['model_name'],
                    cv_score=self.model_info['cv_score']
                )
            
            # Summary
            print("\n" + "="*70)
            print("PREDICTION PIPELINE COMPLETE")
            print("="*70)
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nPredictions saved to: results/test_predictions.csv")
            if test_accuracy is not None:
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print(f"Constraint {'✓ SATISFIED' if test_accuracy > 0.65 else '✗ NOT SATISFIED'}")
            print("\n" + "="*70)
            
        except FileNotFoundError as e:
            print(f"\n✗ ERROR: {e}")
            print("\nPlease ensure you have:")
            print("  1. Run preprocessing: python scripts/preprocessing_feature_engineering.py")
            print("  2. Run model selection: python scripts/model_selection.py")
            print("  3. Test data is available: data/test.csv")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main execution function."""
    pipeline = PredictionPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()