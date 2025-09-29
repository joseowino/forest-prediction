"""
Prediction Module
=================

This module handles loading the trained model and making predictions on the test dataset.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our preprocessing module
import sys
sys.path.append(os.path.dirname(__file__))
from preprocessing_feature_engineering import ForestDataPreprocessor


class ForestPredictor:
    """
    Class to handle predictions on new data using the trained model.
    """
    
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.scale_features = False
        self.model_name = None
        
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model from pickle file.
        """
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.model = self.model_data['model']
            self.scaler = self.model_data.get('scaler', None)
            self.scale_features = self.model_data.get('scale_features', False)
            self.model_name = self.model_data.get('model_name', 'Unknown')
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model type: {self.model_name}")
            logger.info(f"Requires feature scaling: {self.scale_features}")
            
            if 'cv_score' in self.model_data:
                logger.info(f"Cross-validation score: {self.model_data['cv_score']:.4f}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        np.array
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been loaded!")
        
        # Scale features if needed
        if self.scale_features and self.scaler is not None:
            logger.info("Scaling features for prediction...")
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        logger.info(f"Making predictions on {len(X)} samples...")
        predictions = self.model.predict(X_scaled)
        
        logger.info(f"Predictions completed successfully")
        logger.info(f"Predicted classes: {np.unique(predictions)}")
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities if the model supports it.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        np.array
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been loaded!")
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return None
        
        # Scale features if needed
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)
        logger.info("Prediction probabilities computed successfully")
        
        return probabilities
    
    def evaluate(self, X, y_true):
        """
        Evaluate the model on data with known labels.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y_true : pd.Series or np.array
            True labels
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y_true, predictions)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(y_true, predictions)
        logger.info(f"\nClassification Report:\n{report}")
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': report
        }
        
        return results
    
    def save_predictions(self, predictions, output_path, include_ids=None):
        """
        Save predictions to a CSV file.
        
        Parameters:
        -----------
        predictions : np.array
            Predictions to save
        output_path : str
            Path to save the predictions
        include_ids : array-like, optional
            IDs to include in the output
        """
        # Create predictions dataframe
        if include_ids is not None:
            pred_df = pd.DataFrame({
                'Id': include_ids,
                'Cover_Type': predictions
            })
        else:
            pred_df = pd.DataFrame({
                'Id': range(len(predictions)),
                'Cover_Type': predictions
            })
        
        # Save to CSV
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Display summary
        logger.info(f"\nPrediction Summary:")
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"\nClass distribution:")
        class_counts = pd.Series(predictions).value_counts().sort_index()
        for class_label, count in class_counts.items():
            percentage = (count / len(predictions)) * 100
            logger.info(f"  Class {class_label}: {count} ({percentage:.2f}%)")


def main():
    """
    Main function to make predictions on the test set.
    """
    logger.info("="*80)
    logger.info("FOREST COVER TYPE PREDICTION - TEST SET")
    logger.info("="*80)
    
    # Paths
    model_path = '../results/best_model.pkl'
    train_data_path = '../data/train.csv'
    test_data_path = '../data/test.csv'
    output_path = '../results/test_predictions.csv'
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.error("Please run model_selection.py first to train the model")
        return
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        logger.error(f"Test data file not found: {test_data_path}")
        logger.error("Test data should be available on the last day of the project")
        return
    
    # Initialize preprocessor
    logger.info("\nInitializing data preprocessor...")
    preprocessor = ForestDataPreprocessor()
    
    # Load and prepare training data to get feature names
    logger.info("Loading training data to configure preprocessor...")
    X_train, _, y_train, _ = preprocessor.prepare_data(
        train_data_path,
        engineer_features=True,
        scale=False,
        test_size=0.2,
        random_state=42
    )
    
    # Initialize predictor
    logger.info("\nLoading trained model...")
    predictor = ForestPredictor(model_path)
    
    # Load and prepare test data
    logger.info("\nLoading and preparing test data...")
    test_df_raw = pd.read_csv(test_data_path)
    
    # Check if test data has target column
    has_target = preprocessor.target_name in test_df_raw.columns
    
    if has_target:
        logger.info("Test data contains target labels - will evaluate performance")
        y_test = test_df_raw[preprocessor.target_name]
        test_df_raw = test_df_raw.drop(columns=[preprocessor.target_name])
    
    # Engineer features for test data
    test_df = preprocessor.create_engineered_features(test_df_raw)
    
    # Ensure test data has the same features as training data
    missing_features = set(preprocessor.feature_names) - set(test_df.columns)
    extra_features = set(test_df.columns) - set(preprocessor.feature_names)
    
    if missing_features:
        logger.warning(f"Missing features in test data: {missing_features}")
        for feature in missing_features:
            test_df[feature] = 0
    
    if extra_features:
        logger.info(f"Removing extra features from test data: {extra_features}")
    
    # Select and reorder features to match training data
    X_test = test_df[preprocessor.feature_names]
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    logger.info("\n" + "="*60)
    logger.info("MAKING PREDICTIONS")
    logger.info("="*60)
    
    predictions = predictor.predict(X_test)
    
    # Evaluate if we have labels
    if has_target:
        results = predictor.evaluate(X_test, y_test)
        
        # Check if accuracy meets threshold
        if results['accuracy'] >= 0.65:
            logger.info(f"\n✅ Test accuracy ({results['accuracy']:.4f}) meets the threshold (>= 0.65)")
        else:
            logger.warning(f"\n⚠️  Test accuracy ({results['accuracy']:.4f}) is below the threshold (< 0.65)")
    
    # Save predictions
    logger.info("\n" + "="*60)
    logger.info("SAVING PREDICTIONS")
    logger.info("="*60)
    
    # Create IDs if not present in original data
    if 'Id' in test_df_raw.columns:
        ids = test_df_raw['Id'].values
    else:
        ids = range(len(predictions))
    
    predictor.save_predictions(predictions, output_path, include_ids=ids)
    
    logger.info("\n" + "="*80)
    logger.info("PREDICTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_path}")
    
    if has_target:
        logger.info(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
        logger.info("\nUpdate your README.md with this test accuracy score!")
    
    logger.info("\n✅ All done! Check the results directory for outputs.")


if __name__ == "__main__":
    main()