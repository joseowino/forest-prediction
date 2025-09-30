"""
Forest Cover Type Prediction - Preprocessing and Feature Engineering
====================================================================

This module provides functions for loading, preprocessing, and engineering features
for the forest cover classification project.

Author: Developer 1
Date: 2025-09-30
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import os


class ForestDataPreprocessor:
    """
    A preprocessing pipeline for forest cover classification data.
    
    This class handles data loading, feature engineering, and preprocessing
    for both scaling-sensitive and scale-invariant models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.original_features = None
        self.engineered_features = None
        
    def load_data(self, filepath):
        """
        Load data from CSV file with basic quality checks.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Data shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        
        return df
    
    def check_data_quality(self, df):
        """
        Perform basic data quality checks.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        dict
            Dictionary containing quality metrics
        """
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for any missing values
        if df.isnull().sum().sum() > 0:
            print("WARNING: Missing values detected!")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print("✓ No missing values detected")
            
        return quality_report
    
    def engineer_features(self, df):
        """
        Create engineered features from cartographic variables.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with original features
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional engineered features
        """
        print("Engineering features...")
        df_enhanced = df.copy()
        
        # 1. Euclidean distance to hydrology
        # Combines horizontal and vertical distances for true spatial distance
        df_enhanced['Distance_To_Hydrology'] = np.sqrt(
            df['Horizontal_Distance_To_Hydrology']**2 + 
            df['Vertical_Distance_To_Hydrology']**2
        )
        
        # 2. Difference between fire points and roadways
        # May indicate accessibility and fire management
        df_enhanced['Fire_Road_Distance_Diff'] = (
            df['Horizontal_Distance_To_Hydrology'] - 
            df['Horizontal_Distance_To_Roadways']
        )
        
        # 3. Mean distance to all infrastructure
        df_enhanced['Mean_Distance_To_Infrastructure'] = df[[
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points'
        ]].mean(axis=1)
        
        # 4. Elevation-related features
        # Hillshade at different times captures sun exposure patterns
        df_enhanced['Mean_Hillshade'] = df[[
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm'
        ]].mean(axis=1)
        
        # Hillshade variance (terrain complexity indicator)
        df_enhanced['Hillshade_Variance'] = df[[
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm'
        ]].var(axis=1)
        
        # 5. Aspect-related features
        # Convert aspect to sine and cosine for circular nature
        aspect_rad = df['Aspect'] * np.pi / 180
        df_enhanced['Aspect_Sin'] = np.sin(aspect_rad)
        df_enhanced['Aspect_Cos'] = np.cos(aspect_rad)
        
        # 6. Slope-Elevation interaction
        # Steeper slopes at higher elevations may indicate different cover types
        df_enhanced['Slope_Elevation_Interaction'] = (
            df['Slope'] * df['Elevation'] / 1000  # Normalize
        )
        
        # 7. Distance to hydrology relative to elevation
        # Lower values = closer to water at given elevation
        df_enhanced['Hydrology_Elevation_Ratio'] = (
            df_enhanced['Distance_To_Hydrology'] / (df['Elevation'] + 1)
        )
        
        # 8. Wilderness area count (binary features sum)
        wilderness_cols = [col for col in df.columns if 'Wilderness_Area' in col]
        if wilderness_cols:
            df_enhanced['Wilderness_Area_Count'] = df[wilderness_cols].sum(axis=1)
        
        # 9. Soil type count
        soil_cols = [col for col in df.columns if 'Soil_Type' in col]
        if soil_cols:
            df_enhanced['Soil_Type_Count'] = df[soil_cols].sum(axis=1)
        
        # 10. North-facing indicator (north-facing slopes often differ)
        df_enhanced['Is_North_Facing'] = ((df['Aspect'] >= 315) | 
                                          (df['Aspect'] <= 45)).astype(int)
        
        # 11. South-facing indicator
        df_enhanced['Is_South_Facing'] = ((df['Aspect'] >= 135) & 
                                          (df['Aspect'] <= 225)).astype(int)
        
        print(f"✓ Created {len(df_enhanced.columns) - len(df.columns)} new features")
        
        return df_enhanced
    
    def prepare_features_target(self, df, target_col='Cover_Type'):
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str, default='Cover_Type'
            Name of the target column
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        """
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            print(f"Features shape: {X.shape}")
            print(f"Target distribution:\n{y.value_counts().sort_index()}")
            return X, y
        else:
            # Test set without target
            print(f"Features shape: {df.shape}")
            return df, None
    
    def get_feature_groups(self, X):
        """
        Identify different groups of features for selective scaling.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        dict
            Dictionary with feature groups
        """
        # Continuous features that need scaling
        continuous_features = [
            'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Distance_To_Hydrology', 'Fire_Road_Distance_Diff',
            'Mean_Distance_To_Infrastructure', 'Mean_Hillshade',
            'Hillshade_Variance', 'Aspect_Sin', 'Aspect_Cos',
            'Slope_Elevation_Interaction', 'Hydrology_Elevation_Ratio'
        ]
        
        # Binary features (don't need scaling)
        binary_features = [col for col in X.columns if 
                          ('Wilderness_Area' in col or 'Soil_Type' in col or
                           'Is_North_Facing' in col or 'Is_South_Facing' in col)]
        
        # Filter to only include features that exist in X
        continuous_features = [f for f in continuous_features if f in X.columns]
        
        return {
            'continuous': continuous_features,
            'binary': binary_features
        }
    
    def fit_scaler(self, X_train):
        """
        Fit the scaler on training data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix
        """
        feature_groups = self.get_feature_groups(X_train)
        continuous_features = feature_groups['continuous']
        
        if continuous_features:
            self.scaler.fit(X_train[continuous_features])
            print(f"✓ Scaler fitted on {len(continuous_features)} continuous features")
        
    def transform_with_scaling(self, X, fit=False):
        """
        Transform features with scaling (for SVM, KNN, Logistic Regression).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        fit : bool, default=False
            Whether to fit the scaler
            
        Returns:
        --------
        pd.DataFrame
            Scaled feature matrix
        """
        X_scaled = X.copy()
        feature_groups = self.get_feature_groups(X)
        continuous_features = feature_groups['continuous']
        
        if continuous_features:
            if fit:
                X_scaled[continuous_features] = self.scaler.fit_transform(
                    X[continuous_features]
                )
            else:
                X_scaled[continuous_features] = self.scaler.transform(
                    X[continuous_features]
                )
        
        return X_scaled
    
    def transform_without_scaling(self, X):
        """
        Return features without scaling (for tree-based models).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        pd.DataFrame
            Unscaled feature matrix
        """
        return X.copy()
    
    def save_preprocessor(self, filepath='results/preprocessor.pkl'):
        """
        Save the preprocessor object.
        
        Parameters:
        -----------
        filepath : str
            Path to save the preprocessor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath='results/preprocessor.pkl'):
        """
        Load a saved preprocessor object.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved preprocessor
            
        Returns:
        --------
        ForestDataPreprocessor
            Loaded preprocessor object
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded from {filepath}")
        return preprocessor


def create_train_validation_split(X, y, test_size=0.2, random_state=42):
    """
    Create stratified train-validation split.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float, default=0.2
        Proportion of data for validation
    random_state : int, default=42
        Random seed
        
    Returns:
    --------
    X_train, X_val, y_train, y_val
        Split datasets
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Training set class distribution:\n{y_train.value_counts().sort_index()}")
    
    return X_train, X_val, y_train, y_val


def main():
    """
    Main execution function demonstrating the preprocessing pipeline.
    """
    print("="*70)
    print("FOREST COVER TYPE CLASSIFICATION - PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = ForestDataPreprocessor(random_state=42)
    
    # Load training data
    df_train = preprocessor.load_data('data/train.csv')
    
    # Check data quality
    print("\n" + "="*70)
    print("DATA QUALITY CHECK")
    print("="*70)
    quality_report = preprocessor.check_data_quality(df_train)
    
    # Engineer features
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)
    df_enhanced = preprocessor.engineer_features(df_train)
    
    # Prepare features and target
    print("\n" + "="*70)
    print("PREPARING FEATURES AND TARGET")
    print("="*70)
    X, y = preprocessor.prepare_features_target(df_enhanced)
    
    # Create train-validation split
    print("\n" + "="*70)
    print("CREATING TRAIN-VALIDATION SPLIT")
    print("="*70)
    X_train, X_val, y_train, y_val = create_train_validation_split(X, y)
    
    # Prepare scaled and unscaled versions
    print("\n" + "="*70)
    print("PREPARING SCALED AND UNSCALED DATASETS")
    print("="*70)
    
    # Fit scaler on training data
    preprocessor.fit_scaler(X_train)
    
    # Create scaled versions (for SVM, KNN, Logistic Regression)
    X_train_scaled = preprocessor.transform_with_scaling(X_train, fit=False)
    X_val_scaled = preprocessor.transform_with_scaling(X_val, fit=False)
    
    # Unscaled versions (for tree-based models)
    X_train_unscaled = preprocessor.transform_without_scaling(X_train)
    X_val_unscaled = preprocessor.transform_without_scaling(X_val)
    
    print("✓ Scaled datasets prepared (for SVM, KNN, Logistic Regression)")
    print("✓ Unscaled datasets prepared (for Random Forest, Gradient Boosting)")
    
    # Save preprocessor
    print("\n" + "="*70)
    print("SAVING PREPROCESSOR")
    print("="*70)
    preprocessor.save_preprocessor()
    
    # Save processed datasets
    os.makedirs('data/processed', exist_ok=True)
    
    # Save as pickle for efficient loading
    datasets = {
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_train_unscaled': X_train_unscaled,
        'X_val_unscaled': X_val_unscaled,
        'y_train': y_train,
        'y_val': y_val
    }
    
    with open('data/processed/datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    
    print("✓ Processed datasets saved to 'data/processed/datasets.pkl'")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Number of classes: {len(y.unique())}")
    print("\nPreprocessing pipeline complete! ✓")


if __name__ == "__main__":
    main()