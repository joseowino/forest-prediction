"""
Preprocessing and Feature Engineering Module
===========================================

This module handles data preprocessing and feature engineering for the forest cover type prediction project.
It includes functions for loading data, creating new features, and preparing data for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForestDataPreprocessor:
    """
    A comprehensive preprocessor for forest cover type data.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Data shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def identify_feature_types(self, df, target_col=None):
        """
        Identify different types of features in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str, optional
            Name of target column. If None, assumes last column is target.
            
        Returns:
        --------
        dict
            Dictionary containing different feature types
        """
        if target_col is None:
            target_col = df.columns[-1]
            
        features = df.drop(columns=[target_col])
        
        # Identify feature types based on common naming conventions
        wilderness_cols = [col for col in features.columns if 'Wilderness_Area' in col]
        soil_cols = [col for col in features.columns if 'Soil_Type' in col]
        numerical_cols = [col for col in features.columns if col not in wilderness_cols + soil_cols]
        
        feature_types = {
            'numerical': numerical_cols,
            'wilderness': wilderness_cols,
            'soil': soil_cols,
            'target': target_col
        }
        
        logger.info(f"Feature types identified:")
        logger.info(f"  - Numerical: {len(numerical_cols)} features")
        logger.info(f"  - Wilderness: {len(wilderness_cols)} features")
        logger.info(f"  - Soil: {len(soil_cols)} features")
        logger.info(f"  - Target: {target_col}")
        
        return feature_types
    
    def create_engineered_features(self, df):
        """
        Create engineered features based on domain knowledge.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with additional engineered features
        """
        df_eng = df.copy()
        
        # 1. Distance to hydrology (Euclidean distance)
        if 'Horizontal_Distance_To_Hydrology' in df.columns and 'Vertical_Distance_To_Hydrology' in df.columns:
            df_eng['Distance_To_Hydrology'] = np.sqrt(
                df['Horizontal_Distance_To_Hydrology']**2 + 
                df['Vertical_Distance_To_Hydrology']**2
            )
            logger.info("Created feature: Distance_To_Hydrology")
        
        # 2. Road-Fire distance difference
        if 'Horizontal_Distance_To_Fire_Points' in df.columns and 'Horizontal_Distance_To_Roadways' in df.columns:
            df_eng['Fire_Road_Distance_Diff'] = (
                df['Horizontal_Distance_To_Fire_Points'] - 
                df['Horizontal_Distance_To_Roadways']
            )
            logger.info("Created feature: Fire_Road_Distance_Diff")
        
        # 3. Mean distance to infrastructure
        distance_cols = [col for col in df.columns if 'Distance' in col and 'Vertical' not in col]
        if len(distance_cols) > 1:
            df_eng['Mean_Distance_To_Infrastructure'] = df[distance_cols].mean(axis=1)
            logger.info("Created feature: Mean_Distance_To_Infrastructure")
        
        # 4. Elevation-based features
        if 'Elevation' in df.columns:
            df_eng['Elevation_Squared'] = df['Elevation']**2
            df_eng['Elevation_Log'] = np.log1p(df['Elevation'])  # log(1+x) to handle potential zeros
            logger.info("Created features: Elevation_Squared, Elevation_Log")
        
        # 5. Aspect trigonometric transformations (convert circular feature)
        if 'Aspect' in df.columns:
            # Convert aspect to radians
            aspect_rad = df['Aspect'] * np.pi / 180
            df_eng['Aspect_Sin'] = np.sin(aspect_rad)
            df_eng['Aspect_Cos'] = np.cos(aspect_rad)
            logger.info("Created features: Aspect_Sin, Aspect_Cos")
        
        # 6. Slope categories
        if 'Slope' in df.columns:
            df_eng['Slope_Category'] = pd.cut(df['Slope'], 
                                             bins=[0, 10, 20, 30, float('inf')], 
                                             labels=['Flat', 'Moderate', 'Steep', 'Very_Steep'])
            # One-hot encode slope categories
            slope_dummies = pd.get_dummies(df_eng['Slope_Category'], prefix='Slope')
            df_eng = pd.concat([df_eng, slope_dummies], axis=1)
            df_eng.drop('Slope_Category', axis=1, inplace=True)
            logger.info("Created slope category features")
        
        # 7. Hillshade features interaction
        hillshade_cols = [col for col in df.columns if 'Hillshade' in col]
        if len(hillshade_cols) > 1:
            df_eng['Mean_Hillshade'] = df[hillshade_cols].mean(axis=1)
            df_eng['Hillshade_Range'] = df[hillshade_cols].max(axis=1) - df[hillshade_cols].min(axis=1)
            logger.info("Created hillshade interaction features")
        
        # 8. Wilderness area count
        wilderness_cols = [col for col in df.columns if 'Wilderness_Area' in col]
        if wilderness_cols:
            df_eng['Wilderness_Count'] = df[wilderness_cols].sum(axis=1)
            logger.info("Created feature: Wilderness_Count")
        
        # 9. Soil type count (in case multiple soil types are possible)
        soil_cols = [col for col in df.columns if 'Soil_Type' in col]
        if soil_cols:
            df_eng['Soil_Count'] = df[soil_cols].sum(axis=1)
            logger.info("Created feature: Soil_Count")
        
        new_features = [col for col in df_eng.columns if col not in df.columns]
        logger.info(f"Total new features created: {len(new_features)}")
        
        return df_eng
    
    def scale_features(self, X_train, X_test, method='standard'):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        method : str
            Scaling method ('standard' or 'robust')
            
        Returns:
        --------
        tuple
            Scaled training and test features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Identify numerical columns (excluding binary features)
        numerical_cols = []
        for col in X_train.columns:
            if X_train[col].nunique() > 2:  # Not binary
                numerical_cols.append(col)
        
        if numerical_cols:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            
            logger.info(f"Scaled {len(numerical_cols)} numerical features using {method} scaling")
            
            return X_train_scaled, X_test_scaled
        else:
            logger.warning("No numerical features found for scaling")
            return X_train, X_test
    
    def prepare_data(self, file_path, target_col=None, test_size=0.2, random_state=42, 
                     engineer_features=True, scale=False, scaling_method='standard'):
        """
        Complete data preparation pipeline.
        
        Parameters:
        -----------
        file_path : str
            Path to data file
        target_col : str, optional
            Name of target column
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        engineer_features : bool
            Whether to create engineered features
        scale : bool
            Whether to scale features
        scaling_method : str
            Method for scaling ('standard' or 'robust')
            
        Returns:
        --------
        tuple
            X_train, X_test, y_train, y_test
        """
        # Load data
        df = self.load_data(file_path)
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values. Filling with median...")
            df = df.fillna(df.median())
        
        # Engineer features if requested
        if engineer_features:
            df = self.create_engineered_features(df)
        
        # Identify target
        if target_col is None:
            target_col = df.columns[-1]
        
        self.target_name = target_col
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Scale if requested
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test, method=scaling_method)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_test_data(self, file_path, engineer_features=True, scale=False):
        """
        Prepare test data using the same transformations as training data.
        
        Parameters:
        -----------
        file_path : str
            Path to test data file
        engineer_features : bool
            Whether to create engineered features
        scale : bool
            Whether to scale features
            
        Returns:
        --------
        pd.DataFrame
            Prepared test features
        """
        # Load data
        df = self.load_data(file_path)
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values. Filling with median...")
            df = df.fillna(df.median())
        
        # Engineer features if requested
        if engineer_features:
            df = self.create_engineered_features(df)
        
        # Remove target if present
        if self.target_name and self.target_name in df.columns:
            df = df.drop(columns=[self.target_name])
        
        # Ensure all training features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            extra_features = set(df.columns) - set(self.feature_names)
            
            if missing_features:
                logger.warning(f"Missing features in test data: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0
            
            if extra_features:
                logger.warning(f"Extra features in test data (will be removed): {extra_features}")
                df = df[self.feature_names]
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        # Scale if requested and scaler exists
        if scale and self.scaler is not None:
            numerical_cols = []
            for col in df.columns:
                if df[col].nunique() > 2:
                    numerical_cols.append(col)
            
            if numerical_cols:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return df


def main():
    """
    Example usage of the preprocessor.
    """
    # Initialize preprocessor
    preprocessor = ForestDataPreprocessor()
    
    # Prepare training data
    data_path = '../data/train.csv'
    
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(
            data_path,
            engineer_features=True,
            scale=False,  # We'll scale separately for different models
            test_size=0.2,
            random_state=42
        )
        
        logger.info(f"\nFinal dataset shapes:")
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"X_test: {X_test.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"y_test: {y_test.shape}")
        
        logger.info(f"\nTarget distribution in training set:")
        logger.info(y_train.value_counts().sort_index())
    else:
        logger.error(f"Data file not found: {data_path}")


if __name__ == "__main__":
    main()