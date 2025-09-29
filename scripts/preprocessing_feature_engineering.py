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
    
    def prepare_features_target(self, df, target_col=None):
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str, optional
            Name of target column
            
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        if target_col is None:
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        self.target_name = target_col
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Target classes: {sorted(y.unique())}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, scaler_type='standard'):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
        scaler_type : str
            Type of scaler ('standard' or 'robust')
            
        Returns:
        --------
        tuple or pd.DataFrame
            Scaled features
        """
        # Identify numerical columns (exclude binary features)
        numerical_cols = []
        for col in X_train.columns:
            if not (X_train[col].isin([0, 1]).all() and X_train[col].nunique() == 2):
                numerical_cols.append(col)
        
        logger.info(f"Scaling {len(numerical_cols)} numerical features")
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Scale training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_info(self, X):
        """
        Get information about features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
            
        Returns:
        --------
        pd.DataFrame
            Feature information
        """
        info_data = []
        
        for col in X.columns:
            info_data.append({
                'Feature': col,
                'Type': X[col].dtype,
                'Unique_Values': X[col].nunique(),
                'Missing_Values': X[col].isnull().sum(),
                'Mean': X[col].mean() if pd.api.types.is_numeric_dtype(X[col]) else None,
                'Std': X[col].std() if pd.api.types.is_numeric_dtype(X[col]) else None
            })
        
        return pd.DataFrame(info_data)


def preprocess_pipeline(train_path, test_path=None, create_features=True, scale_data=False):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    train_path : str
        Path to training data
    test_path : str, optional
        Path to test data
    create_features : bool
        Whether to create engineered features
    scale_data : bool
        Whether to scale features
        
    Returns:
    --------
    dict
        Dictionary containing processed data and metadata
    """
    preprocessor = ForestDataPreprocessor()
    
    # Load training data
    logger.info("Loading training data...")
    train_df = preprocessor.load_data(train_path)
    
    # Create engineered features
    if create_features:
        logger.info("Creating engineered features...")
        train_df = preprocessor.create_engineered_features(train_df)
    
    # Prepare features and target
    X_train, y_train = preprocessor.prepare_features_target(train_df)
    
    # Process test data if provided
    X_test, y_test = None, None
    if test_path and os.path.exists(test_path):
        logger.info("Loading test data...")
        test_df = preprocessor.load_data(test_path)
        
        if create_features:
            logger.info("Creating engineered features for test data...")
            test_df = preprocessor.create_engineered_features(test_df)
        
        X_test, y_test = preprocessor.prepare_features_target(test_df, 
                                                               target_col=preprocessor.target_name)
    
    # Scale features if requested
    if scale_data:
        logger.info("Scaling features...")
        if X_test is not None:
            X_train, X_test = preprocessor.scale_features(X_train, X_test)
        else:
            X_train = preprocessor.scale_features(X_train)
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names
    }
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Paths
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    
    # Run preprocessing
    try:
        result = preprocess_pipeline(
            train_path=train_path,
            test_path=test_path if os.path.exists(test_path) else None,
            create_features=True,
            scale_data=False  # We'll scale separately for different models
        )
        
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Training samples: {result['X_train'].shape[0]}")
        print(f"Number of features: {result['X_train'].shape[1]}")
        print(f"Number of classes: {result['y_train'].nunique()}")
        
        if result['X_test'] is not None:
            print(f"Test samples: {result['X_test'].shape[0]}")
        
        print("\nClass distribution in training data:")
        print(result['y_train'].value_counts().sort_index())
        
        print("\nFeature info (first 10 features):")
        feature_info = result['preprocessor'].get_feature_info(result['X_train'])
        print(feature_info.head(10))
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        sys.exit(1)