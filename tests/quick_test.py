"""
Quick Test Script
=================

This script performs a quick test of the model selection pipeline with reduced parameters
to verify everything works before running the full grid search.
"""

import sys
import os
sys.path.append('scripts')

from preprocessing_feature_engineering import ForestDataPreprocessor
from model_selection import ForestModelSelector
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_test():
    """
    Run a quick test with minimal parameters.
    """
    logger.info("="*80)
    logger.info("QUICK TEST MODE - Reduced Parameters")
    logger.info("="*80)
    logger.info("\nThis is a quick test to verify the pipeline works correctly.")
    logger.info("For full training, run: python3 scripts/model_selection.py")
    logger.info("="*80 + "\n")
    
    # Check if data exists
    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        logger.error(f"Training data not found: {data_path}")
        logger.error("Please place train.csv in the data/ directory")
        return
    
    # Initialize preprocessor
    logger.info("Step 1: Loading and preprocessing data...")
    preprocessor = ForestDataPreprocessor()
    
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        data_path,
        engineer_features=True,
        scale=False,
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"✓ Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Initialize model selector with quick parameters
    logger.info("\nStep 2: Testing model selection with minimal parameters...")
    