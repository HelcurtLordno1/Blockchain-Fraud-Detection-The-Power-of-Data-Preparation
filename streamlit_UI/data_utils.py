"""
Data Utilities Module

Handles data loading, preprocessing, and feature extraction
for the Ethereum Phishing Detection system.

Author: Fraud Detection Research Team
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pickle_file(file_path: str) -> Optional[Any]:
    """
    Safely load a pickle file with error handling.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object or None if loading fails
        
    Example:
        >>> tags = load_pickle_file('../data/processed_data/account_tags.pkl')
        >>> if tags:
        ...     print(f"Loaded {len(tags)} tags")
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"Pickle file not found: {file_path}")
        logger.error(f"Please ensure you have run the data preparation notebooks first.")
        logger.error(f"Expected location: {path.parent}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded pickle file: {path.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {str(e)}")
        return None


def load_json_file(file_path: str) -> Optional[Dict]:
    """
    Safely load a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary or None if loading fails
        
    Example:
        >>> config = load_json_file('../config/feature_config.json')
        >>> if config:
        ...     features = config.get('selected_features', [])
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"JSON file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded JSON file: {path.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return None


def load_features_csv(file_path: str, required_columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Load features from CSV file with validation.
    
    Args:
        file_path: Path to CSV file
        required_columns: Optional list of required column names
        
    Returns:
        DataFrame or None if loading fails
        
    Example:
        >>> df = load_features_csv(
        ...     '../data/dataset/Data_after_FE/features.csv',
        ...     required_columns=['node_id', 'node_indegree']
        ... )
        >>> if df is not None:
        ...     print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"CSV file not found: {file_path}")
        logger.error(f"Please run the feature engineering notebooks first.")
        logger.error(f"Expected: Phase 3, Step 5 (01_a_features_engineering.ipynb)")
        return None
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return None
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {str(e)}")
        return None


def extract_features_for_address(
    address: str, 
    features_df: pd.DataFrame,
    feature_columns: List[str]
) -> Optional[np.ndarray]:
    """
    Extract feature vector for a specific address.
    
    Args:
        address: Ethereum address or node ID
        features_df: DataFrame containing all features
        feature_columns: List of feature column names to extract
        
    Returns:
        Feature vector as numpy array or None if address not found
        
    Example:
        >>> features = extract_features_for_address(
        ...     '0x1234...', 
        ...     features_df,
        ...     ['node_indegree', 'node_outdegree', ...]
        ... )
        >>> if features is not None:
        ...     prediction = model.predict(features.reshape(1, -1))
    """
    try:
        # Try to find by address or node_id
        if 'address' in features_df.columns:
            row = features_df[features_df['address'] == address]
        elif 'node_id' in features_df.columns:
            row = features_df[features_df['node_id'] == address]
        else:
            logger.error("DataFrame missing 'address' or 'node_id' column")
            return None
        
        if row.empty:
            logger.warning(f"Address not found: {address}")
            return None
        
        # Extract features
        feature_vector = row[feature_columns].values[0]
        logger.debug(f"Extracted {len(feature_vector)} features for {address}")
        return feature_vector
        
    except Exception as e:
        logger.error(f"Error extracting features for {address}: {str(e)}")
        return None


def validate_feature_vector(features: np.ndarray, expected_length: int) -> bool:
    """
    Validate feature vector shape and content.
    
    Args:
        features: Feature vector to validate
        expected_length: Expected number of features
        
    Returns:
        True if valid, False otherwise
        
    Example:
        >>> if validate_feature_vector(features, 34):
        ...     prediction = model.predict(features.reshape(1, -1))
        ... else:
        ...     print("Invalid feature vector!")
    """
    if not isinstance(features, np.ndarray):
        logger.error(f"Features must be numpy array, got {type(features)}")
        return False
    
    if len(features) != expected_length:
        logger.error(f"Expected {expected_length} features, got {len(features)}")
        return False
    
    if np.any(np.isnan(features)):
        logger.warning("Feature vector contains NaN values")
        return False
    
    if np.any(np.isinf(features)):
        logger.warning("Feature vector contains infinite values")
        return False
    
    return True


def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize feature vector (if needed for certain models).
    
    Args:
        features: Input feature vector
        method: Normalization method ('standard', 'minmax', or 'none')
        
    Returns:
        Normalized feature vector
        
    Example:
        >>> normalized = normalize_features(features, method='standard')
    """
    if method == 'none':
        return features
    
    try:
        if method == 'standard':
            # Z-score normalization
            mean = np.mean(features)
            std = np.std(features) + 1e-8
            return (features - mean) / std
        
        elif method == 'minmax':
            # Min-max scaling to [0, 1]
            min_val = np.min(features)
            max_val = np.max(features)
            return (features - min_val) / (max_val - min_val + 1e-8)
        
        else:
            logger.warning(f"Unknown normalization method: {method}, returning original")
            return features
            
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return features


def get_feature_statistics(features_df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for each feature.
    
    Args:
        features_df: DataFrame containing features
        feature_columns: List of feature column names
        
    Returns:
        Dictionary with statistics for each feature
        
    Example:
        >>> stats = get_feature_statistics(df, ['node_indegree', 'node_outdegree'])
        >>> print(f"Mean indegree: {stats['node_indegree']['mean']:.2f}")
    """
    stats = {}
    
    for col in feature_columns:
        if col in features_df.columns:
            try:
                stats[col] = {
                    'mean': float(features_df[col].mean()),
                    'median': float(features_df[col].median()),
                    'std': float(features_df[col].std()),
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max()),
                    'q25': float(features_df[col].quantile(0.25)),
                    'q75': float(features_df[col].quantile(0.75))
                }
            except Exception as e:
                logger.warning(f"Could not compute stats for {col}: {str(e)}")
                continue
    
    logger.info(f"Computed statistics for {len(stats)} features")
    return stats


def check_data_pipeline_status() -> Dict[str, bool]:
    """
    Check which data files exist in the pipeline.
    
    Returns:
        Dictionary indicating which pipeline stages are complete
        
    Example:
        >>> status = check_data_pipeline_status()
        >>> if not status['features_generated']:
        ...     print("Please run feature engineering notebook first!")
    """
    base_path = Path(__file__).parent.parent
    
    status = {
        'raw_data': (base_path / 'data' / 'raw_data' / 'phisher_accounts.txt').exists(),
        'processed_data': (base_path / 'data' / 'processed_data' / 'eoa2seq.pkl').exists(),
        'features_generated': (base_path / 'data' / 'dataset' / 'Data_after_FE' / 'features.csv').exists(),
        'features_selected': (base_path / 'data' / 'dataset' / 'Data_feature_selection' / 'selected_features.json').exists(),
        'model_trained': (base_path / 'models' / 'save_models' / 'xgboost_no_FE_selection.joblib').exists()
    }
    
    logger.info(f"Pipeline status: {sum(status.values())}/{len(status)} stages complete")
    return status


def generate_sample_features(num_features: int, seed: int = 42) -> np.ndarray:
    """
    Generate sample feature vector for testing/demo purposes.
    
    Args:
        num_features: Number of features to generate
        seed: Random seed for reproducibility
        
    Returns:
        Sample feature vector
        
    Example:
        >>> sample = generate_sample_features(34, seed=42)
        >>> prediction = model.predict(sample.reshape(1, -1))
    """
    np.random.seed(seed)
    
    # Generate realistic-looking features based on actual data distributions
    features = np.random.lognormal(mean=0, sigma=2, size=num_features)
    features = np.clip(features, 0, 1000)  # Clip extreme values
    
    logger.info(f"Generated {num_features} sample features")
    return features.astype(np.float32)
