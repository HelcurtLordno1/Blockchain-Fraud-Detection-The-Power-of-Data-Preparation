"""
Model Utilities Module

Handles model loading, prediction, and model-related operations
for the Ethereum Phishing Detection system.

Author: Fraud Detection Research Team
"""

import joblib
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_with_fallback(model_path: str, fallback_path: Optional[str] = None) -> Optional[Any]:
    """
    Load a trained model with defensive file existence checks and fallback option.
    
    Args:
        model_path: Primary path to the model file (.joblib)
        fallback_path: Optional fallback path if primary fails
        
    Returns:
        Loaded model object or None if loading fails
        
    Example:
        >>> model = load_model_with_fallback(
        ...     '../models/save_models/xgboost_no_FE_selection.joblib',
        ...     '../models/save_models/xgboost_with_FE_selection.joblib'
        ... )
        >>> if model:
        ...     predictions = model.predict(features)
    """
    # Check primary path
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        
        # Try fallback
        if fallback_path and Path(fallback_path).exists():
            logger.info(f"Using fallback model: {fallback_path}")
            model_path = fallback_path
        else:
            logger.error("No valid model file found. Please run training notebooks first.")
            logger.error("Expected location: models/save_models/xgboost_*.joblib")
            return None
    
    try:
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def load_model_metadata(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model metadata (feature names, training info, etc.)
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Dictionary containing metadata or None if loading fails
        
    Example:
        >>> metadata = load_model_metadata('../models/save_models/xgboost_metadata.json')
        >>> if metadata:
        ...     feature_names = metadata.get('feature_names', [])
    """
    if not Path(metadata_path).exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded: {len(metadata)} keys")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        return None


def predict_with_probability(
    model: Any, 
    features: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[int, float, str]:
    """
    Make prediction with probability scores and confidence level.
    
    Args:
        model: Trained classifier with predict_proba method
        features: Feature vector (1D or 2D array)
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Tuple of (prediction, probability, confidence_level)
        - prediction: 0 (normal) or 1 (phisher)
        - probability: Probability of being a phisher (0.0-1.0)
        - confidence_level: 'High', 'Medium', or 'Low'
        
    Example:
        >>> pred, prob, conf = predict_with_probability(model, features, threshold=0.5)
        >>> print(f"Prediction: {'Phisher' if pred == 1 else 'Normal'}")
        >>> print(f"Confidence: {prob:.2%} ({conf})")
    """
    try:
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get probability predictions
        probabilities = model.predict_proba(features)[0]
        phisher_prob = probabilities[1]  # Probability of class 1 (phisher)
        
        # Make binary prediction
        prediction = 1 if phisher_prob >= threshold else 0
        
        # Determine confidence level
        if phisher_prob >= 0.8 or phisher_prob <= 0.2:
            confidence = 'High'
        elif phisher_prob >= 0.6 or phisher_prob <= 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        logger.debug(f"Prediction: {prediction}, Prob: {phisher_prob:.4f}, Conf: {confidence}")
        return prediction, phisher_prob, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


def get_feature_importance(
    model: Any, 
    feature_names: Optional[list] = None, 
    top_n: int = 10
) -> Dict[str, float]:
    """
    Extract and sort feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Optional list of feature names
        top_n: Number of top features to return
        
    Returns:
        Dictionary of {feature_name: importance_score}
        
    Example:
        >>> importance = get_feature_importance(model, feature_names, top_n=5)
        >>> for feature, score in importance.items():
        ...     print(f"{feature}: {score:.4f}")
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
        
        importances = model.feature_importances_
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        importance_dict = {
            feature_names[i]: float(importances[i]) 
            for i in indices
        }
        
        logger.info(f"Extracted top {top_n} feature importances")
        return importance_dict
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return {}


def validate_model_compatibility(model: Any, expected_features: int) -> bool:
    """
    Validate that model expects the correct number of features.
    
    Args:
        model: Trained model
        expected_features: Expected number of input features
        
    Returns:
        True if compatible, False otherwise
        
    Example:
        >>> if validate_model_compatibility(model, 34):
        ...     predictions = model.predict(features)
        ... else:
        ...     print("Feature dimension mismatch!")
    """
    try:
        # Try to get n_features from model
        if hasattr(model, 'n_features_in_'):
            actual_features = model.n_features_in_
            is_compatible = actual_features == expected_features
            
            if not is_compatible:
                logger.warning(
                    f"Feature mismatch: model expects {actual_features}, "
                    f"got {expected_features}"
                )
            return is_compatible
        else:
            logger.warning("Cannot determine model feature count")
            return True  # Assume compatible if can't check
            
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        return False
