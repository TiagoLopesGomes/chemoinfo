"""
Metrics module for evaluating model performance.
Contains implementations of various metrics used for model evaluation.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from typing import Union, Any, Dict
import deepchem

def convert_to_numpy(data: Any) -> np.ndarray:
    """Convert input data to numpy array if needed."""
    if hasattr(data, 'values'):  # if it's a pandas Series
        data = data.values
    return data.flatten()

def spearman(preds: Any, target: Any) -> float:
    """
    Calculate Spearman correlation coefficient.
    
    Args:
        preds: Predicted values
        target: True values
        
    Returns:
        float: Spearman correlation coefficient
    """
    preds = convert_to_numpy(preds)
    target = convert_to_numpy(target)
    return spearmanr(preds, target)[0]

def kendall(preds: Any, target: Any) -> float:
    """Calculate Kendall's tau correlation coefficient."""
    preds = convert_to_numpy(preds)
    target = convert_to_numpy(target)
    return kendalltau(preds, target)[0]

def rmse(preds: Any, target: Any) -> float:
    """Calculate Root Mean Square Error."""
    preds = convert_to_numpy(preds)
    target = convert_to_numpy(target)
    return np.sqrt(np.mean((preds - target) ** 2))

def mse(preds: Any, target: Any) -> float:
    """Calculate Mean Square Error."""
    preds = convert_to_numpy(preds)
    target = convert_to_numpy(target)
    return np.mean((preds - target) ** 2)

def evaluate_model_performance(model, 
                             train_data, 
                             valid_data, 
                             test_data,
                             normalizer,
                             featurizer_fn,
                             target: str = "ddr1") -> Dict:
    """
    Evaluate model performance on all datasets including held-out sets.
    
    Args:
        model: Trained model
        train_data: Training dataset
        valid_data: Validation dataset
        test_data: Test dataset
        normalizer: Data normalizer
        featurizer_fn: Function to featurize SMILES strings
        target: Target name for held-out evaluation
        
    Returns:
        Dict containing all evaluation metrics
    """
    from src.utils.data_processing import get_testing_data
    
    results = {}
    predictions = {'all': {}, 'lib': {}}
    
    # Training, validation and test metrics
    for name, data in [("train", train_data), 
                      ("valid", valid_data), 
                      ("test", test_data)]:
        preds = model.predict(data)
        denorm_preds = normalizer.untransform(preds)
        denorm_y = normalizer.untransform(data.y)
        
        results[name] = {
            "rho": spearman(denorm_preds, denorm_y),
            "tau": kendall(denorm_preds, denorm_y),
            "rmse": rmse(denorm_preds, denorm_y),
            "mse": mse(denorm_preds, denorm_y)
        }
    
    # Get held-out data evaluation
    testing_data = get_testing_data(target)
    
    results["all"] = {}
    results["lib"] = {}
    
    for condition in ("on", "off"):
        # All held-out data
        df = testing_data[condition]
        features = featurizer_fn(df["smiles"].values)
        dataset = deepchem.data.NumpyDataset(X=features, y=df["y"].values)
        normalized_dataset = normalizer.transform(dataset)
        preds = model.predict(normalized_dataset)
        denorm_preds = normalizer.untransform(preds)
        
        # Store predictions for plotting
        predictions['all'][condition] = {
            'true': df["y"].values,
            'pred': denorm_preds.flatten()
        }
        
        results["all"][condition] = {
            "rho": spearman(denorm_preds, df["y"]),
            "tau": kendall(denorm_preds, df["y"]),
            "rmse": rmse(denorm_preds, df["y"]),
            "mse": mse(denorm_preds, df["y"])
        }
        
        # In-library held-out data
        testing_data_lib = get_testing_data(target, in_library=True)
        df_lib = testing_data_lib[condition]
        features_lib = featurizer_fn(df_lib["smiles"].values)
        dataset_lib = deepchem.data.NumpyDataset(X=features_lib, y=df_lib["y"].values)
        normalized_dataset_lib = normalizer.transform(dataset_lib)
        preds_lib = model.predict(normalized_dataset_lib)
        denorm_preds_lib = normalizer.untransform(preds_lib)
        
        # Store predictions for plotting
        predictions['lib'][condition] = {
            'true': df_lib["y"].values,
            'pred': denorm_preds_lib.flatten()
        }
        
        results["lib"][condition] = {
            "rho": spearman(denorm_preds_lib, df_lib["y"]),
            "tau": kendall(denorm_preds_lib, df_lib["y"]),
            "rmse": rmse(denorm_preds_lib, df_lib["y"]),
            "mse": mse(denorm_preds_lib, df_lib["y"])
        }
    
    return results, predictions

def convert_results(results_dict):
    """Convert numpy values to regular Python floats."""
    converted = {}
    for key, value in results_dict.items():
        if isinstance(value, dict):
            converted[key] = convert_results(value)
        elif hasattr(value, 'dtype'):  # Check if it's a numpy value
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted
