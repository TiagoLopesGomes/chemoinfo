"""
Data processing utilities for molecular datasets.
Handles data loading, splitting, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
import deepchem
from typing import Tuple, Dict
from joblib import Parallel, delayed
from tqdm import tqdm
from src.config import DATA_ROOT


#taken from https://github.com/insitro/kindel/blob/main/kindel/utils/data.py
def get_testing_data(target: str, in_library: bool = False) -> Dict:
    """
    Load testing data for a specific target.
    
    Args:
        target: Target name (e.g., 'ddr1')
        in_library: Whether to filter for in-library compounds
        
    Returns:
        Dict containing 'on' and 'off' dataframes
    """
    data = {
        "on": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_ondna.csv"), 
            index_col=0
        ).rename({"kd": "y"}, axis="columns"),
        "off": pd.read_csv(
            os.path.join(DATA_ROOT, "heldout", f"{target}_offdna.csv"), 
            index_col=0
        ).rename({"kd": "y"}, axis="columns"),
    }
    
    if in_library:
        data["on"] = data["on"].dropna(subset="molecule_hash")
        data["off"] = data["off"].dropna(subset="molecule_hash")
    return data
