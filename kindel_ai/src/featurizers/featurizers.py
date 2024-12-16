"""
Molecular featurization module with parallel processing capabilities.
Contains implementations of various featurization methods.
"""

import numpy as np
import deepchem
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Any, Union
import pandas as pd




def get_featurizer(featurizer_name: str, size: int = 1024, radius: int = 2):
    """Get featurizer based on name."""
    if featurizer_name == 'morgan':
        return deepchem.feat.CircularFingerprint(radius=radius, size=size)
    elif featurizer_name == 'graph':
        return deepchem.feat.ConvMolFeaturizer()
    elif featurizer_name == 'attentivefp':
        return deepchem.feat.MolGraphConvFeaturizer(use_edges=True)
    elif featurizer_name == 'mpnn':
        return deepchem.feat.WeaveFeaturizer()
    else:
        raise ValueError(f"Unknown featurizer: {featurizer_name}")
    
    

def parallel_featurization(csv_path: str, featurizer_name: str = 'morgan', n_jobs: int = 10) -> deepchem.data.DiskDataset:
    """
    Parallel featurization of molecules from CSV file.
    
    Args:
        csv_path: Path to CSV file containing SMILES
        featurizer_name: Name of featurizer
        n_jobs: Number of parallel jobs
        
    Returns:
        DiskDataset: DeepChem dataset with features
    """
    # Read data
    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].values
    targets = df['target_enrichment'].values
    total_mols = len(smiles_list)
    
    print(f"Starting featurization of {total_mols} molecules...")
    
    # Get featurizer based on name
    featurizer = get_featurizer(featurizer_name)
    
    # For graph-based featurizers
    if featurizer_name in ['attentivefp', 'mpnn', 'graph']:
        print("Using single process for graph featurization...")
        features = []
        batch_size = 10000
        for i in tqdm(range(0, total_mols, batch_size), desc="Featurizing"):
            batch = smiles_list[i:i + batch_size]
            batch_features = featurizer.featurize(batch)
            features.append(batch_features)
        features = np.vstack(features)
        return deepchem.data.DiskDataset.from_numpy(X=features, y=targets.reshape(-1, 1))
    
    # For fingerprint featurizers in parallel
    chunk_size = len(smiles_list) // n_jobs
    smiles_chunks = [smiles_list[i:i + chunk_size] for i in range(0, len(smiles_list), chunk_size)]
    
    features = Parallel(n_jobs=n_jobs)(
        delayed(featurizer.featurize)(chunk) for chunk in tqdm(smiles_chunks, 
                                                             desc="Featurizing",
                                                             total=len(smiles_chunks))
    )
    
    # Combine results
    features = np.vstack(features)
    
    # Create proper DeepChem Dataset
    dataset = deepchem.data.DiskDataset.from_numpy(X=features, 
                                                  y=targets.reshape(-1, 1))
    return dataset

def single_process_featurization(smiles_list: List[str], 
                               featurizer_name: str = 'morgan',
                               size: int = 1024, 
                               radius: int = 2) -> np.ndarray:
    """Single process featurization for held-out data."""
    featurizer = get_featurizer(featurizer_name, size, radius)
    return featurizer.featurize(smiles_list)