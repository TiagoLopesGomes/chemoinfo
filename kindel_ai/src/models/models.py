"""
Models module containing different model implementations.
Each model function returns predictions and evaluation metrics.
"""

import deepchem
from typing import Dict, Any, Tuple
import numpy as np
import torch

def get_device():
    """Get available device (GPU or CPU)."""
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    print("CUDA not available, using CPU")
    return 'cpu'

def train_multitask_dnn(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    n_features: int = 1024,
    layer_sizes: list = [1000, 500],
    dropout: float = 0.25,
    learning_rate: float = 0.001,
    epochs: int = 5
) -> deepchem.models.MultitaskRegressor:
    """Train a Multitask Deep Neural Network model."""
    model = deepchem.models.MultitaskRegressor(
        n_tasks=1,
        n_features=n_features,
        layer_sizes=layer_sizes,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Train model
    model.fit(train_data, nb_epoch=epochs)
    return model

def train_graph_conv(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    batch_size: int = 50,
    batch_normalize: bool = False,
    learning_rate: float = 0.001,
    epochs: int = 5
) -> deepchem.models.GraphConvModel:
    """Train a Graph Convolution model."""
    model = deepchem.models.GraphConvModel(
        n_tasks=1,
        batch_size=batch_size,
        batch_normalize=batch_normalize,
        mode="regression",
        learning_rate=learning_rate
    )
    
    # Train model
    model.fit(train_data, nb_epoch=epochs)
    return model

def train_attentivefp(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    n_features: int = 30,
    n_tasks: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 5
) -> deepchem.models.AttentiveFPModel:
    """Train an Attentive FP model."""
    device = get_device()
    model = deepchem.models.AttentiveFPModel(
        n_tasks=n_tasks,
        num_layers=3,
        num_timesteps=2,
        graph_feat_size=200,
        mode='regression',
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train model with validation monitoring
    for epoch in range(epochs):
        model.fit(train_data, nb_epoch=1)
        print(f"Epoch {epoch+1}/{epochs}")
    
    return model

def train_mpnn(
    train_data: Any,
    valid_data: Any,
    test_data: Any,
    n_tasks: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 5
) -> deepchem.models.MPNNModel:
    """Train a Message Passing Neural Network model."""
    device = get_device()
    print(f"\nInitializing MPNN model on {device}")
    
    model = deepchem.models.MPNNModel(
        n_tasks=n_tasks,
        n_atom_feat=30,
        n_pair_feat=11,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    # Train model with validation monitoring
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.fit(train_data, nb_epoch=1)
        # Add validation metrics here if needed
    
    return model
