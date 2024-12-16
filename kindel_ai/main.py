"""
Main script predicting Kd values on kindel dataset.
Handles data loading, model training, and evaluation.
"""

import os
import yaml
import argparse
import deepchem
import pandas as pd
import warnings

# Suppress specific DeepChem warning. Looks like a bug in DeepChem.
warnings.filterwarnings("ignore", category=UserWarning, message=".*please use MorganGenerator.*")

from src.featurizers.featurizers import single_process_featurization, parallel_featurization
from src.models.models import train_multitask_dnn, train_graph_conv, train_attentivefp, train_mpnn
from src.utils.metrics import evaluate_model_performance, convert_results
from src.config import DATA_ROOT
from src.utils.plots import plot_regression_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate molecular property prediction models')
    parser.add_argument('--dataset', type=str, default='ddr1_1M.csv',
                       help='Dataset file name')
    parser.add_argument('--featurizer', type=str, default='morgan',
                       help='Featurization method')
    parser.add_argument('--model', type=str, default='multitask_dnn',
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read parquet and convert to CSV
    print("Loading parquet data...")
    df = pd.read_parquet(os.path.join(DATA_ROOT, args.dataset))
    
    # Create temporary CSV for featurization
    csv_path = os.path.join(DATA_ROOT, args.dataset.replace('.parquet', '.csv'))
    df.to_csv(csv_path, index=False)
    
    # Use parallel featurization with disk dataset
    print("Featurizing molecules...")
    dataset = parallel_featurization(
        csv_path=csv_path,
        featurizer_name=args.featurizer,
        n_jobs=20
    )
    
    # Split data
    print("Splitting dataset...")
    splitter = deepchem.splits.RandomSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1
    )
    
    # Normalize data
    print("Normalizing data...")
    normalizer = deepchem.trans.NormalizationTransformer(
        transform_y=True,
        dataset=train,
        move_mean=True
    )
    train = normalizer.transform(train)
    valid = normalizer.transform(valid)
    test = normalizer.transform(test)
    
    print(f"Train size: {len(train.ids)}")
    print(f"Valid size: {len(valid.ids)}")
    print(f"Test size: {len(test.ids)}")
    
    # Train model
    print("Training model...")
    if args.model == 'multitask_dnn':
        model = train_multitask_dnn(
            train_data=train,
            valid_data=valid,
            test_data=test
        )
    elif args.model == 'attentivefp':
        model = train_attentivefp(
            train_data=train,
            valid_data=valid,
            test_data=test
        )
    elif args.model == 'mpnn':
        model = train_mpnn(
            train_data=train,
            valid_data=valid,
            test_data=test
        )
    elif args.model == 'graph_conv':
        model = train_graph_conv(
            train_data=train,
            valid_data=valid,
            test_data=test
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Calculate training metrics
    print("Calculating training metrics...")
    metric = deepchem.metrics.Metric(deepchem.metrics.mean_squared_error)
    train_scores = model.evaluate(train, [metric], [normalizer])
    valid_scores = model.evaluate(valid, [metric], [normalizer])
    test_scores = model.evaluate(test, [metric], [normalizer])

    print(f"Train Scores: {train_scores}")
    print(f"Validation Scores: {valid_scores}")
    print(f"Test Scores: {test_scores}")
    
    # Evaluate model
    print("Evaluating model...")
    results, predictions = evaluate_model_performance(
        model=model,
        train_data=train,
        valid_data=valid,
        test_data=test,
        normalizer=normalizer,
        featurizer_fn=lambda x: single_process_featurization(x, featurizer_name=args.featurizer),
        target="ddr1"
    )
    
    # Save results and generate plots
    print("Saving results and generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save YAML results
    output_file = os.path.join(args.output_dir, 
        f"results_{args.featurizer}_{args.model}_{args.dataset.split('.')[0]}.yml")
    converted_results = convert_results(results)
    with open(output_file, "w") as f:
        yaml.dump(converted_results, f, default_flow_style=False, sort_keys=False)

    # Generate both types of plots
    plot_regression_metrics(results=converted_results, 
                           predictions=predictions,
                           output_dir=args.output_dir,
                           model_name=f"{args.featurizer}_{args.model}",
                           dataset_type="extended")
    plot_regression_metrics(results=converted_results,
                           predictions=predictions,
                           output_dir=args.output_dir,
                           model_name=f"{args.featurizer}_{args.model}",
                           dataset_type="lib")
    
    print(f"Results saved to {output_file}")
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
