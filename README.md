# Kindel AI: Prediction of affinity constants from DEL data

This repository contains the code for the Kindel AI project, which aims to predict affinity constants from DEL data. It uses extensively the [DeepChem](https://deepchem.io/) library to build the models and featurizers.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/insitro/kindel_ai.git
cd kindel_ai
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the dataset files from [42basepairs](https://42basepairs.com/browse/s3/kin-del-2024/data):
   - ddr1_1M.parquet
   - mapk14_1M.parquet

2. Place the downloaded files, ddr1_1M.parquet and mapk14_1M.parquet, in the `data` folder.

## Usage example

Run the main script with desired arguments:

```bash
python main.py --dataset ddr1_1M.parquet --featurizer morgan --model multitask_dnn --output_dir ./results
```

This will train a multitask DNN model using Morgan fingerprints and save the results in the `results` folder.

### Arguments
- `--dataset`: Dataset file name 
- `--featurizer`: Featurization method (default: morgan)
- `--model`: Model type (default: multitask_dnn)
- `--output_dir`: Output directory for results (default: ./results)

## DeepChem Models

1. **Multitask DNN**
   - Deep Neural Network for regression
   - Configurable layer sizes and dropout
   - Best for fingerprint-based features

2. **AttentiveFP**
   - Graph neural network with attention mechanism
   - Learns molecular representations directly from graphs
   - Suitable for complex structural patterns

3. **MPNN (Message Passing Neural Network)**
   - Graph-based model using message passing
   - Captures atomic interactions
   - Good for 3D structural information

4. **Graph Convolution**
   - Convolutional operations on molecular graphs
   - Learns local chemical environments
   - Effective for structural patterns

## DeepChem Featurizers

1. **Morgan Fingerprints** (`morgan`)
   - Circular fingerprints
   - Fast and lightweight

2. **Graph Convolution** (`graph`)
   - Converts molecules to graph representation
   - Includes atom and bond features
   - Used with Graph Convolution model

3. **AttentiveFP** (`attentivefp`)
   - Graph-based featurization
   - Includes edge features
   - Optimized for AttentiveFP model

4. **Weave** (`mpnn`)
   - Pair-wise atomic features
   - Includes atomic and pair-wise interactions
   - Designed for MPNN model

## Results
### Model perfomance

#### Multitask DNN results on Kindel Dataset using Morgan fingerprints for the DDR1 kinase dataset

Currently only results for the multitask regressor are presented due to the high computational cost of training the other models. For future work, the results for the other models could be presented. Nonetheless, they are presented here for future reference.

Here, it is reported the MSE (mean squared error) for the training, validation and test datasets, on the top left corner of each plot. This metric follows the definition of the [Kindel](https://github.com/insitro/kindel) repository and the associated [paper](https://arxiv.org/pdf/2410.08938.pdf).

The plots indicate the model-predicted enrichment values versus experimental Kd values (higher enrichment values indicate lower Kd values). It is also reported the Spearman and Pearson correlation coefficients for extended and in-library and on-DNA and off-DNA datasets.


Results are saved in the results folder for both binary and human-readable formats YAML formats, together with generated plots generated, and presented below:
- On-DNA predictions
- Off-DNA predictions
- Training/validation/test metrics


-----
\
*Held-out extended dataset*

![Training Results](/results/multitask_regression/ddr1_predictions_lib_all.png)
\
\
*Held-out in-library dataset predictions*

![Library Results](/results/multitask_regression/ddr1_predictions_extended.png)



- As expected, the multitask DNN model performs best on the on-DNA dataset with higher Spearman correlation coefficients than the off-DNA dataset. The Spearman coeficients, for on-DNA datasets, rivals the authors best results of around 0.7. The results worsen for the off-DNA dataset, with a Spearman coefficient of around 0.4, being worse than the original insitro implementation for all models tried. Bear in mind that the reported Spearman is negative because higher enrichment corresponds to higher affinity, hence lower Kd. Reported is the 1/Kd to aid visualization. Overall the model performed reasonably well with Spearman metrics rivalling the authors original implementation.

Take-home messages:


- Strong performance on on-DNA predictions (Spearman ρ ≈ -0.71)
- Moderate performance on in-library off-DNA predictions (Spearman ρ ≈ -0.50)
- Consistent MSE across train/valid/test splits indicating good generalization

For future work the model could be improved by using exploring different solutions:

- For the off-DNA dataset we observe an outlier (ca. 1.2), so removing these outliers could  improve the Spearman coeficient.
- One could also try to use different featurizers, like the ones refered above for the other models. We could also combine fingerprints with descriptors that are used to assess molecule druglikeness (e.g qed).
- Here I peformed random splits. Using disynthon splits could also improve the results.
- Experiment with deeper networks for more complex feature hierarchies, expecially with the disynthon splits.
- Check more thoroughly for over-fitting (inspite train, valid and test MSE's being similar), by adding regularization or e.g early-stopping.
- Error analysis was not performed due to computational constrainds and in future implementations it should be performed by e.g. using different random splits and calculating average MSE.
- Due to time and computational constraints only results for the DDR1 kinase dataset are presented. Future work could include also the MAPK14 kinase dataset.


## Alternative implementations

This was an implementation from scratch of the multitask DNN model on the KinDel dataset. Please check my other repository where I modified the insitro code available [here](https://github.com/insitro/kindel).

## Acknowledgments

Built on the [Kindel](https://github.com/insitro/kindel) dataset project by Insitro.




