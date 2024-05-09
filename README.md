# DataSP Algorithm Experiments Repository

This repository hosts a series of experiments designed to test the DataSP algorithm, as proposed in the paper [DataSP: A Differential All-to-All Shortest Path Algorithm for Learning Costs and Predicting Paths with Context](https://arxiv.org/abs/2405.04923). The DataSP algorithm represents a differentiable adaptation of the Floyd-Warshall all-to-all Shortest Path algorithm, tailored specifically for path and destination prediction tasks. Below you will find detailed descriptions and instructions for each experiment.

## Experiments Overview

The repository contains four main experiments:

1. **Synthetic Graph Experiment**
2. **Warcraft Map Experiment: Single Source to Target Node**
3. **Warcraft Map Experiment: Multiple Sources to Multiple Targets**
4. **Cabspotting Taxi Dataset Experiment**

## 1. Synthetic Graph Experiment

- **File to Run**: `exp_synthetic_graph.py`
- **Data**: Data is dynamically generated within the main script.
- **Instructions**: Ensure you maintain the same arguments in `argparse` to replicate results as reported in the associated paper. Users are encouraged to modify graph sparsity and node count to explore different configurations. Noise can also be added to the graph's actual edges, which remain hidden during training.

## 2. Warcraft Maps: Single Source-Target

- **File to Run**: `exp_warcraft_1to1.py`
- **Data**: Utilizes data from the Warcraft map editor, as referenced in [PAPER 1](https://arxiv.org/abs/1912.02175) and [PAPER 2](https://arxiv.org/abs/2002.08676). We use 144x144 images and 18x18 grid map layers. Please paste the data files to `./data_warcraft/18x18`.
- **Instructions**: DataSP may require additional time to achieve accuracy levels above 95% due to its inefficiencies in single source-target configurations. Keep the same `argparse` settings for consistent results with those reported.

## 3. Warcraft Maps: Many Source-Targets

- **File to Run**: `exp_warcraft_MtoM.py`
- **Data**: Similar to the single source-target setup but includes preprocessing to establish multiple source-target pairs. Only the first 350 images are used for training, not 10,000. Data should be placed in `./data_warcraft/18x18`.
- **Instructions**: This experiment allows for the addition or removal of noise via the `suboptimals` argument during the preprocessing step. Results are aligned with those presented in the initial experiment.

## 4. Cabspotting Taxi Dataset

- **File to Run**: `exp_cabspotting.py`
- **Data**: The dataset is available at [Crawdad-EPFLMobility](https://ieee-dataport.org/open-access/crawdad-epflmobility). Download and extract text files to `./data_cabspotting/data_raw/`. Follow the preprocessing steps outlined in notebooks with suffix `P1`, `P2`, and `P3` located in `./cabspotting_preprocessing/`.
- **Instructions**: Analyses and reports are based on the first 1000 trip indices from the test data.

