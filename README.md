# A Margin-Maximizing Fine-Grained Ensemble Method

This is the official code repository for the paper "[A Margin-Maximizing Fine-Grained Ensemble Method](https://arxiv.org/abs/2409.12849)".

## Repository Contents

In addition to the implementation code, this repository includes:
- PDF version of the paper
- Datasets used in our experiments
- Detailed requirements specification

## Reproducibility Note

To ensure reproducibility, we have set fixed random seeds in our code. If you use the same versions of Python packages as specified in `requirements.txt`, you should be able to reproduce the results reported in the paper. However, please note that different versions of dependencies may lead to slightly different results.

## Code Verification

It's worth noting that after the paper's publication, co-author Renwei Luo conducted a comprehensive review of the research content and restructured the code. He verified the accuracy of the code and the resulting data multiple times to ensure the reliability and consistency of the results.

## Key Features

- A learnable confidence matrix Θ for fine-grained optimization of base learners across categories
- An innovative margin-based loss function using logsumexp techniques
- Comprehensive experimental and theoretical analysis

## Repository Structure

This repository contains:

- `code.py`: Main implementation of our proposed method
- `A Margin-Maximizing Fine-Grained.pdf`: Full paper in PDF format
- `requirements.txt`: List of project dependencies
- `data/`: Directory containing datasets used in experiments
- `results.csv`: CSV file with experimental results

## Datasets

The `data/` directory contains the following dataset files:

- BASEHOCK.mat
- breast_uni.mat
- chess.mat
- iris.mat
- jaffe.mat
- pathbased.mat
- RELATHE.mat
- wine.mat

## Installation

1. Clone this repository:
```
git clone https://github.com/HaooChenn/A-Margin-Maximizing-Fine-Grained-Ensemble-Method
```
2. Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

To run the experiments:
```
python code.py
```
This will train and evaluate our model along with baseline models on various datasets and save results to `results.csv`.

## Results

Our method outperforms traditional random forests using only one-tenth of the base learners and other state-of-the-art ensemble methods. Detailed results can be found in the paper and `results.csv`.

## Citation

If you find this work useful in your research, please consider citing:
```
@misc{yuan2024marginmaximizingfinegrainedensemblemethod,
      title={A Margin-Maximizing Fine-Grained Ensemble Method}, 
      author={Jinghui Yuan and Hao Chen and Renwei Luo and Feiping Nie},
      year={2024},
      eprint={2409.12849},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.12849}, 
}
```
