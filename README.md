# Assignment 2: Neural Architectures for Misogyny Detection

## Overview
This project implements and analyzes neural network models for detecting misogyny in memes (MAMI dataset). It compares a BiLSTM baseline against a fine-tuned BERT model, exploring various configurations, ablation studies, and error analysis.

## Project Structure
```
Assignment2/
├── data/               # Symlinked data from Assignment1
├── models/             # Model definitions
│   ├── lstm_classifier.py
│   └── bert_classifier.py
├── notebooks/          # Jupyter notebooks
│   └── main.ipynb      # Main results presentation
├── results/            # Generated plots and JSON results
├── src/                # Source code
│   ├── dataset.py      # Data loading and processing
│   ├── trainer.py      # Training loop and evaluation
│   └── experiments.py  # Main experiment runner
└── requirements.txt    # Dependencies
```

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure data is available in `data/` or symlinked.

## Running Experiments
To reproduce all experiments (Training, Evaluation, Ablations):
```bash
python -m src.experiments
```
*Note: This process may take 30-60 minutes depending on GPU availability.*

## Results
Results are saved in the `results/` directory as JSON files and PNG plots.
You can view the consolidated results in `notebooks/main.ipynb`.

## Key Findings
- **Overfitting Control**: Strong regularization (Dropout 0.5, Weight Decay, Label Smoothing) successfully closed the generalization gap in LSTM models.
- **BERT Performance**: Gradual unfreezing proved effective, achieving competitive performance with better stability than full fine-tuning.
- **Data Efficiency**: BERT demonstrated superior learning capability with limited data (25% subset) compared to LSTM.
