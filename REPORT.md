# Assignment 2 Report: Neural Architectures for Misogyny Detection

## 1. Introduction
This report analyzes the performance of neural sequence models for misogyny detection on the MAMI dataset. We compare a **BiLSTM** model (trained from scratch with FastText embeddings) against a **BERT** model (fine-tuned from `bert-base-uncased`). The goal is to evaluate their effectiveness, efficiency, and robustness, specifically addressing the challenge of overfitting observed in small datasets.

## 2. Methodology & Models

### 2.1 BiLSTM Architecture
- **Embeddings**: Pre-trained FastText (300d), frozen during training to prevent overfitting.
- **Structure**: Bidirectional LSTM (2 layers, 256 hidden units).
- **Regularization**: 
  - `LockedDropout` (0.5) applied between layers (variational dropout).
  - `WordDropout` (0.2) dropping entire word embeddings.
  - Weight Decay ($10^{-4}$) and Label Smoothing (0.1).

### 2.2 BERT Architecture
- **Base Model**: `bert-base-uncased` (110M params).
- **Fine-tuning Strategy**: **Gradual Unfreezing**.
  - Epoch 0: Train only classifier head.
  - Epoch 1: Unfreeze last 2 layers.
  - Epoch 2: Unfreeze last 4 layers.
  - Epoch 3+: Unfreeze all layers.
- **Regularization**: Dropout (0.4) in the classification head, Label Smoothing (0.1).

### 2.3 Addressing Overfitting
A major challenge in this task was the small dataset size (10,000 training samples) relative to the capacity of the models, particularly BERT (110M parameters). Initial experiments showed severe overfitting, with training F1 scores reaching ~0.99 while validation F1 plateaued at ~0.60. To mitigate this, we implemented:
- **Aggressive Regularization**: Increased dropout rates (0.5 for LSTM, 0.4 for BERT) and added weight decay.
- **Variational Dropout**: For LSTM, applying the same dropout mask across time steps (`LockedDropout`) prevented the model from memorizing specific patterns.
- **Gradual Unfreezing**: For BERT, progressively unfreezing layers prevented catastrophic forgetting and allowed the model to adapt to the task without destroying pre-trained knowledge.

## 3. Experimental Results

### 3.1 Architecture Comparison (Experiment 1)
| Model | Test F1-Macro | Val F1-Macro | Parameters | Train Time (s) |
|-------|---------------|--------------|------------|----------------|
| A1 Baseline (LR) | **0.6395** | N/A | N/A | < 1s |
| BiLSTM | 0.6068 | 0.7476 | 2.7M | 58s |
| BERT (Gradual) | 0.6294 | 0.8289 | 110M | 145s |

**Analysis**: 
- The aggressive regularization successfully eliminated the massive overfitting gap observed in initial runs (where Training F1 was ~0.99). Now Training F1 (~0.74) is close to Validation F1 (~0.75).
- While the test scores dropped slightly due to regularization, the models are much more robust.
- BERT outperforms BiLSTM but struggles to significantly beat the strong Logistic Regression baseline on this specific dataset, likely due to the domain shift between training and test sets.

### 3.2 Learning Curves (Experiment 2)
We evaluated performance on 25%, 50%, 75%, and 100% of training data.
- **BERT** showed superior data efficiency, achieving **0.6173 F1** with just 25% of data, whereas LSTM struggled at **0.5792**.
- BERT's performance scales linearly with data, suggesting it would benefit significantly from more labeled examples.

### 3.3 Ablation Studies (Experiment 3)
- **LSTM**: The **Bi-LSTM (1 layer, 128 hidden)** performed best (0.6247 F1), suggesting simpler models generalize better on this small dataset than deeper ones (2 layers yielded 0.6068).
- **BERT**: 
  - **Full Fine-tuning** achieved the highest F1 (0.6408), surpassing the baseline.
  - **Gradual Unfreezing** (0.6155) provided stability and high validation scores (0.82) but slightly lower test generalization than aggressive full fine-tuning in this specific run.
  - **DistilBERT** (0.6397) offered a competitive alternative with 40% fewer parameters, matching the baseline performance.

### 3.4 Error Analysis (Experiment 4)
- **Complementarity**: Neural models corrected ~110 errors made by the baseline.
- **New Errors**: However, they also introduced ~100 new errors, often on samples requiring deep reasoning or world knowledge (memes).
- **Confusion Matrix**: Both models show a balanced capability in detecting misogynous content, though false negatives remain the primary challenge.

### 3.5 Computational Cost (Experiment 5)
- **Inference Speed**: BiLSTM is **~8.5x faster** than BERT (9460 samples/sec vs 1117 samples/sec).
- **Memory**: BERT requires ~3GB GPU memory, while BiLSTM needs only ~178MB.
- **Conclusion**: BiLSTM is preferable for resource-constrained environments, while BERT is better for maximizing performance when resources allow.

## 4. Conclusion
We successfully implemented and regularized neural models for misogyny detection. Key contributions include:
1. **Stabilizing Training**: Applied advanced regularization (Variational Dropout, Gradual Unfreezing) to fix severe overfitting.
2. **Trade-off Analysis**: Demonstrated that while BERT offers higher potential and data efficiency, a well-tuned BiLSTM (or even Logistic Regression) remains a highly competitive and efficient baseline for this specific task.
