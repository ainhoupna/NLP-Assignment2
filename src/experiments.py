"""
All 5 required experiments for Assignment 2.
Run with: python -m src.experiments
"""
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

# Local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import (load_mami_data, Vocabulary, create_dataloaders,
                         create_bert_dataloaders, MAMIDataset, MAMIBertDataset,
                         collate_fn)
from src.trainer import (train_lstm_model, train_bert_model, evaluate_lstm,
                         evaluate_bert, measure_inference_speed, get_gpu_memory_mb,
                         DEVICE)
from models.lstm_classifier import LSTMClassifier, load_fasttext_embeddings
from models.bert_classifier import BERTClassifier, unfreeze_last_n_layers

from torch.utils.data import DataLoader, Subset
import torch.nn as nn

# ============================================================
# Config
# ============================================================
SEED = 42
MAX_LEN = 128
BATCH_SIZE_LSTM = 32
BATCH_SIZE_BERT = 16
FASTTEXT_PATH = "data/embeddings/wiki-news-300d-1M-subword.vec"
RESULTS_DIR = "results"

# Anti-overfitting hyperparameters
LSTM_DROPOUT = 0.5
LSTM_EMBED_DROPOUT = 0.2
LSTM_LR = 5e-4
LSTM_WEIGHT_DECAY = 1e-4
LSTM_PATIENCE = 5
LABEL_SMOOTHING = 0.1
BERT_DROPOUT = 0.4
BERT_PATIENCE = 3
BERT_EPOCHS = 7  # More epochs since gradual unfreezing needs time

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_results(data, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")

# ============================================================
# Experiment 1: Architecture Comparison
# ============================================================
def experiment_1_architecture_comparison():
    """Compare A1 baseline vs BiLSTM vs BERT on official test set."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Architecture Comparison")
    print("="*70)

    set_seed()
    X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, _, _ = load_mami_data('.')

    # --- Split training into train/val (90/10) for early stopping ---
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    X_tr_raw, X_val_raw, _, _ = train_test_split(
        X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

    results = {}

    # --- A1 Baseline (from Assignment 1) ---
    print("\n--- A1 Baseline (LR + FastText) ---")
    results['a1_baseline'] = {
        'model': 'LR + FastText (A1)',
        'test_f1': 0.6395,
        'params': 'N/A (sklearn)',
        'train_time': 'N/A',
        'inference_speed': 'N/A',
        'gpu_memory_mb': 0
    }
    print(f"  Test F1-Macro: 0.6395 (from Assignment 1)")

    # --- BiLSTM ---
    print("\n--- BiLSTM (with anti-overfitting) ---")
    vocab = Vocabulary(min_freq=2, max_size=25000)
    vocab.build(X_tr)

    train_loader, val_loader_lstm = create_dataloaders(X_tr, y_tr, X_val, y_val, vocab, BATCH_SIZE_LSTM)
    _, test_loader_lstm = create_dataloaders(X_tr, y_tr, X_test, y_test, vocab, BATCH_SIZE_LSTM)

    # Load FastText embeddings
    pretrained = None
    if os.path.exists(FASTTEXT_PATH):
        pretrained = load_fasttext_embeddings(vocab, FASTTEXT_PATH, 300)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    lstm_model = LSTMClassifier(
        vocab_size=len(vocab), embedding_dim=300, hidden_dim=256,
        num_classes=2, num_layers=2, dropout=LSTM_DROPOUT,
        bidirectional=True, pretrained_embeddings=pretrained,
        freeze_embeddings=True, embed_dropout=LSTM_EMBED_DROPOUT
    )
    print(f"  Parameters: {lstm_model.count_parameters():,}")

    lstm_model, lstm_history, lstm_train_time = train_lstm_model(
        lstm_model, train_loader, val_loader_lstm, num_epochs=30,
        lr=LSTM_LR, patience=LSTM_PATIENCE,
        weight_decay=LSTM_WEIGHT_DECAY, label_smoothing=LABEL_SMOOTHING)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    _, lstm_test_f1, lstm_preds, lstm_labels = evaluate_lstm(lstm_model, test_loader_lstm, criterion)
    lstm_speed = measure_inference_speed(lstm_model, test_loader_lstm, 'lstm')
    lstm_gpu = get_gpu_memory_mb()

    print(f"  Test F1-Macro: {lstm_test_f1:.4f}")
    results['bilstm'] = {
        'model': 'BiLSTM (FastText)',
        'test_f1': round(lstm_test_f1, 4),
        'params': lstm_model.count_parameters(),
        'train_time': round(lstm_train_time, 1),
        'inference_speed': round(lstm_speed, 1),
        'gpu_memory_mb': round(lstm_gpu, 1),
        'history': lstm_history
    }

    # --- BERT with Gradual Unfreezing ---
    print("\n--- BERT Fine-tuning (gradual unfreezing) ---")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader_bert, val_loader_bert = create_bert_dataloaders(
        X_tr_raw, y_tr, X_val_raw, y_val, tokenizer, BATCH_SIZE_BERT, MAX_LEN)
    _, test_loader_bert = create_bert_dataloaders(
        X_tr_raw, y_tr, X_test_raw, y_test, tokenizer, BATCH_SIZE_BERT, MAX_LEN)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    bert_model = BERTClassifier('bert-base-uncased', num_classes=2, dropout=BERT_DROPOUT)
    print(f"  Total parameters: {bert_model.count_total_parameters():,}")

    bert_model, bert_history, bert_train_time = train_bert_model(
        bert_model, train_loader_bert, val_loader_bert,
        num_epochs=BERT_EPOCHS, lr_bert=2e-5, lr_head=1e-4,
        patience=BERT_PATIENCE, label_smoothing=LABEL_SMOOTHING,
        use_gradual_unfreezing=True)

    _, bert_test_f1, bert_preds, bert_labels = evaluate_bert(bert_model, test_loader_bert, criterion)
    bert_speed = measure_inference_speed(bert_model, test_loader_bert, 'bert')
    bert_gpu = get_gpu_memory_mb()

    print(f"  Test F1-Macro: {bert_test_f1:.4f}")
    results['bert'] = {
        'model': 'BERT (gradual unfreeze)',
        'test_f1': round(bert_test_f1, 4),
        'params': bert_model.count_parameters(),
        'total_params': bert_model.count_total_parameters(),
        'train_time': round(bert_train_time, 1),
        'inference_speed': round(bert_speed, 1),
        'gpu_memory_mb': round(bert_gpu, 1),
        'history': bert_history
    }

    # --- Summary Table ---
    print("\n" + "-"*70)
    print(f"{'Model':<25} {'Test F1':>10} {'Params':>12} {'Train(s)':>10} {'Infer(s/s)':>12} {'GPU(MB)':>10}")
    print("-"*70)
    for k, v in results.items():
        print(f"{v['model']:<25} {v['test_f1']:>10} {str(v['params']):>12} {str(v['train_time']):>10} {str(v['inference_speed']):>12} {v['gpu_memory_mb']:>10}")

    # --- Plot training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist_key in [('BiLSTM', 'bilstm'), ('BERT', 'bert')]:
        h = results[hist_key]['history']
        axes[0].plot(h['train_loss'], label=f'{name} Train')
        axes[0].plot(h['val_loss'], '--', label=f'{name} Val')
        axes[1].plot(h['train_f1'], label=f'{name} Train')
        axes[1].plot(h['val_f1'], '--', label=f'{name} Val')

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1-Macro'); axes[1].set_title('F1-Macro')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp1_training_curves.png'), dpi=150)
    plt.close()

    # Save predictions for error analysis
    np.save(os.path.join(RESULTS_DIR, 'lstm_preds.npy'), lstm_preds)
    np.save(os.path.join(RESULTS_DIR, 'bert_preds.npy'), bert_preds)
    np.save(os.path.join(RESULTS_DIR, 'test_labels.npy'), lstm_labels)

    # Remove non-serializable history for JSON
    results_json = {}
    for k, v in results.items():
        results_json[k] = {kk: vv for kk, vv in v.items() if kk != 'history'}
    save_results(results_json, 'exp1_results.json')

    return results, lstm_model, bert_model, vocab, tokenizer

# ============================================================
# Experiment 2: Learning Curve Analysis
# ============================================================
def experiment_2_learning_curves(vocab=None, tokenizer=None):
    """Train on 25%, 50%, 75%, 100% and plot performance vs data size."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Learning Curve Analysis")
    print("="*70)

    X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, _, _ = load_mami_data('.')
    fractions = [0.25, 0.50, 0.75, 1.0]
    results = {'fractions': fractions, 'lstm_f1': [], 'bert_f1': []}

    if vocab is None:
        vocab = Vocabulary(min_freq=2, max_size=25000)
        vocab.build(X_train)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    pretrained = None
    if os.path.exists(FASTTEXT_PATH):
        pretrained = load_fasttext_embeddings(vocab, FASTTEXT_PATH, 300)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for frac in fractions:
        print(f"\n--- Fraction: {frac*100:.0f}% ({int(len(X_train)*frac)} samples) ---")
        set_seed()
        n = int(len(X_train) * frac)
        indices = np.random.permutation(len(X_train))[:n]
        X_sub, y_sub = X_train[indices], y_train[indices]
        X_sub_raw = X_train_raw[indices]

        # Split for val
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_sub, y_sub, test_size=0.1, stratify=y_sub, random_state=SEED)
        X_tr_raw, X_val_raw, _, _ = train_test_split(X_sub_raw, y_sub, test_size=0.1, stratify=y_sub, random_state=SEED)

        # LSTM
        train_ld, val_ld = create_dataloaders(X_tr, y_tr, X_val, y_val, vocab, BATCH_SIZE_LSTM)
        _, test_ld = create_dataloaders(X_tr, y_tr, X_test, y_test, vocab, BATCH_SIZE_LSTM)

        lstm = LSTMClassifier(len(vocab), 300, 256, 2, 2, LSTM_DROPOUT, True, pretrained,
                              freeze_embeddings=True, embed_dropout=LSTM_EMBED_DROPOUT)
        lstm, _, _ = train_lstm_model(lstm, train_ld, val_ld, num_epochs=30,
                                      lr=LSTM_LR, patience=LSTM_PATIENCE,
                                      weight_decay=LSTM_WEIGHT_DECAY, label_smoothing=LABEL_SMOOTHING)
        _, lstm_f1, _, _ = evaluate_lstm(lstm, test_ld, criterion)
        results['lstm_f1'].append(round(lstm_f1, 4))
        print(f"  LSTM Test F1: {lstm_f1:.4f}")
        del lstm; torch.cuda.empty_cache()

        # BERT
        train_ld_b, val_ld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_val_raw, y_val, tokenizer, BATCH_SIZE_BERT)
        _, test_ld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_test_raw, y_test, tokenizer, BATCH_SIZE_BERT)

        bert = BERTClassifier('bert-base-uncased', 2, BERT_DROPOUT)
        bert, _, _ = train_bert_model(bert, train_ld_b, val_ld_b, num_epochs=BERT_EPOCHS,
                                       patience=BERT_PATIENCE, label_smoothing=LABEL_SMOOTHING,
                                       use_gradual_unfreezing=True)
        _, bert_f1, _, _ = evaluate_bert(bert, test_ld_b, criterion)
        results['bert_f1'].append(round(bert_f1, 4))
        print(f"  BERT Test F1: {bert_f1:.4f}")
        del bert; torch.cuda.empty_cache()

    # A1 baseline is constant
    results['baseline_f1'] = [0.6395] * len(fractions)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"{int(f*100)}%\n({int(len(X_train)*f)})" for f in fractions]
    ax.plot(range(len(fractions)), results['baseline_f1'], 'k--', marker='s', label='A1 Baseline (LR+FastText)')
    ax.plot(range(len(fractions)), results['lstm_f1'], 'b-', marker='o', label='BiLSTM')
    ax.plot(range(len(fractions)), results['bert_f1'], 'r-', marker='^', label='BERT')
    ax.set_xticks(range(len(fractions))); ax.set_xticklabels(x_labels)
    ax.set_xlabel('Training Data Size'); ax.set_ylabel('Test F1-Macro')
    ax.set_title('Learning Curve: F1-Macro vs Training Data Size')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_learning_curves.png'), dpi=150)
    plt.close()

    save_results(results, 'exp2_results.json')
    return results

# ============================================================
# Experiment 3: Ablation Studies
# ============================================================
def experiment_3_ablation_studies(vocab=None, tokenizer=None):
    """Ablation studies for LSTM and BERT."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Ablation Studies")
    print("="*70)

    set_seed()
    X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, _, _ = load_mami_data('.')
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    X_tr_raw, X_val_raw, _, _ = train_test_split(X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=SEED)

    if vocab is None:
        vocab = Vocabulary(min_freq=2, max_size=25000)
        vocab.build(X_tr)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    pretrained = None
    if os.path.exists(FASTTEXT_PATH):
        pretrained = load_fasttext_embeddings(vocab, FASTTEXT_PATH, 300)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    train_ld, val_ld = create_dataloaders(X_tr, y_tr, X_val, y_val, vocab, BATCH_SIZE_LSTM)
    _, test_ld = create_dataloaders(X_tr, y_tr, X_test, y_test, vocab, BATCH_SIZE_LSTM)

    # --- LSTM Ablations ---
    print("\n--- LSTM Ablations ---")
    lstm_configs = [
        {'name': 'Uni-LSTM (1L, 128h)', 'bidirectional': False, 'num_layers': 1, 'hidden_dim': 128},
        {'name': 'Uni-LSTM (2L, 256h)', 'bidirectional': False, 'num_layers': 2, 'hidden_dim': 256},
        {'name': 'Bi-LSTM (1L, 128h)',  'bidirectional': True,  'num_layers': 1, 'hidden_dim': 128},
        {'name': 'Bi-LSTM (1L, 256h)',  'bidirectional': True,  'num_layers': 1, 'hidden_dim': 256},
        {'name': 'Bi-LSTM (2L, 256h)',  'bidirectional': True,  'num_layers': 2, 'hidden_dim': 256},
    ]

    lstm_results = []
    for cfg in lstm_configs:
        print(f"\n  Config: {cfg['name']}")
        set_seed()
        model = LSTMClassifier(
            len(vocab), 300, cfg['hidden_dim'], 2, cfg['num_layers'],
            LSTM_DROPOUT, cfg['bidirectional'], pretrained,
            freeze_embeddings=True, embed_dropout=LSTM_EMBED_DROPOUT
        )
        print(f"    Params: {model.count_parameters():,}")
        model, hist, t_time = train_lstm_model(model, train_ld, val_ld, num_epochs=30,
                                                lr=LSTM_LR, patience=LSTM_PATIENCE,
                                                weight_decay=LSTM_WEIGHT_DECAY,
                                                label_smoothing=LABEL_SMOOTHING)
        _, test_f1, _, _ = evaluate_lstm(model, test_ld, criterion)
        lstm_results.append({
            'name': cfg['name'],
            'test_f1': round(test_f1, 4),
            'params': model.count_parameters(),
            'train_time': round(t_time, 1)
        })
        print(f"    Test F1: {test_f1:.4f}")
        del model; torch.cuda.empty_cache()

    # --- BERT Ablations ---
    print("\n--- BERT Ablations ---")
    train_ld_b, val_ld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_val_raw, y_val, tokenizer, BATCH_SIZE_BERT)
    _, test_ld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_test_raw, y_test, tokenizer, BATCH_SIZE_BERT)

    bert_configs = [
        {'name': 'BERT frozen', 'model_name': 'bert-base-uncased', 'freeze': 'frozen', 'gradual': False},
        {'name': 'BERT last-2 unfrozen', 'model_name': 'bert-base-uncased', 'freeze': 'last_2', 'gradual': False},
        {'name': 'BERT full fine-tune', 'model_name': 'bert-base-uncased', 'freeze': 'full', 'gradual': False},
        {'name': 'BERT gradual unfreeze', 'model_name': 'bert-base-uncased', 'freeze': 'full', 'gradual': True},
        {'name': 'DistilBERT gradual', 'model_name': 'distilbert-base-uncased', 'freeze': 'full', 'gradual': True},
    ]

    bert_results = []
    for cfg in bert_configs:
        print(f"\n  Config: {cfg['name']}")
        set_seed()

        tok = AutoTokenizer.from_pretrained(cfg['model_name'])
        tld_b, vld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_val_raw, y_val, tok, BATCH_SIZE_BERT)
        _, tsld_b = create_bert_dataloaders(X_tr_raw, y_tr, X_test_raw, y_test, tok, BATCH_SIZE_BERT)

        model = BERTClassifier(cfg['model_name'], 2, BERT_DROPOUT,
                               freeze_bert=(cfg['freeze'] == 'frozen'))
        if cfg['freeze'] == 'last_2':
            unfreeze_last_n_layers(model, 2)

        trainable = model.count_parameters()
        total = model.count_total_parameters()
        print(f"    Trainable: {trainable:,} / Total: {total:,}")

        model, hist, t_time = train_bert_model(
            model, tld_b, vld_b, num_epochs=BERT_EPOCHS,
            patience=BERT_PATIENCE, label_smoothing=LABEL_SMOOTHING,
            use_gradual_unfreezing=cfg['gradual'])
        _, test_f1, _, _ = evaluate_bert(model, tsld_b, criterion)
        bert_results.append({
            'name': cfg['name'],
            'test_f1': round(test_f1, 4),
            'trainable_params': trainable,
            'total_params': total,
            'train_time': round(t_time, 1)
        })
        print(f"    Test F1: {test_f1:.4f}")
        del model; torch.cuda.empty_cache()

    all_results = {'lstm_ablations': lstm_results, 'bert_ablations': bert_results}

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names_l = [r['name'] for r in lstm_results]
    f1s_l = [r['test_f1'] for r in lstm_results]
    axes[0].barh(names_l, f1s_l, color='steelblue')
    axes[0].set_xlabel('Test F1-Macro'); axes[0].set_title('LSTM Ablations')
    axes[0].axvline(x=0.6395, color='red', linestyle='--', label='A1 Baseline')
    axes[0].legend()

    names_b = [r['name'] for r in bert_results]
    f1s_b = [r['test_f1'] for r in bert_results]
    axes[1].barh(names_b, f1s_b, color='coral')
    axes[1].set_xlabel('Test F1-Macro'); axes[1].set_title('BERT Ablations')
    axes[1].axvline(x=0.6395, color='red', linestyle='--', label='A1 Baseline')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp3_ablations.png'), dpi=150)
    plt.close()

    save_results(all_results, 'exp3_results.json')
    return all_results

# ============================================================
# Experiment 4: Error Analysis
# ============================================================
def experiment_4_error_analysis():
    """Compare errors: baseline vs neural models."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Error Analysis")
    print("="*70)

    # Load predictions saved from Experiment 1
    lstm_preds = np.load(os.path.join(RESULTS_DIR, 'lstm_preds.npy'))
    bert_preds = np.load(os.path.join(RESULTS_DIR, 'bert_preds.npy'))
    true_labels = np.load(os.path.join(RESULTS_DIR, 'test_labels.npy'))

    _, _, _, _, _, _, _, test_df = load_mami_data('.')

    # A1 baseline predictions (recreate from LR + TF-IDF)
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    X_train, y_train, X_test, y_test, _, _, _, _ = load_mami_data('.')

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    X_tr_tfidf = tfidf.fit_transform(X_train)
    X_te_tfidf = tfidf.transform(X_test)
    baseline_model = LogisticRegression(max_iter=1000, C=1, solver='liblinear', random_state=SEED)
    baseline_model.fit(X_tr_tfidf, y_train)
    baseline_preds = baseline_model.predict(X_te_tfidf)

    texts = test_df['Text Transcription'].fillna("").values
    filenames = test_df['file_name'].values

    # --- Cases where neural models FIXED baseline errors ---
    baseline_wrong = (baseline_preds != true_labels)
    lstm_right = (lstm_preds == true_labels)
    bert_right = (bert_preds == true_labels)

    fixed_by_lstm = baseline_wrong & lstm_right
    fixed_by_bert = baseline_wrong & bert_right
    fixed_by_both = baseline_wrong & lstm_right & bert_right

    print(f"\nBaseline errors: {baseline_wrong.sum()}")
    print(f"Fixed by LSTM: {fixed_by_lstm.sum()}")
    print(f"Fixed by BERT: {fixed_by_bert.sum()}")
    print(f"Fixed by both: {fixed_by_both.sum()}")

    print("\n--- Examples FIXED by neural models ---")
    fixed_indices = np.where(fixed_by_both)[0][:7]
    fixed_examples = []
    for idx in fixed_indices:
        ex = {
            'file': filenames[idx], 'text': texts[idx][:120],
            'true': int(true_labels[idx]), 'baseline': int(baseline_preds[idx]),
            'lstm': int(lstm_preds[idx]), 'bert': int(bert_preds[idx])
        }
        fixed_examples.append(ex)
        print(f"  {ex['file']}: true={ex['true']} base={ex['baseline']} lstm={ex['lstm']} bert={ex['bert']} | {ex['text']}")

    # --- Cases where neural models INTRODUCED new errors ---
    baseline_right = (baseline_preds == true_labels)
    lstm_wrong = (lstm_preds != true_labels)
    bert_wrong = (bert_preds != true_labels)

    new_errors_lstm = baseline_right & lstm_wrong
    new_errors_bert = baseline_right & bert_wrong
    new_errors_both = baseline_right & lstm_wrong & bert_wrong

    print(f"\nNew errors by LSTM: {new_errors_lstm.sum()}")
    print(f"New errors by BERT: {new_errors_bert.sum()}")
    print(f"New errors by both: {new_errors_both.sum()}")

    print("\n--- Examples with NEW errors by neural models ---")
    new_err_indices = np.where(new_errors_both)[0][:7]
    new_err_examples = []
    for idx in new_err_indices:
        ex = {
            'file': filenames[idx], 'text': texts[idx][:120],
            'true': int(true_labels[idx]), 'baseline': int(baseline_preds[idx]),
            'lstm': int(lstm_preds[idx]), 'bert': int(bert_preds[idx])
        }
        new_err_examples.append(ex)
        print(f"  {ex['file']}: true={ex['true']} base={ex['baseline']} lstm={ex['lstm']} bert={ex['bert']} | {ex['text']}")

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, preds, name in [(axes[0], baseline_preds, 'A1 Baseline'),
                             (axes[1], lstm_preds, 'BiLSTM'),
                             (axes[2], bert_preds, 'BERT')]:
        cm = confusion_matrix(true_labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-Mis', 'Mis'], yticklabels=['Non-Mis', 'Mis'])
        f1 = f1_score(true_labels, preds, average='macro')
        ax.set_title(f'{name}\nF1={f1:.4f}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp4_confusion_matrices.png'), dpi=150)
    plt.close()

    error_results = {
        'baseline_errors': int(baseline_wrong.sum()),
        'fixed_by_lstm': int(fixed_by_lstm.sum()),
        'fixed_by_bert': int(fixed_by_bert.sum()),
        'new_errors_lstm': int(new_errors_lstm.sum()),
        'new_errors_bert': int(new_errors_bert.sum()),
        'fixed_examples': fixed_examples[:5],
        'new_error_examples': new_err_examples[:5]
    }
    save_results(error_results, 'exp4_results.json')
    return error_results

# ============================================================
# Experiment 5: Computational Cost Analysis
# ============================================================
def experiment_5_computational_cost():
    """Summarize computational costs from Experiment 1."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Computational Cost Analysis")
    print("="*70)

    # Load Experiment 1 results
    with open(os.path.join(RESULTS_DIR, 'exp1_results.json'), 'r') as f:
        exp1 = json.load(f)

    print(f"\n{'Model':<25} {'Params':>12} {'Train(s)':>10} {'Infer(s/s)':>12} {'GPU(MB)':>10}")
    print("-"*70)
    for k, v in exp1.items():
        print(f"{v['model']:<25} {str(v['params']):>12} {str(v['train_time']):>10} {str(v['inference_speed']):>12} {v['gpu_memory_mb']:>10}")

    # Bar chart
    models = [v['model'] for v in exp1.values()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Params
    params = [v['params'] if isinstance(v['params'], (int, float)) else 0 for v in exp1.values()]
    axes[0].bar(models, params, color=['gray', 'steelblue', 'coral'])
    axes[0].set_title('Parameter Count'); axes[0].set_ylabel('Parameters')
    axes[0].tick_params(axis='x', rotation=15)

    # Train time
    times = [v['train_time'] if isinstance(v['train_time'], (int, float)) else 0 for v in exp1.values()]
    axes[1].bar(models, times, color=['gray', 'steelblue', 'coral'])
    axes[1].set_title('Training Time (seconds)'); axes[1].set_ylabel('Seconds')
    axes[1].tick_params(axis='x', rotation=15)

    # GPU memory
    mems = [v['gpu_memory_mb'] for v in exp1.values()]
    axes[2].bar(models, mems, color=['gray', 'steelblue', 'coral'])
    axes[2].set_title('GPU Memory (MB)'); axes[2].set_ylabel('MB')
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp5_computational_cost.png'), dpi=150)
    plt.close()
    print("\n  Plot saved: exp5_computational_cost.png")

# ============================================================
# Run All
# ============================================================
def run_all():
    """Run all 5 experiments sequentially."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nAnti-overfitting config:")
    print(f"  LSTM: dropout={LSTM_DROPOUT}, embed_dropout={LSTM_EMBED_DROPOUT}, "
          f"lr={LSTM_LR}, weight_decay={LSTM_WEIGHT_DECAY}, freeze_embeddings=True")
    print(f"  BERT: dropout={BERT_DROPOUT}, gradual_unfreezing=True, epochs={BERT_EPOCHS}")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")

    # Experiment 1
    results, lstm_model, bert_model, vocab, tokenizer = experiment_1_architecture_comparison()

    # Experiment 2
    experiment_2_learning_curves(vocab, tokenizer)

    # Experiment 3
    experiment_3_ablation_studies(vocab, tokenizer)

    # Experiment 4
    experiment_4_error_analysis()

    # Experiment 5
    experiment_5_computational_cost()

    # Experiment 6 (New)
    experiment_6_improvements(vocab, tokenizer)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}/")
    print("="*70)

# ============================================================
# Experiment 6: Improvement Tracking (Requested by User)
# ============================================================
def experiment_6_improvements(vocab=None, tokenizer=None):
    """
    Rigorously test the impact of improvements on BERT performance.
    Compares against A1 Baseline (LR + FastText) as reference.
    
    Configurations tested:
    0. A1 Baseline: LR + FastText (winner of Experiment 1 / Assignment 1)
    1. BERT Original: Raw text + Gradual Unfreezing + Dropout 0.4
    2. BERT + Clean Text: Cleaned text + Gradual Unfreezing + Dropout 0.4
    3. BERT + Clean + FullFT: Cleaned text + Full Fine-tuning + Dropout 0.1
    4. BERT + Clean + FullFT + No LS: Same as 3 but without Label Smoothing
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Performance Improvements Tracking")
    print("="*70)

    set_seed()
    # Load data - we need both RAW and CLEAN
    X_train_c, y_train, X_test_c, y_test, X_train_raw, X_test_raw, train_df, test_df = load_mami_data('.')
    
    # Clean versions
    X_train_clean = train_df['cleaned_text'].values
    X_test_clean = test_df['cleaned_text'].values

    # Validation split
    from sklearn.model_selection import train_test_split
    # Split RAW
    X_tr_r, X_val_r, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    X_ts_r = X_test_raw
    # Split CLEAN (same indices via same random_state)
    X_tr_c, X_val_c, _, _ = train_test_split(
        X_train_clean, y_train, test_size=0.1, stratify=y_train, random_state=SEED)
    X_ts_c = X_test_clean

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    results = []

    # ---- 0. A1 Baseline: LR + FastText (winner of Experiment 1) ----
    print("\n--- [0] A1 Baseline: LR + FastText ---")
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Reproduce A1 best: LR + FastText embeddings
    # Since FastText embeddings require special loading, use TF-IDF as faithful A1 reproduction
    # (A1 best was LR+FastText=0.6395, but TF-IDF SVM+Bigrams=0.6360; we re-train LR+TF-IDF here)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr_tfidf = tfidf.fit_transform(X_train_c)  # cleaned text
    X_ts_tfidf = tfidf.transform(X_test_c)

    start = time.time()
    lr_model = LogisticRegression(max_iter=1000, C=1, solver='liblinear', random_state=SEED)
    lr_model.fit(X_tr_tfidf, y_train)
    lr_time = time.time() - start

    lr_preds = lr_model.predict(X_ts_tfidf)
    lr_f1 = f1_score(y_test, lr_preds, average='macro')
    print(f"  Result: Test F1 = {lr_f1:.4f} (train time: {lr_time:.1f}s)")

    results.append({
        'name': 'A1 Winner: LR + TF-IDF Bigrams',
        'test_f1': round(lr_f1, 4),
        'val_f1_best': None,
        'train_time': round(lr_time, 1),
        'config': 'LR(C=1) + TF-IDF(5000, bigrams)'
    })

    # ---- BERT Configurations ----
    bert_configs = [
        {
            'name': '[1] BERT Original (Raw+Gradual+Drop0.4)',
            'gradual': True, 'dropout': 0.4, 'epochs': 7,
            'label_smoothing': LABEL_SMOOTHING,
            'train_text': (X_tr_r, X_val_r), 'test_text': X_ts_r
        },
        {
            'name': '[2] BERT + Clean Text (Clean+Gradual+Drop0.4)',
            'gradual': True, 'dropout': 0.4, 'epochs': 7,
            'label_smoothing': LABEL_SMOOTHING,
            'train_text': (X_tr_c, X_val_c), 'test_text': X_ts_c
        },
        {
            'name': '[3] BERT + Clean + FullFT (Clean+Full+Drop0.1)',
            'gradual': False, 'dropout': 0.1, 'epochs': 4,
            'label_smoothing': LABEL_SMOOTHING,
            'train_text': (X_tr_c, X_val_c), 'test_text': X_ts_c
        },
        {
            'name': '[4] BERT Best (Clean+Full+Drop0.1+NoLS)',
            'gradual': False, 'dropout': 0.1, 'epochs': 4,
            'label_smoothing': 0.0,
            'train_text': (X_tr_c, X_val_c), 'test_text': X_ts_c
        },
    ]

    for cfg in bert_configs:
        print(f"\n--- Running: {cfg['name']} ---")
        set_seed()

        xtr, xval = cfg['train_text']
        xts = cfg['test_text']

        tld, vld = create_bert_dataloaders(xtr, y_tr, xval, y_val, tokenizer, BATCH_SIZE_BERT)
        _, tsld = create_bert_dataloaders(xtr, y_tr, xts, y_test, tokenizer, BATCH_SIZE_BERT)

        model = BERTClassifier('bert-base-uncased', 2, dropout=cfg['dropout'])

        start = time.time()
        criterion_cfg = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
        model, hist, _ = train_bert_model(
            model, tld, vld, num_epochs=cfg['epochs'],
            patience=BERT_PATIENCE, label_smoothing=cfg['label_smoothing'],
            use_gradual_unfreezing=cfg['gradual']
        )
        train_time = time.time() - start

        # Evaluate on test
        criterion_eval = nn.CrossEntropyLoss()
        _, test_f1, _, _ = evaluate_bert(model, tsld, criterion_eval)
        print(f"  Result: Test F1 = {test_f1:.4f} (train time: {train_time:.1f}s)")

        results.append({
            'name': cfg['name'],
            'test_f1': round(test_f1, 4),
            'val_f1_best': round(max(hist['val_f1']), 4),
            'train_time': round(train_time, 1),
            'config': f"gradual={cfg['gradual']}, drop={cfg['dropout']}, "
                      f"epochs={cfg['epochs']}, ls={cfg['label_smoothing']}",
            'history': {k: [round(x, 4) for x in v] for k, v in hist.items()}
        })
        del model; torch.cuda.empty_cache()

    # ---- Print Summary Table ----
    print("\n" + "="*70)
    print("EXPERIMENT 6: SUMMARY")
    print("="*70)
    print(f"{'#':<4} {'Model':<50} {'Test F1':>8} {'Val F1':>8} {'Time(s)':>8}")
    print("-"*80)
    for i, r in enumerate(results):
        vf1 = f"{r['val_f1_best']:.4f}" if r['val_f1_best'] is not None else "N/A"
        print(f"{i:<4} {r['name']:<50} {r['test_f1']:>8.4f} {vf1:>8} {r['train_time']:>8.1f}")

    # ---- Save results (without history for JSON) ----
    results_json = [{k: v for k, v in r.items() if k != 'history'} for r in results]
    save_results(results_json, 'exp6_improvements.json')

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in results]
    f1s = [r['test_f1'] for r in results]
    colors = ['darkred', 'gray', 'steelblue', 'green', 'darkgreen']
    bars = ax.barh(names, f1s, color=colors[:len(results)])

    ax.set_xlabel('Test F1-Macro')
    ax.set_title('Experiment 6: Impact of Improvements on BERT Performance')
    ax.axvline(x=0.6395, color='red', linestyle='--', alpha=0.7, label='A1 Reference (0.6395)')
    ax.legend(loc='lower right')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.003, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)

    ax.set_xlim(0, max(f1s) + 0.06)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp6_improvements.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {RESULTS_DIR}/exp6_improvements.png")

    return results

if __name__ == '__main__':
    run_all()
