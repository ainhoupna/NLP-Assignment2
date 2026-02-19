"""
Training infrastructure: training loop, early stopping, evaluation.
Supports both LSTM and BERT models with proper gradient clipping and scheduling.
Anti-overfitting: label smoothing, weight decay, gradual unfreezing (BERT).
"""
import torch
import torch.nn as nn
import numpy as np
import time
import copy
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# ============================================================
# Training functions
# ============================================================
def train_epoch_lstm(model, dataloader, optimizer, criterion, max_grad_norm=1.0):
    """Train one epoch for LSTM model."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        lengths = batch['lengths']

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

def evaluate_lstm(model, dataloader, criterion):
    """Evaluate LSTM model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            lengths = batch['lengths']

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1, np.array(all_preds), np.array(all_labels)

def train_epoch_bert(model, dataloader, optimizer, criterion, scheduler=None, max_grad_norm=1.0):
    """Train one epoch for BERT model."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1

def evaluate_bert(model, dataloader, criterion):
    """Evaluate BERT model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1, np.array(all_preds), np.array(all_labels)

# ============================================================
# Full training pipelines
# ============================================================
def train_lstm_model(model, train_loader, val_loader, num_epochs=30, lr=5e-4,
                     patience=5, max_grad_norm=1.0, weight_decay=1e-4,
                     label_smoothing=0.1):
    """Full training pipeline for LSTM with weight decay and label smoothing."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_f1 = train_epoch_lstm(model, train_loader, optimizer, criterion, max_grad_norm)
        val_loss, val_f1, _, _ = evaluate_lstm(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f} F1={train_f1:.4f} | Val Loss={val_loss:.4f} F1={val_f1:.4f}")

        early_stopping(val_f1, model)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.restore_best(model)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s | Best Val F1: {early_stopping.best_score:.4f}")
    return model, history, train_time

def train_bert_model(model, train_loader, val_loader, num_epochs=5, lr_bert=2e-5,
                     lr_head=1e-4, patience=3, max_grad_norm=1.0, warmup_ratio=0.1,
                     label_smoothing=0.1, use_gradual_unfreezing=True):
    """Full training pipeline for BERT with gradual unfreezing, label smoothing."""
    from models.bert_classifier import gradual_unfreeze_schedule

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        # Gradual unfreezing: adjust which layers are trainable
        if use_gradual_unfreezing:
            gradual_unfreeze_schedule(model, epoch)

        # Re-create optimizer each epoch (param groups change with unfreezing)
        bert_params = [p for p in model.bert.parameters() if p.requires_grad]
        head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())

        param_groups = [{'params': head_params, 'lr': lr_head}]
        if bert_params:
            param_groups.insert(0, {'params': bert_params, 'lr': lr_bert})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Warmup scheduler for this epoch
        total_steps = len(train_loader)
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.1, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        train_loss, train_f1 = train_epoch_bert(model, train_loader, optimizer, criterion, scheduler, max_grad_norm)
        val_loss, val_f1, _, _ = evaluate_bert(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"  Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f} F1={train_f1:.4f} | Val Loss={val_loss:.4f} F1={val_f1:.4f}")

        early_stopping(val_f1, model)
        if early_stopping.should_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    early_stopping.restore_best(model)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s | Best Val F1: {early_stopping.best_score:.4f}")
    return model, history, train_time

# ============================================================
# Inference speed measurement
# ============================================================
def measure_inference_speed(model, dataloader, model_type='lstm', num_batches=None):
    """Measure inference speed in samples/second."""
    model.eval()
    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
            if model_type == 'lstm':
                input_ids = batch['input_ids'].to(DEVICE)
                lengths = batch['lengths']
                _ = model(input_ids, lengths)
            else:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                _ = model(input_ids, attention_mask)
            total_samples += input_ids.size(0)

    elapsed = time.time() - start
    return total_samples / elapsed if elapsed > 0 else 0

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0
