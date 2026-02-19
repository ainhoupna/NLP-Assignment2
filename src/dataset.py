"""
Data pipeline: Vocabulary, PyTorch Dataset, DataLoader for MAMI dataset.
Supports both custom tokenization (for LSTM) and BERT tokenization.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import string
import os

# ============================================================
# Preprocessing (reused from Assignment 1)
# ============================================================
def clean_text(text):
    """Preprocess social media text for the MAMI dataset."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================================
# Vocabulary
# ============================================================
class Vocabulary:
    """Build and manage vocabulary for tokenization."""
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq=2, max_size=25000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.word_freq = Counter()

    def build(self, texts):
        """Build vocabulary from list of texts."""
        for text in texts:
            tokens = text.split()
            self.word_freq.update(tokens)

        # Filter by min_freq and sort by frequency
        filtered = [(w, c) for w, c in self.word_freq.most_common() if c >= self.min_freq]
        for word, _ in filtered[:self.max_size - 2]:  # -2 for PAD and UNK
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary: {len(self.word2idx)} words (from {len(self.word_freq)} unique tokens, min_freq={self.min_freq})")
        return self

    def encode(self, text, max_len=128):
        """Tokenize and numericalize a text string."""
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        return indices

    def __len__(self):
        return len(self.word2idx)

# ============================================================
# PyTorch Dataset for LSTM/Transformer from scratch
# ============================================================
class MAMIDataset(Dataset):
    """PyTorch Dataset for the MAMI misogyny detection task."""

    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.vocab.encode(text, self.max_len)
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': len(indices)
        }

def collate_fn(batch):
    """Custom collate: pad sequences to max length in batch."""
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)

    # Pad to max length in this batch
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {
        'input_ids': padded,
        'labels': labels,
        'lengths': lengths
    }

# ============================================================
# PyTorch Dataset for BERT
# ============================================================
class MAMIBertDataset(Dataset):
    """PyTorch Dataset using HuggingFace tokenizer for BERT."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# Data loading utility
# ============================================================
def load_mami_data(base_dir='.'):
    """Load MAMI training and official test data."""
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'TRAINING', 'training.csv'), sep='\t')
    test_feat = pd.read_csv(os.path.join(base_dir, 'data', 'test', 'Test.csv'), sep='\t')
    test_labels = pd.read_csv(
        os.path.join(base_dir, 'test_labels.txt'), sep='\t', header=None,
        names=['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence']
    )
    test_df = pd.merge(test_feat, test_labels, on='file_name')

    # Clean text
    train_df['cleaned_text'] = train_df['Text Transcription'].fillna("").apply(clean_text)
    test_df['cleaned_text'] = test_df['Text Transcription'].fillna("").apply(clean_text)

    X_train = train_df['cleaned_text'].values
    y_train = train_df['misogynous'].values
    X_test = test_df['cleaned_text'].values
    y_test = test_df['misogynous'].values

    # Keep raw text for BERT (no cleaning, BERT has its own tokenizer)
    X_train_raw = train_df['Text Transcription'].fillna("").values
    X_test_raw = test_df['Text Transcription'].fillna("").values

    print(f"Loaded: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, train_df, test_df

def create_dataloaders(X_train, y_train, X_test, y_test, vocab, batch_size=32, max_len=128):
    """Create DataLoaders for LSTM training."""
    train_dataset = MAMIDataset(X_train, y_train, vocab, max_len)
    test_dataset = MAMIDataset(X_test, y_test, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    return train_loader, test_loader

def create_bert_dataloaders(X_train_raw, y_train, X_test_raw, y_test, tokenizer,
                             batch_size=16, max_len=128):
    """Create DataLoaders for BERT training."""
    train_dataset = MAMIBertDataset(X_train_raw, y_train, tokenizer, max_len)
    test_dataset = MAMIBertDataset(X_test_raw, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
