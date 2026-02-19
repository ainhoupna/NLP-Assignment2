"""
BiLSTM Classifier for text classification (MAMI misogyny detection).
Trained from scratch with optional pre-trained embedding initialization.
Anti-overfitting: embedding dropout, locked dropout, embedding freeze.
"""
import torch
import torch.nn as nn
import numpy as np

class LockedDropout(nn.Module):
    """Variational dropout: same mask across timesteps."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = x.new_ones(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask

class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary text classification."""

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_classes=2, num_layers=2, dropout=0.5,
                 bidirectional=True, pretrained_embeddings=None,
                 freeze_embeddings=True, embed_dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.data[0] = 0  # Keep PAD as zeros
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Embedding dropout (drops entire words)
        self.embed_dropout = nn.Dropout2d(embed_dropout)
        # Locked dropout on LSTM output
        self.locked_dropout = LockedDropout(dropout)

        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: [batch, seq_len]
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]

        # Embedding dropout: treat dim 2 as "channels" for Dropout2d
        embedded = self.embed_dropout(embedded.unsqueeze(3)).squeeze(3)

        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)

        # Concatenate final forward and backward hidden states
        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden[-1]

        # Classification
        output = self.dropout(final_hidden)
        logits = self.fc(output)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_fasttext_embeddings(vocab, fasttext_path, embedding_dim=300, max_vocab=200000):
    """Load FastText embeddings and create embedding matrix for the vocabulary."""
    print(f"Loading FastText embeddings from {fasttext_path}...")
    embeddings = {}
    with open(fasttext_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for i, line in enumerate(f):
            if i >= max_vocab:
                break
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab.word2idx:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector

    # Create embedding matrix
    matrix = np.random.normal(0, 0.1, (len(vocab), embedding_dim)).astype(np.float32)
    matrix[0] = 0  # PAD
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in embeddings:
            matrix[idx] = embeddings[word]
            found += 1

    print(f"  Found {found}/{len(vocab)} words in FastText ({100*found/len(vocab):.1f}%)")
    return torch.tensor(matrix)
