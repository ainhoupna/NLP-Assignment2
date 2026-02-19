"""
BERT-based classifier for text classification (MAMI misogyny detection).
Fine-tunes a pre-trained BERT model with gradual unfreezing and differential LR.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

class BERTClassifier(nn.Module):
    """BERT fine-tuning for binary text classification."""

    def __init__(self, model_name='bert-base-uncased', num_classes=2,
                 dropout=0.4, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())

def _get_encoder_layers(model):
    """Get encoder layers for BERT or DistilBERT."""
    if hasattr(model.bert, 'encoder'):
        return model.bert.encoder.layer  # BERT
    elif hasattr(model.bert, 'transformer'):
        return model.bert.transformer.layer  # DistilBERT
    else:
        raise AttributeError("Cannot find encoder layers in model")

def unfreeze_last_n_layers(model, n):
    """Unfreeze only the last n encoder layers of BERT/DistilBERT."""
    # Freeze all first
    for param in model.bert.parameters():
        param.requires_grad = False
    # Unfreeze last n encoder layers
    encoder_layers = _get_encoder_layers(model)
    layers_to_unfreeze = min(n, len(encoder_layers))
    for layer in encoder_layers[-layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    # Classifier head always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.dropout.parameters():
        param.requires_grad = True

def gradual_unfreeze_schedule(model, epoch):
    """Gradual unfreezing: progressively unfreeze more BERT layers each epoch.
    Epoch 0: classifier head only (BERT frozen)
    Epoch 1: + last 2 encoder layers
    Epoch 2: + last 4 encoder layers
    Epoch 3: + last 8 encoder layers
    Epoch 4+: all layers
    """
    schedule = {0: 0, 1: 2, 2: 4, 3: 8}
    n_unfreeze = schedule.get(epoch, 999)  # 999 = all

    if n_unfreeze == 0:
        # Freeze all BERT, only classifier trainable
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif n_unfreeze >= len(_get_encoder_layers(model)):
        # Unfreeze everything
        for param in model.bert.parameters():
            param.requires_grad = True
    else:
        unfreeze_last_n_layers(model, n_unfreeze)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Epoch {epoch+1}: unfreezing {n_unfreeze} layers → {trainable:,} trainable params")
