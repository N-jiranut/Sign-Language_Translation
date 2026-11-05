"""
PyTorch Neural Network Models for Sign Language Recognition.
โครงสร้างโมเดล Deep Learning (GRU) สำหรับจดจำลำดับท่วงท่าภาษามือ
"""
import torch
import torch.nn as nn


class SignGRU(nn.Module):
    def __init__(self, input_size=112, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len=24, input_size=112)
        out, _ = self.gru(x)
        # Take the output of the final time step
        last_out = out[:, -1, :]
        logits = self.classifier(last_out)
        return logits
