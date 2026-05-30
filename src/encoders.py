"""모델 인코더 — TextCNN / BiLSTM / CNN-LSTM, multi-task head 포함."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 128
PAD_IDX = 0


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, kernels=(3,4,5),
                 n_filters=96, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = n_filters * len(kernels)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)
        feats = []
        for c in self.convs:
            h = F.relu(c(e))
            feats.append(F.max_pool1d(h, h.size(2)).squeeze(2))
        return self.dropout(torch.cat(feats, dim=1))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * 2

    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        mask = (x != PAD_IDX).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.dropout(pooled)


class CNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, n_filters=96,
                 kernel=5, hidden=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(embed_dim, n_filters, kernel, padding=kernel//2)
        self.lstm = nn.LSTM(n_filters, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * 2

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)
        c = F.relu(self.conv(e)).transpose(1, 2)
        out, _ = self.lstm(c)
        mask = (x != PAD_IDX).float().unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.dropout(pooled)


class MultiTaskHead(nn.Module):
    def __init__(self, in_dim, task_config, shared_dim=256, tower_dim=128, dropout=0.3):
        super().__init__()
        self.task_names = list(task_config.keys())
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim), nn.BatchNorm1d(shared_dim),
            nn.ReLU(), nn.Dropout(dropout))
        def make_tower(n):
            return nn.Sequential(
                nn.Linear(shared_dim, tower_dim), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(tower_dim, n))
        self.towers = nn.ModuleDict({
            n: make_tower(task_config[n]["n"]) for n in self.task_names
        })

    def forward(self, z):
        h = self.shared(z)
        return {n: t(h) for n, t in self.towers.items()}


class PersonaModel(nn.Module):
    def __init__(self, encoder, task_config):
        super().__init__()
        self.encoder = encoder
        self.head = MultiTaskHead(in_dim=encoder.out_dim, task_config=task_config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
                with torch.no_grad():
                    m.weight[PAD_IDX].zero_()

    def forward(self, x):
        return self.head(self.encoder(x))


ENCODER_CLASSES = {
    "TextCNN": TextCNN,
    "BiLSTM": BiLSTM,
    "CNNLSTM": CNNLSTM,
}

# 데모용 모델 이름 → 인코더 클래스 매핑
MODEL_REGISTRY = {
    "textcnn_5task":  "TextCNN",
    "bilstm_5task":   "BiLSTM",
    "cnnlstm_5task":  "CNNLSTM",
    "textcnn_mix":    "TextCNN",   # mix는 같은 구조, 다른 학습 데이터
}

MODEL_LABELS = {
    "textcnn_5task":  "TextCNN",
    "bilstm_5task":   "BiLSTM",
    "cnnlstm_5task":  "CNN-LSTM",
    "textcnn_mix":    "TextCNN-Mix (raw+geo 50:50)",
}
