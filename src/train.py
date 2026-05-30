"""학습 루프 + Loss + 평가."""
from __future__ import annotations
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from .encoders import PersonaModel


class UncertaintyWeightedLoss(nn.Module):
    """Kendall et al. 2018 — log σ²를 학습해 자동 균형."""
    def __init__(self, task_names, class_weights=None):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
        self.criterions = nn.ModuleDict()
        for name in task_names:
            w = class_weights.get(name) if class_weights else None
            self.criterions[name] = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits, labels):
        total, per = 0.0, {}
        for i, name in enumerate(self.task_names):
            loss = self.criterions[name](logits[name], labels[name])
            total = total + torch.exp(-self.log_vars[i]) * loss + 0.5 * self.log_vars[i]
            per[name] = loss.item()
        return total, per


def compute_class_weights(train_df, task_config, task_names, device):
    weights = {}
    for name in task_names:
        col, n_classes = task_config[name]["col"], task_config[name]["n"]
        counts = Counter(train_df[col].tolist())
        n = len(train_df)
        w = torch.zeros(n_classes)
        for c, cnt in counts.items():
            w[c] = n / (n_classes * cnt)
        weights[name] = w.to(device)
    return weights


def make_loaders(X_train, X_val, X_test, train_df, val_df, test_df,
                 task_config, task_names, batch_size=512):
    """token_cache 같은 미리 인코딩된 X를 받아 DataLoader 3개 반환."""
    loaders = {}
    for sname, X, sdf, shuffle in [
        ("train", X_train, train_df, True),
        ("val",   X_val,   val_df,   False),
        ("test",  X_test,  test_df,  False),
    ]:
        labels = [torch.tensor(sdf[task_config[n]["col"]].values, dtype=torch.long)
                  for n in task_names]
        ds = TensorDataset(X, *labels)
        loaders[sname] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=0, pin_memory=True)
    return loaders["train"], loaders["val"], loaders["test"]


def evaluate(model, loader, task_names, device):
    model.eval()
    preds = {n: [] for n in task_names}
    trues = {n: [] for n in task_names}
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            labels = batch[1:]
            logits = model(x)
            for i, n in enumerate(task_names):
                preds[n] += logits[n].argmax(1).cpu().tolist()
                trues[n] += labels[i].tolist()
    metrics, f1s = {}, []
    for n in task_names:
        f1 = f1_score(trues[n], preds[n], average="macro", zero_division=0)
        acc = sum(p == t for p, t in zip(preds[n], trues[n])) / len(preds[n])
        metrics[f"{n}_f1"], metrics[f"{n}_acc"] = f1, acc
        f1s.append(f1)
    metrics["avg_f1"] = sum(f1s) / len(task_names)
    return metrics


def train_model(encoder_cls, train_loader, val_loader, test_loader,
                vocab_size, task_config, task_names, class_weights, device,
                n_epochs=10, patience=3, lr=1e-3, verbose=True):
    """학습 → best 가중치 로드 → test 평가. 모델 반환."""
    model = PersonaModel(encoder_cls(vocab_size=vocab_size), task_config).to(device)
    loss_fn = UncertaintyWeightedLoss(task_names, class_weights).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr, weight_decay=1e-5)

    best_f1, best_state, pc = -1, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            labels = {n: batch[i+1].to(device) for i, n in enumerate(task_names)}
            optimizer.zero_grad()
            loss, _ = loss_fn(model(x), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val = evaluate(model, val_loader, task_names, device)
        if verbose:
            print(f"  ep {epoch:2d}  val avg-F1 {val['avg_f1']:.4f}  prov {val['prov_f1']:.4f}")
        if val["avg_f1"] > best_f1:
            best_f1 = val["avg_f1"]
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                if verbose:
                    print(f"  early stop @ ep {epoch}")
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_metrics = evaluate(model, test_loader, task_names, device)
    if verbose:
        print(f"  → test avg {test_metrics['avg_f1']:.4f}  prov {test_metrics['prov_f1']:.4f}")
    return model, {"best_val_f1": best_f1, "test": test_metrics}
