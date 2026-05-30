"""Mix-training Dataset — 민수님 v9 (autoresearch best) 방식의 음절 환경 차용.

핵심: 각 __getitem__마다 raw / masked_geo 중 확률적으로 추첨.
같은 페르소나가 epoch마다 다른 조건으로 노출되어, 모델이 지명 없는 상황에 적응."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MixedConditionDataset(Dataset):
    def __init__(self, encoded_variants: dict, train_df, task_config, task_names,
                 mix_probs={"raw": 0.5, "masked_geo": 0.5}, seed: int = 42):
        """
        encoded_variants: {"raw": Tensor(N,L), "masked_geo": Tensor(N,L)} — 미리 인코딩된 train
        """
        assert abs(sum(mix_probs.values()) - 1.0) < 1e-6
        self.conds = list(mix_probs.keys())
        self.probs = np.array([mix_probs[c] for c in self.conds], dtype=np.float64)
        self.variants = encoded_variants
        self.task_names = task_names
        self.labels = {
            n: torch.tensor(train_df[task_config[n]["col"]].values, dtype=torch.long)
            for n in task_names
        }
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.variants[self.conds[0]].size(0)

    def __getitem__(self, i):
        c = self.rng.choice(self.conds, p=self.probs)
        return (self.variants[c][i], *[self.labels[n][i] for n in self.task_names])


def make_mix_loader(encoded_variants, train_df, task_config, task_names,
                    mix_probs={"raw": 0.5, "masked_geo": 0.5}, batch_size=512, seed=42):
    ds = MixedConditionDataset(encoded_variants, train_df, task_config, task_names,
                                mix_probs=mix_probs, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True)
