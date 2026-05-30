"""음절 토크나이저 — [DEMO], [LOC] 같은 마스킹 토큰을 통째로 보존."""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import torch


PAD_IDX, UNK_IDX = 0, 1
MIN_FREQ = 5
MASK_SET = ["[DEMO]", "[LOC]", "[SEP]", "[NAME]"]
MASK_LEN = {"[DEMO]": 6, "[LOC]": 5, "[SEP]": 5, "[NAME]": 6}


def tokenize_syllable(s: str) -> list[str]:
    """음절 단위 — 마스킹 토큰은 통째로 보존."""
    tokens, i = [], 0
    while i < len(s):
        matched = False
        for mtok in MASK_SET:
            ml = MASK_LEN[mtok]
            if s[i:i+ml] == mtok:
                tokens.append(mtok); i += ml; matched = True; break
        if not matched:
            if s[i].isspace():
                i += 1
            else:
                tokens.append(s[i]); i += 1
    return tokens


class SyllableTokenizer:
    def __init__(self, itos: list[str], max_len: int = 512):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.vocab_size = len(itos)
        self.max_len = max_len

    @classmethod
    def build(cls, texts: list[str], max_len: int = 512, min_freq: int = MIN_FREQ):
        counter = Counter()
        for t in texts:
            counter.update(tokenize_syllable(t))
        itos = ["<pad>", "<unk>"] + [w for w, c in counter.most_common() if c >= min_freq]
        for tok in MASK_SET:
            if tok not in itos:
                itos.append(tok)
        return cls(itos, max_len=max_len)

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(t, UNK_IDX) for t in tokenize_syllable(text)][:self.max_len]
        return ids + [PAD_IDX] * (self.max_len - len(ids))

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def save_meta(self, path: str | Path, extra: dict | None = None):
        meta = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "pad_idx": PAD_IDX,
            "unk_idx": UNK_IDX,
            "itos": self.itos,
        }
        if extra:
            meta.update(extra)
        Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_meta(cls, path: str | Path):
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(meta["itos"], max_len=meta["max_len"])
