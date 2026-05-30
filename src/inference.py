"""모델 로드 + 예측 + saliency — app.py가 import해서 사용.
수정님(shin-sj) saliency 코드를 본인 모델 구조(model.encoder.embed)에 맞춤.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoders import ENCODER_CLASSES, MODEL_REGISTRY, MODEL_LABELS, PersonaModel
from .tokenizer import SyllableTokenizer, tokenize_syllable, PAD_IDX, UNK_IDX
from .data_utils import DEMO_REGEX

# 경로 — repo 루트 기준
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CONDITIONS = ["raw", "masked", "masked_geo"]
CONDITION_LABELS = {
    "raw": "Raw 원본",
    "masked": "Masked (X대·친족어)",
    "masked_geo": "Masked+Geo (+지명)",
}
TASK_LABELS_KO = {
    "sex": "성별", "age": "연령대", "prov": "광역시도",
    "life": "라이프스타일", "marital": "결혼여부",
}


def load_meta() -> dict:
    """models/meta.json (tokenizer 정보 + label maps + geo_terms 포함)."""
    return json.loads((MODELS_DIR / "meta.json").read_text(encoding="utf-8"))


def load_tokenizer() -> SyllableTokenizer:
    meta = load_meta()
    return SyllableTokenizer(meta["itos"], max_len=meta["max_len"])


def build_geo_pattern_from_meta(meta: dict) -> re.Pattern:
    return re.compile("|".join(sorted(meta["geo_terms"], key=len, reverse=True)))


def mask_text(text: str, condition: str, geo_pattern: re.Pattern) -> str:
    if not text:
        return ""
    t = str(text)
    if condition == "raw":
        return t
    t = DEMO_REGEX.sub("[DEMO]", t)
    if condition == "masked":
        return t
    return geo_pattern.sub("[LOC]", t)


def load_model(model_name: str, device="cpu"):
    """models/<name>.pt 로드 → PersonaModel."""
    meta = load_meta()
    encoder_cls_name = MODEL_REGISTRY[model_name]
    encoder_cls = ENCODER_CLASSES[encoder_cls_name]
    encoder = encoder_cls(vocab_size=meta["vocab_size"])
    model = PersonaModel(encoder, task_config=meta["task_config"])
    state = torch.load(MODELS_DIR / f"{model_name}.pt", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_one(model, tokenizer: SyllableTokenizer, text: str, condition: str,
                geo_pattern: re.Pattern, device="cpu"):
    """단일 추론 — 5 task 예측. (results, masked_text) 반환."""
    meta = load_meta()
    masked = mask_text(text, condition, geo_pattern)
    ids = torch.tensor([tokenizer.encode(masked)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
    result = {}
    for task in meta["task_names"]:
        probs = F.softmax(logits[task], dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        result[task] = {
            "pred_label": meta["label_maps"][task][idx],
            "pred_idx": idx,
            "confidence": float(probs[idx]),
            "probs": probs.tolist(),
            "classes": meta["label_maps"][task],
        }
    return result, masked


def compute_saliency(model, tokenizer: SyllableTokenizer, text: str, condition: str,
                     task: str, geo_pattern: re.Pattern, device="cpu"):
    """글자별 saliency (input × gradient).
    수정님 코드 그대로, 본인 모델 구조(model.encoder.embed)에 맞춤.

    반환: (char_scores [len(masked)], pred_idx, masked_text)
    """
    meta = load_meta()
    masked = mask_text(text, condition, geo_pattern)
    if not masked:
        return np.array([]), 0, masked

    model.eval()
    ids = torch.tensor(
        [tokenizer.encode(masked)], dtype=torch.long, device=device
    )

    # forward hook으로 embedding 출력 캡처 — 수정님 방식 그대로
    captured = {}

    def hook(module, inp, out):
        out.retain_grad()
        captured["emb"] = out

    # ⚠️ 본인 모델은 model.encoder.embed (수정님은 model.embed)
    handle = model.encoder.embed.register_forward_hook(hook)
    try:
        out = model(ids)
    finally:
        handle.remove()

    logits = out[task][0]
    pred_idx = int(logits.argmax().item())
    target = logits[pred_idx]

    emb = captured["emb"]  # [1, L, D]
    grad = torch.autograd.grad(target, emb, retain_graph=False)[0]  # [1, L, D]
    saliency = (grad * emb).abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # 실제 토큰 길이만 (pad 제외)
    actual_len = int((ids[0] != PAD_IDX).sum().item())
    token_scores = saliency[:actual_len]

    # 토큰 → 글자 매핑 (음절은 1글자=1토큰, [DEMO]/[LOC]는 여러 글자)
    tokens = tokenize_syllable(masked)[:meta["max_len"]]
    char_scores = np.zeros(len(masked))
    pos = 0
    for i, tok in enumerate(tokens):
        if i >= len(token_scores):
            break
        while pos < len(masked) and masked[pos].isspace():
            pos += 1
        ln = len(tok)
        if pos + ln <= len(masked):
            char_scores[pos:pos+ln] = token_scores[i]
        pos += ln

    return char_scores, pred_idx, masked


def load_personas_30() -> dict:
    """results/personas_30.json."""
    return json.loads((RESULTS_DIR / "personas_30.json").read_text(encoding="utf-8"))


# 데모용 모델 리스트
MODELS_3 = ["textcnn_5task", "bilstm_5task", "cnnlstm_5task"]
MODELS_MIX = ["textcnn_5task", "textcnn_mix"]