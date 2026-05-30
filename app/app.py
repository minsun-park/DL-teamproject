"""Streamlit 데모 — 한국어 페르소나 분류 + Stereotype 측정 (박민선).

디자인 통합:
- Saliency 시각화: 수정님(shin-sj) presentation2 스타일 (kw1-kw4 4단계 빨강 강도)
- 데모 UI: 본인 2차_발표 스타일 (3-column 토글, cues 태그, 5-task pred 그리드, lec)

주의: HTML 문자열은 모두 단일 라인으로 작성 (멀티라인 들여쓰기는 markdown이
code block으로 오해해 escape하므로).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.inference import (
    CONDITIONS, CONDITION_LABELS, TASK_LABELS_KO, MODELS_3, MODELS_MIX,
    load_tokenizer, load_model, load_meta, predict_one,
    compute_saliency, load_personas_30, build_geo_pattern_from_meta,
)
from src.encoders import MODEL_LABELS

st.set_page_config(page_title="페르소나 분류 데모 (박민선)", layout="wide")
DEVICE = "cpu"

# ============================================================
# 통합 CSS — 모두 한 블록으로
# ============================================================
CUSTOM_CSS = """
<style>
:root {
  --bg: #F8FAFC;
  --paper: #FFFFFF;
  --ink: #0F172A;
  --ink-soft: #334155;
  --ink-muted: #64748B;
  --rule: #E2E8F0;
  --soft: #F1F5F9;
  --accent: #1E40AF;
  --green: #059669;
  --red: #DC2626;
  --amber: #D97706;
}
.sal-text { font-family: 'Pretendard', -apple-system, sans-serif; font-size: 16px; line-height: 2.1; padding: 16px 20px; background: var(--paper); border: 1px solid var(--rule); border-radius: 8px; color: var(--ink-soft); word-break: break-all; }
.kw1 { background: rgba(220,38,38,0.95); color: #fff; padding: 2px 5px; border-radius: 3px; font-weight: 700; }
.kw2 { background: rgba(220,38,38,0.75); color: #fff; padding: 2px 5px; border-radius: 3px; font-weight: 600; }
.kw3 { background: rgba(220,38,38,0.55); padding: 2px 5px; border-radius: 3px; font-weight: 600; }
.kw4 { background: rgba(220,38,38,0.35); padding: 2px 5px; border-radius: 3px; }
.sal-legend { display: flex; gap: 10px; font-size: 12px; color: var(--ink-muted); margin-top: 10px; align-items: center; flex-wrap: wrap; }
.sal-legend .swatch { display: inline-block; width: 18px; height: 14px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
.persona-truth { background: var(--soft); border-radius: 8px; padding: 12px 16px; margin: 0 0 14px 0; }
.persona-truth .label { font-size: 11px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 6px; }
.persona-truth .tag { display: inline-block; background: var(--paper); border: 1px solid var(--rule); border-radius: 4px; padding: 2px 8px; margin-right: 6px; font-size: 13px; font-weight: 500; color: var(--ink); }
.cues { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.cue { background: #FEF9C3; border: 1px solid #FDE68A; border-radius: 6px; padding: 4px 10px; font-size: 12px; color: #92400E; }
.cue strong { color: #7C2D12; margin-right: 4px; }
.cue .arrow { color: #A16207; margin: 0 4px; }
.preds { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 12px 0; }
.pred { border: 1px solid var(--rule); border-radius: 8px; padding: 12px 8px; text-align: center; position: relative; background: var(--paper); }
.pred.correct { border-left: 3px solid var(--green); }
.pred.wrong { border-left: 3px solid var(--red); }
.pred .task { font-size: 10px; color: var(--ink-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 5px; }
.pred .ans { font-size: 15px; font-weight: 700; margin-bottom: 3px; color: var(--ink); }
.pred .conf { font-size: 11px; color: var(--ink-muted); }
.pred .truth { font-size: 11px; color: var(--ink-muted); margin-top: 4px; padding-top: 4px; border-top: 1px dashed var(--rule); }
.pred .mark { position: absolute; top: 6px; right: 6px; font-size: 12px; font-weight: 700; }
.lec { background: #F1F5F9; border: 1px dashed #94A3B8; border-radius: 8px; padding: 11px 15px; font-size: 12.5px; color: var(--ink-soft); margin-top: 10px; line-height: 1.7; }
.lec strong { color: var(--ink); }
.toggle-label { font-size: 11px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; margin-bottom: 4px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 캐싱
# ============================================================
@st.cache_resource(show_spinner="토크나이저 로드 중...")
def cached_tokenizer():
    return load_tokenizer()


@st.cache_resource
def cached_meta():
    return load_meta()


@st.cache_resource
def cached_geo_pattern():
    return build_geo_pattern_from_meta(load_meta())


@st.cache_resource(show_spinner="모델 로드 중...")
def cached_model(model_name: str):
    return load_model(model_name, device=DEVICE)


@st.cache_data
def cached_personas():
    return load_personas_30()


# ============================================================
# Saliency HTML — 수정님 kw1-kw4 4단계
# ============================================================
def saliency_html(text: str, scores: np.ndarray, max_chars: int = 600) -> str:
    text = text[:max_chars]
    n = min(len(text), len(scores), max_chars)
    if n == 0:
        return '<div class="sal-text">(빈 입력)</div>'
    smax = float(scores[:n].max()) if scores[:n].max() > 0 else 1.0

    parts = ['<div class="sal-text">']
    for i in range(n):
        ch = text[i]
        s = float(scores[i]) / smax
        if ch == "\n":
            parts.append("<br>"); continue
        if ch == " ":
            parts.append("&nbsp;"); continue
        esc = ch.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if s > 0.75:
            parts.append(f'<span class="kw1">{esc}</span>')
        elif s > 0.50:
            parts.append(f'<span class="kw2">{esc}</span>')
        elif s > 0.25:
            parts.append(f'<span class="kw3">{esc}</span>')
        elif s > 0.10:
            parts.append(f'<span class="kw4">{esc}</span>')
        else:
            parts.append(esc)
    parts.append('</div>')
    legend = '<div class="sal-legend"><span><span class="swatch" style="background:rgba(220,38,38,0.95)"></span>매우높음</span><span><span class="swatch" style="background:rgba(220,38,38,0.75)"></span>높음</span><span><span class="swatch" style="background:rgba(220,38,38,0.55)"></span>중간</span><span><span class="swatch" style="background:rgba(220,38,38,0.35)"></span>낮음</span><span style="margin-left:8px;">→ 빨강이 진할수록 그 글자가 답에 결정적이었다는 뜻</span></div>'
    parts.append(legend)
    return "".join(parts)


# ============================================================
# 렌더러들 — 모두 단일 라인 HTML
# ============================================================
def render_persona_truth(persona: dict):
    label = persona['label']
    parts = label.replace(' · ', ' ').split()
    tags = ''.join(f'<span class="tag">{p}</span>' for p in parts)
    html = f'<div class="persona-truth"><div class="label">📌 선택된 페르소나 · 정답</div><div>{tags}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_cues(persona: dict):
    cues = persona.get("cues") or persona.get("stereotype_cues")
    if not cues:
        return
    parts = []
    for c in cues:
        if isinstance(c, dict):
            word = c.get("word", "")
            task = c.get("task", "")
            implies = c.get("implies", "")
            if task or implies:
                parts.append(f'<span class="cue"><strong>{word}</strong><span class="arrow">→</span>{task} {implies}</span>')
            else:
                parts.append(f'<span class="cue"><strong>{word}</strong></span>')
        else:
            parts.append(f'<span class="cue">{c}</span>')
    label = '<div class="toggle-label">📍 Stereotype 단서 — 이 단어들이 모델 예측에 영향</div>'
    html = label + '<div class="cues">' + ''.join(parts) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_preds_grid(result: dict, truth_map: dict, task_names: list):
    cards = []
    for task in task_names:
        r = result[task]
        pred = r["pred_label"]
        conf = r["confidence"]
        truth = truth_map.get(task)
        correct = (truth == pred) if truth else None
        css = "correct" if correct is True else ("wrong" if correct is False else "")
        mark = "✓" if correct is True else ("✗" if correct is False else "")
        mark_color = "#059669" if correct is True else ("#DC2626" if correct is False else "")

        truth_html = f'<div class="truth">실제: {truth}</div>' if truth else ''
        mark_html = f'<span class="mark" style="color:{mark_color}">{mark}</span>' if mark else ''

        card = f'<div class="pred {css}">{mark_html}<div class="task">{TASK_LABELS_KO[task]}</div><div class="ans">{pred}</div><div class="conf">{conf:.0%}</div>{truth_html}</div>'
        cards.append(card)

    html = '<div class="preds">' + ''.join(cards) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_lec(html_content: str):
    html = f'<div class="lec">{html_content}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================
st.title("📚 한국어 페르소나 분류 데모")
st.markdown(
    "**LLM이 생성한 한국어 페르소나 텍스트로 인구통계를 분류하고, "
    "LLM이 학습한 사회적 stereotype을 정량 측정한다** (박민선)"
)
st.caption(f"Nemotron-Personas-Korea · 5-task · 음절 토크나이저 · Device = {DEVICE}")

mode1, mode2 = st.tabs([
    "🔬 3 모델 비교 (TextCNN / BiLSTM / CNN-LSTM)",
    "⚡ TextCNN 고도화 (Mix-training)",
])


# ============================================================
# MODE 1 — 3 모델 비교
# ============================================================
with mode1:
    st.subheader("3 모델 비교 — 페르소나·모델·조건을 토글하며 비교")
    st.caption(
        "30개 페르소나 중 하나를 고르고, 모델과 마스킹 조건을 토글하면 예측이 어떻게 달라지는지 "
        "확인할 수 있습니다. saliency로 모델이 어떤 글자를 보고 답했는지 빨간색 강도로 표시됩니다."
    )

    personas = cached_personas()
    meta = cached_meta()
    geo_pat = cached_geo_pattern()

    col1, col2, col3 = st.columns([2.2, 1.6, 1.6])
    with col1:
        st.markdown('<div class="toggle-label">① 페르소나 (30개)</div>', unsafe_allow_html=True)
        p_options = [f"[{i}] {p['label']}" for i, p in enumerate(personas["personas"])]
        p_idx = st.selectbox(
            "p1", range(len(p_options)),
            format_func=lambda i: p_options[i],
            key="m1_p", label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="toggle-label">② 모델</div>', unsafe_allow_html=True)
        m1_model = st.radio(
            "m1", MODELS_3,
            format_func=lambda m: MODEL_LABELS[m],
            key="m1_model", horizontal=True, label_visibility="collapsed",
        )
    with col3:
        st.markdown('<div class="toggle-label">③ 마스킹 조건</div>', unsafe_allow_html=True)
        m1_cond = st.radio(
            "c1", CONDITIONS,
            format_func=lambda c: CONDITION_LABELS[c],
            key="m1_cond", horizontal=True, label_visibility="collapsed",
        )

    persona = personas["personas"][p_idx]
    raw_text = persona["texts"]["raw"]

    render_persona_truth(persona)
    render_cues(persona)

    tok = cached_tokenizer()
    model = cached_model(m1_model)
    result, masked = predict_one(model, tok, raw_text, m1_cond, geo_pat, device=DEVICE)

    truth_map = {}
    if "TextCNN" in persona["models"]:
        for task in meta["task_names"]:
            truth_map[task] = persona["models"]["TextCNN"].get("raw", {}).get(task, {}).get("true")

    st.markdown(
        f"##### 예측 결과 — `{MODEL_LABELS[m1_model]}` × `{CONDITION_LABELS[m1_cond]}`"
    )
    render_preds_grid(result, truth_map, meta["task_names"])

    lec_map = {
        "textcnn_5task": "<strong>TextCNN</strong> — (k×d) 커널로 단어 방향만 슬라이딩해 임베딩 의미 보존. k=3/4/5 multi-scale + max-over-time pooling. <em>n-gram 검출에 강함.</em>",
        "bilstm_5task": "<strong>BiLSTM</strong> — 양방향 인코딩, gate(sigmoid/tanh)로 정보 흐름 제어. cell state gradient flow ≡ ResNet skip. <em>긴 의존성에 강함.</em>",
        "cnnlstm_5task": "<strong>CNN-LSTM Hybrid</strong> — Conv1d로 n-gram 추출 후 BiLSTM. 공간+시간 결합. <em>'합치면 더 좋아지나'의 검증.</em>",
    }
    render_lec(lec_map.get(m1_model, ""))

    with st.expander(f"🔍 모델이 실제로 본 입력 ({CONDITION_LABELS[m1_cond]})"):
        st.text(masked[:1500] + ("..." if len(masked) > 1500 else ""))

    with st.expander("📊 task별 확률 분포 (top-8)"):
        bar_cols = st.columns(5)
        for i, task in enumerate(meta["task_names"]):
            with bar_cols[i]:
                st.markdown(f"**{TASK_LABELS_KO[task]}**")
                r = result[task]
                df = pd.DataFrame({"class": r["classes"], "prob": r["probs"]})
                df = df.sort_values("prob", ascending=False).head(8)
                st.bar_chart(df.set_index("class"), height=200)

    st.markdown("---")
    st.markdown("### 🔍 Saliency — 모델이 어떤 글자를 보고 답했나")
    st.caption(
        "각 task의 예측에 어떤 글자가 기여했는지 글자 단위로 시각화. "
        "마스킹 조건을 바꾸면 모델이 의존하는 단서가 어떻게 달라지는지 직접 확인할 수 있습니다."
    )

    sal_cols = st.columns(2)
    for i, task in enumerate(meta["task_names"]):
        with sal_cols[i % 2]:
            with st.expander(
                f"**{TASK_LABELS_KO[task]}** → `{result[task]['pred_label']}` "
                f"({result[task]['confidence']:.0%})",
                expanded=(task == "prov"),
            ):
                try:
                    scores, _, masked_text = compute_saliency(
                        model, tok, raw_text, m1_cond, task, geo_pat, device=DEVICE)
                    st.markdown(saliency_html(masked_text, scores), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"saliency 계산 실패: {e}")


# ============================================================
# MODE 2 — TextCNN 고도화 (Mix-training)
# ============================================================
with mode2:
    st.subheader("TextCNN 고도화 — Mix-training으로 한계 극복")
    st.caption(
        "같은 TextCNN 구조에서, **학습 데이터를 raw+masked_geo 50:50으로 섞으면** "
        "(민수님 v9 mix-training 방식 차용) 지명을 가려도 stereotype을 학습합니다. "
        "동일 페르소나로 두 버전을 직접 비교해보세요."
    )

    personas = cached_personas()
    meta = cached_meta()
    geo_pat = cached_geo_pattern()

    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        st.markdown('<div class="toggle-label">페르소나</div>', unsafe_allow_html=True)
        p_options = [f"[{i}] {p['label']}" for i, p in enumerate(personas["personas"])]
        p_idx2 = st.selectbox(
            "p2", range(len(p_options)),
            format_func=lambda i: p_options[i],
            key="m2_p", label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="toggle-label">마스킹 조건</div>', unsafe_allow_html=True)
        m2_cond = st.radio(
            "c2", CONDITIONS,
            format_func=lambda c: CONDITION_LABELS[c],
            index=2, key="m2_cond", horizontal=True, label_visibility="collapsed",
        )

    persona = personas["personas"][p_idx2]
    raw_text = persona["texts"]["raw"]
    render_persona_truth(persona)

    st.info(
        "💡 **Masked+Geo**가 기본 — 기본 TextCNN은 광역시도 신뢰도가 무너지지만, "
        "Mix는 살아남는 효과를 가장 잘 보여줍니다."
    )

    tok = cached_tokenizer()
    results = {}
    for mname in MODELS_MIX:
        model = cached_model(mname)
        r, _ = predict_one(model, tok, raw_text, m2_cond, geo_pat, device=DEVICE)
        results[mname] = r

    truth_map2 = {}
    if "TextCNN" in persona["models"]:
        for task in meta["task_names"]:
            truth_map2[task] = persona["models"]["TextCNN"].get("raw", {}).get(task, {}).get("true")

    for mname in MODELS_MIX:
        st.markdown(f"##### {MODEL_LABELS[mname]}")
        render_preds_grid(results[mname], truth_map2, meta["task_names"])

    st.markdown("##### 광역시도 예측 확률 — 모델별 비교")
    st.caption(
        "Mix-training이 지명 없이도 stereotype으로 추론할 수 있게 만드는지 직접 확인."
    )
    prov_cols = st.columns(2)
    for i, mname in enumerate(MODELS_MIX):
        with prov_cols[i]:
            st.markdown(f"**{MODEL_LABELS[mname]}**")
            prov = results[mname]["prov"]
            df = pd.DataFrame({"class": prov["classes"], "prob": prov["probs"]})
            df = df.sort_values("prob", ascending=False).head(10)
            st.bar_chart(df.set_index("class"), height=240)
            st.caption(f"예측: **{prov['pred_label']}** ({prov['confidence']:.1%})")

    render_lec("<strong>해석.</strong> 기본 TextCNN은 학습 시 지명을 직접 본 모델이라, 지명이 가려지면 광역시도 신뢰도가 무너집니다. 반면 Mix-training TextCNN은 학습 데이터의 50%가 이미 지명이 가려진 상태였기에, <strong>지명 없이도 stereotype으로 추론하는 법을 배웠습니다.</strong> → 박민선의 leakage 측정(14.5%)과 김민수의 학습 가능성(78.6%)이 합쳐져 <em>\"지명 의존도는 크지만, 학습 전략을 바꾸면 79% 가능\"</em>이라는 결론.")
