"""데이터 로드 · 5-task 라벨 정의 · stratified split · 3-condition 마스킹.
본인 1차/2차 노트북에서 검증된 코드를 모듈로 정리.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 라이프스타일 키워드 사전 (weak labeling)
# ============================================================
LIFESTYLE_KEYWORDS = {
    "패션-스포츠":   ["헬스","러닝","골프","크로스핏","운동복","트레이닝","PT","필라테스"],
    "라이프-DIY":    ["등산","캠핑","정원","목공","텃밭","낚시","산행","야영"],
    "패션-스타일":   ["쇼핑","옷","코디","액세서리","패션","스타일링"],
    "뷰티-스타일":   ["화장품","메이크업","피부관리","네일","스킨케어"],
    "음식-미식":     ["맛집","디저트","와인","베이킹","요리","카페","브런치"],
    "학습-자기계발": ["독서","강의","공부","자격증","어학","책을","학습"],
    "가족-여가":     ["가족여행","자녀","손주","가족과","아이와","손녀","손자"],
    "디지털-IT":     ["게임","코딩","스마트폰","유튜브","컴퓨터","인터넷","온라인"],
}
LIFE_LABELS = list(LIFESTYLE_KEYWORDS.keys())
LIFE_MAP = {l: i for i, l in enumerate(LIFE_LABELS)}

AGE_BINS = [0, 29, 39, 49, 59, 69, 200]
AGE_LABELS = ["20대", "30대", "40대", "50대", "60대", "70+"]

MARITAL_LABELS = ["배우자있음", "미혼", "사별", "이혼"]
MARITAL_MAP = {l: i for i, l in enumerate(MARITAL_LABELS)}


def label_lifestyle(row):
    text = " ".join([
        str(row.get("hobbies_and_interests","")),
        str(row.get("sports_persona","")),
        str(row.get("hobbies_and_interests_list","")),
    ])
    counts = {c: sum(text.count(kw) for kw in kws) for c, kws in LIFESTYLE_KEYWORDS.items()}
    best_cat, best_n = max(counts.items(), key=lambda x: x[1])
    return LIFE_MAP[best_cat] if best_n > 0 else -1


def load_and_label(parquet_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """parquet 로드 → 5 task 라벨 부여 → drop → 메타 반환."""
    df = pd.read_parquet(parquet_path)
    print(f"원본 shape: {df.shape}")

    df["y_sex"] = df["sex"].map({"여자": 0, "남자": 1})

    df["age_bucket"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["y_age"] = df["age_bucket"].map({l: i for i, l in enumerate(AGE_LABELS)}).astype(int)

    prov_list = sorted(df["province"].unique().tolist())
    df["y_prov"] = df["province"].map({p: i for i, p in enumerate(prov_list)})

    df["y_life"] = df.apply(label_lifestyle, axis=1)
    n_before = len(df)
    df = df[df["y_life"] >= 0].reset_index(drop=True)
    print(f"라이프 매칭 실패 drop: {n_before - len(df)} → {df.shape}")

    df["y_marital"] = df["marital_status"].map(MARITAL_MAP)

    meta = {
        "prov_list": prov_list,
        "age_labels": AGE_LABELS,
        "life_labels": LIFE_LABELS,
        "marital_labels": MARITAL_LABELS,
        "task_names": ["sex", "age", "prov", "life", "marital"],
        "task_config": {
            "sex":     {"col": "y_sex",     "n": 2},
            "age":     {"col": "y_age",     "n": 6},
            "prov":    {"col": "y_prov",    "n": len(prov_list)},
            "life":    {"col": "y_life",    "n": len(LIFE_LABELS)},
            "marital": {"col": "y_marital", "n": 4},
        },
    }
    return df, meta


# ============================================================
# 분할
# ============================================================
def stratified_split(df: pd.DataFrame, seed: int = 42):
    """sex × age 기준 stratified 80/10/10."""
    df = df.copy()
    df["_strat"] = df["y_sex"] * 6 + df["y_age"]
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["_strat"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["_strat"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# 마스킹
# ============================================================
DEMO_REGEX = re.compile(
    r"\d{1,2}대|"
    r"아내|남편|아들|딸|손주|손녀|손자|"
    r"며느리|사위|시어머니|시아버지|어머니|아버지"
)


def build_geo_pattern(df: pd.DataFrame, prov_list: list[str]) -> tuple[re.Pattern, list[str]]:
    """광역시도 + 시군구 + 약칭으로 마스킹 정규식 빌드."""
    districts_raw = df["district"].dropna().unique().tolist()
    sigungu = set()
    for d in districts_raw:
        parts = d.split("-")
        if len(parts) >= 2:
            full = parts[-1]
            sigungu.add(full)
            stripped = re.sub(r"(시|군|구)$", "", full)
            if len(stripped) >= 2:
                sigungu.add(stripped)
    prov_aliases = {
        "전라남":["전남"],"전라북":["전북"],"경상남":["경남"],
        "경상북":["경북"],"충청남":["충남"],"충청북":["충북"],
    }
    geo_terms = set(prov_list) | sigungu
    for p, aliases in prov_aliases.items():
        geo_terms.update(aliases)
    terms_sorted = sorted(geo_terms, key=len, reverse=True)
    pattern = re.compile("|".join(terms_sorted))
    return pattern, terms_sorted


def mask_text(text, geo_pattern, condition="raw"):
    if pd.isna(text):
        return ""
    t = str(text)
    if condition == "raw":
        return t
    t = DEMO_REGEX.sub("[DEMO]", t)
    if condition == "masked":
        return t
    t = geo_pattern.sub("[LOC]", t)
    return t


NARRATIVE_COLS = [
    "persona", "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "family_persona",
    "cultural_background", "skills_and_expertise",
    "hobbies_and_interests", "career_goals_and_ambitions",
]


def build_input(row, geo_pattern, condition="raw", sep=" [SEP] "):
    parts = [mask_text(row[c], geo_pattern, condition) for c in NARRATIVE_COLS]
    return sep.join(parts)
