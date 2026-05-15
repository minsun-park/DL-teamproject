"""
Nemotron-Personas-Korea에서 10만 행을 재현 가능한 방식으로 샘플링.
SEED 고정으로 누가 실행해도 동일한 10만 행이 나옴.
"""

from datasets import load_dataset
import pandas as pd
import os
import hashlib

SEED = 42
N_SAMPLES = 100_000
OUTPUT_PATH = "data/nemotron_100k_seed42.parquet"

print("Hugging Face에서 데이터셋 로드 중...")
ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train")
print(f"전체 행 수: {len(ds):,}")

print("Pandas로 변환 중...")
df = ds.to_pandas()

print(f"{N_SAMPLES:,} 행 샘플링 (seed={SEED})...")
df_sample = df.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df_sample.to_parquet(OUTPUT_PATH, index=False)

size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"\n저장 완료: {OUTPUT_PATH}")
print(f"파일 크기: {size_mb:.1f} MB")
print(f"행 수: {len(df_sample):,}")
print(f"컬럼 수: {len(df_sample.columns)}")

with open(OUTPUT_PATH, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"MD5: {md5}")