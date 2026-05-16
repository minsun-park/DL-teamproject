# DL-teamproject

고려대학교 인공지능융합 대학원 딥러닝 텀프로젝트 repo입니다.

## 주제

**한국어 페르소나 narrative로부터 인구통계 속성 다중 예측 + Stereotype Leakage 분석**

NVIDIA가 공개한 한국어 페르소나 데이터셋(Nemotron-Personas-Korea)을 사용해, LLM이 생성한 자연어 서사(직업·취미·스포츠·예술 등 11개 narrative 컬럼)만 보고 인물의 성별·연령대·광역시도·직업 카테고리를 예측하는 멀티태스크 분류 모델을 구현합니다. 그리고 어떤 narrative 컬럼이 어떤 demographic 정보를 가장 많이 누설하는지(Stereotype Leakage) 정량 분석합니다.

요건: **RNN · CNN · DNN 모두 활용** (TextCNN / BiLSTM / CNN-LSTM Hybrid 비교 예정)

## 데이터

- **원본**: [nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) (CC BY 4.0)
- **샘플링**: 전체 100만 행 중 10만 행 (random seed=42, 단순 random sampling)
- **파일**: `data/nemotron_100k_seed42.parquet` (191 MB, Git LFS로 관리)
- **MD5**: `63a16d805468ca76330fd5740b8501f4`
- **컬럼 수**: 26개 (uuid + 인구통계 14 + LLM 생성 narrative 11)

## 환경 셋업

### 1. Git LFS 설치 (필수)

데이터 파일이 Git LFS로 관리되어 있어, clone 전에 LFS가 설치되어 있어야 합니다.

```bash
# macOS
brew install git-lfs

# 그 외 OS는 https://git-lfs.com 에서 다운로드
```

설치 후:
```bash
git lfs install
```

### 2. Repo Clone

```bash
git clone https://github.com/minsun-park/DL-teamproject.git
cd DL-teamproject
```

Clone 시 LFS 파일(parquet 데이터)도 자동으로 받아집니다 (191MB).

### 3. Python 가상환경 + 패키지 설치

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas datasets pyarrow huggingface_hub
```

### 4. 데이터 로드 확인

```python
import pandas as pd
df = pd.read_parquet("data/nemotron_100k_seed42.parquet")
print(df.shape)  # (100000, 26)
print(df.columns.tolist())
```

## 데이터 재현 방법

만약 데이터를 직접 다시 만들고 싶다면:

```bash
python sample_data.py
```

위 스크립트는 동일 seed(42)로 10만 행을 다시 샘플링합니다. 
출력되는 MD5가 `63a16d805468ca76330fd5740b8501f4` 와 같으면 동일 데이터 확인.

처음 실행 시 Hugging Face에서 Nemotron 원본 데이터(약 2GB)를 다운로드하므로 5-15분 소요됩니다.

## 작업 진행 계획

- **1단계** · 셋이 함께 데이터 라벨 매핑·평가 기준 합의 (1-2일)
- **2단계** · 각자 풀 파이프라인 (TextCNN + BiLSTM + Hybrid 모두) 구현 (7-10일)
- **3단계** · 결과 모아서 비교 (2-3일)
- **4단계** · 통합 데모 코드 (한 명, 2-3일)
- **5단계** · 발표 자료 + 녹화 (모두 함께)

## 1단계 미팅에서 확정할 사항

다음 항목들은 아직 미정이며, 첫 sync 미팅에서 합의 후 별도 commit으로 정의 예정입니다.

- **Train/Val/Test 분할** — 현재안: 8/1/1, stratified split
- **라벨 매핑 규칙**
  - 연령대 5-bucket (20대/30대/40대/50대/60+ 등)
  - 직업 8-class (KOSCO 대분류 기반)
  - 라이프스타일 8-class (hobbies·sports 키워드 기반 weak labeling)
- **토크나이저** — Mecab 형태소 / 음절 / WordPiece 중 비교 후 결정
- **학습 환경** — Colab Pro / Kaggle GPU / 학교 클러스터 중 결정
- **평가 metric** — Macro-F1 기본, 학습 시간 보조

## 파일 구조

```
DL-teamproject/
├── README.md                              # 이 파일
├── .gitignore                             # venv, __pycache__ 등 제외
├── .gitattributes                         # parquet은 Git LFS로 관리
├── sample_data.py                         # 10만 행 샘플링 스크립트
└── data/
    └── nemotron_100k_seed42.parquet       # 샘플 데이터 (LFS)
```

향후 추가 예정:
- `mapping/` — 라벨 매핑 사전 (JSON)
- `src/` — 공통 코드 (data_loader, eval, tokenizer)
- `notebooks/` — EDA, 실험 노트북
- `models/` — 각자 모델 구현

## 라이선스

- 코드: 학내 과제용
- 데이터: 원본 Nemotron-Personas-Korea는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
