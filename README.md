# DScover 가이드 프로젝트 — Predicting Student Health Risk

Kaggle Playground Series S6E7 대회를 바탕으로 진행한 26-2 Dscover 가이드프로젝트 파이프라인입니다.

## 최종 결과

| 지표 | 점수 |
|---|---|
| Private Score | **0.95045** |
| Public Score | 0.95050 |
| 평가 지표 | Balanced Accuracy |
| 베이스라인 대비 | +0.00100 (CatBoost K=10) |

## 파이프라인 구조

```
전처리 & 피처 엔지니어링
        ↓
┌───────────────────────────────────┐
│  TE-HGBC           RealMLP Bag   FT-Transformer     │
│  (10-Fold)       (시드 42/123/2026)  (7-Fold)        │
└───────────────────────────────────┘
        ↓
   3종 균등 블렌딩
        ↓
   submission.csv
```

## 핵심 방법론

**Per-value Target Encoding**
수치형 변수도 `astype(str)`로 변환해 exact value 기준 인코딩. Synthetic 데이터의 반복값 패턴을 활용해 GBDT 히스토그램 비닝의 정밀도 손실 보완. 가장 큰 단일 성능 도약(+0.00058).

**Unweighted 학습 + Decision-time 보정**
`class_weight="balanced"` 대신 unweighted로 학습해 진짜 사후확률을 학습시킨 뒤, 예측 시점에 클래스별 승수를 grid search로 탐색.

**구조적으로 다른 모델 패밀리 블렌딩**
GBDT끼리는 예측 일치율 97~99%로 블렌딩 이득이 없었고, RealMLP↔FT-Transformer 간 93.7% 일치율에서 처음으로 유효한 블렌딩 이득 발생.

**단순 균등 평균**
비율 최적화(SLSQP, Nelder-Mead)는 OOF에 과적합되어 역효과. 1:1:1 균등 평균이 일관되게 가장 강했음.

## 모델 구성

| 모델 | OOF | 역할 |
|---|---|---|
| TE-HGBC             | 0.95025 | Per-value TE + HistGradientBoosting, unweighted |
| RealMLP Seed Bag    | 0.95067 | PBLD 임베딩 신경망, 시드 3개 평균 |
| FT-Transformer      | 0.95031 | Feature Tokenizer + Transformer |
| **3종 균등 블렌딩**  | **0.95064** | **→ Private 0.95045** |

## 실행 방법

### 환경
- Google Colab (GPU T4 권장)
- Python 3.10+

### 패키지 설치
```bash
pip install masamlp==0.3.0 catstat==0.4.0 kagglehub
```

### Kaggle API 설정
1. [kaggle.com](https://kaggle.com) → Settings → API → Create New Token
2. Colab Secrets에 `KAGGLE_USERNAME`, `KAGGLE_KEY` 등록
3. [대회 규칙](https://www.kaggle.com/competitions/playground-series-s6e7/rules)에서 Accept Rules

### 실행
```python
# 경로 설정
DATA_DIR = "/content/drive/MyDrive/your-folder"
SAVE_DIR = "/content/drive/MyDrive/your-folder"

# 재학습 플래그 (npy 파일 있으면 자동 로드)
FORCE_RETRAIN_TE  = False
FORCE_RETRAIN_MLP = False
FORCE_RETRAIN_FTT = False
```

`final_pipeline_fixed.ipynb`를 순서대로 실행하면 됩니다.

### 소요 시간 (T4 기준)
| 상황 | 시간 |
|---|---|
| npy 파일 전부 있음 | ~5분 |
| 전체 처음부터 학습 | ~1시간 20분 |

## 공유 파일 목록

팀원 간 공유 시 아래 npy 파일을 함께 전달하세요.

```
oof_te_sleep_bmi.npy / test_pred_te_sleep_bmi.npy / te_sleep_bmi_weights.npy
oof_realmlp_bag.npy  / test_pred_realmlp_bag.npy
oof_ftt_v2.npy       / test_pred_ftt_v2.npy       / ftt_v2_nm_weights.npy
y_true.npy / label_encoder_classes.pkl
```
