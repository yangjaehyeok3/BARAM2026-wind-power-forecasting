# 재현 가이드 — 파이프라인 · 환경 · 결정성

> 대회 데이터와 학습 가중치는 재배포 금지 규정상 저장소에 없다.
> 아래는 **전체 파이프라인이 어떻게 실행되는지**와, **최종 제출 파일이 오차 0으로 재현됨을 확인**한 경로의 기록이다.

## 1. 파이프라인 한눈에

```
대회 제공 데이터 (LDAPS·GFS·SCADA·발전량 라벨)
   +  외부 수집 데이터 (전량 D-2 발표분 — docs/EXTERNAL_DATA.md)
        ▼  src/features.py · scripts/ (피처 빌드·수집)
   도메인 피처 186열 + 격자 토큰 캐시
        ▼  train.py → src/dl/ (앵커 학습, GPU ~28분/앵커)
   앵커 모델 산출물 (OOF·테스트 예측)
        ▼  inference.py (완전 결정적 — 난수·GPU·외부호출 없음)
   최종 제출 파일
```

최종 조립(`inference.py`)은 기저 예측 위에 ① 앙상블의 **순서 정보만** 레벨중립 투영으로 이식하고
② 근정격 밴드 보정(리더보드 통제실험으로 측정한 응답곡선 기반)을 더한다. 모든 단계 `[0, 설비용량]` 클리핑.

## 2. 실행

```bash
pip install -r requirements.txt   # torch는 CUDA 빌드 별도 설치
python train.py --all             # 학습: 피처 캐시 + 앵커 모델 (GPU)
python inference.py --out submission_final.csv   # 추론: CPU 수 초
```

## 3. 결정성 — 단계별로 다르게 설계

| 단계 | 결정성 | 비고 |
|---|---|---|
| 피처 캐시 빌드 | 결정적 | |
| 앵커 학습 | 시드 고정, GPU 연산 비결정성 잔존 | cuDNN/atomic reduction |
| 최종 조립 (`inference.py`) | **완전 결정적** | 난수 없음, float64 고정 누적 |

이 구조 덕에 학습 산출물만 있으면 `inference.py` 한 줄로 최종 제출 파일이 **비트 일치(최대차 ~1e-11 kWh)** 로 재현된다 — 실제 실행 로그로 확인한 사실이다.
세부 주의점 하나: 앵커 산출물이 float32라 평균을 float32로 누적하면 ~1e-3 kWh가 어긋난다 — float64 승격 후 계산해야 비트 일치가 된다.

## 4. 개발 환경

| 항목 | 값 |
|---|---|
| OS / GPU | Windows 11 · NVIDIA RTX 5090 32GB (CUDA 12.8) |
| Python | 3.11 |
| 핵심 | torch 2.11, numpy 2.4, pandas 2.3, scipy 1.17, scikit-learn 1.6 |
| 보조 실험 | lightgbm 4.6 · xgboost 3.2 · tabpfn 2.0 (모델 비교 실험용, 최종 경로 미사용) |
| 외부 NWP 처리 | xarray · netCDF4 · pyarrow |

전체 목록: [`requirements.txt`](../requirements.txt)

## 5. 이 저장소에서 재현 가능한 범위

| 수준 | 가능 여부 | 조건 |
|---|---|---|
| 코드 구조·레시피 열람 | ✅ 전부 | 이 저장소만으로 |
| 피처 빌드 → 학습 → 추론 전 경로 실행 | ⚙️ 가능 | 대회 제공 데이터 + 외부 데이터 재수집(docs/EXTERNAL_DATA.md의 스크립트) 필요 |
| 제출 파일 오차 0 재현 | 기록으로 확인 | 학습 산출물 필요 — 2026-08-14 실행 로그로 비트 일치 확인 |
