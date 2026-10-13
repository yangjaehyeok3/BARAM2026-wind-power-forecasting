# 외부 데이터 — 출처 · 수집 · 합법성 검증

외부 예보는 전량 **대상일 이틀 전(D-2) 발표분**만 사용했다 — 대회 기준시점(대상일 전일 14:00 KST)보다 최소 15시간 이전에 공개된 자료라는 뜻이다. 수집 스크립트가 요청 파라미터에 발표시점(`previous_day2`)을 명시하고, 응답의 시각 메타데이터를 그대로 보존해 사후 검증이 가능하게 했다.

## 1. 최종 모델이 실제로 사용한 데이터

| 데이터 | 출처 | 수집 스크립트 |
|---|---|---|
| JMA GSM/MSM 등 다중 글로벌 모델, 지상 10m 바람 | Open-Meteo Previous Runs API | `scripts/fetch_newnwp_pd2.py` |
| JMA GSM 한반도·황해 15개 격자점 | Open-Meteo | `scripts/fetch_regional_jma_pd2.py` |
| ECMWF/ICON/GFS 100m 바람 | Open-Meteo | `src/fetch_ecmwf_pd1.py` (pd2 모드) |
| GEFS 앙상블 (불확실성 폭) | NOAA AWS Open Data | `src/gefs_fetch.py` |
| UM-KR 격자 파생 11열 | 기상청 | 원본 격자 샤드(440MB)에서 추출 |

**왜 JMA인가**: ① 2022~2025 전 기간 아카이브 존재(ECMWF/ICON은 커버리지 탈락) ② 발표시점 검증 가능 ③ LDAPS와 **독립된 물리 코어** — 탈상관 오차가 앙상블 가치를 만든다. 효과: 나셀 풍속 MAE −7.3%, 총점 +0.0018.

## 2. 수집했으나 채택하지 않은 데이터 (투명성 기록)

| 데이터 | 미채택 사유 |
|---|---|
| JMA MSM (RISH 아카이브) · 기상청 MOS | 통제 실험에서 이득 없음 |
| 기상청 지상관측 (ASOS 태백·AWS 3지점) | 구세대 모델에서만 사용, 최종 경로 제외 |
| Open-Meteo previous_day1 계열 | **자진 폐기** — D-1 발표분은 기준시점 위반 소지가 있어 전량 D-2로 재수집 |
| CIRA MLWP (GraphCast/Pangu) | **자진 폐기** — S3 객체 게시시각을 직접 조회해 2025년 예보가 수 주 뒤 일괄 백필된 것을 확인, 발표시점 요건 미충족 |

마지막 두 건은 규정 위반이 아니라고 우길 수도 있는 회색지대였지만, **공개시각을 직접 검증해 스스로 제외**했다.

## 3. 라이선스

| 출처 | 라이선스 |
|---|---|
| Open-Meteo | CC BY-NC 4.0 (원출처: 각국 기상기관 공개자료) |
| NOAA GEFS | Public Domain (U.S. Government Work) |
| 기상청 API허브 / 기상자료개방포털 | 공공누리 제1유형 (출처표시) |

## 4. 재수집

```bash
python scripts/fetch_regional_jma_pd2.py   # JMA 지역 격자 D-2 발표분
python src/gefs_fetch.py                   # GEFS 앙상블
```
