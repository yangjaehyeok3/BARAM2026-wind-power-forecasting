# -*- coding: utf-8 -*-
"""광역 예보장 수집 — 사이트 1점이 아니라 한반도·황해·일본 서부를 격자로 샘플.

착안(사용자): 2025 서풍 91.6% → 서쪽 상류의 기상이 수 시간 뒤 가덕산에 도달.
지금까지 쓴 격자는 반경 6km(LDAPS 16셀)뿐 = 종관 규모 정보 없음.
여기서는 100~600km 규모로 15개 지점을 뽑아 상류/하류·기압경도·전선 위치를 잡는다.

지점 설계 (가덕산 37.28N 128.96E 기준):
  상류(서·북서) 4점: 황해~중국연안  / 근상류(서) 3점: 한반도 내륙
  남·북 4점: 종관 기압경도용        / 하류(동) 2점: 동해 (대조군)
  사이트 1점
합법성: previous_day2 = 대상일 하루 전 발표분. 전부 예보값(관측 아님).
출력: data_ext/regional/*.csv, 병합 regional_jma_pd2_all.csv
"""
import warnings; warnings.filterwarnings("ignore")
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import requests
import numpy as np, pandas as pd

ODIR = "data_ext/regional_jma_pd2"
os.makedirs(ODIR, exist_ok=True)
SITE = (37.2833, 128.96)
# (이름, lat, lon) — 상류 우선
PTS = [
    ("site", 37.28, 128.96),
    ("w1", 37.28, 127.5), ("w2", 37.28, 126.0), ("w3", 37.28, 124.5), ("w4", 37.28, 122.5),
    ("nw1", 38.5, 126.5), ("nw2", 39.5, 124.5),
    ("sw1", 36.0, 126.5), ("sw2", 35.0, 125.0),
    ("n1", 39.0, 128.5), ("s1", 35.5, 128.5),
    ("e1", 37.28, 131.0), ("e2", 37.28, 133.5),
    ("ne1", 38.8, 130.5), ("se1", 35.5, 131.0),
]
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
VARS = ("wind_speed_10m_previous_day2,wind_direction_10m_previous_day2,"
        "surface_pressure_previous_day2,temperature_2m_previous_day2")
MODEL = "jma_gsm"
YEARS = [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
         ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")]


def fetch(lat, lon, a, b, tries=5):
    for k in range(tries):
        try:
            r = requests.get(URL, params=dict(latitude=lat, longitude=lon, hourly=VARS,
                                              start_date=a, end_date=b, models=MODEL,
                                              timezone="Asia/Seoul"), timeout=120)
            if r.status_code == 200:
                d = pd.DataFrame(r.json()["hourly"])
                d = d.rename(columns={c: c.replace("_previous_day2", "") for c in d.columns})
                d["forecast_kst_dtm"] = pd.to_datetime(d["time"])
                return d.drop(columns=["time"])
            if r.status_code == 429:
                time.sleep(40 * (k + 1)); continue
            print("   HTTP %s %s" % (r.status_code, r.text[:80])); return None
        except Exception as e:
            print("   retry %d %s" % (k, str(e)[:60])); time.sleep(6 * (k + 1))
    return None


for nm, la, lo in PTS:
    out = os.path.join(ODIR, "%s.csv" % nm)
    if os.path.exists(out):
        print("%s: 있음(skip)" % nm); continue
    parts = []
    for a, b in YEARS:
        d = fetch(la, lo, a, b)
        if d is None or d.drop(columns=["forecast_kst_dtm"]).notna().mean().mean() < 0.05:
            print("  %s %s 실패/빈값" % (nm, a[:4])); continue
        parts.append(d)
        time.sleep(1.2)
    if not parts:
        print("%s 전 연도 실패" % nm); continue
    z = pd.concat(parts, ignore_index=True).drop_duplicates("forecast_kst_dtm").sort_values("forecast_kst_dtm")
    z.to_csv(out, index=False, encoding="utf-8-sig")
    print("%s(%.2f,%.2f) -> %d행 유효 %.0f%%"
          % (nm, la, lo, len(z), 100 * z.drop(columns=["forecast_kst_dtm"]).notna().mean().mean()))

frames = []
for nm, la, lo in PTS:
    f = os.path.join(ODIR, "%s.csv" % nm)
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]).set_index("forecast_kst_dtm")
    d.columns = ["%s__%s" % (nm, c) for c in d.columns]
    frames.append(d)
Z = pd.concat(frames, axis=1).sort_index()
Z.to_csv("regional_jma_pd2_all.csv", encoding="utf-8-sig")
print()
print("병합 regional_jma_pd2_all.csv: %d행 %d열" % Z.shape)
print("연도별 유효율:")
print(Z.notna().groupby(Z.index.year).mean().mean(1).round(3).to_string())
