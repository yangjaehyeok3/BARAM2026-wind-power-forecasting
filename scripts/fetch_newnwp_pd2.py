# -*- coding: utf-8 -*-
"""신규 NWP 8종 수집 (open-meteo Historical Forecast API, previous_day2 = 하루전 발표분).

모델: jma_gsm, jma_msm, kma_gdps, kma_ldps, ukmo_global_deterministic_10km,
      cma_grapes_global, gem_global, bom_access_global
변수: wind_speed_10m, wind_direction_10m, temperature_2m (+ 가능시 gusts)
기간: 2022-01-01 ~ 2025-12-31 (연 단위 청크, 이어받기 지원)
합법성: previous_day2 = 대상일 D 기준 D-1 발표 예보. 실적/정비 정보 아님.
출력: data_ext/newnwp/{model}.csv, 최종 병합 newnwp_pd2_all.csv
"""
import warnings; warnings.filterwarnings("ignore")
import sys, io, os, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import requests
import numpy as np, pandas as pd

LAT, LON = 37.2833, 128.96
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ODIR = "data_ext/newnwp_pd2"
os.makedirs(ODIR, exist_ok=True)
MODELS = ["jma_gsm", "jma_msm"]
BASEV = ["wind_speed_10m_previous_day2", "wind_direction_10m_previous_day2",
         "temperature_2m_previous_day2"]
GUSTM = {"kma_gdps", "kma_ldps", "cma_grapes_global", "gem_global", "bom_access_global"}
YEARS = [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
         ("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-12-31")]


def fetch(model, a, b, tries=5):
    v = list(BASEV) + (["wind_gusts_10m_previous_day2"] if model in GUSTM else [])
    for k in range(tries):
        try:
            r = requests.get(URL, params=dict(latitude=LAT, longitude=LON, hourly=",".join(v),
                                              start_date=a, end_date=b, models=model,
                                              timezone="Asia/Seoul"), timeout=120)
            if r.status_code == 200:
                j = r.json()["hourly"]
                d = pd.DataFrame(j)
                d = d.rename(columns={c: c.replace("_previous_day2", "") for c in d.columns})
                d["forecast_kst_dtm"] = pd.to_datetime(d["time"])
                return d.drop(columns=["time"])
            if r.status_code == 429:
                time.sleep(30 * (k + 1)); continue
            print("   %s %s..%s HTTP %s %s" % (model, a, b, r.status_code, r.text[:90]))
            return None
        except Exception as e:
            print("   %s retry %d: %s" % (model, k, str(e)[:70]))
            time.sleep(5 * (k + 1))
    return None


for m in MODELS:
    out = os.path.join(ODIR, "%s.csv" % m)
    if os.path.exists(out):
        print("%s: 이미 있음 (skip)" % m); continue
    parts = []
    for a, b in YEARS:
        d = fetch(m, a, b)
        if d is None or len(d) == 0:
            print("  %s %s 실패/빈값" % (m, a[:4])); continue
        nn = d.drop(columns=["forecast_kst_dtm"]).notna().mean().mean()
        print("  %s %s: %d행 유효 %.0f%%" % (m, a[:4], len(d), 100 * nn))
        if nn > 0.05:
            parts.append(d)
        time.sleep(1.5)
    if not parts:
        print("%s: 전 연도 실패" % m); continue
    z = pd.concat(parts, ignore_index=True).drop_duplicates("forecast_kst_dtm").sort_values("forecast_kst_dtm")
    z.columns = ["%s_%s" % (m.split("_")[0], c) if c != "forecast_kst_dtm" else c for c in z.columns]
    z.to_csv(out, index=False, encoding="utf-8-sig")
    print("%s -> %s (%d행 %s..%s)" % (m, out, len(z), z.forecast_kst_dtm.min(), z.forecast_kst_dtm.max()))

# 병합
frames = []
for m in MODELS:
    f = os.path.join(ODIR, "%s.csv" % m)
    if os.path.exists(f):
        d = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
        d = d.set_index("forecast_kst_dtm")
        d.columns = ["%s__%s" % (m, c.split("_", 1)[-1]) for c in d.columns]
        frames.append(d)
if frames:
    Z = pd.concat(frames, axis=1).sort_index()
    Z.to_csv("newnwp_pd2_all.csv", encoding="utf-8-sig")
    print()
    print("병합 완료 newnwp_pd2_all.csv: %d행 %d열" % (Z.shape[0], Z.shape[1]))
    print("연도별 유효율:")
    print(Z.notna().groupby(Z.index.year).mean().mean(1).round(3).to_string())
