# -*- coding: utf-8 -*-
"""Leak-legal multi-model NWP fetch (Open-Meteo previous-runs API).

Fetch ECMWF-IFS / ICON / GFS / KMA-GDPS 100m+10m wind & 2m temp at
previous_day1 (D-1 lead, matches competition's 13:00 D-1 availability, lead 12-35h)
and previous_day2 (D-2, extra-conservative fallback) for the FULL span
2023-01-01 .. 2025-12-31 so train (2023-24) and test (2025) share one consistent
leak-free source. License: Open-Meteo CC-BY (public, free) -> rule-compliant.
Location = Taebaek Gadeoksan 37.2833,128.96. timezone Asia/Seoul => hourly == KST.
"""
import requests, time, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

LAT, LON = 37.2833, 128.96
MODELS = ["ecmwf_ifs025", "icon_global", "gfs_global", "kma_gdps"]
LEADS = [1, 2]                       # D-1 (matched/legal) + D-2 (conservative)
VARS = ["wind_speed_100m", "wind_direction_100m", "wind_speed_10m", "temperature_2m"]
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root


def fetch(model, d0, d1):
    hourly = []
    for v in VARS:
        for n in LEADS:
            hourly.append(f"{v}_previous_day{n}")
    p = dict(latitude=LAT, longitude=LON, hourly=",".join(hourly), models=model,
             start_date=d0, end_date=d1, timezone="Asia/Seoul", wind_speed_unit="ms")
    for attempt in range(6):
        try:
            r = requests.get(URL, params=p, timeout=180)
            if r.status_code == 200:
                return r.json()["hourly"]
            print("  retry", model, r.status_code, str(r.json())[:140])
        except Exception as e:
            print("  err", model, repr(e)[:100])
        time.sleep(6)
    raise RuntimeError(f"fail {model} {d0}..{d1}")


def build_span(d0, d1):
    frames = []
    for m in MODELS:
        h = fetch(m, d0, d1)
        df = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(h["time"])})
        short = {"ecmwf_ifs025": "ecmwf", "icon_global": "icon",
                 "gfs_global": "gfs", "kma_gdps": "kma"}[m]
        for v in VARS:
            vshort = (v.replace("wind_speed_100m", "ws100").replace("wind_direction_100m", "wd100")
                       .replace("wind_speed_10m", "ws10").replace("temperature_2m", "t2m"))
            for n in LEADS:
                key = f"{v}_previous_day{n}"
                if key in h:
                    df[f"{vshort}_{short}_d{n}"] = h[key]
        nn = df.filter(like="ws100").notna().mean().mean()
        print(f"  {m}: cols={df.shape[1]-1} ws100_nonnull={nn:.2f}")
        frames.append(df.set_index("forecast_kst_dtm"))
        time.sleep(1)
    out = pd.concat(frames, axis=1).reset_index()
    return out


if __name__ == "__main__":
    # Full consistent span. Split saved by train(2023-24) / test(2025) for pipeline use.
    print("=== fetching 2023-2025 (this takes ~1-2 min) ===")
    allspan = build_span("2023-01-01", "2025-12-31")
    allspan["year"] = allspan["forecast_kst_dtm"].dt.year
    tr = allspan[allspan["year"].isin([2023, 2024])].drop(columns="year")
    te = allspan[allspan["year"] == 2025].drop(columns="year")
    ftr = os.path.join(OUTDIR, "ext_multinwp_train.csv")
    fte = os.path.join(OUTDIR, "ext_multinwp_test.csv")
    tr.to_csv(ftr, index=False, encoding="utf-8-sig")
    te.to_csv(fte, index=False, encoding="utf-8-sig")
    print(f"saved {ftr} shape={tr.shape} range {tr.forecast_kst_dtm.min()}..{tr.forecast_kst_dtm.max()}")
    print(f"saved {fte} shape={te.shape} range {te.forecast_kst_dtm.min()}..{te.forecast_kst_dtm.max()}")
