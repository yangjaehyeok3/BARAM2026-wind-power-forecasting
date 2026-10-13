# -*- coding: utf-8 -*-
"""공용 유틸: 데이터 로드, 기상 격자 집계, 대회 평가지표."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

# 태백 가덕산 발전소 대략 중심 좌표 (info.xlsx 터빈 좌표 평균)
FARM_LAT, FARM_LON = 37.2833, 128.9600

# 그룹별 터빈 모델 / 파워커브 파라미터 (info.xlsx)
GROUP_TURBINE = {
    "kpx_group_1": dict(model="VESTAS_V126", n=6, rated_kw=3600, hub=117, rotor=126),
    "kpx_group_2": dict(model="VESTAS_V126", n=6, rated_kw=3600, hub=117, rotor=126),
    "kpx_group_3": dict(model="UNISON_U136", n=5, rated_kw=4200, hub=117, rotor=136),
}


def load_labels():
    df = pd.read_csv(TRAIN_DIR / "train_labels.csv", encoding="utf-8-sig")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    return df.rename(columns={"kst_dtm": "forecast_kst_dtm"})


def load_sample_submission():
    df = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df


def _num_value_cols(df):
    drop = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id",
            "latitude", "longitude"}
    return [c for c in df.columns if c not in drop]


def aggregate_weather(df, prefix, how="mean"):
    """forecast_kst_dtm 단위로 격자 집계.
    how='mean'(전체평균) / 'nearest'(최근접 격자) / 'idw'(가까운 4격자 역거리가중)."""
    df = df.copy()
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    vcols = _num_value_cols(df)
    if how == "nearest":
        d2 = (df["latitude"] - FARM_LAT) ** 2 + (df["longitude"] - FARM_LON) ** 2
        df = df.assign(_d=d2)
        idx = df.groupby("forecast_kst_dtm")["_d"].idxmin()
        agg = df.loc[idx, ["forecast_kst_dtm"] + vcols].set_index("forecast_kst_dtm")
    elif how == "idw":
        d2 = (df["latitude"] - FARM_LAT) ** 2 + (df["longitude"] - FARM_LON) ** 2
        df = df.assign(_d=d2)
        df["_rank"] = df.groupby("forecast_kst_dtm")["_d"].rank(method="first")
        df = df[df["_rank"] <= 4].copy()                     # 가장 가까운 4격자
        df["_w"] = 1.0 / (df["_d"] + 1e-6)                   # 역거리제곱 가중
        for c in vcols:
            df[c] = df[c] * df["_w"]
        g = df.groupby("forecast_kst_dtm")
        agg = g[vcols].sum().div(g["_w"].sum(), axis=0)
    else:
        agg = df.groupby("forecast_kst_dtm")[vcols].mean()
    agg.columns = [f"{prefix}_{c}" for c in agg.columns]
    return agg.reset_index()


def load_weather(split="train", how="mean"):
    """LDAPS + GFS 병합된 시간단위 기상 피처 프레임 반환."""
    d = TRAIN_DIR if split == "train" else TEST_DIR
    ld = pd.read_csv(d / f"ldaps_{split}.csv", encoding="utf-8-sig")
    gf = pd.read_csv(d / f"gfs_{split}.csv", encoding="utf-8-sig")
    ld = aggregate_weather(ld, "ldaps", how)
    gf = aggregate_weather(gf, "gfs", how)
    m = ld.merge(gf, on="forecast_kst_dtm", how="outer")
    m = m.merge(spatial_gfs(split), on="forecast_kst_dtm", how="left")
    m = m.merge(grid_spread(split), on="forecast_kst_dtm", how="left")
    m = m.sort_values("forecast_kst_dtm").reset_index(drop=True)
    return _fix_corrupt(m)


# 격자 앙상블 spread(불확실성/엔트로피): 격자간·레벨간 풍속 불일치. 특히 g3 병목 개선.
def grid_spread(split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    ld = pd.read_csv(d / f"ldaps_{split}.csv", encoding="utf-8-sig")
    gf = pd.read_csv(d / f"gfs_{split}.csv", encoding="utf-8-sig")
    ld["forecast_kst_dtm"] = pd.to_datetime(ld["forecast_kst_dtm"])
    gf["forecast_kst_dtm"] = pd.to_datetime(gf["forecast_kst_dtm"])
    ld["_ws"] = np.hypot(ld["heightAboveGround_10_10u"], ld["heightAboveGround_10_10v"])
    g = ld.groupby("forecast_kst_dtm")
    o = pd.DataFrame(index=sorted(ld["forecast_kst_dtm"].unique()))
    o["ld_ws_std"] = g["_ws"].std()                              # 격자간 풍속 불일치
    o["ld_ws_range"] = g["_ws"].max() - g["_ws"].min()
    o["ld_sp_std"] = g["meanSea_0_prmsl"].std()
    gf["_ws850"] = np.hypot(gf["isobaricInhPa_850_u"], gf["isobaricInhPa_850_v"])
    gf["_ws100"] = np.hypot(gf["heightAboveGround_100_100u"], gf["heightAboveGround_100_100v"])
    gf["_ws700"] = np.hypot(gf["isobaricInhPa_700_u"], gf["isobaricInhPa_700_v"])
    gg = gf.groupby("forecast_kst_dtm")
    o["gf_ws850_std"] = gg["_ws850"].std()
    lv = pd.concat([gg["_ws100"].mean(), gg["_ws850"].mean(), gg["_ws700"].mean()], axis=1)
    o["lvl_ws_std"] = lv.std(axis=1)                             # 100/850/700 레벨간 불일치
    o = o.interpolate(limit_direction="both").ffill().bfill()    # 손상 hour NaN 대치
    return o.reset_index(names="forecast_kst_dtm")


# GFS 원본 3x3 격자 공간피처 (idw로 뭉개면 소실되는 공간경도/안정도). 지역특화·5폴드검증.
# 격자: West열 lon128.75={1,4,7}, East열 lon129.25={3,6,9}, 중심=grid5(발전소).
def spatial_gfs(split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    g = pd.read_csv(d / f"gfs_{split}.csv", encoding="utf-8-sig")
    g["forecast_kst_dtm"] = pd.to_datetime(g["forecast_kst_dtm"])
    WEST, EAST = [1, 4, 7], [3, 6, 9]
    def piv(c): return g.pivot_table(index="forecast_kst_dtm", columns="grid_id", values=c)
    P = piv("meanSea_0_prmsl"); T2 = piv("heightAboveGround_2_2t")
    T8 = piv("isobaricInhPa_850_t"); T7 = piv("isobaricInhPa_700_t")
    out = pd.DataFrame({
        # 횡산(서-동) 해면기압 경도 [Pa]: +면 서고→활강 서풍(양간지풍/gap-wind 구동력)
        "prmsl_grad_we": P[WEST].mean(axis=1) - P[EAST].mean(axis=1),
        # 저층 역전/안정도 [K]: 850hPa기온 - 지상2m기온, +면 역전(디커플링)
        "inv_850_2m": T8[5] - T2[5],
        # 상층 정적안정도 [K]: (700-850기온)+건조단열(14.7), +면 안정
        "stab_700_850": (T7[5] - T8[5]) + 14.7,
    })
    return out.reset_index()


# 물리적으로 불가능한 센티넬(0 등) → NaN 후 시간보간. test 3시간 sp=0/t=0 손상 대응.
_PHYS_FLOOR = {  # 컬럼 부분일치: 이 값 미만이면 손상으로 간주
    "surface_0_sp": 5e4, "meanSea_0_prmsl": 5e4,          # 기압 Pa
    "heightAboveGround_2_t": 200.0, "heightAboveGround_2_2t": 200.0,   # 기온 K
    "heightAboveGround_2_dpt": 180.0, "heightAboveGround_2_2d": 180.0,  # 이슬점 K
    "isobaricInhPa_850_t": 200.0, "isobaricInhPa_700_t": 200.0, "isobaricInhPa_500_t": 180.0,
}


def _fix_corrupt(m):
    m = m.copy()
    for key, floor in _PHYS_FLOOR.items():
        for c in [col for col in m.columns if col.endswith(key)]:
            bad = m[c] < floor
            if bad.any():
                m.loc[bad, c] = np.nan
                m[c] = m[c].interpolate(limit_direction="both").ffill().bfill()
    return m


def metric(answer_df, pred_df):
    """대회 공식 지표. (total, one_minus_nmae, ficr) 반환. 그룹별 세부도 함께."""
    group_nmae, group_ficr = [], []
    for col in TARGET_COLS:
        actual = answer_df[col].to_numpy(dtype=float)
        forecast = pred_df[col].to_numpy(dtype=float)
        cap = CAPACITY_KWH[col]
        valid = actual >= cap * 0.10
        actual, forecast = actual[valid], forecast[valid]
        err = np.abs(forecast - actual) / cap
        group_nmae.append(np.mean(err))
        unit = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        group_ficr.append(np.sum(actual * unit) / np.sum(actual * 4.0))
    one_minus_nmae = 1 - np.mean(group_nmae)
    ficr = np.mean(group_ficr)
    total = 0.5 * one_minus_nmae + 0.5 * ficr
    return dict(total=total, one_minus_nmae=one_minus_nmae, ficr=ficr,
                group_nmae=group_nmae, group_ficr=group_ficr)
