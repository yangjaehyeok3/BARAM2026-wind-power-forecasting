# -*- coding: utf-8 -*-
"""물리 기반 파생변수 생성 (태백 가덕산 전용). train/inference 공용.
모든 피처는 예보(LDAPS/GFS)에서만 생성 — 실측/타깃 시차 미사용(리키지 방지)."""
import numpy as np
import pandas as pd

HUB = 117.0            # 세 그룹 모두 hub 117m
ALPHA = 0.12           # 멱법칙 지수: SCADA 26,207시간으로 최적화(편향 최소, MAE최소). 산악=0.2~0.25는 오답
import os as _os
_ALPHAV = _os.environ.get("ALPHAV", "0") == "1"   # 풍속의존 α (아래 ws_hub 참조)
R_DRY = 287.05
R_VAP = 461.5


def _wind_speed(u, v):
    return np.hypot(u, v)


def _wind_dir(u, v):
    # 기상학적 풍향(바람이 불어오는 방향, degree)
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def air_density(temp_k, pressure_pa, rh_pct):
    """습윤공기 밀도 (kg/m^3). rh_pct 0~100."""
    t = np.clip(temp_k, 230, 330)
    p = np.clip(pressure_pa, 5e4, 1.1e5)
    # 포화수증기압 (Tetens, Pa)
    tc = t - 273.15
    es = 610.78 * np.exp(17.27 * tc / (tc + 237.3))
    e = np.clip(rh_pct, 0, 100) / 100.0 * es
    pd_ = p - e
    return pd_ / (R_DRY * t) + e / (R_VAP * t)


def build_features(wx: pd.DataFrame) -> pd.DataFrame:
    """common.load_weather() 결과(시간단위 LDAPS+GFS 병합)를 받아 피처 프레임 반환."""
    d = wx.copy().sort_values("forecast_kst_dtm").reset_index(drop=True)
    dt = pd.to_datetime(d["forecast_kst_dtm"])
    f = pd.DataFrame({"forecast_kst_dtm": dt})

    # ---- LDAPS 풍속 (주력) ----
    lu, lv = d["ldaps_heightAboveGround_10_10u"], d["ldaps_heightAboveGround_10_10v"]
    f["ws10"] = _wind_speed(lu, lv)
    f["wd10"] = _wind_dir(lu, lv)
    f["wd_sin"] = np.sin(np.radians(f["wd10"]))
    f["wd_cos"] = np.cos(np.radians(f["wd10"]))
    # ★ALPHAV (2026-07-31 감사): 고정 α 는 **강풍대에서 허브풍속을 크게 과대추정**한다.
    #   SCADA 로 역산한 실제 α (ws10 -> 117m 나셀풍속) 는 풍속에 따라 두 배 넘게 변한다:
    #     ws_hub 0-4 → 0.1606 · 4-6 → 0.0937 · 6-8 → 0.1254 · 10-12 → 0.1054
    #            12-14 → 0.0843 · **14+ → 0.0589**  (전체 중앙 0.1139 ≈ 현행 0.12)
    #   그 결과 편향(예측−실측)이 12-14 구간 **+0.986**, 14+ 구간 **+2.301 m/s**.
    #   = 근정격(채점 집중 구간)을 직접 때리는 결함. nrM3 가 −0.048 로 눌러야 했던 이유.
    #   교정: ws10 구간별 α 를 SCADA 로 적합(연도교차 검증 3셀 전부 MAE −0.05,
    #        고풍속 편향 +1.557 → +0.278 로 82% 감소).
    #   주의: 고정 α 는 ws10 의 단조변환이라 **순서(NMAE)를 못 바꾸고 레벨(FICR)만** 건드린다.
    #        가변 α 는 구간별로 다르게 늘려 순서까지 바꾼다.
    #   ALPHAV=0 이면 기존과 비트단위 동일.
    if _ALPHAV:
        _b = np.array([0., 1.5, 2.5, 3.5, 4.5, 6., 8., 50.])
        _t = np.array([0.3895, 0.1669, 0.0849, 0.1040, 0.1260, 0.1212, 0.0815])
        _i = np.clip(np.digitize(np.nan_to_num(f["ws10"].to_numpy()), _b) - 1, 0, len(_t) - 1)
        f["ws_hub"] = f["ws10"] * (HUB / 10.0) ** _t[_i]
    else:
        f["ws_hub"] = f["ws10"] * (HUB / 10.0) ** ALPHA
    f["ws_hub2"] = f["ws_hub"] ** 2
    f["ws_hub3"] = f["ws_hub"] ** 3          # 출력 비례항 (P ∝ v^3)

    # 50m 바람 (U/V max·min → 평균·레인지=돌풍성)
    u50 = 0.5 * (d["ldaps_heightAboveGround_50_50MUmax"] + d["ldaps_heightAboveGround_50_50MUmin"])
    v50 = 0.5 * (d["ldaps_heightAboveGround_50_50MVmax"] + d["ldaps_heightAboveGround_50_50MVmin"])
    f["ws50"] = _wind_speed(u50, v50)
    f["ws50_urange"] = d["ldaps_heightAboveGround_50_50MUmax"] - d["ldaps_heightAboveGround_50_50MUmin"]
    f["ws50_vrange"] = d["ldaps_heightAboveGround_50_50MVmax"] - d["ldaps_heightAboveGround_50_50MVmin"]
    f["gust_range50"] = np.hypot(f["ws50_urange"], f["ws50_vrange"])   # 돌풍성/난류 지표
    # 5m 경계층 바람
    f["ws5_bl"] = _wind_speed(d["ldaps_heightAboveGround_5_XBLWS"], d["ldaps_heightAboveGround_5_YBLWS"])
    # 연직 시어 (10m→50m)
    f["shear_10_50"] = (f["ws50"] - f["ws10"]) / (np.log(50/10))

    # ---- 공기밀도 & 물리출력 ----
    f["rho"] = air_density(d["ldaps_heightAboveGround_2_t"], d["ldaps_surface_0_sp"], d["ldaps_heightAboveGround_2_r"])
    f["wind_power_density"] = 0.5 * f["rho"] * f["ws_hub3"]        # W/m^2 스케일
    f["rho_ws_hub3"] = f["rho"] * f["ws_hub3"]

    # ---- 기온/습도/안정도 ----
    f["t2"] = d["ldaps_heightAboveGround_2_t"]
    f["dewpoint"] = d["ldaps_heightAboveGround_2_dpt"]
    f["dewpoint_spread"] = d["ldaps_heightAboveGround_2_t"] - d["ldaps_heightAboveGround_2_dpt"]
    f["rh"] = d["ldaps_heightAboveGround_2_r"]
    f["q"] = d["ldaps_heightAboveGround_2_q"]
    f["blh"] = d["ldaps_etc_0_blh"]
    f["sp"] = d["ldaps_surface_0_sp"]
    f["prmsl"] = d["ldaps_meanSea_0_prmsl"]
    # 결빙 리스크 (기온<1C & 습도>90 & 강수)
    f["icing"] = ((d["ldaps_heightAboveGround_2_t"] <= 274.15) &
                  (d["ldaps_heightAboveGround_2_r"] >= 90) &
                  (d["ldaps_surface_0_ncpcp"] > 0)).astype(int)
    # 구름/강수 (변동성 보조)
    for c in ["ldaps_etc_0_hcc","ldaps_etc_0_mcc","ldaps_etc_0_lcc","ldaps_surface_0_ncpcp"]:
        if c in d: f[c.replace("ldaps_","")] = d[c]

    # ---- GFS 보조: 상층바람/기압경도/돌풍 (능선 고지대 핵심) ----
    if "gfs_heightAboveGround_100_100u" in d:
        f["gfs_ws100"] = _wind_speed(d["gfs_heightAboveGround_100_100u"], d["gfs_heightAboveGround_100_100v"])
        f["gfs_ws80"]  = _wind_speed(d["gfs_heightAboveGround_80_u"], d["gfs_heightAboveGround_80_v"])
        u850, v850 = d["gfs_isobaricInhPa_850_u"], d["gfs_isobaricInhPa_850_v"]
        u700, v700 = d["gfs_isobaricInhPa_700_u"], d["gfs_isobaricInhPa_700_v"]
        f["gfs_ws850"] = _wind_speed(u850, v850)
        f["gfs_ws700"] = _wind_speed(u700, v700)
        f["gfs_ws500"] = _wind_speed(d["gfs_isobaricInhPa_500_u"], d["gfs_isobaricInhPa_500_v"])
        f["gfs_ws850_3"] = f["gfs_ws850"] ** 3            # 상층풍 출력비례항
        # 상층 풍향 (850hPa) — 능선은 상층류 방향이 speedup 결정
        wd850 = _wind_dir(u850, v850)
        f["wd850_sin"] = np.sin(np.radians(wd850)); f["wd850_cos"] = np.cos(np.radians(wd850))
        f["gfs_gust"]  = d["gfs_surface_0_gust"]
        f["gfs_tcc"]   = d["gfs_atmosphere_0_tcc"]
        # 대기 안정도: 850hPa 기온 - 지상 2m 기온 (음수=불안정/혼합↑)
        f["lapse_850_2m"] = d["gfs_isobaricInhPa_850_t"] - d["ldaps_heightAboveGround_2_t"]
        f["t850"] = d["gfs_isobaricInhPa_850_t"]
        f["gh500"] = d["gfs_isobaricInhPa_500_gh"]
        # 연직 시어: 상층(850) - 지상풍 (자유대기 결합도)
        f["shear_hub_850"] = f["gfs_ws850"] - f["ws_hub"]
        # LDAPS hub와 GFS850 블렌드 (두 소스 결합 hub풍속 추정)
        f["ws_blend"] = 0.6 * f["ws_hub"] + 0.4 * f["gfs_ws850"]
        f["ws_blend3"] = f["ws_blend"] ** 3
        # LDAPS 대비 GFS 풍속비 (예보 일치도)
        # ★FEATFIX (2026-08-02 DL 적합성 감사): 원비율은 저풍속에서 폭발 (skew 22.9,
        #   표준화가 못 고침). 로그비로 대칭화. ncpcp(0비율 88%, skew 18.2)도 log1p.
        if _os.environ.get("FEATFIX", "0") == "1":
            f["gfs_ldaps_ws_ratio"] = np.log1p(f["gfs_ws100"]) - np.log1p(f["ws_hub"])
            if "surface_0_ncpcp" in f: f["surface_0_ncpcp"] = np.log1p(np.clip(f["surface_0_ncpcp"], 0, None))
        else:
            f["gfs_ldaps_ws_ratio"] = f["gfs_ws100"] / (f["ws_hub"] + 1e-6)

    # ★FEATDIR (2026-08-02 지형감사): 섹터별 실측 speed-up 보정 허브풍속.
    #   SCADA실측/NWP외삽 비가 방향별로 ±30% (동풍 g3 0.65 vs 서풍 1.00) — 가덕산 능선 차폐.
    #   표는 2022+2023(g3 는 램프제외 2023H2)로 적합·원형스무딩 = 2024 OOF·2025 추론 모두 leak-safe.
    #   연도 안정성: g3 0.879 · g1/g2 0.61~0.85 (지형=불변 신호 확인). dirspeedup.csv 참조.
    if _os.environ.get("FEATDIR", "0") == "1":
        _sp = pd.read_csv(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                        "dirspeedup.csv"), encoding="utf-8-sig")
        _si = np.clip((f["wd10"].to_numpy() // 30).astype(int), 0, 11)
        for _g in (1, 2, 3):
            _t = _sp["spd_g%d" % _g].to_numpy()
            f["ws_hub_eff_g%d" % _g] = f["ws_hub"] * _t[_si]
            f["ws_hub_eff3_g%d" % _g] = f["ws_hub_eff_g%d" % _g] ** 3
    # ★FEATRISK (2026-08-02): 지표 정렬 위험 피처. FICR 은 |오차| <= 용량 6% 의 밴드 적중이고,
    #   오차의 주원인은 풍속 예보오차 σv 이며 그것이 출력오차로 증폭되는 배율이 파워커브 기울기다.
    #     예상 출력오차 ≈ |dCF/dv| × σv,   밴드적중확률 ≈ erf(0.06 / (√2·예상오차))
    #   같은 풍속오차라도 급경사(7-8 m/s, dCF/dv 최대 0.15)에서는 밴드를 놓치고 정격 근처(0.01~0.02)
    #   에서는 맞는다. **곱셈 상호작용이라 NN 이 소량 데이터로 스스로 형성하기 어렵다** -> 명시 주입.
    #   curveslope.csv = train(2022-24) 라벨로 적합한 CF(v)·dCF/dv 표 (ALPHA 와 동일한 leak-safe 규약).
    if _os.environ.get("FEATRISK", "0") == "1":
        _cs = pd.read_csv(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                        "curveslope.csv"), encoding="utf-8-sig")
        _wsg = _cs["ws"].to_numpy()
        _sig = f["ld_ws_std"].to_numpy(float) if "ld_ws_std" in f else np.full(len(f), np.nan)
        if not np.isfinite(_sig).any():
            _sig = 0.5 * f["gust_range50"].to_numpy(float)      # 대체 프록시
        _sig = np.clip(np.nan_to_num(_sig, nan=float(np.nanmedian(_sig))), 0.05, 5.0)
        _v = np.nan_to_num(f["ws_hub"].to_numpy(float))
        from scipy.special import erf as _erf
        for _g in (1, 2, 3):
            _dp = np.interp(_v, _wsg, _cs["dpdv_g%d" % _g].to_numpy())
            f["dpdv_g%d" % _g] = _dp
            _ee = np.clip(_dp * _sig, 1e-4, None)               # 예상 출력오차 (CF 단위)
            f["experr_g%d" % _g] = _ee
            f["bandp_g%d" % _g] = _erf(0.06 / (np.sqrt(2.0) * _ee))   # 밴드적중 확률 추정
    # ---- 풍향 섹터 & 상호작용 (태백=서풍 74% 지배, 서풍일수록 고발전) ----
    f["is_west"] = (((f["wd10"] >= 202.5) & (f["wd10"] < 292.5))).astype(int)   # W~SW 섹터
    f["ws_hub3_west"] = f["ws_hub3"] * f["is_west"]        # 서풍×출력 상호작용
    f["ws_hub_wd_cos"] = f["ws_hub"] * f["wd_cos"]         # 동서성분 가중 풍속
    f["ws_hub_wd_sin"] = f["ws_hub"] * f["wd_sin"]
    # 8방위 섹터 인덱스 (트리가 섹터별 speedup 학습)
    f["wd_sector"] = (((f["wd10"] + 22.5) % 360) // 45).astype(int)

    # ---- 상층풍 × 풍향 상호작용 (능선: 상층류 방향·강도가 speedup 결정) ----
    if "gfs_heightAboveGround_100_100u" in d:
        wd850 = _wind_dir(u850, v850)
        # 층간 풍향 정렬도: cos(wd850-wd10). 1=상하층 정렬(순압/강결합), <0=시어/전선
        f["dir_agree_850_10"] = np.cos(np.radians(wd850 - f["wd10"]))
        # 상층 서풍 성분 (동서축 투영; 서풍=음수). 능선 speedup은 서풍류에서 최대
        f["gfs_ws850_ew"] = f["gfs_ws850"] * f["wd850_sin"]
        f["gfs_ws700_ew"] = f["gfs_ws700"] * np.sin(np.radians(_wind_dir(u700, v700)))
        f["gfs_ws850_west"] = f["gfs_ws850"] * f["is_west"]        # 상층 서풍 강도
        # 블렌드 hub풍속을 동서축에 투영 (서풍일수록 큰 음수 → 고발전)
        f["ws_blend_ew"] = f["ws_blend"] * f["wd_sin"]
        f["ws_blend3_west"] = f["ws_blend3"] * f["is_west"]
        # 상하층 풍속 결합 × 정렬도 (정렬+강풍일 때 실제 hub도달 강함)
        f["couple_hub_850"] = f["ws_blend"] * f["dir_agree_850_10"]

    # ---- 지역특화 GFS 공간피처 (양간지풍 기압경도·저층안정도) — 원본격자, 5폴드검증 ----
    for c in ["prmsl_grad_we", "inv_850_2m", "stab_700_850"]:
        if c in d: f[c] = d[c].values
    # ---- 격자 앙상블 spread(예보 불확실성/엔트로피) — g1·g3(병목) 개선 ----
    for c in ["ld_ws_std", "ld_ws_range", "ld_sp_std", "gf_ws850_std", "lvl_ws_std"]:
        if c in d: f[c] = d[c].values

    # ---- 지표 순복사 (열플럭스→경계층 혼합→hub고도 운동량 하향수송, 열적 산곡풍) ----
    # 5시드·5폴드 전부 양(+)이고 과적합 갭을 줄이는 유일한 robust 추가 (SCADA검증)
    if "ldaps_surface_0_NDNSW" in d:
        f["net_sw"] = d["ldaps_surface_0_NDNSW"]
        f["net_lw"] = d["ldaps_surface_0_NDNLW"]
        f["net_rad"] = d["ldaps_surface_0_NDNSW"] + d["ldaps_surface_0_NDNLW"]

    # ---- 시간 주기성 ----
    f["hour"] = dt.dt.hour
    f["month"] = dt.dt.month
    f["doy"] = dt.dt.dayofyear
    f["hour_sin"] = np.sin(2*np.pi*f["hour"]/24); f["hour_cos"] = np.cos(2*np.pi*f["hour"]/24)
    f["month_sin"] = np.sin(2*np.pi*f["month"]/12); f["month_cos"] = np.cos(2*np.pi*f["month"]/12)
    f["doy_sin"] = np.sin(2*np.pi*f["doy"]/365); f["doy_cos"] = np.cos(2*np.pi*f["doy"]/365)

    # ---- 예보 내부 rolling (타깃 아님 → 리키지 안전) ----
    for w in [3, 6, 12]:
        f[f"ws_hub_rm{w}"] = f["ws_hub"].rolling(w, min_periods=1).mean()
        f[f"ws_hub_rstd{w}"] = f["ws_hub"].rolling(w, min_periods=1).std().fillna(0)
    f["ws_hub_diff1"] = f["ws_hub"].diff().fillna(0)          # 증감
    f["ws_hub_accel"] = f["ws_hub_diff1"].diff().fillna(0)    # 가속도

    # ★STAB (2026-08-11): 안정도 의존 수직외삽 허브풍속. 고정 α=0.12 는 안정도에 따라
    #   .05(불안정)~.40(강안정) 으로 움직이는 실제 셰어를 못 담는다 (data_prep _LD_XWIDE 주석).
    #   전부 예보(LDAPS/GFS) 채널만 사용 — leak-safe. STAB 미설정 시 이 블록은 실행되지 않아
    #   기존 출력과 비트단위 동일.
    #   부지 특이점: 태백 가덕산 ~1050m ASL -> 850hPa(~1450m)가 허브(~1170m ASL) 바로 위라
    #   inv_850_2m(T850-T2)가 사실상 허브층 안정도다. Δz~400m 로 위치온도 보정.
    if _os.environ.get("STAB", "0") == "1":
        _t2k = np.clip(d["ldaps_heightAboveGround_2_t"].to_numpy(float), 230., 330.)
        _blh_ = np.nan_to_num(d["ldaps_etc_0_blh"].to_numpy(float), nan=400.0)
        _nrad_ = (np.nan_to_num(d["ldaps_surface_0_NDNSW"].to_numpy(float))
                  + np.nan_to_num(d["ldaps_surface_0_NDNLW"].to_numpy(float))) \
            if "ldaps_surface_0_NDNSW" in d else np.zeros(len(f))
        _dz = 400.0                                          # 850hPa - 부지 지표 고도차 [m]
        _dth = (d["inv_850_2m"].to_numpy(float) if "inv_850_2m" in d
                else np.zeros(len(f))) + 0.0098 * _dz        # 위치온도차 Δθ [K]
        _u2 = np.maximum(np.nan_to_num(f["ws50"].to_numpy(float)), 0.5) ** 2
        _ri = (9.81 / _t2k) * _dth * _dz / _u2               # 벌크 리처드슨 근사
        _ri = (pd.Series(_ri, index=f.index).replace([np.inf, -np.inf], np.nan)
               .interpolate(limit_direction="both").fillna(0.0).to_numpy())
        f["st_ri"] = np.clip(_ri, -5.0, 5.0)
        # 합성 안정도 [-1,1]: Ri(+안정) + blh 낮음(=안정) + 순복사 음(야간냉각=안정)
        _stab = (np.tanh(f["st_ri"].to_numpy() / 1.0)
                 + np.tanh((400.0 - _blh_) / 400.0)
                 - np.tanh(_nrad_ / 100.0)) / 3.0
        f["st_stab"] = np.clip(_stab, -1.0, 1.0)
        # α 매핑 (문헌 상수 고정: 중립 0.12, 강안정 →0.34, 불안정 →0.05 클립)
        f["st_alpha"] = np.clip(0.12 + 0.22 * f["st_stab"], 0.05, 0.40)
        f["st_ws_hub"] = f["ws50"] * (HUB / 50.0) ** f["st_alpha"]
        f["st_ws_hub_cu"] = f["st_ws_hub"] ** 3               # 파워커브 근사 (P∝v^3)
        # 예보 2레벨(10/50m)로 직접 잰 셰어지수 (안정도의 독립 관측치, 실측 아님)
        _am = (np.log(np.maximum(np.nan_to_num(f["ws50"].to_numpy(float)), 0.3)
                      / np.maximum(np.nan_to_num(f["ws10"].to_numpy(float)), 0.3))
               / np.log(5.0))
        f["st_alpha_meas"] = np.clip(_am, -0.3, 0.8)
        f["st_ws_hub_m"] = f["ws50"] * (HUB / 50.0) ** np.clip(_am, 0.0, 0.5)

    return f
