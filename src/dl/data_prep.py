# -*- coding: utf-8 -*-
"""BARAM 2026 풍력 DL 파이프라인 — 데이터 준비 & 로더 (leakage-safe).

subC(대회 최고 LB 0.6419) 재현용 입력 텐서를 만든다.

무엇을 만드나
--------------
1. 스칼라 피처 행렬 X  : src/features.build_features (예보 LDAPS/GFS 파생, 실측/타깃 미사용).
2. 격자 토큰 텐서 GT/LT: NWP 원본 격자(GFS 3x3=9토큰, LDAPS 4x5→16토큰)를 토큰으로.
   각 토큰은 (lat,lon) 위치 + 변수벡터. Transformer 트렁크의 입력.
3. 두 피처셋 버전:
     v2 (86 feat) = 현재 features.py 전체 (grid_spread 5피처 포함)  -> 멤버 v2_gated_attn / v2_base / ssl
     v1 (81 feat) = v2에서 grid_spread 5피처 제거                    -> 멤버 v1_attn
   (GT/LT/gpos/lpos/Y/year/keys 는 v1==v2 로 동일; X 컬럼 수만 다름.)

리키지/규정 준수
----------------
- 입력은 전부 **예보(forecast, 12~35h lead)** 에서만 파생. 실측 SCADA·타깃 시차 미사용.
- SCADA 실측은 오직 train 구간(2022~2024)에서 features.ALPHA(허브풍속 지수) 상수 하나를
  사전 최적화하는 데만 쓰였고(코드에 상수로 박제), 여기서는 어떤 실측도 로드하지 않는다.
- 2025 test 입력은 예측(추론)에만 쓰이며 학습·정규화 통계 산출에 절대 섞이지 않는다.
- 결측/손상 시각 대치는 common._fix_corrupt (물리 하한 → 시간보간) 로 예보 내부에서만 처리.

저장 위치: models/  (data_train_v1.npz, data_test_v1.npz, data_train_v2.npz, data_test_v2.npz)
"""
import sys, io, os
from pathlib import Path
import os
import numpy as np
import pandas as pd

# --- src/ (common, features) 임포트 경로 확보 ---
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import common as C
import features as F

ROOT = SRC.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 대회 상수
CAP = np.array([21600., 21600., 21000.], np.float32)      # 그룹별 설비용량 (kWh)
FOLDS = [([2022], 2023), ([2022, 2023], 2024)]            # 시계열 확장창 CV (fold A, B)
GROUP_FOLDS = {0: [0, 1], 1: [0, 1], 2: [1]}              # 그룹별 평가 fold (g3는 2023~만 라벨)

# v2에만 있는 grid_spread 피처 (v1 파생 시 제거) — common.grid_spread 산출
GRID_SPREAD_FEATS = ["ld_ws_std", "ld_ws_range", "ld_sp_std", "gf_ws850_std", "lvl_ws_std"]

# ------------------------------------------------------------------ 격자 토큰화
# LDAPS grid_id -> 4x5 격자 (row=lat 내림차순; 모서리 4칸은 관측 없음)
LD_MAP = {1: (0, 1), 2: (0, 2), 3: (0, 3),
          4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3), 8: (1, 4),
          9: (2, 0), 10: (2, 1), 11: (2, 2), 12: (2, 3), 13: (2, 4),
          14: (3, 1), 15: (3, 2), 16: (3, 3)}
# ★공간 토큰 경로는 원래 **바람 4 + blh = 5채널**만 받았다. LDAPS 원본은 30채널이고,
# 나머지(기온·습도·기압·운량·지형고도·XBLWS/YBLWS)는 단일점 스칼라로만 존재해
# **공간 구조로는 모델에 들어간 적이 없다**. LDAPS 는 세 그룹을 구분하는 유일한 소스이므로
# 안정도·밀도·지형의 공간 변화를 못 본다. LDCH=wide 로 확장해 검정한다.
_LD_BASE = ["heightAboveGround_10_10u", "heightAboveGround_10_10v", "u50", "v50", "etc_0_blh"]
# 지형 파생(복잡지형 풍력예보 표준): TPI(능선/계곡), 경사크기, 풍향투영 경사(=업슬로프/다운슬로프).
# Sx(풍상측 노출)의 격자판 근사로 **∇h·ŵ** 를 쓴다 — 값이 양수면 바람이 오르막을 타는 상황.
_LD_WIDE = _LD_BASE + ["heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS",
                       "ws50_range", "heightAboveGround_2_t", "heightAboveGround_2_q",
                       "surface_0_sp", "meanSea_0_prmsl", "surface_0_h", "etc_0_lcc",
                       "terr_tpi", "terr_slope", "terr_upslope"]
# ★_LD_XWIDE (2026-07-31): 셀간 공간변동 순위로 고른 **미사용** 채널 추가.
#   실측 셀간변동(시각내 SD / 전체 SD): 50MVmin .217 / 50MVmax .210 / 50MUmax .181 / 50MUmin .180
#   -> 사용중인 blh(.210)·10u(.189) 와 동급 이상인데 토큰에 없었다.
#   u50/v50 은 이들의 **평균**이고 ws50_range 는 크기범위만 담는다. max/min 을 U·V 성분으로
#   각각 실으면 **방향 변동성**까지 들어간다. 50m 는 허브(117m)에 가장 가까운 LDAPS 레벨.
#   추가로 안정도 축: 2_r(습도 .156) · VLCDC(하층운 .161) · NDNLW(순장파복사 .130).
#   우리는 10m->117m 외삽에 고정 α=0.12 를 쓰는데, 안정도가 α 를 .05~.4 로 움직인다.
_LD_XWIDE = _LD_WIDE + ["heightAboveGround_50_50MUmax", "heightAboveGround_50_50MUmin",
                        "heightAboveGround_50_50MVmax", "heightAboveGround_50_50MVmin",
                        "heightAboveGround_2_r", "etc_0_VLCDC", "surface_0_NDNLW"]
_LDCH = os.environ.get("LDCH", "base")
LD_CH = _LD_XWIDE if _LDCH == "xwide" else (_LD_WIDE if _LDCH == "wide" else _LD_BASE)
_GF_BASE = ["heightAboveGround_10_10u", "heightAboveGround_10_10v",
            "heightAboveGround_100_100u", "heightAboveGround_100_100v",
            "isobaricInhPa_850_u", "isobaricInhPa_850_v",
            "isobaricInhPa_700_u", "isobaricInhPa_700_v",
            "meanSea_0_prmsl", "heightAboveGround_2_2t"]
# ★_GF_XWIDE (2026-07-31): 격자간 공간변동 순위로 고른 **미사용** GFS 채널.
#   실측(시각내 SD / 전체 SD): 80_v .554 / PBL_v .459 / 80_u .395 / gust .370 / 2r .356 / PBL_u .336
#   -> 80m 바람이 **사용중인 100m(.542/.381)보다 변동이 크다.**
#   물리 근거:
#     (a) 80m: 허브 117m 인데 현재 10m·100m 두 점뿐 -> 80m 추가로 **3점 전단**.
#         10->117m 외삽의 고정 α=0.12 를 데이터로 대체할 수 있다.
#     (b) surface_0_gust: 우리 σ 분석의 병목이 "시간평균 예보오차"인데
#         gust 는 **시간 내 변동**을 담는 유일한 채널이다.
#     (c) PBL u/v: 경계층 상단 바람 = 혼합 강도 지표.
#   주: planetaryBoundaryLayer_0_VRATE 는 |v|>1e4 가 25,698개로 이상치가 실재한다.
#       (넣으려면 정제 필요. 현재 미포함)
_GF_XWIDE = _GF_BASE + ["heightAboveGround_80_u", "heightAboveGround_80_v",
                        "surface_0_gust", "planetaryBoundaryLayer_0_u",
                        "planetaryBoundaryLayer_0_v", "heightAboveGround_2_2r"]
GF_CH = _GF_XWIDE if os.environ.get("GFCH", "base") == "xwide" else _GF_BASE
# GFS 3x3 격자 중심좌표 (grid_id 1..9 = row-major)
GPOS = np.array([[37.50, 128.75], [37.50, 129.00], [37.50, 129.25],
                 [37.25, 128.75], [37.25, 129.00], [37.25, 129.25],
                 [37.00, 128.75], [37.00, 129.00], [37.00, 129.25]], np.float32)
# LDAPS 격자 중심좌표 (grid_id 1..16)
LDPOS = {1: (37.3032, 128.9443), 2: (37.3027, 128.9617), 3: (37.3022, 128.9790),
         4: (37.2899, 128.9263), 5: (37.2894, 128.9437), 6: (37.2888, 128.9610),
         7: (37.2883, 128.9784), 8: (37.2878, 128.9958),
         9: (37.2760, 128.9257), 10: (37.2755, 128.9430), 11: (37.2750, 128.9604),
         12: (37.2745, 128.9778), 13: (37.2740, 128.9951),
         14: (37.2617, 128.9424), 15: (37.2612, 128.9598), 16: (37.2607, 128.9771)}
LPOS = np.array([LDPOS[i] for i in range(1, 17)], np.float32)
DATA = ROOT / "data"


def _build_grid_tensor(split, keys):
    """keys = forecast_kst_dtm 리스트(행 순서). GFS/LDAPS 원본 격자를 토큰 텐서로.
    반환: GT [N,9,10], LT [N,16,5]  (모두 예보값)."""
    key_idx = {k: i for i, k in enumerate(keys)}
    N = len(keys)
    d = DATA / split
    # ---- GFS 3x3 ----
    gf = pd.read_csv(d / f"gfs_{split}.csv", encoding="utf-8-sig")
    gf["forecast_kst_dtm"] = pd.to_datetime(gf["forecast_kst_dtm"])
    gf = gf[gf["forecast_kst_dtm"].isin(key_idx)]
    G = np.zeros((N, len(GF_CH), 3, 3), np.float32)
    ri = gf["forecast_kst_dtm"].map(key_idx).to_numpy()
    rr = ((gf["grid_id"].to_numpy() - 1) // 3); cc = ((gf["grid_id"].to_numpy() - 1) % 3)
    for ci, ch in enumerate(GF_CH):
        G[ri, ci, rr, cc] = gf[ch].to_numpy(np.float32)
    GT = G.reshape(N, len(GF_CH), 9).transpose(0, 2, 1).copy()      # [N,9,Cgf]
    # ---- LDAPS 4x5 ----
    ld = pd.read_csv(d / f"ldaps_{split}.csv", encoding="utf-8-sig")
    ld["forecast_kst_dtm"] = pd.to_datetime(ld["forecast_kst_dtm"])
    ld = ld[ld["forecast_kst_dtm"].isin(key_idx)].copy()
    # ★지형 피처 (복잡지형 풍력예보 표준): 고도 h, 상대고도 TPI(= h - 주변평균),
    # 풍향의존 노출도 Sx(= 풍상측 지형이 얼마나 높은가). 태백 복잡지형에서 국지증폭의 주 요인이고
    # 셀간변동 비율이 h 99.8% / sp 67.0% 로 전 채널 중 최고인데 토큰 경로에 없었다.
    ld["ws50_range"] = np.hypot(ld["heightAboveGround_50_50MUmax"], ld["heightAboveGround_50_50MVmax"]) -                        np.hypot(ld["heightAboveGround_50_50MUmin"], ld["heightAboveGround_50_50MVmin"])
    ld["u50"] = 0.5 * (ld["heightAboveGround_50_50MUmax"] + ld["heightAboveGround_50_50MUmin"])
    ld["v50"] = 0.5 * (ld["heightAboveGround_50_50MVmax"] + ld["heightAboveGround_50_50MVmin"])
    if "terr_tpi" in LD_CH:
        # ★지형 파생 (복잡지형 풍력예보 표준). h 는 정적이므로 셀별 상수.
        #   TPI  = h - 4근방 평균  (양수=능선, 음수=계곡)
        #   slope= |grad h|
        #   upslope = grad h 를 풍향 단위벡터에 투영 (양수=바람이 오르막을 탐) = Sx 의 격자판 근사
        _h = ld.groupby("grid_id")["surface_0_h"].first()
        _H = np.full((4, 5), np.nan, np.float32)
        for _g, (_r, _c) in LD_MAP.items():
            if _g in _h.index: _H[_r, _c] = float(_h.loc[_g])
        _H = np.where(np.isfinite(_H), _H, np.nanmean(_H))
        _nb = sum(np.roll(np.roll(_H, dr, 0), dc, 1) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))) / 4.0
        _TPI = _H - _nb
        _gy, _gx = np.gradient(_H)          # y=lat 축(내림차순), x=lon 축
        _SL = np.hypot(_gy, _gx)
        _rc2 = ld["grid_id"].map(LD_MAP)
        _rr = np.array([t[0] for t in _rc2]); _cc = np.array([t[1] for t in _rc2])
        ld["terr_tpi"] = _TPI[_rr, _cc]
        ld["terr_slope"] = _SL[_rr, _cc]
        # 풍향 투영: u=동, v=북. lon 증가=동(+x), lat 내림차순이므로 북은 -y
        _u = ld["u50"].to_numpy(float); _v = ld["v50"].to_numpy(float)
        _sp = np.maximum(np.hypot(_u, _v), 0.1)
        ld["terr_upslope"] = (_gx[_rr, _cc] * (_u / _sp) - _gy[_rr, _cc] * (_v / _sp))
    L = np.full((N, len(LD_CH), 4, 5), np.nan, np.float32)
    ri = ld["forecast_kst_dtm"].map(key_idx).to_numpy()
    rc = ld["grid_id"].map(LD_MAP)
    rows = np.array([x[0] for x in rc]); cols = np.array([x[1] for x in rc])
    for ci, ch in enumerate(LD_CH):
        L[ri, ci, rows, cols] = ld[ch].to_numpy(np.float32)
    # 관측 없는 4 모서리는 (n,ch)별 존재 셀 평균으로 채움
    for ci in range(len(LD_CH)):
        sl = L[:, ci, :, :].reshape(N, -1)
        mean = np.nanmean(sl, axis=1, keepdims=True)
        inds = np.where(np.isnan(sl))
        sl[inds] = np.take(mean, inds[0])
        L[:, ci, :, :] = sl.reshape(N, 4, 5)
    # ★지형 파생: 4x5 격자에서 h 의 TPI/경사/풍향투영. h 는 정적이므로 셀별 상수.
    if "terr_tpi" in LD_CH:
        H = np.full((4, 5), np.nan, np.float32)
        for gid, (r, c) in LD_MAP.items():
            v = ld.loc[ld["grid_id"] == gid, "surface_0_h"]
            if len(v): H[r, c] = float(v.iloc[0])
        Hf = np.where(np.isfinite(H), H, np.nanmean(H))
        nb = np.zeros_like(Hf); cnt = np.zeros_like(Hf)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sh = np.roll(np.roll(Hf, dr, 0), dc, 1)
            nb += sh; cnt += 1
        TPI = Hf - nb / np.maximum(cnt, 1)
        gy, gx = np.gradient(Hf)                       # y=lat(내림차순), x=lon
        SLOPE = np.hypot(gy, gx)
    LT = np.zeros((N, 16, len(LD_CH)), np.float32)
    for gid, (r, c) in LD_MAP.items():
        LT[:, gid - 1, :] = L[:, :, r, c]
    return GT, LT


def _build_split(split):
    """한 split(train/test)의 v2 배열 dict 생성."""
    wx = C.load_weather(split, how="idw")
    feat = F.build_features(wx).sort_values("forecast_kst_dtm").reset_index(drop=True)
    keys = list(feat["forecast_kst_dtm"])
    Fcols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    X = feat[Fcols].to_numpy(np.float32)
    GT, LT = _build_grid_tensor(split, keys)
    year = feat["forecast_kst_dtm"].dt.year.to_numpy()
    if split == "train":
        lab = C.load_labels().set_index("forecast_kst_dtm")
        Y = np.full((len(keys), 3), np.nan, np.float32)
        for gi, col in enumerate(C.TARGET_COLS):
            Y[:, gi] = lab[col].reindex(keys).to_numpy(np.float32)
    else:
        Y = np.zeros((len(keys), 3), np.float32)
    return dict(X=X, GT=GT, LT=LT, gpos=GPOS, lpos=LPOS, Y=Y, year=year,
                keys=np.array([str(k) for k in keys]), Fcols=np.array(Fcols))


def _save(d, path):
    np.savez(path, **d)


def build_all(verbose=True):
    """v2(전체) + v1(grid_spread 제거) train/test npz 를 models/ 에 저장.
    ★DATASUF (2026-08-11): env DATASUF 가 설정되면 파일명에 접미사를 붙인다.
      FEATFIX 등 features.py env 변형을 **기존 캐시를 덮지 않고** 별도 캐시로 빌드하기 위함."""
    _suf = os.environ.get("DATASUF", "")
    for split in ["train", "test"]:
        v2 = _build_split(split)
        _save(v2, MODELS / f"data_{split}_v2{_suf}.npz")
        # ---- v1 = v2 - grid_spread 5피처 (컬럼 순서 유지) ----
        f2 = list(v2["Fcols"])
        keep = [i for i, c in enumerate(f2) if c not in GRID_SPREAD_FEATS]
        v1 = dict(v2)
        v1["X"] = v2["X"][:, keep]
        v1["Fcols"] = np.array([f2[i] for i in keep])
        _save(v1, MODELS / f"data_{split}_v1{_suf}.npz")
        if verbose:
            print(f"[{split}] v2 X{v2['X'].shape} v1 X{v1['X'].shape} "
                  f"GT{v2['GT'].shape} LT{v2['LT'].shape} Y{v2['Y'].shape} "
                  f"years={sorted(set(v2['year'].tolist()))}", flush=True)
    if verbose:
        print("[OK] data_prep 완료 -> models/data_{train,test}_{v1,v2}.npz", flush=True)


# ------------------------------------------------------------------ 로더 & 변환
def load_data(split, ver):
    """models/ 에 저장된 npz 로드 (nan_to_num 적용). ver in {'v1','v2'}.
    DATASUF 설정시 해당 접미사 캐시를 읽는다 (기본 "" = 기존과 비트동일)."""
    _suf = os.environ.get("DATASUF", "")
    d = np.load(MODELS / f"data_{split}_{ver}{_suf}.npz", allow_pickle=True)
    # ★NaNFIX (2026-07-31 감사): test LDAPS 에 **전 격자 결측 시각 3개**가 있다.
    #   2025-04-08 17:00 / 06-18 18:00 (14열) · 07-18 06:00 (19열)
    #   결측 열에 50MUmax/min·50MVmax/min 이 포함돼 **u50·v50 이 통째로 죽는다**(_LD_BASE 5채널 중 2개).
    #   기존 nan_to_num 은 이를 **0 m/s** 로 채운다 — 물리적으로 무의미하고 모델은 '무풍'으로 읽는다.
    #   특히 07-18 06:00 은 ws_hub 9.002 m/s(근정격 바람)인데 토큰 50m 바람만 0 = 입력 자체가 모순.
    #   교정: 토큰 텐서는 **시간축 선형보간 후 남은 것만 0**. 스칼라 X 는 기존대로(이미 파생단계에서 처리).
    #   NANFIX=0 으로 끄면 기존과 비트단위 동일.
    _gt = d["GT"].astype(np.float32); _lt = d["LT"].astype(np.float32)
    if os.environ.get("NANFIX", "1") == "1":
        for _T in (_gt, _lt):
            bad = ~np.isfinite(_T)
            if not bad.any():
                continue
            # [N, cell, ch] -> 셀·채널별로 시간축 보간
            n, nc, ch = _T.shape
            idx = np.arange(n, dtype=np.float64)
            for c in range(nc):
                for k in range(ch):
                    col = _T[:, c, k]
                    m = np.isfinite(col)
                    if m.all() or m.sum() < 2:
                        continue
                    _T[:, c, k] = np.interp(idx, idx[m], col[m]).astype(np.float32)
    return dict(
        X=np.nan_to_num(d["X"].astype(np.float32)),
        GT=np.nan_to_num(_gt),
        LT=np.nan_to_num(_lt),
        gpos=d["gpos"].astype(np.float32), lpos=d["lpos"].astype(np.float32),
        Y=d["Y"].astype(np.float32), year=d["year"],
        Fcols=list(d["Fcols"]), keys=d["keys"])


def revin_center(arr, eps=1e-5):
    """RevIN-center: 샘플별·채널별로 토큰(T)축 평균을 뺀다 (scale 유지).
    격자 토큰의 절대 오프셋(동기적 예보 바이어스)을 제거해 상대 공간패턴만 학습."""
    return (arr - arr.mean(axis=1, keepdims=True)).astype(np.float32)


def apply_revin_center(data):
    """GT/LT 에 RevIN-center 적용한 새 dict 반환 (subC 최고 구성)."""
    d = dict(data)
    d["GT"] = revin_center(data["GT"])
    d["LT"] = revin_center(data["LT"])
    return d


# ------------------------------------------------------------------ 학습 유틸
def fold_indices(data, fold):
    tr_years, va_year = fold
    year = data["year"]
    tr = np.where(np.isin(year, tr_years))[0]
    va = np.where(year == va_year)[0]
    return tr, va


def standardize(tr, va, other=None):
    """train fold 통계(mu,sd)로 표준화. 3D 텐서는 평탄화 후 dim별 표준화."""
    mu = tr.reshape(len(tr), -1).mean(0)
    sd = tr.reshape(len(tr), -1).std(0) + 1e-6

    def ap(a):
        sh = a.shape
        flat = a.reshape(len(a), -1)
        return ((flat - mu) / sd).reshape(sh).astype(np.float32)
    outs = [ap(tr), ap(va)]
    if other is not None:
        outs.append(ap(other))
    return outs


def clean_mask(X, ycf, Fcols, gi):
    """다운타임/curtailment 제거 (train 전용, CF 단위). production 'orig' 규칙.
    - downtime: ws>5 & CF<0.03  (강풍인데 무출력=정지)
    - near0   : ws>4 & CF==0
    - curtail : ws>=11 & CF<0.10
    평가대상(CF>=0.10)은 절대 지우지 않는다."""
    ws = X[:, Fcols.index("ws_hub")]
    y = ycf
    downtime = (ws > 5) & (y < 0.03)
    near0 = (ws > 4) & (y == 0.0)
    curtail = (ws >= 11) & (y < 0.10)
    return ~(downtime | near0 | curtail)


def group_score_np(y_cf, p_cf):
    """대회 group_score = 0.5*(1-NMAE) + 0.5*FICR. y_cf,p_cf: 한 그룹 CF 배열(라벨만)."""
    v = y_cf >= 0.10
    if v.sum() == 0:
        return np.nan
    e = np.abs(p_cf[v] - y_cf[v])                       # CF 오차 = |p-a|/cap
    nm = e.mean()
    u = np.select([e <= 0.06, e <= 0.08], [4., 3.], default=0.)
    fi = np.sum(y_cf[v] * u) / np.sum(y_cf[v] * 4.)
    return 0.5 * (1 - nm) + 0.5 * fi


# ------------------------------------------------------------------ 예보-윈도 (±kh)
# 윈도 채널: 주변 예보시각의 원(raw) 동적 신호 (손수 만든 rolling stat 아님).
WIN_FEATS = ["ws_hub", "ws50", "gfs_ws100", "gfs_ws850", "gfs_ws700", "shear_hub_850",
             "ws_blend", "wd_sin", "wd_cos", "t2", "sp", "prmsl", "rho", "blh"]


def year_blocks(year):
    """연속 연도블록별 (start,end) 포함 행범위. 이웃 참조가 fold를 넘지 않게."""
    return {y: (np.where(year == y)[0].min(), np.where(year == y)[0].max())
            for y in np.unique(year)}


def build_window(Xraw, year, Fcols, feats, offsets, clamp_year=True):
    """Xraw[N,F] 원본에서 이웃 예보시각 윈도 W[N, len(offsets), len(feats)] 구성.
    이웃 행은 같은 연도블록 내로 clamp (fold 교차·리키지 방지, 입력 전용)."""
    cols = [Fcols.index(f) for f in feats]
    N_ = len(Xraw)
    blocks = year_blocks(year)
    lo = np.empty(N_, np.int64); hi = np.empty(N_, np.int64)
    for y, (a, b) in blocks.items():
        m = year == y; lo[m] = a; hi[m] = b
    base = np.arange(N_)
    W = np.empty((N_, len(offsets), len(feats)), np.float32)
    for j, off in enumerate(offsets):
        idx = base + off
        idx = np.clip(idx, lo, hi) if clamp_year else np.clip(idx, 0, N_ - 1)
        W[:, j, :] = Xraw[idx][:, cols]
    return W


def std_window(Wtr, *others):
    """train 윈도 통계(전체 행·시점)로 채널별 표준화."""
    mu = Wtr.reshape(-1, Wtr.shape[-1]).mean(0)
    sd = Wtr.reshape(-1, Wtr.shape[-1]).std(0) + 1e-6

    def ap(W):
        return ((W - mu) / sd).astype(np.float32)
    return [ap(Wtr)] + [ap(o) for o in others]


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    build_all()
