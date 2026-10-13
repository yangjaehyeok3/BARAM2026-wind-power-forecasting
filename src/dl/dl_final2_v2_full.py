# -*- coding: utf-8 -*-
"""★앵커 3축 교정 재학습 파일럿 (2026-08-01, 사용자 승인: "손실함수 축으로 앵커 재학습까지").

배경 — 배포 주변분포의 지배 앵커 f2(dl_final2)는 g1·g3 에 가중 0.7 인데:
  1. 손실이 구세대 ver=1 (L1 − lam·soft_ficr). 최강 레버였던 ver=2 밴드손실
     (FICRPEN/focal, seq 계열 +0.012)이 앵커에는 한 번도 적용된 적 없다.
  2. mm 피처가 원본 오염판(mm_train.csv: mm_ecmwf 가짜 + 2022 std 87.6% 퇴화 = 연도인덱스).
  3. NANFIX 이전에 구워짐 (test 3시각 u50/v50=0).
= "오염된 베이스 위에 모델 형성" 가설이 앵커 수준에서 사실.

이 스크립트 = dl_final2_full.py 와 **동일 구조·동일 멤버·동일 시드**, 차이는:
  MMSUF 환경변수 (기본 "_f" = 정제 mm)      <- 축 2
  T.BASE 에 ver=2 손실 + tau 어닐링 주입     <- 축 1
  NANFIX 는 data_prep 기본값(env 로 제어)    <- 축 3
손실 레시피는 train_seqbase.py L395 와 동일:
  loss_fn(ver=2, tau=0.025→0.008 기하어닐링, w_actual=True, bias_lam=0.05,
          focal=1.0, tail=0.15, nmae_w=1−FICRW, ficr_w=FICRW) + env FICRPEN=1
차이점 명시: seq 의 L1 워밍업(ep<8)은 train_dl 에 없음(미구현, 파일럿 한계로 기록).
SSL(g3)은 tau 어닐링 없이 고정 tau=0.008 (mae.py 는 tau_sched 미지원).

평가: models/dl_final2_v2/dl_oof.npz 저장 후
  F2DIR=dl_final2_v2 로 order_lab/audit_bundle_provenance 재실행 = 체인 교체 평가.
판정은 대회 지표로만 한다 (대리지표 금지 — baram2026-invented-criteria).

실행: env FICRPEN=1 [MMSUF=_f] [FICRW=0.5] [NANFIX=1] python src/dl/dl_final2_v2_full.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import data_prep as D
import train_dl as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MMSUF = os.environ.get("MMSUF", "_f")
FICRW = float(os.environ.get("FICRW", "0.5"))
MM = ["mm_icon", "mm_gfs", "mm_ecmwf", "mm_mean", "mm_std", "mm_u", "mm_v"]
FUSED = ["fused_g1", "fused_g2", "fused_g3"]
GEFS = ["gefs_ws_hub", "gefs_ws10", "gefs_spr_hub", "gefs_u_hub", "gefs_v_hub"]


def _load(csv, cols):
    d = pd.read_csv(os.path.join(ROOT, csv), encoding="utf-8-sig")
    d["k"] = pd.to_datetime(d["forecast_kst_dtm"]).astype(str)
    return d.set_index("k")[cols]


_MM = {s: _load("mm_%s%s.csv" % (s, MMSUF), MM) for s in ["train", "test"]}
_FUSENAME = os.environ.get("FUSEFILE", "v2")
_FU = {s: _load("fused_%s_%s.csv" % (s, _FUSENAME), FUSED) for s in ["train", "test"]}
_GEFSFILE = os.environ.get("GEFSFILE", "gefsfeat.csv")
_GE = _load(_GEFSFILE, GEFS)
# ★UMKR (2026-08-01): 검증된 살아있는 피처축(감사 +0.00182, FICR +0.00242)을 앵커에 통합.
#   기본 0 = 기존과 비트단위 동일. UMKR=1 이면 ext_umkr_patch 11열을 입력에 추가.
_UMKR = os.environ.get("UMKR", "0") == "1"
if _UMKR:
    _uk_raw = pd.read_csv(os.path.join(ROOT, "ext_umkr_patch.csv"), encoding="utf-8-sig")
    _uk_raw["k"] = pd.to_datetime(_uk_raw["forecast_kst_dtm"]).astype(str)
    _UKC = [c for c in _uk_raw.columns if c not in ("forecast_kst_dtm", "k")]
    _UK = _uk_raw.set_index("k")[_UKC]
    print("[umkr] %d cols ON (anchor)" % len(_UKC), flush=True)
# ★DOMF (2026-08-04): 이번주 발견 피처축 통합 — LOYO-BU(유닛분해 상향식 3열)+BLH+LDAPS갭+풍향 7열.
#   기본 0 = 기존과 비트동일. train측 BU는 연도교차 적합(타깃인코딩 누수 차단), test측은 전기간 적합.
_DOMF = os.environ.get("DOMF", "0") == "1"
if _DOMF:
    _dv = os.environ.get("DOMV", "")            # ""=v1(7열) / "2"=v2(융합풍속+융합BU 9열)
    _dmc = None
    _dm = {}
    for s in ["train", "test"]:
        _raw = pd.read_csv(os.path.join(ROOT, "ext_domain%s_%s.csv" % (_dv, s)), encoding="utf-8-sig")
        _raw["k"] = pd.to_datetime(_raw["forecast_kst_dtm"]).astype(str)
        if _dmc is None:
            _dmc = [c for c in _raw.columns if c not in ("forecast_kst_dtm", "k")]
        _dm[s] = _raw.set_index("k")[_dmc]
    print("[domf%s] %d cols ON (anchor): %s" % (_dv, len(_dmc), _dmc), flush=True)
# ★AUX (2026-08-08): 중간 감독 타깃 주입. AUXW=0(기본)이면 로드조차 하지 않아 기존과 비트 동일.
#   aux_targets.csv = 그룹별 나셀풍속(25m/s 정규화) + 가용률. train 라벨 전용, 추론엔 미사용.
_AUXW = float(os.environ.get("AUXW", "0"))
_AUXC = ["aux_ws_g1", "aux_av_g1", "aux_ws_g2", "aux_av_g2", "aux_ws_g3", "aux_av_g3"]
_G3UNIT_AUX = os.environ.get("G3UNIT_AUX", "0") == "1"
if _AUXW > 0 and _G3UNIT_AUX:
    _tz = np.load(os.path.join(ROOT, "experiments", "turbine_labels.npz"), allow_pickle=True)
    _g3i = np.where(_tz["group"].astype(int) == 2)[0]
    _TKEY = {str(k): i for i, k in enumerate(_tz["keys"])}
    _TY = (_tz["Y17"][:, _g3i] / _tz["capt"][_g3i][None, :]).astype(np.float32)
    _TM = _tz["M17"][:, _g3i].astype(bool) & np.isfinite(_TY)
    _TY = np.where(_TM, _TY, np.nan).astype(np.float32)
    os.environ["AUXN"] = "5"
    print("[aux] exact G3 five-unit CF targets ON (Unison o=-6, AUXW=%.3f, blend=%s)" %
          (_AUXW, os.environ.get("G3_UNIT_BLEND", "0")), flush=True)
elif _AUXW > 0:
    _ax = pd.read_csv(os.path.join(ROOT, "aux_targets.csv"), encoding="utf-8-sig")
    _ax["k"] = pd.to_datetime(_ax["forecast_kst_dtm"]).astype(str)
    _AX = _ax.set_index("k")[_AUXC]
    os.environ["AUXN"] = str(len(_AUXC))          # 모델 보조헤드 출력 차원
    print("[aux] %d targets ON (AUXW=%.3f)" % (len(_AUXC), _AUXW), flush=True)

_orig = D.load_data


def _patched(split, ver):
    d = _orig(split, ver); keys = [str(k) for k in d["keys"]]
    parts = [
        _MM[split].reindex(keys)[MM].ffill().bfill().to_numpy(np.float32),
        _FU[split].reindex(keys)[FUSED].ffill().bfill().to_numpy(np.float32),
        _GE.reindex(keys)[GEFS].ffill().bfill().to_numpy(np.float32),
    ] + ([_UK.reindex(keys).ffill().bfill().to_numpy(np.float32)] if _UMKR else []) \
      + ([_dm[split].reindex(keys).ffill().bfill().to_numpy(np.float32)] if _DOMF else [])
    ext = np.nan_to_num(np.concatenate(parts, axis=1))
    d = dict(d); d["X"] = np.concatenate([d["X"], ext], axis=1).astype(np.float32)
    if _AUXW > 0:
        # test 에는 라벨이 없다 -> 전부 NaN(마스크로 자동 배제). train 만 실제 감독.
        if _G3UNIT_AUX:
            d["AUX"] = (np.stack([_TY[_TKEY[k]] for k in keys]).astype(np.float32)
                        if split == "train" else np.full((len(keys), 5), np.nan, np.float32))
        else:
            d["AUX"] = (_AX.reindex(keys).to_numpy(np.float32) if split == "train"
                        else np.full((len(keys), len(_AUXC)), np.nan, np.float32))
    d["Fcols"] = list(d["Fcols"]) + MM + FUSED + GEFS + (_UKC if _UMKR else []) \
        + (_dmc if _DOMF else [])
    # ★RAMPFIX (2026-08-02): g3 시운전 램프(2023-07-01 이전) 라벨을 NaN 마스크.
    #   앵커는 raw-y 학습(AVAILNORM 없음)이라 램프 오염이 정격밴드 조건부 평균을
    #   최빈값 아래로 끌어내린다 (g3 정격 4점적중 0.118 vs g2 0.653). train 만 적용.
    if split == "train" and os.environ.get("RAMPFIX", "0") == "1":
        _dt = pd.to_datetime(pd.Index(keys))
        _rm = np.asarray(_dt < pd.Timestamp("2023-07-01"))
        Y = d["Y"].astype(np.float32).copy(); Y[_rm, 2] = np.nan; d["Y"] = Y
        print("[rampfix] g3 pre-2023-07 %d행 라벨 마스크" % int(_rm.sum()), flush=True)
    # ★AVCAL (2026-08-02): 유닛레벨 가용률로 타깃을 잠재출력화 (y_pot = y/avail).
    #   근거: 정격 라벨 = 가지혼합(k5 57%@.971/k4 25%@.771/...) -> 혼합평균 .837 = 모델 예측 .84.
    #   y_pot 는 정격 IQR .259->.096 (2.7배 응축). 서빙은 pot×스케줄(달력/기후) 되곱 — 별도 조립.
    #   avail<0.20 행은 라벨 마스크(0/5 등 나눗셈 불능). 기본 0 = 기존과 비트동일.
    if split == "train" and os.environ.get("AVCAL", "0") == "1":
        _at = pd.read_csv(os.path.join(ROOT, os.environ.get("AVFILE2", "avail_frac_units.csv")),
                          encoding="utf-8-sig")
        _at["k"] = pd.to_datetime(_at["forecast_kst_dtm"]).astype(str)
        _av = _at.set_index("k")[["avail_g1", "avail_g2", "avail_g3"]].reindex(keys)\
                 .to_numpy(np.float32)
        Y = d["Y"].astype(np.float32).copy()          # ★kWh 원단위 (train_dl 이 /CAP 함)
        _bad = _av < 0.20
        _cap = 1.05 * np.asarray(D.CAP, np.float32).reshape(1, -1)
        Y = np.where(_bad, np.nan,
                     np.minimum(Y / np.clip(_av, 0.20, 1.0), _cap)).astype(np.float32)
        d["Y"] = Y
        print("[avcal] 잠재타겟 on: 평균배율 %.4f, 마스크 %d셀" %
              (float(np.nanmean(1 / np.clip(_av, 0.20, 1.0))), int(_bad.sum())), flush=True)
    return d


if __name__ == "__main__":
    sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    D.load_data = _patched
    if os.environ.get("REVERSE_FOLD", "0") == "1":
        D.FOLDS = [([2024], 2023)]
        D.GROUP_FOLDS = {0: [0], 1: [0], 2: [0]}
        print("[diagnostic] reverse fold: train 2024 -> validate 2023", flush=True)
    T.MODELS_DL = D.MODELS / os.environ.get("OUT", "dl_final2_v2")
    T.MODELS_DL.mkdir(parents=True, exist_ok=True)
    # LOSSV=1 이면 원본 ver=1 손실 그대로 (같은 세대 통제군용 — 구 dl_final2 와의 비교는
    # 세대드리프트 −0.0038 때문에 무효. 판정은 반드시 같은 세대 쌍으로).
    if os.environ.get("LOSSV", "2") == "2":
        assert os.environ.get("FICRPEN", "0") == "1", "FICRPEN=1 필요 (model.py 모듈 플래그)"
        # ★ASYM (2026-08-06): 이봉(가용성) 구간에서 "아래 봉우리를 골라라"는 지시.
        #   진단: g3 예측>=0.84 에서 실측이 0.954(5/5, 51%) / 0.775(4/5, 41%) 두 봉우리인데
        #   모델은 중간 0.864 를 찍어 4점률 0.196. 대칭 밴드손실은 봉우리 선택을 지시하지 않는다.
        #   asym>0 이면 밴드 보상용 유효오차 e_eff=|e|+asym·relu(p-y) 로 **과대예측만** 벌해
        #   불확실할 때 하단 봉우리로 붙는다. LB 증거(g3 근정격 하강 최대레버 t=-6.94)와 정합.
        _asym = float(os.environ.get("ASYM", "0"))
        # ★HPO 훅 (2026-08-08): 손실 하이퍼를 env 로 노출. 미설정 시 기존값과 비트 동일.
        _tau0 = float(os.environ.get("TAU0", "0.025"))
        _tau1 = float(os.environ.get("TAU1", "0.008"))
        _focal = float(os.environ.get("FOCAL", "1.0"))
        _tail = float(os.environ.get("TAIL", "0.15"))
        _blam = float(os.environ.get("BIASLAM", "0.05"))
        T.BASE = dict(T.BASE,
                      loss_kw=dict(ver=2, tau=_tau1, w_actual=True, bias_lam=_blam,
                                   focal=_focal, asym=_asym, tail=_tail,
                                   nmae_w=1 - FICRW, ficr_w=FICRW),
                      tau_sched=(_tau0, _tau1))
        lk = ("ver=2 (FICRW=%.2f, ASYM=%.2f, FICRPEN=1, tau %.4f->%.4f, focal=%.2f, tail=%.2f, biaslam=%.3f)"
              % (FICRW, _asym, _tau0, _tau1, _focal, _tail, _blam))
    else:
        lk = "ver=1 원본 (통제군)"
    print("[%s] MM%s(7)+FUSED%s(3)+GEFS(5)  %s  NANFIX=%s"
          % (os.environ.get("OUT", "dl_final2_v2"), MMSUF, _FUSENAME, lk,
             os.environ.get("NANFIX", "1")), flush=True)
    T.main(smoke=os.environ.get("SMOKE", "0") == "1")
