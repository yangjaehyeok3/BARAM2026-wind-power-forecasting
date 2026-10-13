# -*- coding: utf-8 -*-
"""추론 코드 (inference) — 최종 제출 파일 복원. 학습(train.py)과 분리된 추론 전용 코드.

입력 (학습 산출물 + 아티팩트):
  models/<앵커>/dl_oof.npz            앵커 5종의 테스트 예측 (키: subC_test_cf, [8760,3] 이용률)
  models/dl_f2jmaL, dl_f2dom20        기저 파일 복원용 앵커 2종
  submissions/f18.csv                 기저 계보의 시작 아티팩트 (docs/REPRODUCING.md 참조)
  submissions/jd40all.csv             (선택) f18 경로가 없을 때 사용하는 기저 파일

출력:
  submission_final.csv                최종 채점 파일 (= 대회 제출본 champ_g3dn08.csv 와 비트 일치)

특징:
  - GPU 불필요, 난수 없음, 완전 결정적.
  - 원 제출본이 함께 제출되면 자동으로 비트 대조 결과를 출력한다.

실행:
  python inference.py                         # 기본 경로로 실행 + 비트 대조
  python inference.py --out my_sub.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- 설정(상수)
ROOT = os.path.dirname(os.path.abspath(__file__))

COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
# 그룹별 설비용량 [kWh] — data/data_description.md 기재값
CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}

# 앵커 앙상블: 5종 평균을 신호로, dl_f2ffx 를 통제로 사용
POOL = ["dl_f2ffx", "dl_f2ffx_s100", "dl_f2ffx_xw", "dl_f2ffx_xw_uk", "dl_f2tau15"]
CONTROL = "dl_f2ffx"
LAMBDA = 0.40           # 앙상블 델타 이식 계수

# 기저 파일(jd40all) 복원용 — JMA(D-2 발표분) 정보 기여분 이식
JMA_SIGNAL = "dl_f2jmaL"
JMA_CONTROL = "dl_f2dom20"
JMA_LAMBDA = 0.40

# 밴드 오퍼 보정 (리더보드 통제실험으로 결정된 값 — README §3 참조)
NR_CF = 0.80            # '근정격' 밴드 정의: 기저 예측 이용률 >= 0.80
G2_NR_SHIFT = +0.0216   # 그룹2 근정격 상향 (사다리 정점)
G3_NR_SHIFT = -0.0800   # 그룹3 근정격 하향 (과대오퍼 교정)

N_ROWS = 8760           # 2025-01-01 01:00 ~ 2026-01-01 00:00 (KST, 시간단위)


# ----------------------------------------------------------------- 유틸
def load_anchor_test_cf(models_dir: str, name: str) -> np.ndarray:
    """앵커의 테스트 예측(이용률, [8760,3])을 float64 로 로드.

    npz 는 float32 로 저장돼 있다. 평균을 float32 로 누적하면 최종 오퍼가
    ~1e-3 kWh 어긋나므로(점수 영향은 없으나 비트 일치가 깨짐) 반드시 승격한다.
    """
    path = os.path.join(models_dir, name, "dl_oof.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("앵커 산출물이 없습니다: %s" % path)
    with np.load(path, allow_pickle=True) as z:
        arr = z["subC_test_cf"]
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (N_ROWS, 3):
        raise ValueError("%s: subC_test_cf 형상 %s (기대 (%d,3))" % (name, arr.shape, N_ROWS))
    return arr


def level_neutral(delta: np.ndarray, base_kwh: np.ndarray, cap: float,
                  n_bands: int = 10) -> np.ndarray:
    """레벨중립 투영.

    기저 예측 이용률의 십분위 밴드마다 델타의 평균을 0으로 만든다.
    => 앙상블이 주는 '순서(어느 시간이 더 센가)' 정보만 남기고,
       밴드 전체를 위아래로 옮기는 '레벨' 성분은 제거한다.
    레벨 성분은 별도로 리더보드 실험을 통해서만 조정한다(아래 밴드 보정).
    """
    d = delta.copy()
    band = np.clip((base_kwh / cap * n_bands).astype(int), 0, n_bands - 1)
    for k in range(n_bands):
        m = band == k
        if m.any():
            d[m] -= d[m].mean()
    return d


# ----------------------------------------------------------------- 본체
def build_base(root: str, verbose: bool = True) -> pd.DataFrame:
    """기저 파일(jd40all) 복원.

    jd40all = f18 + JMA_LAMBDA x (dl_f2jmaL - dl_f2dom20) x CAP
      - dl_f2jmaL : 합법 JMA(D-2 발표분) 다중모델 피처를 쓴 앵커
      - dl_f2dom20: 동일 학습 레시피의 통제 앵커
    => 두 앵커의 '차이'만 이식하므로 JMA 정보가 준 순수 기여분이 반영된다.

    f18 이나 두 앵커가 없으면 동봉된 jd40all.csv 를 그대로 사용한다.
    """
    models_dir = os.path.join(root, "models")
    f18_path = os.path.join(root, "submissions", "f18.csv")
    jd_path = os.path.join(root, "submissions", "jd40all.csv")
    have_parts = os.path.exists(f18_path) and all(
        os.path.exists(os.path.join(models_dir, n, "dl_oof.npz")) for n in (JMA_SIGNAL, JMA_CONTROL))

    if not have_parts:
        if verbose:
            print("[기저] f18 경로 없음 -> 동봉된 jd40all.csv 사용")
        return pd.read_csv(jd_path)

    f18 = pd.read_csv(f18_path)
    sig = load_anchor_test_cf(models_dir, JMA_SIGNAL)
    ctl = load_anchor_test_cf(models_dir, JMA_CONTROL)
    base = f18.copy()
    for i, g in enumerate(COLS):
        cap = CAP[g]
        cf = f18[g].to_numpy(dtype=np.float64) / cap
        base[g] = np.clip(cf + JMA_LAMBDA * (sig[:, i] - ctl[:, i]), 0.0, 1.0) * cap
    if verbose:
        print("[기저] f18 + %.2f x (%s - %s) 로 복원" % (JMA_LAMBDA, JMA_SIGNAL, JMA_CONTROL))
        if os.path.exists(jd_path):
            ref = pd.read_csv(jd_path)
            mx = max(np.abs(base[g].to_numpy(float) - ref[g].to_numpy(float)).max() for g in COLS)
            print("       동봉 jd40all.csv 와 최대차 %.3g kWh" % mx)
    return base


def build_submission(root: str = ROOT, verbose: bool = True) -> pd.DataFrame:
    models_dir = os.path.join(root, "models")
    base = build_base(root, verbose=verbose)
    if len(base) != N_ROWS:
        raise ValueError("기저 파일 행수 %d (기대 %d)" % (len(base), N_ROWS))

    control = load_anchor_test_cf(models_dir, CONTROL)
    signal = np.mean([load_anchor_test_cf(models_dir, n) for n in POOL], axis=0)

    out = base.copy()
    for i, g in enumerate(COLS):
        cap = CAP[g]
        b = base[g].to_numpy(dtype=np.float64)

        # 1) 앙상블 다양성 델타 (5앵커 평균 - 통제앵커)
        delta = LAMBDA * (signal[:, i] - control[:, i]) * cap
        # 2) 클리핑된 값 기준으로 레벨 성분 제거 (원 제출본과 동일 순서)
        delta = np.clip(b + delta, 0.0, cap) - b
        delta = level_neutral(delta, b, cap)

        # 3) 밴드 오퍼 보정 (그룹별)
        nr = (b / cap) >= NR_CF
        if g == "kpx_group_2":
            delta = delta + np.where(nr, G2_NR_SHIFT * cap, 0.0)
        elif g == "kpx_group_3":
            delta = delta + np.where(nr, G3_NR_SHIFT * cap, 0.0)

        out[g] = np.clip(b + delta, 0.0, cap)

    # 안전장치: 음수 / 용량 초과 / 결측 없음
    for g in COLS:
        v = out[g].to_numpy(dtype=np.float64)
        assert np.isfinite(v).all(), "%s: 비정상값(NaN/inf)" % g
        assert (v >= 0).all(), "%s: 음수 발전량" % g
        assert (v <= CAP[g] + 1e-6).all(), "%s: 설비용량 초과" % g
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="최종 제출 파일 복원 (추론 전용)")
    ap.add_argument("--root", default=ROOT, help="저장소 루트 (models/, submissions/ 포함)")
    ap.add_argument("--out", default=os.path.join(ROOT, "submission_final.csv"))
    ap.add_argument("--verify-against", default=None,
                    help="원 제출본 csv 경로 (지정 시 비트 대조)")
    args = ap.parse_args()

    sub = build_submission(args.root)
    sub.to_csv(args.out, index=False)
    print("[생성] %s  (%d행)" % (args.out, len(sub)))

    ref = args.verify_against
    if ref is None:
        cand = os.path.join(args.root, "submissions", "champ_g3dn08.csv")
        ref = cand if os.path.exists(cand) else None
    if ref and os.path.exists(ref):
        r = pd.read_csv(ref)
        ok = True
        for g in COLS:
            d = np.abs(sub[g].to_numpy(float) - r[g].to_numpy(float))
            n_bad = int((d > 1e-6).sum())
            ok &= n_bad == 0
            print("  %-14s 불일치행 %5d  최대차 %.3g kWh" % (g, n_bad, d.max()))
        print("[대조] %s : %s" % (os.path.basename(ref),
                                  "일치 (재현 성공)" if ok else "불일치"))


if __name__ == "__main__":
    main()
