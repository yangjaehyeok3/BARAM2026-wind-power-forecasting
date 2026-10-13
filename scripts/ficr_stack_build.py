# -*- coding: utf-8 -*-
"""FICR 셀 스택 — 겹침제거 탐욕선택 + lawshot 위 프로브/조합 파일 빌드.

깊이 상한 -0.040 (법칙: 얕은다운만 산다). 셀은 (그룹,월집합,밴드) 단위로
탐욕 선택하되 같은 그룹 내 행 겹침 발생 시 후순위 셀은 미겹침 행에만 적용.
출력: 누적 OOF 사다리 + submissions/f15..f19.csv (개별) + maxficr.csv (전체스택).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("BASEDIR", "dl_final2_swafull"); os.environ.setdefault("W", "0.5")
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import numpy as np, pandas as pd

_r = sys.stdout; sys.stdout = open(os.devnull, "w", encoding="utf-8")
import order_lab as OL
sys.stdout.close(); sys.stdout = _r
A = OL.get()
CAP = np.array([21600., 21600., 21000.]); C = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]

MSETS = {"m1-4": [1, 2, 3, 4], "m5-6": [5, 6], "m7-8": [7, 8], "m9-10": [9, 10],
         "m11-12": [11, 12], "dead8": [1, 2, 3, 4, 7, 8, 11, 12]}
BANDS = [(0.10, 0.25), (0.25, 0.40), (0.40, 0.55), (0.55, 0.70),
         (0.70, 0.84), (0.84, 0.895), (0.895, 0.95), (0.95, 1.05)]
SHIFTS = [-0.010, -0.015, -0.020, -0.025, -0.030, -0.035, -0.040]
MAXN = 1200


def parts(y, p, m):
    e = np.abs(p - y)[m]; yy = y[m]
    nm = 1 - e.mean()
    cr = np.where(e <= 0.06, 4., np.where(e <= 0.08, 3., 0.))
    fc = (yy * cr).sum() / max((yy * 4).sum(), 1e-9)
    return 0.5 * (nm + fc)


# --- 탐욕 스택 (OOF) ---
P = {gi: A[gi]["po"].copy() for gi in range(3)}
Y = {gi: A[gi]["y"] for gi in range(3)}
MO = {gi: pd.DatetimeIndex(A[gi]["dt"]).month.values for gi in range(3)}
BASE = {gi: np.isfinite(P[gi]) & np.isfinite(Y[gi]) & (Y[gi] >= 0.10) for gi in range(3)}
USED = {gi: np.zeros(len(P[gi]), bool) for gi in range(3)}
S0 = {gi: parts(Y[gi], P[gi], BASE[gi]) for gi in range(3)}
chosen = []
for it in range(12):
    best = None
    for gi in range(3):
        s_cur = parts(Y[gi], P[gi], BASE[gi])
        for mnm, ms in MSETS.items():
            inm = np.isin(MO[gi], ms)
            for lo, hi in BANDS:
                cell = BASE[gi] & inm & (P[gi] >= lo) & (P[gi] < hi) & ~USED[gi]
                n = int(cell.sum())
                if n < 100 or n > MAXN:
                    continue
                for sh in SHIFTS:
                    p2 = P[gi].copy(); p2[cell] = np.clip(p2[cell] + sh, 0, 1.05)
                    dS = parts(Y[gi], p2, BASE[gi]) - s_cur
                    if best is None or dS > best[0]:
                        best = (dS, gi, mnm, lo, hi, sh, n, cell)
    if best is None or best[0] < 0.0002:
        break
    dS, gi, mnm, lo, hi, sh, n, cell = best
    P[gi][cell] = np.clip(P[gi][cell] + sh, 0, 1.05)
    USED[gi] |= cell
    chosen.append(dict(g=gi + 1, mset=mnm, lo=lo, hi=hi, shift=sh, n=n, dS=dS, dMacro=dS / 3))
    print("선택 %2d: g%d %s [%.2f,%.3f) %+.3f n=%d dS=%+.5f dMacro=%+.5f"
          % (it + 1, gi + 1, mnm, lo, hi, sh, n, dS, dS / 3))
tot = sum(c["dMacro"] for c in chosen)
print("누적 OOF dMacro: %+.5f (겹침제거 탐욕)" % tot)

# --- lawshot 위 파일 빌드 ---
b = pd.read_csv("submissions/lawshot.csv", encoding="utf-8-sig")
key = pd.to_datetime(b["forecast_kst_dtm"]); mo = key.dt.month.values
V = b[C].to_numpy() / CAP

reg = []
used_f = np.zeros(V.shape, bool)
Vstack = V.copy()
for i, c in enumerate(chosen):
    gi = c["g"] - 1
    inm = np.isin(mo, MSETS[c["mset"]])
    cell = inm & (Vstack[:, gi] >= c["lo"]) & (Vstack[:, gi] < c["hi"]) & ~used_f[:, gi]
    # 개별 프로브 (lawshot 원본 위)
    cell_solo = inm & (V[:, gi] >= c["lo"]) & (V[:, gi] < c["hi"])
    o = b.copy()
    o[C[gi]] = np.where(cell_solo, np.clip(V[:, gi] + c["shift"], 0, 1.05), V[:, gi]) * CAP[gi]
    fn = "submissions/f%d.csv" % (15 + i)
    o.to_csv(fn, index=False, encoding="utf-8-sig")
    # 스택 적용
    Vstack[cell, gi] = np.clip(Vstack[cell, gi] + c["shift"], 0, 1.05)
    used_f[cell, gi] = True
    reg.append(dict(file=os.path.basename(fn), g=c["g"], mset=c["mset"],
                    band="[%.2f,%.3f)" % (c["lo"], c["hi"]), shift=c["shift"],
                    rows_solo=int(cell_solo.sum()), rows_stack=int(cell.sum()),
                    oof_dMacro=round(c["dMacro"], 5)))
    print("빌드 %s: 개별 %d행 / 스택 %d행" % (fn, cell_solo.sum(), cell.sum()))

o = b.copy()
for gi in range(3):
    o[C[gi]] = Vstack[:, gi] * CAP[gi]
o.to_csv("submissions/maxficr.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(reg).to_csv("docs/ficr_stack_registry_0806.csv", index=False, encoding="utf-8-sig")

# 무결성
for fn in ["submissions/maxficr.csv"] + ["submissions/f%d.csv" % (15 + i) for i in range(len(chosen))]:
    v = pd.read_csv(fn, encoding="utf-8-sig")[C].to_numpy() / CAP
    d = int((np.abs(v - V) > 1e-9).sum())
    assert not np.isnan(v).any() and v.max() <= 1.0501
    print("무결성 %s: 변경 %d칸, NaN 없음, 최대 %.3f" % (os.path.basename(fn), d, v.max()))
print("완료 — registry: docs/ficr_stack_registry_0806.csv")
