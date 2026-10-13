# -*- coding: utf-8 -*-
"""상관 구조 도표 2종 — 타깃 상관 barh + 가족 블록 히트맵, 월×시 발전 히트맵.
출력: docs/figs_domain/e4_corr.png, e5_monthhour.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 145
plt.rcParams["savefig.bbox"] = "tight"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figs_domain")
NAVY, RED, BLU, GREY = "#1a2332", "#e12900", "#5d77f8", "#8a8f98"
CAPS = [21600.0, 21600.0, 21000.0]


def load_all():
    lab = pd.read_csv(os.path.join(ROOT, "data/train/train_labels.csv"))
    y = np.nanmean(lab[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].to_numpy(float)
                   / np.array(CAPS), axis=1)
    z = np.load(os.path.join(ROOT, "models", "data_train_v2.npz"), allow_pickle=True)
    Xb = pd.DataFrame(np.asarray(z["X"], float), columns=[str(c) for c in z["Fcols"]])
    dom = pd.read_csv(os.path.join(ROOT, "ext_domain25_train.csv"))
    dom = dom.drop(columns=[c for c in dom.columns if "dtm" in c.lower() or "time" in c.lower()],
                   errors="ignore")
    n = min(len(Xb), len(dom), len(y))
    X = pd.concat([Xb.iloc[:n].reset_index(drop=True),
                   dom.iloc[:n].reset_index(drop=True)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y[:n], lab.iloc[:n]


def fam(c):
    if c.startswith(("wsfu", "ws_iqr")):
        return "SCADA 대리", "#d95f02"
    if c.startswith("pc_"):
        return "파워커브", "#7570b3"
    if c.startswith(("gx_", "gy_", "gsd", "gup", "gdn", "gud")):
        return "격자 공간", "#1b9e77"
    if c.startswith(("jma_", "wj_")):
        return "JMA", "#e7298a"
    if c.startswith("rg_"):
        return "광역", "#66a61e"
    if c.startswith("wk_"):
        return "wake", "#a6761d"
    if c.startswith("ph_"):
        return "물리 위상", "#666666"
    return "기본/기타", "#2f6f9f"


def e4_corr(X, y):
    cors = {}
    for c in X.columns:
        v = X[c].to_numpy(float)
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() < 5000 or np.nanstd(v[m]) < 1e-9:
            continue
        cors[c] = float(np.corrcoef(v[m], y[m])[0, 1])
    S = pd.Series(cors).dropna()
    ranked = S.reindex(S.abs().sort_values(ascending=False).index)
    # 다양성 필터: 이미 뽑힌 피처와 상관 0.98 이상인 근중복은 건너뜀
    keep = []
    for c in ranked.index:
        v = X[c].to_numpy(float)
        dup = False
        for k in keep:
            w = X[k].to_numpy(float)
            m = np.isfinite(v) & np.isfinite(w)
            if m.sum() > 5000 and abs(np.corrcoef(v[m], w[m])[0, 1]) >= 0.98:
                dup = True
                break
        if not dup:
            keep.append(c)
        if len(keep) == 14:
            break
    top = ranked.reindex(keep)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    names = top.index[::-1]
    vals = top.to_numpy()[::-1]
    cols = [fam(c)[1] for c in names]
    ax.barh(range(len(vals)), vals, color=cols, height=0.66)
    ax.axvline(0, color="black", lw=0.9)
    for i, (v, c) in enumerate(zip(vals, names)):
        ax.text(v + (0.012 if v >= 0 else -0.012), i, "%.2f" % v,
                va="center", ha="left" if v >= 0 else "right", fontsize=8.6)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8.8)
    ax.set_xlabel("발전량(3그룹 평균 이용률)과의 상관", fontsize=10.5)
    ax.set_title("타깃 상관 상위 14개 피처 (색 = 피처 가족)", fontsize=11.5, color=NAVY)
    ax.set_xlim(min(vals.min() - 0.1, -0.15), vals.max() + 0.12)
    ax.grid(axis="x", alpha=0.25)
    seen = {}
    for c in names:
        f, col = fam(c)
        seen[f] = col
    hs = [plt.Rectangle((0, 0), 1, 1, color=v) for v in seen.values()]
    ax.legend(hs, seen.keys(), fontsize=8, ncol=len(seen),
              loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)

    picks = [c for c in ["ws_hub", "ws50", "rho_ws_hub3", "wsfu_v", "wsfu_u",
                         "pc_g1_cf", "pc_g3_cf", "jma_wmean", "wj_g1_mean",
                         "rg_wmean", "gsd_ws50", "gx_ws50", "dom_blh", "wk_g1_eff"]
             if c in X.columns]
    M = X[picks].astype(float)
    C = M.corr().to_numpy()
    ax = axes[1]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(picks)):
        for j in range(len(picks)):
            ax.text(j, i, "%.1f" % C[i, j], ha="center", va="center",
                    fontsize=6.8, color="white" if abs(C[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(picks)))
    ax.set_xticklabels(picks, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(picks)))
    ax.set_yticklabels(picks, fontsize=8)
    ax.set_title("대표 피처 14종의 상호 상관 (가족 순 정렬)", fontsize=11.5, color=NAVY)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(os.path.join(OUT, "e4_corr.png"))
    plt.close(fig)
    return top


def e5_monthhour(lab):
    d = lab.copy()
    d["dt"] = pd.to_datetime(d["kst_dtm"])
    cf = np.nanmean(d[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].to_numpy(float)
                    / np.array(CAPS), axis=1)
    d["cf"] = cf
    d["mo"] = d.dt.dt.month
    d["hr"] = d.dt.dt.hour
    P = d.pivot_table(index="mo", columns="hr", values="cf", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    im = ax.imshow(P.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(["%d시" % h for h in range(0, 24, 2)], fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(["%d월" % m for m in P.index], fontsize=9)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.iloc[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.4,
                    color="white" if v > 0.32 else "#333333")
    ax.set_title("월 × 시각 평균 이용률 (2022-2024, 3그룹 평균)", fontsize=11.5, color=NAVY)
    fig.colorbar(im, ax=ax, shrink=0.9, label="평균 이용률")
    fig.savefig(os.path.join(OUT, "e5_monthhour.png"))
    plt.close(fig)
    return P


if __name__ == "__main__":
    X, y, lab = load_all()
    print("X", X.shape)
    top = e4_corr(X, y)
    print(top.head(8))
    P = e5_monthhour(lab)
    print("month-hour min %.2f max %.2f" % (P.min().min(), P.max().max()))
    print("saved ->", OUT)
