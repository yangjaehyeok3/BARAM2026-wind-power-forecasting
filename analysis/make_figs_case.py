# -*- coding: utf-8 -*-
"""사례·성분·공간 도표 3종 (전부 실데이터 계산). 출력 docs/figs_domain/
사용: BASEDIR=dl_final2_swafull W=0.5 python src/make_figs_case.py
"""
import os
import sys
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
def _sup(fig, *a, **k):
    pass


OUT = os.path.join(ROOT, "docs", "figs_domain")
NAVY, RED, GREEN, BLUE, ORANGE, GREY = "#1a2332", "#c8402f", "#3a8f3a", "#2f6f9f", "#e08214", "#8a8f98"

sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments")]
os.environ.setdefault("BASEDIR", "dl_final2_swafull")
os.environ.setdefault("W", "0.5")
import order_lab  # noqa: E402

A = order_lab.get()


def group_score(y, p):
    m = y >= 0.10
    e = np.abs(y - p)
    price = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    w = np.where(m, y, 0.0)
    ficr = (w * price).sum() / max((w * 4.0).sum(), 1e-9)
    nmae = (e[m]).mean()
    return 0.5 * (1 - nmae) + 0.5 * ficr


def anchor_oof(name):
    z = np.load(os.path.join(ROOT, "models", name, "dl_oof.npz"), allow_pickle=True)
    return [np.asarray(z["prod_oof_p%d" % k], float) for k in range(3)]


# ------------------------------------------------------------ E1 사례 일
def e1_casedays():
    k = 0  # 그룹1
    y = np.asarray(A[k]["y"], float)
    p = np.asarray(A[k]["po"], float)
    dt = pd.to_datetime(A[k]["dt"])
    df = pd.DataFrame({"dt": dt, "y": y, "p": p})
    df["date"] = df.dt.dt.date
    g = df.groupby("date").agg(n=("y", "size"), my=("y", "mean"), mp=("p", "mean"),
                               mae=("y", lambda s: 0))
    # 일별 지표
    daily = []
    for d, sub in df.groupby("date"):
        if len(sub) < 24:
            continue
        e = np.abs(sub.y - sub.p)
        m = sub.y >= 0.10
        if m.sum() < 20:
            continue
        hit = float(((e <= 0.08) & m).sum() / m.sum())
        daily.append((d, float(sub.y.mean()), float(sub.p.mean()), float(e[m].mean()), hit))
    D = pd.DataFrame(daily, columns=["date", "my", "mp", "mae", "hit"])
    # 잘 맞은 windy 날: 발전 많고 hit 높음
    good = D[(D.my > 0.45)].sort_values("hit", ascending=False).iloc[0]
    # 빗나간 날: 예보 높은데 실제 낮음 (down-bust)
    D["bust"] = D.mp - D.my
    bad = D[(D.mp > 0.45)].sort_values("bust", ascending=False).iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3), sharey=True)
    fig.subplots_adjust(top=0.76, wspace=0.08)
    for ax, row, ttl, col in [(axes[0], good, "예보가 맞은 날", GREEN),
                              (axes[1], bad, "예보가 빗나간 날", RED)]:
        sub = df[df.date == row.date].sort_values("dt")
        hrs = sub.dt.dt.hour + (sub.dt.dt.date != row.date).astype(int) * 24
        hrs = np.arange(len(sub))
        ax.fill_between(hrs, sub.y - 0.08, sub.y + 0.08, color=GREY, alpha=0.25,
                        label="정산 허용창 (실제 ±8%)")
        ax.plot(hrs, sub.y, color=NAVY, lw=2.4, label="실제 발전량")
        ax.plot(hrs, sub.p, color=col, lw=2.2, ls="--", label="하루전 예측")
        e = np.abs(sub.y - sub.p); m = sub.y >= 0.10
        hit = ((e <= 0.08) & m).sum(); tot = int(m.sum())
        ax.set_title("%s   %s\n창 통과 %d/%d시간" % (ttl, row.date, hit, tot),
                     fontsize=12.5, color=col)
        ax.set_xlabel("시각"); ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["01시", "07시", "13시", "19시", "24시"])
        ax.grid(alpha=0.22); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("이용률 (발전량/설비용량)")
    axes[0].legend(fontsize=9.5, loc="lower right")
    _sup(fig, "그룹1 사례 — 하루전 예측이 감당할 수 있는 날과 없는 날",
                 fontsize=13.5, color=NAVY, y=0.99)
    fig.savefig(os.path.join(OUT, "e1_casedays.png")); plt.close(fig)
    return str(good.date), str(bad.date)


# ------------------------------------------------------------ E2 개선 워터폴
def e2_waterfall():
    ys = [np.asarray(A[k]["y"], float) for k in range(3)]
    stages = [("다중 NWP 기본", "dl_f2dom20"),
              ("+ JMA 광역 예보 추가", "dl_f2jmaL"),
              ("+ 피처 정제·확장\n(왜도 보정·지형·안정도)", "dl_f2ffx")]
    scores = []
    for lab, name in stages:
        ps = anchor_oof(name)
        s = np.mean([group_score(ys[k], ps[k][:len(ys[k])]) for k in range(3)])
        scores.append((lab, float(s)))

    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    xs = np.arange(len(scores))
    base = scores[0][1]
    for i, (lab, sc) in enumerate(scores):
        col = BLUE if i == 0 else (GREEN if sc >= scores[i - 1][1] else RED)
        ax.bar(i, sc - 0.640, bottom=0.640, color=col, width=0.58,
               alpha=0.95 if i in (0, len(scores) - 1) else 0.8)
        ax.text(i, sc + 0.0006, "%.4f" % sc, ha="center", fontsize=11.5, fontweight="bold",
                color=NAVY)
        if i > 0:
            d = sc - scores[i - 1][1]
            ax.text(i, sc - 0.0022, "%+.4f" % d, ha="center", fontsize=11,
                    color="white", fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in scores], fontsize=10.5)
    ax.set_ylim(0.640, max(s[1] for s in scores) + 0.004)
    ax.set_ylabel("내부 검증 점수 (연도 홀드아웃)")
    ax.set_title("성분별 기여 (동일 검증셋 · 동일 채점식)", fontsize=12, color=NAVY)
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(os.path.join(OUT, "e2_waterfall.png")); plt.close(fig)
    return scores


# ------------------------------------------------------------ E3 공간 상관
def e3_spatial():
    usecols = ["forecast_kst_dtm", "grid_id", "latitude", "longitude",
               "heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    ld = pd.read_csv(os.path.join(ROOT, "data/train/ldaps_train.csv"), usecols=usecols)
    ld["ws"] = np.hypot(ld["heightAboveGround_10_10u"], ld["heightAboveGround_10_10v"])
    pos = ld.groupby("grid_id")[["latitude", "longitude"]].first()
    piv = ld.pivot_table(index="forecast_kst_dtm", columns="grid_id", values="ws")
    piv.index = pd.to_datetime(piv.index)

    lab = pd.read_csv(os.path.join(ROOT, "data/train/train_labels.csv"))
    lab["dt"] = pd.to_datetime(lab["kst_dtm"])
    lab = lab.set_index("dt")
    CAPS = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for ax, g in zip(axes, CAPS):
        cf = (lab[g] / CAPS[g]).reindex(piv.index)
        cors = {}
        for gid in piv.columns:
            s = piv[gid]
            m = s.notna() & cf.notna()
            cors[gid] = np.corrcoef(s[m], cf[m])[0, 1]
        lats = pos["latitude"]; lons = pos["longitude"]
        cvals = [cors[i] for i in pos.index]
        sc = ax.scatter(lons, lats, c=cvals, s=650, cmap="YlOrRd", marker="s",
                        vmin=min(cvals) - 0.01, vmax=max(cvals) + 0.005,
                        edgecolor=NAVY, linewidth=0.7)
        best = max(cors, key=cors.get)
        for gid in pos.index:
            ax.text(lons[gid], lats[gid], "%.2f" % cors[gid], ha="center", va="center",
                    fontsize=8.2, color=NAVY, fontweight="bold")
        ax.scatter([lons[best]], [lats[best]], s=980, facecolor="none",
                   edgecolor=BLUE, linewidth=2.6, marker="s")
        ax.set_title("%s   최고 상관 r=%.2f (남동 격자)" % (g.replace("kpx_group_", "그룹"), cors[best]),
                     fontsize=11.5, color=NAVY)
        ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)" if g == "kpx_group_1" else "")
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.2f" % v))
        ax.set_aspect(1 / np.cos(np.deg2rad(37.3)))
    _sup(fig, "예보 격자(4×5)별 바람-발전량 상관 — 남동쪽 격자가 지배하고 서쪽과 r 차이 0.3\n"
                 "단일 격자 선택 대신 격자 전체를 공간 토큰으로 입력하는 근거", fontsize=13, color=NAVY)
    fig.savefig(os.path.join(OUT, "e3_spatial.png")); plt.close(fig)


if __name__ == "__main__":
    print("E1", e1_casedays())
    print("E2", e2_waterfall())
    e3_spatial()
    print("saved -> ", OUT)
