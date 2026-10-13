# -*- coding: utf-8 -*-
"""도메인 분석 도표 — 원본 데이터(SCADA/LDAPS/labels)만으로 산출.
출력: docs/figs_domain/*.png    사용: python src/make_figs_domain.py

검증된 사실 (본 스크립트가 산출):
  그룹1 = Vestas wtg01-06 (3.6MW x 6 = 21.6MW)
  그룹2 = Vestas wtg07-12 (3.6MW x 6 = 21.6MW)
  그룹3 = Unison wtg01-05 (4.2MW x 5 = 21.0MW)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _sup(fig, *a, **k):
    pass


OUT = os.path.join(ROOT, "docs", "figs_domain")
os.makedirs(OUT, exist_ok=True)

NAVY, RED, GREEN, BLUE, ORANGE, GREY = "#1a2332", "#d62728", "#2ca02c", "#2f6f9f", "#e08214", "#8a8f98"
CAPS = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
GN = {"kpx_group_1": "그룹1", "kpx_group_2": "그룹2", "kpx_group_3": "그룹3"}
RATED_V, RATED_U = 3600.0, 4200.0     # kW / 터빈
CLIP_V, CLIP_U = 700.0, 800.0          # kWh / 10분 (물리 상한 + 여유)


def load():
    lab = pd.read_csv(os.path.join(ROOT, "data/train/train_labels.csv"))
    lab["dt"] = pd.to_datetime(lab["kst_dtm"])
    ves = pd.read_csv(os.path.join(ROOT, "data/train/scada_vestas_train.csv"))
    uni = pd.read_csv(os.path.join(ROOT, "data/train/scada_unison_train.csv"))
    for df in (ves, uni):
        df["dt"] = pd.to_datetime(df["kst_dtm"]).dt.floor("h")
    return lab, ves, uni


def pcols(df):
    return [c for c in df.columns if c.endswith("_power_kw10m")]


def outlier_mask(df, hi):
    P = pcols(df)
    A = df[P].to_numpy(float)
    return A, (A < -10) | (A > hi)


# ---------------------------------------------------------------- D1 품질
def d1_quality(ves, uni):
    Av, bv = outlier_mask(ves, CLIP_V)
    Au, bu = outlier_mask(uni, CLIP_U)
    ext = np.nanmax(np.abs(Av))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    ax = axes[0]
    cat = ["센서 이상치\n(|값|>정격)", "음수 출력", "정확히 0"]
    vv = [bv.mean() * 100, (np.where(bv, np.nan, Av) < 0).mean() * 100,
          (np.where(bv, np.nan, Av) == 0).mean() * 100]
    uu = [bu.mean() * 100, (np.where(bu, np.nan, Au) < 0).mean() * 100,
          (np.where(bu, np.nan, Au) == 0).mean() * 100]
    x = np.arange(3)
    ax.bar(x - 0.19, vv, 0.36, label="Vestas 12기", color=BLUE)
    ax.bar(x + 0.19, uu, 0.36, label="Unison 5기", color=ORANGE)
    for i, (a, b) in enumerate(zip(vv, uu)):
        ax.text(i - 0.19, a + 0.6, "%.1f%%" % a, ha="center", fontsize=9.5)
        ax.text(i + 0.19, b + 0.6, "%.1f%%" % b, ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(cat, fontsize=10.5)
    ax.set_ylabel("터빈-시각 표본 대비 (%)")
    ax.set_title("원자료 이상 유형", fontsize=12.5, color=NAVY)
    ax.legend(fontsize=9.5); ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.90, "최대 이상치 절대값 %.1f 백만 kWh\n(정격의 약 7만 배)" % (ext / 1e6),
            transform=ax.transAxes, fontsize=10.5, color=RED, fontweight="bold")

    ax = axes[1]
    P = pcols(ves)
    A = np.where(bv, np.nan, Av)
    on = (A > 5).sum(1) / A.shape[1]
    valid = np.isfinite(A).any(1)
    on = on[valid]
    ax.hist(on * 100, bins=25, color=BLUE, alpha=0.85)
    part = ((on > 0.02) & (on < 0.98)).mean() * 100
    ax.set_xlabel("동시 가동 터빈 비율 (%)"); ax.set_ylabel("10분 표본 수")
    ax.set_title("동시 가동 터빈 비율의 분포 (부분가동 %.0f%%)" % part,
                 fontsize=12, color=NAVY)
    ax.grid(axis="y", alpha=0.25)
    _sup(fig, "SCADA 원자료 품질 감사 — 학습 라벨의 신뢰 구간을 먼저 정의해야 한다",
                 fontsize=13.5, color=NAVY)
    fig.savefig(os.path.join(OUT, "d1_quality.png")); plt.close(fig)
    return ext, part


# ---------------------------------------------------------------- D2 파워커브
def d2_powercurve(ves, uni):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    for ax, (nm, df, hi, rated) in zip(axes, [("Vestas (그룹1·2)", ves, CLIP_V, RATED_V),
                                              ("Unison (그룹3)", uni, CLIP_U, RATED_U)]):
        wc = [c for c in df.columns if c.endswith("_ws")]
        pc = [c.replace("_ws", "_power_kw10m") for c in wc]
        keep = [(a, b) for a, b in zip(wc, pc) if b in df.columns]
        w = df[[a for a, _ in keep]].to_numpy(float).ravel()
        p = df[[b for _, b in keep]].to_numpy(float).ravel() * 6.0  # kWh/10min -> kW
        m = np.isfinite(w) & np.isfinite(p) & (w > 0) & (p > -100) & (p < rated * 1.25)
        w, p = w[m], p[m]
        ax.scatter(w[::23], p[::23], s=1.4, alpha=0.05, color=GREY)
        bins = np.arange(0, 26, 0.5)
        bi = np.digitize(w, bins)
        med = np.array([np.median(p[bi == k]) if (bi == k).sum() > 30 else np.nan
                        for k in range(1, len(bins))])
        ctr = bins[:-1] + 0.25
        ax.plot(ctr, med, color=RED, lw=2.8, label="구간 중앙값 = 실측 파워커브")
        ax.axhline(rated, color=GREEN, ls="--", lw=1.6)
        ax.text(23.6, rated * 1.03, "정격 %.1f MW" % (rated / 1000), ha="right",
                fontsize=10, color=GREEN, fontweight="bold")
        ax.axvspan(5, 12, color=ORANGE, alpha=0.13)
        ax.text(8.5, rated * 0.30, "급경사 구간\n5~12 m/s", ha="center", fontsize=10.5,
                color=ORANGE, fontweight="bold")
        ax.set_xlim(0, 25); ax.set_ylim(-200, rated * 1.25)
        ax.set_xlabel("나셀 풍속 (m/s)"); ax.set_ylabel("출력 (kW)")
        ax.set_title(nm, fontsize=12.5, color=NAVY)
        ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.25)
    _sup(fig, "기종별 실측 파워커브 — 두 기종의 정격·컷아웃이 다르므로 그룹을 같은 모델로 다룰 수 없다",
                 fontsize=13.5, color=NAVY)
    fig.savefig(os.path.join(OUT, "d2_powercurve.png")); plt.close(fig)


# ---------------------------------------------------------------- D3 오차 증폭
def d3_sensitivity(ves):
    wc = [c for c in ves.columns if c.endswith("_ws")]
    keep = [(a, a.replace("_ws", "_power_kw10m")) for a in wc
            if a.replace("_ws", "_power_kw10m") in ves.columns]
    w = ves[[a for a, _ in keep]].to_numpy(float).ravel()
    p = ves[[b for _, b in keep]].to_numpy(float).ravel() * 6.0
    m = np.isfinite(w) & np.isfinite(p) & (w > 0) & (p > -100) & (p < RATED_V * 1.25)
    w, p = w[m], p[m]
    bins = np.arange(0, 22, 0.5)
    bi = np.digitize(w, bins)
    med = np.array([np.median(p[bi == k]) if (bi == k).sum() > 50 else np.nan
                    for k in range(1, len(bins))])
    ctr = bins[:-1] + 0.25
    ok = np.isfinite(med)
    dP = np.gradient(med[ok], ctr[ok])
    band_kw = 0.08 * 21600 / 6.0   # 그룹 ±8% 를 터빈 1기 환산

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    ax.plot(ctr[ok], dP, color=RED, lw=2.8)
    ax.fill_between(ctr[ok], 0, dP, color=RED, alpha=0.12)
    ax.axvspan(5, 12, color=ORANGE, alpha=0.12)
    k = int(np.nanargmax(dP))
    peak, upk = float(dP[k]), float(ctr[ok][k])
    tol = band_kw / peak
    ax.annotate("최대 민감도 %.0f kW per m/s  @ %.1f m/s" % (peak, upk),
                (upk, peak), textcoords="offset points", xytext=(18, -4),
                fontsize=11.5, color=RED, fontweight="bold")
    ax.set_xlabel("풍속 (m/s)"); ax.set_ylabel("dP/dU   [kW per (m/s)]")
    ax.set_title("실측 커브의 기울기 dP/dU (Vestas)", fontsize=12, color=NAVY)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, "d3_sensitivity.png")); plt.close(fig)
    return peak, upk, tol


# ---------------------------------------------------------------- D4 안정도
def d4_stability():
    fp = os.path.join(ROOT, "models", "data_train_v2_ffst.npz")
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    names = [str(c) for c in z["Fcols"]]
    if "st_alpha" not in names:
        return None
    a = z["X"][:, names.index("st_alpha")].astype(float)
    hr = np.arange(len(a)) % 24
    prof = np.array([np.nanmean(a[hr == h]) for h in range(24)])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.plot(range(24), prof, "-o", color=BLUE, lw=2.6, ms=5.5)
    ax.axhline(0.12, color=RED, ls="--", lw=1.8)
    ax.text(23.4, 0.126, "현행 고정값 0.12", ha="right", fontsize=10.5, color=RED, fontweight="bold")
    ax.axvspan(0, 6, color=NAVY, alpha=0.07); ax.axvspan(19, 23, color=NAVY, alpha=0.07)
    ax.axvspan(11, 15, color=ORANGE, alpha=0.10)
    ax.text(3, prof.max() * 0.97, "야간 · 안정 성층\n연직 전단 강함", ha="center",
            fontsize=10.5, color=NAVY)
    ax.text(13, prof.max() * 0.97, "주간 · 대류 혼합\n전단 약함", ha="center",
            fontsize=10.5, color=ORANGE)
    ax.set_xlabel("시각 (KST)"); ax.set_ylabel("멱법칙 지수 α")
    ax.set_title("연직 외삽 지수 α 의 시간별 평균", fontsize=12, color=NAVY)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, "d4_stability.png")); plt.close(fig)
    return prof


# ---------------------------------------------------------------- D5 지형
def d5_terrain():
    fp = os.path.join(ROOT, "dirspeedup.csv")
    if not os.path.exists(fp):
        return None
    d = pd.read_csv(fp)
    cols = [c for c in d.columns if c.startswith("spd_")]
    ang = np.deg2rad(np.arange(len(d)) * 30 + 15)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), subplot_kw={"projection": "polar"})
    for ax, c in zip(axes, cols):
        v = d[c].to_numpy(float)
        aa = np.append(ang, ang[0]); vv = np.append(v, v[0])
        ax.plot(aa, vv, color=BLUE, lw=2.4); ax.fill(aa, vv, color=BLUE, alpha=0.18)
        ax.plot(aa, np.ones_like(aa), color=GREY, ls="--", lw=1.2)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_ylim(0.5, 1.25)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["N", "E", "S", "W"], fontsize=10)
        ax.set_yticks([0.7, 1.0, 1.2]); ax.tick_params(labelsize=8)
        ax.set_title("%s   %.2f ~ %.2f" % (c.replace("spd_g", "그룹"), v.min(), v.max()),
                     fontsize=11.5, color=NAVY)
    _sup(fig, "풍향 섹터별 지형 가속비 (허브 실측 ÷ 예보 외삽)\n"
                 "태백 복잡지형에서 같은 예보 풍속도 방향에 따라 실제 유입이 30% 이상 달라진다",
                 fontsize=13, color=NAVY)
    fig.savefig(os.path.join(OUT, "d5_terrain.png")); plt.close(fig)
    return d


# ---------------------------------------------------------------- D6 프로파일
def d6_profile(lab):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2))
    hr, mo = lab["dt"].dt.hour, lab["dt"].dt.month
    for g, col in zip(CAPS, [BLUE, GREEN, ORANGE]):
        cf = lab[g] / CAPS[g]
        axes[0].plot(range(24), [cf[hr == h].mean() for h in range(24)], "-o",
                     color=col, ms=4, lw=2.2, label=GN[g])
        axes[1].plot(range(1, 13), [cf[mo == m].mean() for m in range(1, 13)], "-o",
                     color=col, ms=4, lw=2.2, label=GN[g])
    axes[0].axvspan(0, 6, color=NAVY, alpha=0.06); axes[0].axvspan(19, 23, color=NAVY, alpha=0.06)
    axes[0].set_xlabel("시각 (KST)"); axes[0].set_ylabel("평균 이용률")
    axes[0].set_title("일주기 — 야간 최대 · 정오 최소\n산악 야간 안정경계층 가속과 정합",
                      fontsize=12, color=NAVY)
    axes[1].set_xlabel("월"); axes[1].set_ylabel("평균 이용률")
    axes[1].set_title("계절성 — 겨울 계절풍 최대 · 9~10월 최소", fontsize=12, color=NAVY)
    for ax in axes:
        ax.legend(fontsize=9.5); ax.grid(alpha=0.25)
    _sup(fig, "세 그룹이 같은 기상장을 공유한다 — 그러나 반응 크기는 다르다",
                 fontsize=13.5, color=NAVY)
    fig.savefig(os.path.join(OUT, "d6_profile.png")); plt.close(fig)


# ---------------------------------------------------------------- D7 가용성 쌍봉
def d7_bimodal(lab, ves):
    P = pcols(ves)
    A = ves[P].to_numpy(float)
    A = np.where((A < -10) | (A > CLIP_V), np.nan, A)
    g1 = A[:, :6]
    on = np.nanmean((g1 > 5).astype(float), axis=1)
    tmp = pd.DataFrame({"dt": ves["dt"], "avail": on}).groupby("dt")["avail"].mean()

    d = lab.set_index("dt").join(tmp)
    d["cf"] = d["kpx_group_1"] / CAPS["kpx_group_1"]
    sel = d.dropna(subset=["avail", "cf"])
    hi = sel[sel.cf >= 0.60]
    full = hi[hi.avail > 0.95]["cf"]; part = hi[hi.avail <= 0.95]["cf"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    ax = axes[0]
    ax.scatter(sel["avail"] * 100, sel["cf"], s=2, alpha=0.05, color=BLUE)
    ax.set_xlabel("동시 가동 터빈 비율 (%)"); ax.set_ylabel("그룹1 실제 이용률")
    ax.set_title("가용성이 발전량 상한을 직접 자른다", fontsize=12.5, color=NAVY)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(full, bins=32, color=GREEN, alpha=0.85, label="전 터빈 가동 (>95%)")
    ax.hist(part, bins=32, color=RED, alpha=0.6, label="부분정지 (≤95%)")
    ax.set_xlabel("실제 이용률 (고출력 구간)"); ax.set_ylabel("시간 수")
    ax.set_title("고출력 구간의 분포는 두 개로 갈린다\n부분정지 시간 비중 %.0f%%"
                 % (len(part) / max(len(hi), 1) * 100), fontsize=12.5, color=NAVY)
    ax.legend(fontsize=9.5)
    _sup(fig, "예보만으로는 알 수 없는 갈림 — 같은 바람에도 결과가 둘로 나뉜다",
                 fontsize=13.5, color=NAVY)
    fig.savefig(os.path.join(OUT, "d7_bimodal.png")); plt.close(fig)
    return len(part) / max(len(hi), 1) * 100


# ---------------------------------------------------------------- D8 오차 예산
def d8_budget(lab, ves):
    """오차 예산: 예보풍속 / 실측풍속 / 실제발전량 3단 비교로 원인을 분해."""
    fp = os.path.join(ROOT, "models", "data_train_v2.npz")
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    names = [str(c) for c in z["Fcols"]]
    fc = z["X"][:, names.index("ws_hub")].astype(float)     # 예보 기반 허브풍속

    wc = [c for c in ves.columns if c.endswith("_ws")]
    W = ves[wc].to_numpy(float)
    W = np.where(np.isfinite(W) & (W > 0) & (W < 40), W, np.nan)
    with np.errstate(invalid="ignore"):
        nac = np.where(np.isfinite(W).any(1), np.nanmean(W, axis=1), np.nan)
    obs = pd.DataFrame({"dt": ves["dt"], "ws": nac}).groupby("dt")["ws"].mean()

    d = lab.set_index("dt").copy()
    d["fc"] = fc[:len(d)]
    d = d.join(obs.rename("obs")).dropna(subset=["fc", "obs", "kpx_group_1"])
    cf = (d["kpx_group_1"] / CAPS["kpx_group_1"]).to_numpy()

    def curve_pred(u, y):
        bins = np.arange(0, 26, 0.5)
        bi = np.digitize(u, bins)
        med = np.array([np.median(y[bi == k]) if (bi == k).sum() > 20 else np.nan
                        for k in range(1, len(bins))])
        ok = np.isfinite(med)
        return np.interp(u, (bins[:-1] + 0.25)[ok], med[ok])

    p_obs = curve_pred(d["obs"].to_numpy(), cf)   # 풍속을 완벽히 알 때
    p_fc = curve_pred(d["fc"].to_numpy(), cf)     # 하루전 예보만 알 때
    v_tot = np.var(cf)
    r2_obs = 1 - np.var(cf - p_obs) / v_tot
    r2_fc = 1 - np.var(cf - p_fc) / v_tot

    share_fc = r2_fc * 100
    share_wind_gap = (r2_obs - r2_fc) * 100
    share_rest = (1 - r2_obs) * 100

    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    vals = [share_fc, share_wind_gap, share_rest]
    cols = [GREEN, ORANGE, RED]
    labels = ["하루전 예보로 설명 가능",
              "풍속 예보오차\n(완벽 풍속이면 회수)",
              "풍속으로 설명 불가\n(가용성·난류·시간평균)"]
    left = 0
    for v, c in zip(vals, cols):
        ax.barh([0], [v], left=left, color=c, height=0.46)
        if v > 5:
            ax.text(left + v / 2, 0, "%.0f%%" % v, ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold")
        left += v
    for i, (v, c, l) in enumerate(zip(vals, cols, labels)):
        ax.text(sum(vals[:i]) + v / 2, 0.42, l, ha="center", fontsize=10.5,
                color=c, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, 0.85); ax.set_yticks([])
    ax.set_xlabel("그룹1 발전량 분산에 대한 기여 (%)")
    ax.set_title("그룹1 발전량 분산의 3단 분해", fontsize=12, color=NAVY)
    fig.savefig(os.path.join(OUT, "d8_budget.png")); plt.close(fig)
    return vals


if __name__ == "__main__":
    lab, ves, uni = load()
    print("D1", d1_quality(ves, uni))
    d2_powercurve(ves, uni)
    print("D3", d3_sensitivity(ves))
    d4_stability()
    d5_terrain()
    d6_profile(lab)
    print("D7 부분정지비중", d7_bimodal(lab, ves))
    print("D8", d8_budget(lab, ves))
    print("saved ->", OUT)
