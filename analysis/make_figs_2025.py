# -*- coding: utf-8 -*-
"""2025년(예측 대상) 기상 특성 — 학습 기간과의 비교. 출력 docs/figs_domain/e6_year2025.png"""
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
NAVY, RED, GREY = "#1a2332", "#e12900", "#8a8f98"
YCOL = {2022: "#b7c6d9", 2023: "#8aa5c4", 2024: "#5d80ab", 2025: "#e12900"}


def load(split):
    use = ["forecast_kst_dtm", "grid_id",
           "heightAboveGround_10_10u", "heightAboveGround_10_10v"]
    d = pd.read_csv(os.path.join(ROOT, "data", split, "ldaps_%s.csv" % split), usecols=use)
    d = d[d.grid_id == 15]  # 남동쪽 최고 상관 격자
    u = d["heightAboveGround_10_10u"].to_numpy(float)
    v = d["heightAboveGround_10_10v"].to_numpy(float)
    ws = np.hypot(u, v)
    wd = (np.degrees(np.arctan2(-u, -v))) % 360.0   # 기상학 풍향(불어오는 방향)
    yr = pd.to_datetime(d["forecast_kst_dtm"]).dt.year.to_numpy()
    return ws, wd, yr


def main():
    ws1, wd1, y1 = load("train")
    ws2, wd2, y2 = load("test")
    ws = np.concatenate([ws1, ws2]); wd = np.concatenate([wd1, wd2])
    yr = np.concatenate([y1, np.full_like(y2, 2025)])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    ax = axes[0]
    bins = np.linspace(0, 16, 33)
    stats = {}
    for y in (2022, 2023, 2024, 2025):
        m = yr == y
        h, _ = np.histogram(ws[m], bins=bins, density=True)
        ax.plot(bins[:-1] + 0.25, h, color=YCOL[y], lw=2.6 if y == 2025 else 1.7,
                label="%d%s" % (y, " (예측 대상)" if y == 2025 else ""))
        stats[y] = (float(np.nanmean(ws[m])), float(np.nanmean((ws[m] >= 8))))
    ax.set_xlabel("10m 예보 풍속 (m/s)"); ax.set_ylabel("확률밀도")
    ax.set_title("연도별 예보 풍속 분포 (남동 격자)", fontsize=12, color=NAVY)
    ax.legend(fontsize=9.5); ax.grid(alpha=0.25)

    ax = axes[1]
    sec_edges = np.arange(0, 361, 45)
    sec_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    width = 0.2
    for k, y in enumerate((2022, 2023, 2024, 2025)):
        m = yr == y
        cnt, _ = np.histogram(wd[m], bins=sec_edges)
        share = cnt / cnt.sum() * 100
        ax.bar(np.arange(8) + (k - 1.5) * width, share, width, color=YCOL[y],
               label=str(y))
    ax.set_xticks(range(8)); ax.set_xticklabels(sec_names, fontsize=10)
    ax.set_ylabel("빈도 (%)")
    ax.set_title("연도별 풍향 분포", fontsize=12, color=NAVY)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
    fig.savefig(os.path.join(OUT, "e6_year2025.png"))
    plt.close(fig)

    print("연도별 평균 풍속 / 8m/s 이상 비중:")
    for y, (m, hi) in stats.items():
        print("  %d: %.2f m/s  /  %.1f%%" % (y, m, hi * 100))
    for y in (2022, 2023, 2024, 2025):
        m = yr == y
        cnt, _ = np.histogram(wd[m], bins=sec_edges)
        share = cnt / cnt.sum() * 100
        print("  %d 풍향: W %.1f%%  NW %.1f%%  SW %.1f%%" % (y, share[6], share[7], share[5]))


if __name__ == "__main__":
    main()
