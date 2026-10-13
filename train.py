# -*- coding: utf-8 -*-
"""학습 코드 (train) — 최종 제출에 사용된 앵커 모델 5종 학습.

추론(inference.py)과 분리된 학습 전용 코드.

수행 내용
  1) 피처 캐시 빌드 (원본 data/ + 외부 NWP -> models/data_*_v2*.npz)
  2) 앵커 5종 학습 (각 ~28분, RTX 5090 기준) -> models/<이름>/dl_oof.npz, *.pt

산출물 중 추론에 실제로 필요한 것은 각 앵커의 dl_oof.npz["subC_test_cf"] 이며,
inference.py 가 이를 읽어 최종 제출 파일을 결정적으로 복원한다.

주의: GPU 연산 비결정성(cuDNN/atomic)으로 재학습 결과가 원 산출물과
소수점 하위에서 달라질 수 있다. 채점 점수 복원은 함께 제출한 npz +
inference.py 경로로 보장된다(비트 일치 검증 완료).

실행
  python train.py --list                 # 학습 대상 목록 확인
  python train.py --cache                # 피처 캐시만 빌드
  python train.py --all                  # 캐시 + 앵커 5종 전부 (약 2.5시간)
  python train.py --anchor dl_f2tau15    # 특정 앵커만
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = os.path.join(ROOT, "src", "dl", "data_prep.py")
TRAIN = os.path.join(ROOT, "src", "dl", "dl_final2_v2_full.py")
MODELS = os.path.join(ROOT, "models")

# 모든 앵커 공통 설정 (원 학습 런처 run_anchor2.ps1 과 동일)
COMMON_ENV = {
    "FICRPEN": "1",     # 점수식 FICR 이중밴드(6%/8%) 정렬 페널티 사용
    "FICRW": "0.50",    # 손실 배합: (1-FICRW)*nMAE + FICRW*(1-softFICR)
    "ASYM": "0",        # 과대/과소 비대칭 페널티 미사용
    "MMSUF": "_f",      # 다중모델 NWP 피처 버전
    "NANFIX": "1",      # 격자 토큰 결측: 시간축 선형보간 후 잔여만 0
    "RAMPFIX": "0",
    "LOSSV": "2",       # 손실 v2
    "DOMF": "1",
    "DOMV": "25",       # 도메인 피처 세트 v25 (186열)
    "GFILM": "1",       # 그룹별 FiLM 변조 사용
    "GFILM_LR": "30",
}

# 피처 캐시 3종 (DATASUF -> 빌드 시 환경변수)
CACHES = {
    "": {},                                        # 기본 캐시
    "_ff": {"FEATFIX": "1"},                       # 왜도 로그변환 교정
    "_ffxw": {"FEATFIX": "1", "LDCH": "xwide"},    # + LDAPS 격자 24채널
}

# 앵커 5종 (이름 -> 개별 환경변수). inference.py 의 POOL 과 반드시 일치해야 한다.
ANCHORS = {
    "dl_f2ffx":         {"DATASUF": "_ff"},
    "dl_f2ffx_s100":    {"DATASUF": "_ff", "SEED0": "100"},
    "dl_f2ffx_xw":      {"DATASUF": "_ffxw", "LDCH": "xwide"},
    "dl_f2ffx_xw_uk":   {"DATASUF": "_ffxw", "LDCH": "xwide", "UMKR": "1"},
    "dl_f2tau15":       {"TAU1": "0.015"},         # 기본 캐시 + FICR 밴드 평활폭 조정
}


def _run(script: str, env_extra: dict, log_name: str) -> None:
    env = dict(os.environ)
    # 이전 실행의 잔류 환경변수 오염 방지
    for k in ("DATASUF", "LDCH", "UMKR", "SEED0", "TAU1", "FEATFIX", "OUT"):
        env.pop(k, None)
    env.update({k: str(v) for k, v in env_extra.items()})
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = os.path.join(ROOT, log_name)
    print("[실행] %s  env=%s" % (os.path.basename(script),
                                 {k: v for k, v in env_extra.items()}))
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run([sys.executable, script], env=env, cwd=ROOT,
                              stdout=fp, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError("실패 (exit %d) — 로그: %s" % (proc.returncode, log_path))
    print("       완료 %.1f분  로그: %s" % (dt / 60.0, log_name))


def build_caches(force: bool = False) -> None:
    for suf, extra in CACHES.items():
        marker = os.path.join(MODELS, "data_train_v2%s.npz" % suf)
        if os.path.exists(marker) and not force:
            print("[캐시] %s 이미 존재 — 건너뜀" % (marker,))
            continue
        env = dict(extra)
        if suf:
            env["DATASUF"] = suf
        _run(PREP, env, "logs_dataprep%s.txt" % (suf or "_base"))


def train_anchor(name: str) -> None:
    if name not in ANCHORS:
        raise KeyError("알 수 없는 앵커: %s (가능: %s)" % (name, ", ".join(ANCHORS)))
    env = dict(COMMON_ENV)
    env.update(ANCHORS[name])
    env["OUT"] = name
    _run(TRAIN, env, "logs_%s.txt" % name)
    out_npz = os.path.join(MODELS, name, "dl_oof.npz")
    print("       산출물: %s (%s)" % (out_npz, "확인" if os.path.exists(out_npz) else "없음!"))


def main() -> None:
    ap = argparse.ArgumentParser(description="앵커 모델 학습 (추론과 분리)")
    ap.add_argument("--list", action="store_true", help="학습 대상 목록만 출력")
    ap.add_argument("--cache", action="store_true", help="피처 캐시만 빌드")
    ap.add_argument("--rebuild-cache", action="store_true", help="캐시 강제 재빌드")
    ap.add_argument("--anchor", default=None, help="특정 앵커 1종만 학습")
    ap.add_argument("--all", action="store_true", help="캐시 + 앵커 5종 전부")
    args = ap.parse_args()

    if args.list:
        print("피처 캐시:")
        for suf, extra in CACHES.items():
            print("  DATASUF=%-7s %s" % (suf or "(기본)", extra))
        print("앵커 (학습 순서):")
        for n, e in ANCHORS.items():
            print("  %-18s %s" % (n, e))
        return

    if args.cache or args.all:
        build_caches(force=args.rebuild_cache)
    if args.anchor:
        train_anchor(args.anchor)
    elif args.all:
        for n in ANCHORS:
            train_anchor(n)
    elif not args.cache:
        ap.print_help()


if __name__ == "__main__":
    main()
