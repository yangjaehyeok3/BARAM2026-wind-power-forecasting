# -*- coding: utf-8 -*-
"""BARAM 2026 풍력 DL — subC(대회 최고 LB 0.6419) 학습 & 가중치 저장.

subC 구성
---------
  g1, g2 = "prodwin" : 6-멤버 등가중 앙상블
        3 no-window (GridTokenTransformerV2, RevIN-center) + 3 window (WinGridTokenTransformerV2)
        멤버 3종 config = {v2_gated_attn, v2_base, v1_attn}, 각 16 seed.
  g3     = "ssl"     : MAE(train-only) 사전학습 트렁크 미세조정 (mae.train_fold_mae), 10 seed.

recipe (검증된 값 고정)
  BASE  = AdamW, soft-FICR(metric, lam1.0, tau0.004, w_actual=False), lr1.5e-3, wd5e-4,
          width128, drop0.3, epochs170(조기종료 patience30), batch1024.
  WINKW = 예보 윈도 offsets ±6h, win_dim48, win_ksize5.
  MK(ssl)= mask_ratio0.6, pt_epochs100, pt_lr1.5e-3, enc_lr_mult0.4.

동작
----
  1) 각 멤버를 2-fold OOF 로 학습 → per-group group_score (재현 검증) + median best-epoch 산출.
  2) 같은 recipe 로 **전체 train(2022~2024)** 재학습 → 시드별 state_dict 를 models/dl/ 에 저장,
     test(2025) CF 예측 생성.
  3) subC OOF(g1/g2 prodwin, g3 ssl) 및 test CF 를 models/dl/dl_oof.npz 로 저장.
  실행: python src/dl/train_dl.py            (전체 16/10 seed)
        python src/dl/train_dl.py --smoke    (2 seed 스모크: 정합성/저장 확인용)

leakage-safe: 입력은 예보 파생만, 표준화 통계는 train fold 로만, test 는 예측에만.
"""
import sys, io, os, time, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_prep as D
from model import (GridTokenTransformerV2, WinGridTokenTransformerV2, loss_fn, dist_ce_loss, DEV)
import mae as MAE

# ★AVW 용 가용률 표 (train 구간 SCADA 산출). AVW=1.0(기본)이면 로드조차 하지 않는다.
_AVAIL = None
if float(os.environ.get("AVW", "1.0")) < 1.0:
    import pandas as _pd
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "avail_frac_v6.csv")
    _a = _pd.read_csv(_p, encoding="utf-8-sig")
    _a["k"] = _pd.to_datetime(_a["forecast_kst_dtm"]).astype(str)
    _AVAIL = _a.drop_duplicates("k").set_index("k")[["avail_g1", "avail_g2", "avail_g3"]]
    print("[avw] 부분정지 다운웨이트 ON (AVW=%s, %d행)" % (os.environ["AVW"], len(_AVAIL)), flush=True)

MODELS_DL = D.MODELS / "dl"
MODELS_DL.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ recipe 상수
BASE = dict(loss_kind="metric", lam=1.0, epochs=170, lr=1.5e-3, wd=5e-4, width=128,
            drop=0.3, patience=30, loss_kw=dict(w_actual=False, tau=0.004))
WINKW = dict(win_offsets=list(range(-6, 7)), win_dim=48, win_ksize=5)
# 멤버: (태그, 데이터버전, 트렁크 config)
MEMBERS = [("v2_gated_attn", "v2", dict(dim=64, heads=4, layers=2, pool="attn", gated=True)),
           ("v2_base",       "v2", dict(dim=64, heads=4, layers=2, pool="mean")),
           ("v1_attn",       "v1", dict(dim=64, heads=4, layers=2, pool="attn"))]
# ssl(g3) config
SSL_TK = dict(dim=64, heads=4, layers=2, pool="mean")
MK = dict(mask_ratio=0.6, pt_epochs=100, pt_lr=1.5e-3, enc_lr_mult=0.4)


# ================================================================= prodwin 학습기
def train_fold(data, tr_idx, va_idx, tr_kw, win_offsets=None, win_dim=48, win_ksize=5,
               loss_kind="metric", lam=1.0, seeds=(0, 1, 2, 3, 4), epochs=170,
               lr=1.5e-3, wd=5e-4, width=128, drop=0.3, batch=1024, patience=30,
               verbose=False, test_arrays=None, loss_kw=None, return_states=False,
               anom_w=None, tau_sched=None):
    """격자 토큰 Transformer 한 fold 학습 (AdamW + soft-FICR, val group_score 조기종료).
    win_offsets=None => no-window(GridTokenTransformerV2), 지정 => window(WinGridTokenTransformerV2).
    return_states=True 면 시드별 best state_dict 리스트 포함(전체학습 저장용)."""
    loss_kw = loss_kw or {}; tk = tr_kw
    # ★AVW (2026-08-08): 부분정지 라벨 다운웨이트.
    #   clean_mask 는 "평가대상(CF>=0.10)은 절대 안 지운다" 규칙이라 **완전정지만** 걸러진다.
    #   그런데 부분정지(6기 중 4기 가동 등)가 학습행의 g1 25.1% / g2 21.4% / g3 36.7% 다.
    #   같은 날씨에 다른 출력이 라벨로 들어가 파워커브를 흐린다.
    #   → 학습 구간엔 SCADA 가 있으니 **테스트 식별 없이** 가중만 낮출 수 있다
    #     (우리 메타패턴: "식별 불필요 레버만 이긴다").
    #   AVW=1.0(기본)이면 완전 비활성 = 기존과 비트 동일.
    if anom_w is None:
        _avw = float(os.environ.get("AVW", "1.0"))
        if _avw < 1.0 and _AVAIL is not None:
            _k = [str(k) for k in data["keys"]]
            anom_w = {}
            for gi in range(3):
                a = _AVAIL["avail_g%d" % (gi + 1)].reindex(_k).to_numpy(np.float32)
                # 결측(=가용률 모름)은 건드리지 않는다(1.0). 전기가동 1.0, 부분정지 AVW.
                anom_w[gi] = np.where(np.isfinite(a) & (a < 0.999), _avw, 1.0).astype(np.float32)
    X = data["X"]; Y = data["Y"] / D.CAP; Fcols = data["Fcols"]; year = data["year"]
    A = data["GT"]; B = data["LT"]
    Xtr, Xva = D.standardize(X[tr_idx], X[va_idx])[:2]
    Atr, Ava = D.standardize(A[tr_idx], A[va_idx])[:2]
    Btr, Bva = D.standardize(B[tr_idx], B[va_idx])[:2]
    Ytr = Y[tr_idx]; Yva = Y[va_idx]
    Mtr = (~np.isnan(Ytr)).astype(np.float32); Ytr_f = np.nan_to_num(Ytr)
    for gi in range(3):
        lab = ~np.isnan(Ytr[:, gi]); keep = np.zeros(len(Ytr), bool)
        if lab.sum() > 0:
            keep[lab] = D.clean_mask(X[tr_idx][lab], Ytr_f[lab, gi], Fcols, gi)
        Mtr[:, gi] *= keep.astype(np.float32)
        # physics-anomaly downweight (train-loss weight ONLY; eval/Mva untouched).
        # anom_w[gi] is a full-length CF-row weight array (1.0 normal, <1 anomaly).
        if anom_w is not None:
            Mtr[:, gi] *= anom_w[gi][tr_idx].astype(np.float32)
    Mva = (~np.isnan(Yva)).astype(np.float32); Yva_f = np.nan_to_num(Yva)

    # ---- ★AUX (2026-08-08): 중간 감독 타깃 (그룹별 나셀풍속/가용률) ----
    #   AUXW=0(기본)이면 완전히 비활성 = 기존과 비트 동일. 추론 경로는 건드리지 않는다.
    _AUXW = float(os.environ.get("AUXW", "0"))
    tA = tAM = None
    if _AUXW > 0 and data.get("AUX") is not None:
        _a = data["AUX"][tr_idx]
        tA = torch.tensor(np.nan_to_num(_a).astype(np.float32)).to(DEV)
        tAM = torch.tensor((~np.isnan(_a)).astype(np.float32)).to(DEV)

    # ---- 예보 윈도 텐서 (raw X, 연도블록 clamp = fold-safe/입력전용) ----
    win_ch = 0; Wtr = Wva = Wte = None
    if win_offsets is not None:
        feats = D.WIN_FEATS
        Wfull = D.build_window(X, year, Fcols, feats, list(win_offsets), clamp_year=True)
        if test_arrays is not None:
            Wte_r = D.build_window(test_arrays["X"], test_arrays["year"], Fcols, feats,
                                   list(win_offsets), clamp_year=True)
            Wtr, Wva, Wte = D.std_window(Wfull[tr_idx], Wfull[va_idx], Wte_r)
        else:
            Wtr, Wva = D.std_window(Wfull[tr_idx], Wfull[va_idx])
        win_ch = len(feats)

    T = lambda a: torch.tensor(a).to(DEV)
    tX = T(Xtr); tY = T(Ytr_f); tM = T(Mtr); vX = T(Xva)
    tG = T(Atr); tL = T(Btr); vG = T(Ava); vL = T(Bva)
    tW = T(Wtr) if Wtr is not None else None; vW = T(Wva) if Wva is not None else None
    if test_arrays is not None:
        Xte = D.standardize(X[tr_idx], X[va_idx], test_arrays["X"])[2]
        Ate = D.standardize(A[tr_idx], A[va_idx], test_arrays["GT"])[2]
        Bte = D.standardize(B[tr_idx], B[va_idx], test_arrays["LT"])[2]
        teX = T(Xte); teG = T(Ate); teL = T(Bte); teW = T(Wte) if Wte is not None else None

    d_in = X.shape[1]; n = len(tX)
    scalar_group_mask = None
    if os.environ.get("SITE_MASK", "0") == "1":
        scalar_group_mask = np.ones((3, d_in), dtype=np.float32)
        for column_index, column_name in enumerate(Fcols):
            for source_group in range(1, 4):
                if column_name == "fused_g%d" % source_group:
                    for target_group in range(3):
                        if target_group != source_group - 1:
                            scalar_group_mask[target_group, column_index] = 0.0
        print("[site-mask] masked scalar cells=%d" % int((scalar_group_mask == 0).sum()), flush=True)
    val_preds = []; tr_preds = []; te_preds = []; best_epochs = []; states = []
    vprob_sum = None; tprob_sum = None                  # ★SAVEDIST 시드확률 누적
    win = win_offsets is not None

    def new_model(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        common = dict(dim=tk.get("dim", 64), heads=tk.get("heads", 4), layers=tk.get("layers", 2),
                      drop=drop, width=width, pool=tk.get("pool", "mean"),
                      norm_first=tk.get("norm_first", False), gated=tk.get("gated", False),
                      ff_mult=tk.get("ff_mult", 2), scalar_group_mask=scalar_group_mask)
        if win:
            return WinGridTokenTransformerV2(d_in, A.shape[2], B.shape[2], data["gpos"],
                                             data["lpos"], win_ch, win_dim=win_dim,
                                             win_ksize=win_ksize, **common).to(DEV)
        return GridTokenTransformerV2(d_in, A.shape[2], B.shape[2], data["gpos"],
                                      data["lpos"], **common).to(DEV)

    def fwd(model, Xe, Ge, Le, We):
        return model(Xe, Ge, Le, We) if win else model(Xe, Ge, Le)

    def eval_model(model, Xe, Ge, Le, We):
        model.eval()
        with torch.no_grad():
            return fwd(model, Xe, Ge, Le, We).cpu().numpy()

    # ★SAVEDIST (2026-08-10): 분포헤드의 시드집계 교정.
    #   결함: 시드별 util-argmax 점예측을 산술평균하면 '모드들의 평균'이 되어 다봉 모드선택이
    #   집계단계에서 파괴된다(분포헤드의 존재이유 소멸). 교정 = 시드별 softmax 확률을 평균한 뒤
    #   미세 오퍼격자에서 효용 argmax 1회. SAVEDIST=0(기본)이면 기존 경로 비트동일.
    #   (g3 ssl 경로 mae.train_fold_mae 는 미적용 — 게이트 통과시 확장.)
    _SDIST = os.environ.get("SAVEDIST", "0") == "1"

    def eval_probs(model, Xe, Ge, Le, We):
        model.eval()
        with torch.no_grad():
            _ = fwd(model, Xe, Ge, Le, We)
            return torch.softmax(model._dist, -1).cpu().numpy().astype(np.float32)

    for seed in seeds:
        model = new_model(seed)
        # ★GFILM_LR (2026-08-08): GFILM=1 실측 결과 **기전이 작동하지 않았다** —
        #   목표였던 그룹 결합 완화가 0(예측상관 0.9751->0.9748, 그룹차이 표현력 54%->53%).
        #   원인: 설계상 초기화가 통제군과 함수 동일(gw=head.weight, g=1, b=0)인데
        #   med_ep~15 짧은 학습에서 옵티마이저가 그 자유도를 쓰지 않는다.
        #   교정: FiLM/그룹헤드 파라미터에만 lr 배수를 준다. 1.0(기본)이면 기존과 동일.
        _gmul = float(os.environ.get("GFILM_LR", "1.0"))
        _smul = float(os.environ.get("STRUCT_LR", "1.0"))
        if _gmul != 1.0 or _smul != 1.0:
            gkeys = ("film.", "gw", "gb")
            skeys = ("smot_g.", "smot_l.", "group_pool.")
            gp = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) or ("." + k) in n for k in gkeys)]
            sp = [p for n, p in model.named_parameters()
                  if any(n.startswith(k) or ("." + k) in n for k in skeys)]
            op = [p for n, p in model.named_parameters()
                  if not any(n.startswith(k) or ("." + k) in n for k in gkeys + skeys)]
            groups = [{"params": op, "lr": lr}]
            if gp:
                groups.append({"params": gp, "lr": lr * _gmul})
            if sp:
                groups.append({"params": sp, "lr": lr * _smul})
            opt = torch.optim.AdamW(groups, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best_gs = -1e9; best_state = None; best_ep = 0; bad = 0
        swa_top = []; _SWA = int(os.environ.get("SWA", "0"))  # SWA: top-K val상태 평균(flat minima)
        _SAM = float(os.environ.get("SAM_RHO", "0"))  # SAM: sharpness-aware min(학습중 평평한해 제약, SWA에 가산)
        for ep in range(epochs):
            model.train(); perm = torch.randperm(n)
            # tau 어닐링(옵션): tau_sched=(tau_hi,tau_lo) → 기하보간으로 계단 근사에 점진 수렴.
            # None(기본)이면 loss_kw 원본 그대로 → subC 재현불변.
            if tau_sched is not None:
                frac = ep / max(epochs - 1, 1)
                cur_tau = tau_sched[0] * (tau_sched[1] / tau_sched[0]) ** frac
                lk_ep = dict(loss_kw); lk_ep["tau"] = cur_tau
            else:
                lk_ep = loss_kw
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                bW = tW[idx] if tW is not None else None
                p = fwd(model, tX[idx], tG[idx], tL[idx], bW)
                if getattr(model, "distb", 0) > 0:
                    # ★DIST: 분포헤드 CE (loss_kind/lam/tau 전부 무시 — 다이얼 없는 설계)
                    loss = dist_ce_loss(model._dist, tY[idx], tM[idx], model.distb)
                else:
                    loss = loss_fn(p, tY[idx], tM[idx], loss_kind, lam, **lk_ep)
                if tA is not None and getattr(model, "_aux", None) is not None:
                    am = tAM[idx]
                    loss = loss + _AUXW * ((model._aux - tA[idx]).abs() * am).sum() / am.sum().clamp_min(1.0)
                if _SAM > 0:                              # SAM two-step: perturb to worst-case, grad there, step orig
                    loss.backward()
                    with torch.no_grad():
                        gn = torch.sqrt(sum((pr.grad * pr.grad).sum() for pr in model.parameters() if pr.grad is not None)) + 1e-12
                        sc = _SAM / gn; ew = []
                        for pr in model.parameters():
                            if pr.grad is not None:
                                e = pr.grad * sc; pr.add_(e); ew.append((pr, e))
                    opt.zero_grad()
                    p2 = fwd(model, tX[idx], tG[idx], tL[idx], bW)
                    (dist_ce_loss(model._dist, tY[idx], tM[idx], model.distb)
                     if getattr(model, "distb", 0) > 0
                     else loss_fn(p2, tY[idx], tM[idx], loss_kind, lam, **lk_ep)).backward()
                    with torch.no_grad():
                        for pr, e in ew: pr.sub_(e)
                    opt.step()
                else:
                    loss.backward(); opt.step()
            pv = eval_model(model, vX, vG, vL, vW)
            gss = []
            for gi in range(3):
                lab = Mva[:, gi] > 0
                if lab.sum() == 0: continue
                g = D.group_score_np(Yva_f[lab, gi], pv[lab, gi])
                if not np.isnan(g): gss.append(g)
            gsm = np.mean(gss) if gss else -1e9
            if _SWA and (len(swa_top) < _SWA or gsm > swa_top[-1][0]):  # top-K val 상태 수집
                _st = {k: v.detach().clone() for k, v in model.state_dict().items()}
                swa_top.append((gsm, _st)); swa_top.sort(key=lambda x: -x[0]); swa_top = swa_top[:_SWA]
            if gsm > best_gs + 1e-5:
                best_gs = gsm
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_ep = ep; bad = 0
            else:
                bad += 1
                if bad >= patience: break
        if _SWA and len(swa_top) > 1:  # top-K 가중치 평균 = SWA (input-noise NMAE대가 없이 일반화↑)
            _ks = list(swa_top[0][1].keys())
            best_state = {k: sum(s[k].float() for _, s in swa_top) / len(swa_top) for k in _ks}
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in best_state.items()})
        val_preds.append(eval_model(model, vX, vG, vL, vW))
        tr_preds.append(eval_model(model, tX, tG, tL, tW))
        if test_arrays is not None:
            te_preds.append(eval_model(model, teX, teG, teL, teW))
        if _SDIST and getattr(model, "distb", 0) > 0:
            vp = eval_probs(model, vX, vG, vL, vW)
            vprob_sum = vp if vprob_sum is None else vprob_sum + vp
            if test_arrays is not None:
                tp = eval_probs(model, teX, teG, teL, teW)
                tprob_sum = tp if tprob_sum is None else tprob_sum + tp
        if return_states:
            states.append({k: v.detach().cpu().clone() for k, v in best_state.items()})
        best_epochs.append(best_ep)
        if verbose:
            print(f"    seed{seed} best_ep{best_ep} val_gs{best_gs:.4f}", flush=True)
    out = dict(val=np.mean(val_preds, 0), val_seedstd=np.std(val_preds, 0),
               tr=np.mean(tr_preds, 0), Mva=Mva, Yva=Yva_f, Mtr=Mtr, Ytr=Ytr_f,
               tr_idx=tr_idx, va_idx=va_idx, best_epochs=best_epochs)
    if os.environ.get("SAVESEEDS"): out["val_preds"] = np.stack(val_preds)  # seed수렴 실험용(조건부)
    if test_arrays is not None:
        out["test"] = np.mean(te_preds, 0)
    if vprob_sum is not None:                           # ★SAVEDIST: 확률평균 → 미세격자 1회 디코드
        qv = vprob_sum / len(seeds)
        out["val"] = _dist_fine_decode(qv)
        out["val_prob"] = qv.astype(np.float16)
        if tprob_sum is not None:
            qt = tprob_sum / len(seeds)
            out["test"] = _dist_fine_decode(qt)
            out["test_prob"] = qt.astype(np.float16)
    if return_states:
        out["states"] = states
    return out


def _dist_fine_decode(q, lo=0.0, hi=1.05, noffer=421):
    """q [n,3,K] 확률 → 미세 오퍼격자 효용 argmax [n,3].
    효용 = -|c_k - p| + 0.25·credit(|c_k - p|) (dist_ce_loss/모델 dist_util 과 동일 상수)."""
    K = q.shape[-1]
    edges = np.linspace(lo, hi, K + 1); cent = 0.5 * (edges[:-1] + edges[1:])
    offers = np.linspace(lo, hi, noffer)
    d = np.abs(offers[:, None] - cent[None, :])
    credit = np.where(d <= 0.06, 4.0, np.where(d <= 0.08, 3.0, 0.0))
    U = (-d + 0.25 * credit).astype(np.float32)          # [noffer, K]
    eu = q.reshape(-1, K).astype(np.float32) @ U.T       # [n*3, noffer]
    return offers[eu.argmax(1)].reshape(q.shape[0], q.shape[1]).astype(np.float32)


def run_oof(data, tr_kw, win_offsets=None, seeds=(0, 1, 2, 3, 4), anom_w=None,
            tau_sched=None, **kw):
    """2-fold OOF. results[fi] 반환. anom_w=None(기본,subC) 또는 그룹별 행-가중 dict.
    tau_sched=None(기본)이면 subC 재현불변; (tau_hi,tau_lo)면 학습중 τ 어닐링."""
    results = {}
    for fi, fold in enumerate(D.FOLDS):
        tr, va = D.fold_indices(data, fold)
        results[fi] = train_fold(data, tr, va, tr_kw, win_offsets=win_offsets,
                                 seeds=seeds, anom_w=anom_w, tau_sched=tau_sched, **kw)
    return results


# ================================================================= 앙상블 지표
def ensemble_group(member_results, gi):
    """멤버 result-dict 리스트 → 그룹 gi 의 등가중 앙상블 y,p,gs (fold 연결)."""
    ys = []; ps = []
    for fi in D.GROUP_FOLDS[gi]:
        lab = member_results[0][fi]["Mva"][:, gi] > 0
        y = member_results[0][fi]["Yva"][lab, gi]
        p = np.mean([m[fi]["val"][lab, gi] for m in member_results], 0)
        ys.append(y); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p, D.group_score_np(y, p)


# ================================================================= 저장 유틸
def save_member(name, tr_kw, ver, win, states, epochs_used):
    torch.save(dict(name=name, tr_kw=tr_kw, ver=ver, win=win,
                    win_offsets=WINKW["win_offsets"] if win else None,
                    win_dim=WINKW["win_dim"], win_ksize=WINKW["win_ksize"],
                    epochs_used=epochs_used, states=states),
               MODELS_DL / f"{name}.pt")


# ================================================================= 메인
def main(smoke=False):
    t_all = time.time()
    n_seed = 2 if smoke else int(os.environ.get("NSEED", "16"))
    n_seed_ssl = 2 if smoke else int(os.environ.get("NSEED_SSL", "10"))
    _s0 = int(os.environ.get("SEED0", "0"))
    SEEDS = tuple(range(_s0, _s0 + n_seed)); SEEDS_SSL = tuple(range(_s0, _s0 + n_seed_ssl))
    print(f"[train_dl] {'SMOKE ' if smoke else ''}prodwin seeds={n_seed} ssl seeds={n_seed_ssl} dev={DEV}", flush=True)

    # ---- 데이터 로드 + RevIN-center (버전별 캐시) ----
    cache = {}
    def get(ver):
        if ver not in cache:
            cache[ver] = (D.apply_revin_center(D.load_data("train", ver)),
                          D.apply_revin_center(D.load_data("test", ver)))
        return cache[ver]

    keys_ref = get("v2")[1]["keys"]
    nowin_oofs = []; win_oofs = []; nowin_test = []; win_test = []

    # ---- g1/g2 prodwin: 멤버별 nowin + win, OOF → median epoch → 전체학습(저장) ----
    _member_limit = int(os.environ.get("MEMBER_LIMIT", str(len(MEMBERS))))
    _nowindow_only = os.environ.get("NOWINDOW", "0") == "1"
    _skip_ssl = os.environ.get("SKIPSSL", "0") == "1"
    for tag, ver, tk in MEMBERS[:_member_limit]:
        tr, te = get(ver); allidx = np.arange(len(tr["Y"]))
        # no-window
        t = time.time()
        o0 = run_oof(tr, tk, win_offsets=None, seeds=SEEDS, **BASE)
        me0 = max(int(np.median([e for fi in o0 for e in o0[fi]["best_epochs"]])), 15)
        R0 = train_fold(tr, allidx, allidx, tk, win_offsets=None, seeds=SEEDS,
                        **dict(BASE, epochs=me0, patience=10**9), test_arrays=te,
                        return_states=True)
        save_member(f"nowin_{tag}", tk, ver, False, R0["states"], me0)
        nowin_oofs.append(o0); nowin_test.append(R0["test"])
        g0 = [D.group_score_np(*ensemble_group([o0], gi)[:2]) for gi in range(3)]
        print(f"[NOWIN/{tag}] g {np.round(g0,4)} med_ep {me0} [{time.time()-t:.0f}s]", flush=True)
        if not _nowindow_only:
            # window
            t = time.time()
            o1 = run_oof(tr, tk, seeds=SEEDS, **dict(BASE, **WINKW))
            me1 = max(int(np.median([e for fi in o1 for e in o1[fi]["best_epochs"]])), 15)
            R1 = train_fold(tr, allidx, allidx, tk, seeds=SEEDS,
                            **dict(BASE, epochs=me1, patience=10**9, **WINKW), test_arrays=te,
                            return_states=True)
            save_member(f"win_{tag}", tk, ver, True, R1["states"], me1)
            win_oofs.append(o1); win_test.append(R1["test"])
            g1 = [D.group_score_np(*ensemble_group([o1], gi)[:2]) for gi in range(3)]
            print(f"[WIN  /{tag}] g {np.round(g1,4)} med_ep {me1} [{time.time()-t:.0f}s]", flush=True)

    # prodwin 6-멤버 OOF (등가중) + test CF (T6 = 0.5*(nowin_mean + win_mean))
    prod6 = nowin_oofs + win_oofs
    prod_gs = np.array([ensemble_group(prod6, gi)[2] for gi in range(3)])
    prod_oof = {gi: ensemble_group(prod6, gi) for gi in range(3)}
    Tn = np.mean(nowin_test, 0)
    T6 = Tn if _nowindow_only else 0.5 * (Tn + np.mean(win_test, 0))
    print(f"\n[PRODWIN 6-mem] OOF g {prod_gs.round(4)} mean {prod_gs.mean():.4f}", flush=True)

    # ---- g3 ssl: MAE train-only 사전학습 미세조정 ----
    if _skip_ssl:
        ssl_gs = prod_gs.copy(); ssl_oof = prod_oof; Tssl = T6.copy(); me_ssl = 0
        print("[SSL skipped] using production branch for diagnostic output", flush=True)
    else:
        trv2, tev2 = get("v2"); allidx = np.arange(len(trv2["Y"]))
        t = time.time()
        ssl_res = {}
        for fi, fold in enumerate(D.FOLDS):
            trx, va = D.fold_indices(trv2, fold)
            ssl_res[fi] = MAE.train_fold_mae(trv2, trx, va, SSL_TK, seeds=SEEDS_SSL, **dict(BASE, **MK))
        me_ssl = max(int(np.median([e for fi in ssl_res for e in ssl_res[fi]["best_epochs"]])), 15)
        Rs = MAE.train_fold_mae(trv2, allidx, allidx, SSL_TK, seeds=SEEDS_SSL,
                                **dict(BASE, epochs=me_ssl, patience=10**9, **MK),
                                test_arrays=tev2, return_states=True)
        torch.save(dict(name="ssl", tr_kw=SSL_TK, ver="v2", win=False, epochs_used=me_ssl,
                        mk=MK, states=Rs["states"]), MODELS_DL / "ssl.pt")
        ssl_gs = np.array([ensemble_group([ssl_res], gi)[2] for gi in range(3)])
        ssl_oof = {gi: ensemble_group([ssl_res], gi) for gi in range(3)}
        Tssl = Rs["test"]
        print(f"[SSL g3] OOF g {ssl_gs.round(4)} med_ep {me_ssl} [{time.time()-t:.0f}s]", flush=True)

    # ---- subC 조립: g1/g2 = prodwin, g3 = ssl ----
    subC_gs = np.array([prod_gs[0], prod_gs[1], ssl_gs[2]])
    subC_test = np.column_stack([T6[:, 0], T6[:, 1], Tssl[:, 2]]).astype(np.float32)
    print("\n" + "=" * 60, flush=True)
    print(f"subC OOF per-group: g1={subC_gs[0]:.4f} g2={subC_gs[1]:.4f} g3={subC_gs[2]:.4f} "
          f"mean={subC_gs.mean():.4f}", flush=True)
    print(f"reference        : g1=0.6461 g2=0.6606 g3=0.6249 mean=0.6439", flush=True)
    print(f"subC test CF mean per-grp: {subC_test.mean(0).round(4)} (ref 0.388 0.412 0.353)", flush=True)

    # ---- OOF/test 저장 ----
    np.savez(MODELS_DL / "dl_oof.npz", keys=keys_ref, CAP=D.CAP,
             subC_test_cf=subC_test, prod_test_cf=T6.astype(np.float32),
             ssl_test_cf=Tssl.astype(np.float32),
             prod_gs=prod_gs, ssl_gs=ssl_gs, subC_gs=subC_gs,
             **{f"prod_oof_y{gi}": prod_oof[gi][0] for gi in range(3)},
             **{f"prod_oof_p{gi}": prod_oof[gi][1] for gi in range(3)},
             **{f"ssl_oof_y{gi}": ssl_oof[gi][0] for gi in range(3)},
             **{f"ssl_oof_p{gi}": ssl_oof[gi][1] for gi in range(3)})
    print(f"\n[saved] weights -> {MODELS_DL}  (nowin_*/win_*/ssl .pt)", flush=True)
    print(f"[saved] {MODELS_DL/'dl_oof.npz'}  [total {time.time()-t_all:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-seed 스모크 (정합성/저장 확인)")
    args = ap.parse_args()
    main(smoke=args.smoke)
