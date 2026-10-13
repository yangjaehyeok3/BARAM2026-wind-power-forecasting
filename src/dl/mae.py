# -*- coding: utf-8 -*-
"""BARAM 2026 풍력 DL — TRAIN-ONLY 마스크드 사전학습(MAE/SimMIM) → g3 'ssl' 멤버.

목적
----
격자 토큰 트렁크(GridTokenTransformerV2 인코더)를 **train 입력(2022~2024)만으로** 자기지도
사전학습한 뒤, 그 가중치를 초기값으로 소량 학습률(enc_lr_mult)로 미세조정한다.
데이터가 부족한 g3(2023~ 라벨)에서 subC 최고 성능(OOF g3=0.6249)을 낸 멤버.

Pretext (SimMIM 스타일)
-----------------------
- 25개 격자셀 토큰 중 mask_ratio 비율을 학습형 mask token 으로 치환(위치/소스 임베드는 유지).
- 전체 토큰을 인코딩 → 소스별 경량 디코더가 원래의 (토큰,변수) 표준화값을 복원.
- 손실 = 마스킹된 토큰에 대한 MSE 만.

규정 준수 (train-only)
----------------------
- 사전학습 코퍼스 = **train fold 입력 토큰만**(pt_extra=None). 2025 test 입력은 어디에도 미사용.
- 표준화 통계도 train fold 로만 산출. 완전 leakage-safe.
- 인코더 submodule 이름이 GridTokenTransformerV2 와 1:1 대응 → state_dict 그대로 이식.

미세조정 루프는 train_dl.train_fold(no-window) 와 동일 recipe(AdamW + soft-FICR, 조기종료)이며
인코더에만 낮은 LR(enc_lr_mult)을 주는 차등 학습률 + 사전학습 초기화가 추가된 형태.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from model import GridTokenTransformerV2, Block, loss_fn, dist_ce_loss, DEV
import data_prep as D

# 인코더(사전학습 이식 대상) 파라미터 접두사
ENC_PREFIXES = ("emb_g", "emb_l", "src", "pos_mlp", "blocks", "final_norm")


# ------------------------------------------------------------------ MAE 인코더
class MAEEncoder(nn.Module):
    """GridTokenTransformerV2 의 토큰 파이프라인 + MAE mask token + 소스별 복원 디코더."""
    def __init__(self, cg, cl, gpos, lpos, dim=64, heads=4, layers=2, drop=0.3,
                 norm_first=False, gated=False, ff_mult=2, dec_layers=1, dec_drop=0.0):
        super().__init__()
        self.dim = dim
        self.emb_g = nn.Linear(cg, dim); self.emb_l = nn.Linear(cl, dim)
        self.src = nn.Embedding(2, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        pos = np.concatenate([gpos, lpos], 0).astype(np.float32)
        pn = pos.copy(); pn[:, 0] = (pn[:, 0] - 37.25) / 0.25; pn[:, 1] = (pn[:, 1] - 129.0) / 0.25
        self.register_buffer("posn", torch.tensor(pn))
        sid = np.array([0] * gpos.shape[0] + [1] * lpos.shape[0])
        self.register_buffer("sid", torch.tensor(sid, dtype=torch.long))
        self.ng = gpos.shape[0]; self.ntok = pos.shape[0]
        self.blocks = nn.ModuleList([Block(dim, heads, ff_mult, drop, norm_first, gated)
                                     for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim) if norm_first else nn.Identity()
        # --- MAE 전용 (미세조정 시 폐기) ---
        self.mask_tok = nn.Parameter(torch.randn(dim) * 0.02)

        def dec(cout):
            layers_ = []
            for _ in range(dec_layers):
                layers_ += [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dec_drop)]
            layers_ += [nn.Linear(dim, cout)]
            return nn.Sequential(*layers_)
        self.dec_g = dec(cg); self.dec_l = dec(cl)
        self.cg = cg; self.cl = cl

    def forward_mae(self, GT, LT, mask):
        """GT[B,ng,cg], LT[B,nl,cl] 표준화; mask[B,ntok] bool(True=마스킹)."""
        tg = self.emb_g(GT); tl = self.emb_l(LT)
        tok = torch.cat([tg, tl], 1)
        mt = self.mask_tok.view(1, 1, -1)
        tok = torch.where(mask.unsqueeze(-1), mt, tok)
        z = tok + self.pos_mlp(self.posn).unsqueeze(0) + self.src(self.sid).unsqueeze(0)
        for blk in self.blocks:
            z = blk(z, None)
        z = self.final_norm(z)
        rg = self.dec_g(z[:, :self.ng])
        rl = self.dec_l(z[:, self.ng:])
        return rg, rl

    def enc_state(self):
        """GridTokenTransformerV2 인코더 파라미터로 매핑되는 state_dict 부분집합."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if k.startswith(ENC_PREFIXES)}


def pretrain_encoder(GTs, LTs, gpos, lpos, tr_kw, seed=0, mask_ratio=0.6,
                     pt_epochs=100, pt_lr=1.5e-3, wd=1e-4, batch=1024, drop=0.3,
                     dec_layers=1, verbose=False):
    """표준화 토큰 GTs,LTs 로 마스크드 복원 사전학습. 인코더 state_dict 반환."""
    torch.manual_seed(seed); np.random.seed(seed)
    tk = tr_kw
    cg = GTs.shape[2]; cl = LTs.shape[2]
    enc = MAEEncoder(cg, cl, gpos, lpos,
                     dim=tk.get("dim", 64), heads=tk.get("heads", 4),
                     layers=tk.get("layers", 2), drop=drop,
                     norm_first=tk.get("norm_first", False), gated=tk.get("gated", False),
                     ff_mult=tk.get("ff_mult", 2), dec_layers=dec_layers).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=pt_lr, weight_decay=wd)
    n = GTs.shape[0]; ntok = enc.ntok; ng = enc.ng
    tG = torch.tensor(GTs).to(DEV); tL = torch.tensor(LTs).to(DEV)
    steps = pt_epochs * max(1, (n + batch - 1) // batch)
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(pt_epochs):
        enc.train(); perm = torch.randperm(n)
        tot = 0.0; nb = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            g = tG[idx]; l = tL[idx]; B = g.shape[0]
            mask = torch.rand(B, ntok, device=DEV) < mask_ratio
            allm = mask.all(1)
            if allm.any(): mask[allm, 0] = False               # >=1 가시 토큰 보장
            nonem = (~mask).all(1)
            if nonem.any(): mask[nonem, 0] = True               # >=1 마스킹 토큰 보장
            rg, rl = enc.forward_mae(g, l, mask)
            mg = mask[:, :ng].unsqueeze(-1).float()
            ml = mask[:, ng:].unsqueeze(-1).float()
            lg = (((rg - g) ** 2) * mg).sum() / mg.sum().clamp_min(1.0) / cg
            ll = (((rl - l) ** 2) * ml).sum() / ml.sum().clamp_min(1.0) / cl
            loss = lg + ll
            opt.zero_grad(); loss.backward(); opt.step(); sc.step()
            tot += loss.item(); nb += 1
        if verbose and (ep % 20 == 0 or ep == pt_epochs - 1):
            print(f"      [pt seed{seed}] ep{ep} recon {tot/max(1,nb):.4f}", flush=True)
    return enc.enc_state()


# ------------------------------------------------------------------ 미세조정
def train_fold_mae(data, tr_idx, va_idx, tr_kw, loss_kind="metric", lam=1.0,
                   seeds=(0, 1, 2, 3, 4), epochs=170, lr=1.5e-3, wd=5e-4,
                   width=128, drop=0.3, batch=1024, patience=30, verbose=False,
                   test_arrays=None, loss_kw=None,
                   mask_ratio=0.6, pt_epochs=100, pt_lr=1.5e-3, pt_wd=1e-4,
                   dec_layers=1, enc_lr_mult=0.4, return_states=False, anom_w=None,
                   tau_sched=None):   # tau_sched: 미지원(고정 tau). train_dl.BASE 공유를 위해 수용만.
    """MAE 사전학습 초기화 + 차등 LR 미세조정 (train-only, pt_extra 없음).
    return_states=True 면 시드별 최종 state_dict 리스트를 함께 반환(가중치 저장용)."""
    loss_kw = loss_kw or {}; tk = tr_kw
    X = data["X"]; Y = data["Y"] / D.CAP; Fcols = data["Fcols"]
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
        if anom_w is not None:                       # physics-anomaly train-loss downweight
            Mtr[:, gi] *= anom_w[gi][tr_idx].astype(np.float32)
    Mva = (~np.isnan(Yva)).astype(np.float32); Yva_f = np.nan_to_num(Yva)
    T = lambda a: torch.tensor(a).to(DEV)
    tX = T(Xtr); tY = T(Ytr_f); tM = T(Mtr); vX = T(Xva)
    tG = T(Atr); tL = T(Btr); vG = T(Ava); vL = T(Bva)
    if test_arrays is not None:
        Xte = D.standardize(X[tr_idx], X[va_idx], test_arrays["X"])[2]
        Ate = D.standardize(A[tr_idx], A[va_idx], test_arrays["GT"])[2]
        Bte = D.standardize(B[tr_idx], B[va_idx], test_arrays["LT"])[2]
        teX = T(Xte); teG = T(Ate); teL = T(Bte)
    # 사전학습 코퍼스 = train fold 토큰만 (train-only, 표준화도 train 통계)
    ptG = Atr; ptL = Btr

    d_in = X.shape[1]; n = len(tX)
    val_preds = []; tr_preds = []; te_preds = []; best_epochs = []; states = []

    def new_model(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        return GridTokenTransformerV2(
            d_in, A.shape[2], B.shape[2], data["gpos"], data["lpos"],
            dim=tk.get("dim", 64), heads=tk.get("heads", 4), layers=tk.get("layers", 2),
            drop=drop, width=width, pool=tk.get("pool", "mean"),
            norm_first=tk.get("norm_first", False), gated=tk.get("gated", False),
            ff_mult=tk.get("ff_mult", 2)).to(DEV)

    def eval_model(model, Xe, Ge, Le):
        model.eval()
        with torch.no_grad():
            return model(Xe, Ge, Le).cpu().numpy()

    steps_per = max(1, (n + batch - 1) // batch)
    for seed in seeds:
        init_state = pretrain_encoder(ptG, ptL, data["gpos"], data["lpos"], tk,
                                      seed=seed, mask_ratio=mask_ratio, pt_epochs=pt_epochs,
                                      pt_lr=pt_lr, wd=pt_wd, batch=batch, drop=drop,
                                      dec_layers=dec_layers, verbose=verbose)
        model = new_model(seed)
        msd = model.state_dict()
        msd.update({k: v.to(msd[k].dtype) for k, v in init_state.items() if k in msd})
        model.load_state_dict(msd)
        # ★GFILM_LR (2026-08-08): 배포 조립은 **g3 = ssl** 이라 이 경로가 g3 를 결정한다.
        #   train_dl.train_fold 에는 GFILM 파라미터 lr 배수를 넣었으나 이 SSL 옵티마이저는
        #   별도라 배수가 안 걸렸다 → g3 만 그 이득을 못 받고 있었다(prodwin g3 +0.0054 는
        #   배포되지 않는 경로의 값). 여기서 동일 배수를 적용한다. 1.0(기본)이면 기존과 동일.
        _gmul = float(os.environ.get("GFILM_LR", "1.0"))
        _gk = ("gw", "gb", "film.")
        _isg = lambda nm: any(nm.startswith(k) or ("." + k) in nm for k in _gk)
        enc_params = [p for nm, p in model.named_parameters()
                      if nm.startswith(ENC_PREFIXES) and not _isg(nm)]
        hd_params = [p for nm, p in model.named_parameters()
                     if not nm.startswith(ENC_PREFIXES) and not _isg(nm)]
        gf_params = [p for nm, p in model.named_parameters() if _isg(nm)]
        groups = [{"params": hd_params, "lr": lr},
                  {"params": enc_params, "lr": lr * enc_lr_mult}]
        if gf_params:
            groups.append({"params": gf_params, "lr": lr * _gmul})
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
        best_gs = -1e9; best_state = None; best_ep = 0; bad = 0
        swa_top = []; _SWA = int(os.environ.get("SWA", "0"))  # SWA: top-K val상태 평균(flat minima)
        for ep in range(epochs):
            model.train(); perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                p = model(tX[idx], tG[idx], tL[idx])
                if getattr(model, "distb", 0) > 0:
                    # ★DIST: 분포헤드 CE — argmax 디코드된 p 는 기울기가 없으므로 반드시 logits 로
                    loss = dist_ce_loss(model._dist, tY[idx], tM[idx], model.distb)
                else:
                    loss = loss_fn(p, tY[idx], tM[idx], loss_kind, lam, **loss_kw)
                loss.backward(); opt.step()
            pv = eval_model(model, vX, vG, vL)
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
        val_preds.append(eval_model(model, vX, vG, vL))
        tr_preds.append(eval_model(model, tX, tG, tL))
        if test_arrays is not None:
            te_preds.append(eval_model(model, teX, teG, teL))
        if return_states:
            states.append({k: v.detach().cpu().clone() for k, v in best_state.items()})
        best_epochs.append(best_ep)
        if verbose:
            print(f"    seed{seed} best_ep{best_ep} val_gs{best_gs:.4f}", flush=True)
    vp = np.mean(val_preds, 0); vstd = np.std(val_preds, 0); tp = np.mean(tr_preds, 0)
    out = dict(val=vp, val_seedstd=vstd, tr=tp, Mva=Mva, Yva=Yva_f, Mtr=Mtr, Ytr=Ytr_f,
               tr_idx=tr_idx, va_idx=va_idx, best_epochs=best_epochs)
    if test_arrays is not None:
        out["test"] = np.mean(te_preds, 0)
    if return_states:
        out["states"] = states
    return out
