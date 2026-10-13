# -*- coding: utf-8 -*-
"""BARAM 2026 풍력 DL — 모델 아키텍처 & 목적함수 (subC 재현).

구성 요소
--------
- Block / AttnPool            : Transformer 인코더 블록(사전/사후 norm, GEGLU 선택) + 마스킹 attention-pool.
- GridTokenTransformerV2      : 격자 토큰(9 GFS + 16 LDAPS) self-attention 트렁크.
    * 각 격자셀을 토큰으로: 소스별 선형임베드 + (lat,lon) 위치 MLP + 소스타입 임베드.
    * pooling {mean,cls,attn}, 토큰 드롭아웃(마스킹), 입력 가우시안 노이즈 옵션.
    * pooled 벡터 + 스칼라 피처 X 를 concat → MLP 트렁크 → 3그룹 head(sigmoid=CF).
    * g1/g2 no-window 멤버 3개 및 g3 ssl 멤버가 사용.
- WinGridTokenTransformerV2   : 위 트렁크 + 예보 ±kh 윈도 1D-CNN 인코더(WinEncoder) 융합.
    * g1/g2 window 멤버 3개가 사용 (subC 6-멤버 앙상블의 window 절반).
- EMA                         : 가중치 지수이동평균 (옵션; subC 기본 recipe 는 미사용).
- loss_fn                     : L1 + soft-FICR (미분가능 대회지표 근사).

RevIN-center 는 data_prep.apply_revin_center 로 토큰에 사전 적용된다(모델 밖).
모든 입력은 예보 파생 → leakage-safe (data_prep 참고).
"""
import numpy as np
import os
import os
import torch
import torch.nn as nn

torch.set_num_threads(16)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FICRDEN = os.environ.get("FICRDEN", "0") == "1"   # FICR 분모 고정 (공식 산식 정렬)
FICRPEN = os.environ.get("FICRPEN", "0") == "1"   # focal 을 값이 아니라 페널티에 (최적점 보존)
# ★OUTPULL: 밴드 **밖** 연속 인력항. 계단을 시그모이드로 평활하면 |e| 가 tau 의 몇 배만 벗어나도
#   기울기가 0 이 된다(tau=0.008 이면 |e|=0.10 에서 이미 소멸). 그런데 잃은 정산금의 20~23% 가
#   |e| 0.06~0.10 = **회수 가능 구간**에 있다. tau 를 키우면 밴드 자체가 흐려지므로,
#   경계는 날카롭게 둔 채 밖에서만 당기는 항을 따로 붙인다.
#   포화형(OUTCAP)으로 둔다: |e| > 0.08+OUTCAP 인 가망없는 시간은 **비용은 남고 기울기는 0**
#   (거리비례로 세게 당기면 가망없는 시간이 기울기를 지배해 σ 분석과 반대가 된다).
#   NMAE 항도 선형 인력이지만 **균일가중**이다. 이 항은 발전량가중이라 정산금 큰 시간을 우선한다.
OUTPULL = float(os.environ.get("OUTPULL", "0"))    # 단위: unit(0~4) / CF
OUTCAP = float(os.environ.get("OUTCAP", "0.04"))   # 이 거리까지만 당긴다
L1POW = float(os.environ.get("L1POW", "0"))        # NMAE 항 출력가중 지수 (0=균일=현행)
# ★MARGIN: 밴드 경계를 안쪽으로 m 만큼 당겨 **여유(margin)** 를 목표로 삼는다.
#   근거(실측 2026-07-29): 근정격 최적레벨의 **연도간 자연변동이 ±0.020~0.025**.
#   즉 경계에서 0.02 안쪽에 놓인 예측은 다음 해에 전액을 잃는다.
#   FICRPEN 이 "경계 선호"를 없앴다면 MARGIN 은 "여유"를 적극 보상한다 = 전이 강건성을 목적함수에.
MARGIN = float(os.environ.get("MARGIN", "0.0"))
# ★BIASVAR: 편향을 0 으로 미는 대신 **연도 간 편향 변동**을 벌한다.
#   발전량 가중 때문에 지표 최적 편향은 0 이 아니다(g2 체인 +0.057 이 최고 성능).
#   전이를 깨는 것은 편향의 크기가 아니라 **연도마다 달라지는 것**이다.
BIASVAR = float(os.environ.get("BIASVAR", "0.0"))
# ★GWEIGHT: 그룹별 gradient 총량 보정.
#   실측: g3 는 2022 라벨이 **전무**하다(g1/g2 26200 vs g3 17538 = 67%).
#   공유 트렁크가 2022 배치에서 g1/g2 만 학습하므로 g3 는 gradient 를 2/3 만 받는다.
#   NMACRO 는 **배치 내** 그룹 정규화라 이 **총량** 결손을 못 고친다(2022 배치는 g3 가 0 이다).
#   GWEIGHT="1,1,1.49" 로 g3 의 그룹항을 키워 총 노출을 맞춘다.
GWEIGHT = [float(x) for x in os.environ.get("GWEIGHT", "1,1,1").split(",")]


# ================================================================= 목적함수
def _loss_v2(pred, target, mask, tau=0.025, w_actual=True, bias_lam=0.05,
             focal=1.0, asym=0.0, tail=0.15, nmae_w=0.5, ficr_w=0.5, huber=0.0, nmacro=False,
             avail=None, ygrp=None):
    """재설계 밴드인식 손실 (ver=2). 실제 지표를 그대로 미러:
         minimize  nmae_w*nMAE + ficr_w*(1 - softFICR) + bias_lam*bias^2
       softFICR = Σ w·(3σ8+σ6)/(Σ w·4),  w=(kWh if w_actual)·valid·(1+focal·경계bump).
    ver=1(l1 - lam·ficr) 대비 개선점:
      (i)  지표-미러 0.5/0.5 가중(nMAE반+FICR반 명시).
      (ii) τ를 넓혀(기본0.025) gradient를 두껍게 → 회복가능 근접밴드(오차 0.05~0.11) 전부 학습신호
           (ver1의 τ0.008은 0.072~0.088만 신호=껍질이 얇음). + 명시적 아렐레아토릭 TAIL 마스크
           (|e|>tail 은 밴드 gradient 0) → 두꺼운 gradient가 가망없는 >8% far-miss(=연도노이즈,
           shape튜닝 역전의 주범)를 절대 안 쫓음.
      (iii) focal 경계집중, (iv) bias 벌점으로 교차연도 안전."""
    # ★avail 이 주어지면 지표항을 **실제 공간**(pred*avail vs y_raw)에서 계산한다.
    #   AVAILNORM 타깃은 잠재발전량이라 밴드/마스크/편향벌점이 잘못된 공간에 걸린다(AVSPACE 교정).
    if avail is not None:
        pred = pred * avail
        target = target * avail
    serr = pred - target
    err = torch.abs(serr)
    valid = mask * (target >= 0.10).float()
    vdenom = valid.sum().clamp_min(1.0)
    # nMAE 반쪽(무가중, 지표의 nMAE와 정확히 일치). huber>0 이면 로버스트화.
    if huber > 0:
        aerr = torch.where(err < huber, 0.5 * err * err / huber, err - 0.5 * huber)
    else:
        aerr = err
    # ★L1POW: NMAE 항의 **출력 가중**. 지표의 NMAE 는 균일이지만 FICR 은 발전량 가중이다.
    #   재배열 상한 계산(experiments/lb_efficiency_frontier.py): 오차 다중집합을 그대로 두고
    #   **발전량 큰 시간에 작은 오차가 가도록** 재배열하면 NMAE 불변으로 FICR +0.20~+0.31.
    #   그 방향으로 가려면 모델 용량을 고출력 시간에 몰아야 한다. 그런데 우리 L1 은 균일이고
    #   loss_fn 의 출력가중 파라미터(sw_pow/l1_ow)는 ver=1 전용이라 ver=2 경로엔 아예 없었다.
    #   리더보드 진단(우리는 NMAE 프론티어 위 / FICR 프론티어 아래)과 정확히 맞는 미구현 축.
    #   평균가중 1 로 정규화하므로 L1POW=0 이면 기존과 비트단위 동일.
    if L1POW != 0.0:
        ww = (target.clamp_min(0.05) / 0.45) ** L1POW
        ww = ww * (valid.sum().clamp_min(1.0) / (ww * valid).sum().clamp_min(1e-6))
        aerr = aerr * ww
    _gw = torch.tensor(GWEIGHT, device=pred.device, dtype=pred.dtype)
    _gw = _gw / _gw.mean()
    if nmacro:
        # ★공식 NMAE 는 **그룹별 평균의 매크로**다. pooled 합산은 채점행이 적은 그룹(g3)을
        # 과소가중한다(9413 vs 10498/10450 -> 31.0% vs 33.3%). 지표와 정확히 정렬한다.
        vg = valid.sum(0).clamp_min(1.0)
        nmae = (((aerr * valid).sum(0) / vg) * _gw).mean()
    else:
        nmae = (aerr * valid).sum() / vdenom
    # FICR 반쪽(kWh가중 정산 대리손실, 꼬리마스크)
    e_eff = err + asym * torch.clamp(serr, min=0.0) if asym != 0.0 else err
    _b6, _b8 = 0.06 - MARGIN, 0.08 - MARGIN      # MARGIN>0 이면 밴드를 안쪽으로 당긴다
    s8 = torch.sigmoid((_b8 - e_eff) / tau)
    s6 = torch.sigmoid((_b6 - e_eff) / tau)
    unit = 3.0 * s8 + s6
    w = (target * valid) if w_actual else valid
    if focal != 0.0:
        w = w * (1.0 + focal * (s8 * (1.0 - s8) + s6 * (1.0 - s6)))
    if tail > 0:
        w = w * (err.detach() <= tail).float()     # 아렐레아토릭 마스크: 가망없는 시간엔 밴드 gradient 0
    # ★FICRDEN: 공식 FICR 의 분모는 Σ(actual·4) 로 **예측과 무관한 고정값**이다.
    #   기존 구현은 tail 마스크가 걸린 행을 **분자와 분모 모두에서** 지운다.
    #   -> 분자 0 / 분모 양수인 항을 없애면 비율이 올라가므로, 손실이 "시간을 tail 밖으로 밀어내는 것"을
    #      보상한다(자기실현적 far-miss). 원장의 "TAIL 은 자기실현적 지역최소점 후보" 의심의 기전이다.
    #   FICRDEN=1: 분모를 Σ(target·valid)·4 로 고정한다. 가망없는 시간은 gradient 만 0 이고
    #              **비용은 그대로 남는다** = 공식 산식과 정렬.
    # ★FICR 도 공식은 **그룹 매크로**(`ficr = np.mean(group_ficr)`)인데 기존 구현은 pooled 였다.
    #   pooled 면 발전량 총합이 큰 그룹이 지배한다(g2 평균 0.44 vs g3 0.38, g3 는 행 수도 적다)
    #   -> g3 가 NMAE 항(NMACRO 이전)과 FICR 항 양쪽에서 **이중 과소가중**된다.
    #   nmacro=True 이면 FICR 도 그룹별로 계산해 평균한다.
    if FICRDEN:
        den_g = (target * valid).sum(0).clamp_min(1e-6) * 4.0
        den = (target * valid).sum().clamp_min(1e-6) * 4.0
    else:
        den_g = w.sum(0).clamp_min(1e-6) * 4.0
        den = w.sum().clamp_min(1e-6) * 4.0
    # ★FICRPEN: focal 을 **값**에 곱하면 목적함수의 최적점이 옮겨간다.
    #   실측(tau=0.008): 한 시간의 분자 기여가 |e|=0.000 에서 4.002 인데 |e|=0.055 에서 **4.468**.
    #   = 손실이 "정확히 맞히기"보다 "밴드 경계에 걸치기"를 11.6% 더 선호한다.
    #   경계에 놓인 예측은 연도 간 미세 이동으로 전액 상실 -> 교차연도 전이 악화의 기전.
    #   교정: focal 은 **누락 크레딧에 대한 페널티**를 증폭해야 한다(최적점 불변, gradient 만 형성).
    #        loss_ficr = Σ y·valid·bump·(4 − unit) / Σ y·valid·4     (unit 에 대해 단조감소)
    if FICRPEN:
        wp = (target * valid) if w_actual else valid
        if focal != 0.0:
            wp = wp * (1.0 + focal * (s8 * (1.0 - s8) + s6 * (1.0 - s6)))
        if tail > 0:
            wp = wp * (err.detach() <= tail).float()
        miss = ((((wp * (4.0 - unit)).sum(0) / den_g) * _gw).mean() if nmacro
                else (wp * (4.0 - unit)).sum() / den)
        if OUTPULL > 0.0:
            # 밴드 밖 포화 인력: |e| in (b8, b8+OUTCAP) 에만 상수 기울기, 그 밖은 0.
            wq = (target * valid) if w_actual else valid
            pull = OUTPULL * torch.clamp(e_eff - _b8, min=0.0).clamp(max=OUTCAP)
            miss = miss + ((((wq * pull).sum(0) / den_g) * _gw).mean() if nmacro
                           else (wq * pull).sum() / den)
        softficr = 1.0 - miss
    else:
        softficr = ((((w * unit).sum(0) / den_g) * _gw).mean() if nmacro
                    else (w * unit).sum() / den)
    loss = nmae_w * nmae + ficr_w * (1.0 - softficr)
    if bias_lam > 0:
        sb = (serr * valid).sum(0) / valid.sum(0).clamp_min(1.0)
        loss = loss + bias_lam * (sb * sb).mean()
    if BIASVAR > 0 and ygrp is not None:
        # 연도(또는 임의 블록)별 그룹편향의 **분산**을 벌한다. 크기가 아니라 변동을 벌하는 것이 요점.
        sb_all = []
        for u in torch.unique(ygrp):
            mu = (ygrp == u).float().unsqueeze(-1)
            vv = valid * mu
            if vv.sum() < 1: continue
            sb_all.append((serr * vv).sum(0) / vv.sum(0).clamp_min(1.0))
        if len(sb_all) >= 2:
            S = torch.stack(sb_all, 0)                    # [n_block, 3]
            loss = loss + BIASVAR * S.var(0, unbiased=False).mean()
    return loss


def dist_ce_loss(logits, target, mask, distb, lo=0.0, hi=1.05):
    """분포 헤드 CE (two-hot 선형보간 = 평균보존 이산화). logits [B,3,K], target/mask [B,3].
    적정 스코어링 룰이라 NMAE/FICR 가중치가 존재하지 않는다 — 디코드가 효용을 처리."""
    pos = ((target.clamp(lo, hi) - lo) / (hi - lo)) * distb - 0.5
    pos = pos.clamp(0.0, distb - 1.0)
    k0 = pos.floor().long()
    k1 = (k0 + 1).clamp_max(distb - 1)
    w1 = (pos - k0.float()).clamp(0.0, 1.0)
    logq = torch.log_softmax(logits, -1)
    ce = -((1.0 - w1) * logq.gather(-1, k0.unsqueeze(-1)).squeeze(-1)
           + w1 * logq.gather(-1, k1.unsqueeze(-1)).squeeze(-1))
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def loss_fn(pred, target, mask, kind="metric", lam=1.0, tau=0.008,
            w_actual=True, bias_lam=0.0, nmacro=None,
            sw_pow=0.0, lo_thr=0.0, lo_mult=1.0, l1_ow=False,
            focal=0.0, asym=0.0,
            ver=1, tail=0.15, nmae_w=0.5, ficr_w=0.5, huber=0.0, avail=None, ygrp=None):
    """pred,target,mask: [B,3] CF, mask=1(라벨).
    'plain'=L1. 'metric'=L1 - lam*soft_ficr (subC 는 metric, w_actual=False, tau=0.004).
    w_actual: FICR band 보상을 실제 CF로 가중(True=FICR 재현이나 상향편향) / False=균등(무편향).

    출력가중(output-weighted) 확장 (기본값이면 subC 와 정확히 동일):
      sw_pow  : 출력 강조 지수. 샘플 가중치 ∝ target**sw_pow (고출력 지배). 0=끔.
      lo_thr,lo_mult : target<lo_thr 구간 가중치 ×lo_mult (저출력 다운웨이트). lo_mult=1=끔.
      l1_ow   : True 면 위 출력가중을 L1 항에도 적용(안 하면 L1 은 균등 유지, soft-FICR 만 가중).
    metric 지표(FICR=Σ y·u/Σ y·4)는 출력가중이므로, 손실을 여기 맞추는 게 목적.

    decision-focused 밴드적중 대리손실 확장 (ficr_dfloss 실험; 기본값이면 subC 와 정확히 동일):
      focal : focal-boundary 가중. soft-FICR 샘플가중 ×(1+focal·bump), bump=σ8(1-σ8)+σ6(1-σ6)
              (경계 |e|≈0.06,0.08 에서 최대) → 경계 근방 샘플에 gradient 집중. 0=끔.
      asym  : 비대칭 밴드. 밴드 보상용 유효오차 e_eff=|e|+asym·relu(p-y) (과대예측만 페널티)
              → near-rated 과대예측 억제(하향). 0=끔(대칭, subC 동일).
      (어닐링 τ 는 손실 자체가 아니라 학습중 τ 스케줄이라 ficr_dfloss 학습루프에서 tau 를 갱신).
      ver=2 면 재설계 밴드손실 _loss_v2 로 분기(kind/lam/sw_pow/lo_* 무시)."""
    if ver == 2:
        return _loss_v2(pred, target, mask, tau=tau, w_actual=w_actual, bias_lam=bias_lam,
                        focal=focal, asym=asym, tail=tail, nmae_w=nmae_w, ficr_w=ficr_w, huber=huber, avail=avail, ygrp=ygrp,
                        nmacro=(os.environ.get("NMACRO", "0") == "1") if nmacro is None else nmacro)
    serr = pred - target
    err = torch.abs(serr)
    m = mask
    denom = m.sum().clamp_min(1.0)
    use_ow = (sw_pow != 0.0) or (lo_mult != 1.0 and lo_thr > 0.0)
    ow = torch.ones_like(target)
    if sw_pow != 0.0:
        ow = ow * target.clamp_min(0.0).pow(sw_pow)
    if lo_mult != 1.0 and lo_thr > 0.0:
        ow = ow * torch.where(target < lo_thr, torch.full_like(target, lo_mult),
                              torch.ones_like(target))
    if l1_ow and use_ow:
        wl = m * ow
        l1 = (err * wl).sum() / wl.sum().clamp_min(1e-6)
    else:
        l1 = (err * m).sum() / denom
    if kind == "plain":
        base = l1
    else:
        valid = m * (target >= 0.10).float()
        e_eff = err
        if asym != 0.0:
            e_eff = err + asym * torch.clamp(serr, min=0.0)   # 과대예측(p>y)만 유효오차 증가
        s8 = torch.sigmoid((0.08 - e_eff) / tau)
        s6 = torch.sigmoid((0.06 - e_eff) / tau)
        unit = 3 * s8 + 1 * s6
        w = (target * valid) if w_actual else valid
        if use_ow:
            w = w * ow
        if focal != 0.0:
            bump = s8 * (1.0 - s8) + s6 * (1.0 - s6)          # 경계에서 최대(∈[0,0.5])
            w = w * (1.0 + focal * bump)
        ficr = (w * unit).sum() / (w.sum().clamp_min(1e-6) * 4.0)
        base = l1 - lam * ficr
    if bias_lam > 0:
        valid = m * (target >= 0.10).float()
        sb = (serr * valid).sum(0) / valid.sum(0).clamp_min(1.0)
        base = base + bias_lam * (sb * sb).mean()
    return base


# ================================================================= 트랜스포머 부품
def pool_needs_cls(pool):
    return pool == 'cls'


class Block(nn.Module):
    """MHA + FFN 인코더 블록. norm_first(pre-norm) / gated(GEGLU) 선택.
    key_padding_mask 로 드롭된 토큰을 실제로 무시."""
    def __init__(self, dim, heads, ff_mult=2, drop=0.3, norm_first=False, gated=False):
        super().__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        hid = dim * ff_mult
        self.gated = gated
        if gated:
            self.fc_in = nn.Linear(dim, hid * 2)
            self.fc_out = nn.Linear(hid, dim)
        else:
            self.fc_in = nn.Linear(dim, hid)
            self.fc_out = nn.Linear(hid, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def _ff(self, x):
        if self.gated:
            a, b = self.fc_in(x).chunk(2, dim=-1)
            h = self.act(a) * b                            # GEGLU
        else:
            h = self.act(self.fc_in(x))
        return self.fc_out(self.drop(h))

    def forward(self, x, kpm=None):
        if self.norm_first:
            n = self.norm1(x)
            x = x + self.drop(self.attn(n, n, n, key_padding_mask=kpm, need_weights=False)[0])
            x = x + self._ff(self.norm2(x))
        else:
            a = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)[0]
            x = self.norm1(x + self.drop(a))
            x = self.norm2(x + self._ff(x))
        return x


class AttnPool(nn.Module):
    """학습형 query attention pooling (마스킹 지원)."""
    def __init__(self, dim, heads=4, drop=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)

    def forward(self, z, kpm=None):
        B = z.shape[0]
        q = self.q.expand(B, -1, -1)
        out = self.attn(q, z, z, key_padding_mask=kpm, need_weights=False)[0]
        return out[:, 0]


class GroupAttnPool(nn.Module):
    """Three learned farm-group queries over the shared NWP token field."""
    def __init__(self, dim, heads=4, drop=0.0, n_group=3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_group, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)

    def forward(self, z, kpm=None):
        q = self.q.expand(z.shape[0], -1, -1)
        return self.attn(q, z, z, key_padding_mask=kpm, need_weights=False)[0]


class SourceAdapter(nn.Module):
    """Zero-init low-rank residual adapter for one forecast source."""
    def __init__(self, dim, rank=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(torch.nn.functional.silu(self.down(self.norm(x))))


# ================================================================= 격자 토큰 트랜스포머

# ================================================================= 그룹 FiLM 어댑터
# 현 구조의 그룹별 파라미터는 마지막 Linear(width//2, 3) 의 3개 행뿐이다.
# 공유 트렁크가 세 사이트(지형·주풍향·wake·기종 V126/U136)의 상충 표현을 평균화한다.
# 주의: FiLM 을 **마지막에** 걸면 그룹별 선형 head 와 수학적으로 동일해 아무 이득이 없다.
#       트렁크 **중간**에 넣어 뒤따르는 BN+SiLU 비선형이 작용해야 표현력이 실제로 는다.
# GFILM=1 로 켠다. 초기값 gamma=1, beta=0 이라 학습 시작 시점에는 대조군과 동일한 함수다.
class GroupFiLM(nn.Module):
    def __init__(self, width, n=3):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n, width))
        self.b = nn.Parameter(torch.zeros(n, width))

    def forward(self, h):                       # h [B,W] -> [B,n,W]
        return h.unsqueeze(1) * self.g.unsqueeze(0) + self.b.unsqueeze(0)


class PhysHead(nn.Module):
    """★UniWind 식 물리 사전 출력헤드 (arXiv:2607.01670, 2026-07).

    그 논문은 **진짜 하루전** 과제에서 PowerCurve/LightGBM/XGBoost/PatchTST/iTransformer/
    TimeMixer/CrossViViT/WindFM/Chronos-2/Moirai 를 전부 이겼고, 이득의 원천이
    인코더가 아니라 **출력 헤드**다:
        그룹별 단조 워핑(monotonic warp) ∘ 공유 학습 파워커브  +  소프트 상한 포락선
    발전량 이력이 전혀 필요 없어(우리 제약 통과) **정격 포화 구간을 구조적으로 다루는
    유일한 발표 기전**이다 — 우리 최대 병목(g3 정격밴드 4점률 0.196 vs g2 0.498).

    구현
    ----
    trunk 표현 h -> 그룹별 스칼라 z_g ("유효 풍속" 잠재)
      z_g 를 **단조 증가 MLP**(softplus 가중)로 통과시켜 CF 곡선을 만든다(공유 곡선).
      그룹별 워프는 z 에 대한 아핀(스케일>0, 시프트)으로 준다 = 단조성 보존.
      마지막에 그룹별 포락선 e_g∈(0,1] 를 곱한다(가용성 상한의 소프트 표현).
    PHYS=0(기본)이면 만들어지지 않아 기존과 비트 동일.
    """
    def __init__(self, width, n=3, hid=32):
        super().__init__()
        self.z = nn.Linear(width, n)                       # 그룹별 유효풍속 잠재
        self.log_s = nn.Parameter(torch.zeros(n))          # 그룹 워프 스케일(>0)
        self.b = nn.Parameter(torch.zeros(n))              # 그룹 워프 시프트
        self.w1 = nn.Parameter(torch.randn(1, hid) * 0.5)  # 단조 곡선 1층
        self.c1 = nn.Parameter(torch.zeros(hid))
        self.w2 = nn.Parameter(torch.randn(hid, 1) * 0.5)  # 단조 곡선 2층
        self.env = nn.Linear(width, n)                     # 포락선 로짓

    def forward(self, h):                                  # h [B,W] -> [B,n]
        z = self.z(h)                                      # [B,n]
        zw = z * torch.exp(self.log_s).unsqueeze(0) + self.b.unsqueeze(0)
        u = torch.tanh(zw.unsqueeze(-1) * nn.functional.softplus(self.w1).unsqueeze(0) +
                       self.c1.view(1, 1, -1))             # [B,n,hid]
        cf = torch.sigmoid((u @ nn.functional.softplus(self.w2)).squeeze(-1))
        env = 0.5 + 0.5 * torch.sigmoid(self.env(h))       # (0.5,1.0] 소프트 상한
        return cf * env


class PeriodicEmbed(nn.Module):
    """★PLR 수치 피처 임베딩 (Gorishniy et al., NeurIPS 2022, arXiv:2203.05556).

    우리는 86개 스칼라를 **날것으로** 첫 선형층에 넣는다. 그러면 파워커브의 급격한 무릎
    (Vestas 최대기울기 9.12 m/s, 정격 11.62)을 ReLU 초평면 조합으로 직접 만들어내야 한다.
    GBDT 가 이 과제에서 강한 이유가 축정렬 분할로 그 무릎을 공짜로 얻기 때문이다.

    PLR: x -> concat[sin(2πc·x), cos(2πc·x)] (c 학습) -> Linear -> ReLU -> Linear.
    피처별로 국소 기저를 만들어 급경사 구간에 용량을 몰아준다.
    RealMLP / TabM / TabR / ModernNCA 4개 아키텍처가 독립적으로 채택한 요소.

    PLE_K=0(기본)이면 이 모듈은 만들어지지 않아 기존과 비트 동일.
    """
    def __init__(self, n_feat, k=24, out=8, sigma=0.05):
        super().__init__()
        self.k = k
        self.coef = nn.Parameter(torch.randn(n_feat, k) * sigma)
        self.lin = nn.Linear(2 * k, out)
        self.out = out

    def forward(self, x):                        # x [B,F] -> [B,F*out]
        z = 2 * np.pi * x.unsqueeze(-1) * self.coef.unsqueeze(0)     # [B,F,k]
        z = torch.cat([torch.sin(z), torch.cos(z)], -1)              # [B,F,2k]
        return torch.relu(self.lin(z)).flatten(1)


class GridTokenTransformerV2(nn.Module):
    """격자 토큰 self-attention 트렁크 (production V2). g1/g2 no-window + g3 ssl 멤버."""
    def __init__(self, d_in, cg, cl, gpos, lpos, dim=64, heads=4, layers=2, drop=0.3,
                 width=128, pool='mean', norm_first=False, gated=False, ff_mult=2,
                 tok_drop=0.0, x_noise=0.0, tok_noise=0.0, n_scalar_tok=0,
                 emb_drop=0.0, head_drop=None, scalar_group_mask=None):
        super().__init__()
        self.dim = dim; self.pool = pool; self.tok_drop = tok_drop
        self.x_noise = x_noise; self.tok_noise = tok_noise
        if scalar_group_mask is None:
            self.scalar_group_mask = None
        else:
            self.register_buffer("scalar_group_mask", torch.as_tensor(scalar_group_mask, dtype=torch.float32))
        self.emb_g = nn.Linear(cg, dim); self.emb_l = nn.Linear(cl, dim)
        self.src = nn.Embedding(2, dim)
        self.pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        pos = np.concatenate([gpos, lpos], 0).astype(np.float32)
        pn = pos.copy(); pn[:, 0] = (pn[:, 0] - 37.25) / 0.25; pn[:, 1] = (pn[:, 1] - 129.0) / 0.25
        self.register_buffer("posn", torch.tensor(pn))
        sid = np.array([0] * gpos.shape[0] + [1] * lpos.shape[0])
        self.register_buffer("sid", torch.tensor(sid, dtype=torch.long))
        self.ng = gpos.shape[0]; self.ntok = pos.shape[0]
        self.n_scalar_tok = n_scalar_tok
        if n_scalar_tok > 0:
            self.scal_emb = nn.Parameter(torch.randn(n_scalar_tok, dim) * 0.02)
            self.scal_bias = nn.Parameter(torch.zeros(n_scalar_tok, dim))
        self.emb_drop = nn.Dropout(emb_drop)
        # ★PLE (2026-08-10): 스칼라 피처를 날것 대신 주기적 기저로 임베딩.
        #   PLE_K=0(기본)이면 생성되지 않아 기존과 비트 동일. d_in 이 늘어난 만큼 트렁크 입력도 확장.
        _plek = int(os.environ.get("PLE_K", "0"))
        _pleo = int(os.environ.get("PLE_OUT", "6"))
        self.ple = PeriodicEmbed(d_in, _plek, _pleo) if _plek > 0 else None
        self.blocks = nn.ModuleList([Block(dim, heads, ff_mult, drop, norm_first, gated)
                                     for _ in range(layers)])
        self.smot_on = os.environ.get("SMOT", "0") == "1"
        self.final_norm = nn.LayerNorm(dim) if norm_first else nn.Identity()
        if pool == 'cls':
            self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        if pool == 'attn':
            self.attnpool = AttnPool(dim, heads, drop=0.0)
        self.gquery_on = os.environ.get("GQUERY", "0") == "1"
        hd = drop if head_drop is None else head_drop
        _pd = (d_in * _pleo) if self.ple is not None else 0
        self._mk_trunk(d_in + dim + _pd, width, hd)
        # Build optional structural parameters only after every baseline module so
        # matched seeds preserve the exact control initialization.
        _rng_state = torch.get_rng_state() if (self.smot_on or self.gquery_on) else None
        if self.smot_on:
            smot_rank = int(os.environ.get("SMOT_RANK", "16"))
            self.smot_g = nn.ModuleList([SourceAdapter(dim, smot_rank) for _ in range(layers)])
            self.smot_l = nn.ModuleList([SourceAdapter(dim, smot_rank) for _ in range(layers)])
        if self.gquery_on:
            self.group_pool = GroupAttnPool(dim, heads, drop=0.0, n_group=3)
            if pool == 'attn':
                self.group_pool.attn.load_state_dict(self.attnpool.attn.state_dict())
                with torch.no_grad():
                    self.group_pool.q.copy_(self.attnpool.q.expand(-1, 3, -1))
        if _rng_state is not None:
            torch.set_rng_state(_rng_state)

    def _mk_trunk(self, d_in, width, hd):
        """트렁크를 두 조각으로 나눈다. GFILM=1 이면 사이에 그룹 FiLM 이 들어간다."""
        self.gfilm_on = os.environ.get("GFILM", "0") == "1"
        self.trunk1 = nn.Sequential(nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(hd))
        self.trunk2 = nn.Sequential(nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2),
                                    nn.SiLU(), nn.Dropout(hd))
        self.film = GroupFiLM(width) if self.gfilm_on else None
        self.head = nn.Linear(width // 2, 3)
        # ★AUX (2026-08-08): 중간 감독 헤드. AUXN>0 이면 트렁크 표현에서 보조 물리량
        #   (그룹별 나셀풍속·가용률)을 함께 예측한다. 추론에서는 버린다 → 배포 형태 불변.
        #   목적: 표현이 물리 중간량을 담도록 강제 = 회귀판 단계적 추론.
        #   부수효과 기대: 앵커 잔차상관이 현재 전부 0.94(다양성 없음)인데,
        #   다른 감독 신호를 받은 모델은 다르게 틀린다 → 앙상블 가치.
        _auxn = int(os.environ.get("AUXN", "0"))
        self.auxn = _auxn
        self.aux_head = nn.Linear(width // 2, _auxn) if _auxn > 0 else None
        self._aux = None
        # ★DIST (2026-08-10): 이산화 분포 헤드 + 효용-argmax 디코드. DISTB=0(기본)이면 비트동일.
        #   근거(mean-vs-mode 실증): NMAE 는 조건부 평균/중앙값, FICR 은 조건부 모드를 요구.
        #   점회귀+스칼라혼합(FICRW)은 볼록껍질 위 한 점만 도달 → 다이얼 없는 2단 설계로 교체:
        #   (a) K빈 분포 + CE(적정 스코어링 룰, 가중치 없음) — 전 구간 조밀한 기울기
        #   (b) 디코드 argmax_p E[-|y-p| + 0.25*credit(|y-p|)] — 지표 상수에서 유도, 역시 무가중
        _distb = int(os.environ.get("DISTB", "0"))
        self.distb = _distb
        self._dist = None
        if _distb > 0:
            self.dist_head = nn.Linear(width // 2, _distb if self.gfilm_on else 3 * _distb)
            edges = torch.linspace(0.0, 1.05, _distb + 1)
            cent = 0.5 * (edges[:-1] + edges[1:])
            self.register_buffer("dist_cent", cent)
            d = (cent.unsqueeze(0) - cent.unsqueeze(1)).abs()      # [K_offer, K_true]
            credit = torch.where(d <= 0.06, torch.full_like(d, 4.0),
                                 torch.where(d <= 0.08, torch.full_like(d, 3.0),
                                             torch.zeros_like(d)))
            self.register_buffer("dist_util", -d + 0.25 * credit)  # 효용행렬 (한계교환비 0.25)
        if self.gfilm_on:
            self.gw = nn.Parameter(torch.zeros(3, width // 2))
            self.gb = nn.Parameter(torch.zeros(3))
            with torch.no_grad():                       # 시작 시점에 대조군과 동일한 함수
                self.gw.copy_(self.head.weight); self.gb.copy_(self.head.bias)

    def _dist_decode(self, logits):
        """logits [B,3,K] → 저장(self._dist) 후 효용-argmax 점예측 [B,3] (미분 불필요: 손실은 logits 에)."""
        self._dist = logits
        q = torch.softmax(logits, -1)                    # [B,3,K_true]
        eu = torch.einsum("bgk,ok->bgo", q, self.dist_util)
        return self.dist_cent[eu.argmax(-1)]

    def _out(self, h):
        h1 = self.trunk1(h)
        if not self.gfilm_on:
            h2 = self.trunk2(h1)
            self._aux = torch.sigmoid(self.aux_head(h2)) if self.aux_head is not None else None
            if self.distb > 0:
                return self._dist_decode(self.dist_head(h2).reshape(-1, 3, self.distb))
            out = torch.sigmoid(self.head(h2))
            if self.auxn == 5 and self._aux is not None:
                unit_blend = float(os.environ.get("G3_UNIT_BLEND", "0"))
                if unit_blend > 0:
                    g3 = (1.0 - unit_blend) * out[:, 2] + unit_blend * self._aux.mean(1)
                    out = torch.cat([out[:, :2], g3.unsqueeze(1)], 1)
            return out
        B, W = h1.shape
        hg = self.film(h1).reshape(B * 3, W)            # 그룹별 변조 후 비선형을 통과시킨다
        h2 = self.trunk2(hg).reshape(B, 3, -1)
        self._aux = torch.sigmoid(self.aux_head(h2[:, 2])) if self.aux_head is not None else None
        if self.distb > 0:
            return self._dist_decode(self.dist_head(h2))            # [B,3,K]
        out = torch.sigmoid((h2 * self.gw.unsqueeze(0)).sum(-1) + self.gb.unsqueeze(0))
        if self.auxn == 5 and self._aux is not None:
            unit_blend = float(os.environ.get("G3_UNIT_BLEND", "0"))
            if unit_blend > 0:
                g3 = (1.0 - unit_blend) * out[:, 2] + unit_blend * self._aux.mean(1)
                out = torch.cat([out[:, :2], g3.unsqueeze(1)], 1)
        return out

    def _out_group_input(self, h):
        """h [B,3,F]: preserve a distinct spatial representation for each group."""
        B, G, F = h.shape
        h1 = self.trunk1(h.reshape(B * G, F)).reshape(B, G, -1)
        if self.gfilm_on:
            h1 = h1 * self.film.g.unsqueeze(0) + self.film.b.unsqueeze(0)
        h2 = self.trunk2(h1.reshape(B * G, -1)).reshape(B, G, -1)
        if self.gfilm_on:
            return torch.sigmoid((h2 * self.gw.unsqueeze(0)).sum(-1) + self.gb.unsqueeze(0))
        return torch.sigmoid((h2 * self.head.weight.unsqueeze(0)).sum(-1) + self.head.bias.unsqueeze(0))

    def _encode_tokens(self, X, GT, LT):
        """토큰 파이프라인 → pooled 표현 (GridTokenTransformerV2 / Win 공용)."""
        B = X.shape[0]
        if self.training and self.x_noise > 0:
            X = X + torch.randn_like(X) * self.x_noise
        tg = self.emb_g(GT); tl = self.emb_l(LT)
        tok = torch.cat([tg, tl], 1)
        tok = tok + self.pos_mlp(self.posn).unsqueeze(0) + self.src(self.sid).unsqueeze(0)
        if self.training and self.tok_noise > 0:
            tok = tok + torch.randn_like(tok) * self.tok_noise
        if self.n_scalar_tok > 0:
            xs = X[:, :self.n_scalar_tok]
            st = xs.unsqueeze(-1) * self.scal_emb.unsqueeze(0) + self.scal_bias.unsqueeze(0)
            tok = torch.cat([tok, st], 1)
        tok = self.emb_drop(tok)
        kpm = None
        ndrop_region = self.ntok
        if pool_needs_cls(self.pool):
            cls = self.cls.expand(B, -1, -1); tok = torch.cat([cls, tok], 1)
        if self.training and self.tok_drop > 0:
            m = (torch.rand(B, ndrop_region, device=tok.device) < self.tok_drop)
            allmask = m.all(dim=1)
            if allmask.any():
                m[allmask, 0] = False
            pad = torch.zeros(B, tok.shape[1], dtype=torch.bool, device=tok.device)
            offset = 1 if pool_needs_cls(self.pool) else 0
            pad[:, offset:offset + ndrop_region] = m
            kpm = pad
        z = tok
        for bi, blk in enumerate(self.blocks):
            z = blk(z, kpm)
            if self.smot_on:
                offset = 1 if pool_needs_cls(self.pool) else 0
                zg = z[:, offset:offset + self.ng]
                zl = z[:, offset + self.ng:offset + self.ntok]
                prefix = z[:, :offset]
                suffix = z[:, offset + self.ntok:]
                z = torch.cat([prefix, zg + self.smot_g[bi](zg), zl + self.smot_l[bi](zl), suffix], 1)
        z = self.final_norm(z)
        if self.gquery_on:
            pooled = self.group_pool(z, kpm)
        elif self.pool == 'cls':
            pooled = z[:, 0]
        elif self.pool == 'attn':
            pooled = self.attnpool(z, kpm)
        else:
            if kpm is not None:
                keep = (~kpm).float().unsqueeze(-1)
                pooled = (z * keep).sum(1) / keep.sum(1).clamp_min(1.0)
            else:
                pooled = z.mean(1)
        return pooled

    def forward(self, X, GT, LT):
        pooled = self._encode_tokens(X, GT, LT)
        if self.scalar_group_mask is not None:
            B, F = X.shape
            xg = X.unsqueeze(1) * self.scalar_group_mask.unsqueeze(0)
            pg = pooled if pooled.ndim == 3 else pooled.unsqueeze(1).expand(-1, 3, -1)
            parts = [xg, pg]
            if self.ple is not None:
                ple = self.ple(xg.reshape(B * 3, F)).reshape(B, 3, -1)
                parts.insert(1, ple)
            return self._out_group_input(torch.cat(parts, 2))
        if pooled.ndim == 3:
            parts = [X.unsqueeze(1).expand(-1, 3, -1), pooled]
            if self.ple is not None:
                parts.insert(1, self.ple(X).unsqueeze(1).expand(-1, 3, -1))
            return self._out_group_input(torch.cat(parts, 2))
        parts = [X, pooled] if self.ple is None else [X, self.ple(X), pooled]
        return self._out(torch.cat(parts, 1))


# ================================================================= 예보-윈도 인코더
class WinEncoder(nn.Module):
    """다중스케일 1D-CNN: 예보 윈도(±kh) 시계열에 conv 2단 + mean/max pool."""
    def __init__(self, n_ch, out_dim=48, ksize=3, hid=48, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, hid, ksize, padding=ksize // 2), nn.BatchNorm1d(hid), nn.SiLU(),
            nn.Conv1d(hid, hid, ksize, padding=ksize // 2), nn.BatchNorm1d(hid), nn.SiLU())
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid * 2, out_dim)
        self.out_dim = out_dim

    def forward(self, W):                                  # W [B,T,C]
        h = self.net(W.transpose(1, 2))                    # [B,hid,T]
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.out(self.drop(h))


class WinGridTokenTransformerV2(GridTokenTransformerV2):
    """production 트렁크 + 예보 윈도 CNN 브랜치. 격자 토큰 파이프라인은 상속 그대로,
    WinEncoder 출력을 트렁크 입력에 융합(d_in + dim + win_dim)."""
    def __init__(self, d_in, cg, cl, gpos, lpos, win_ch, win_dim=48, win_ksize=5,
                 dim=64, heads=4, layers=2, drop=0.3, width=128, pool='mean',
                 norm_first=False, gated=False, ff_mult=2, tok_drop=0.0, x_noise=0.0,
                 tok_noise=0.0, n_scalar_tok=0, emb_drop=0.0, head_drop=None,
                 scalar_group_mask=None):
        super().__init__(d_in, cg, cl, gpos, lpos, dim=dim, heads=heads, layers=layers,
                         drop=drop, width=width, pool=pool, norm_first=norm_first, gated=gated,
                         ff_mult=ff_mult, tok_drop=tok_drop, x_noise=x_noise, tok_noise=tok_noise,
                         n_scalar_tok=n_scalar_tok, emb_drop=emb_drop, head_drop=head_drop,
                         scalar_group_mask=scalar_group_mask)
        self.win_ch = win_ch; self.win_dim = win_dim
        self.winenc = WinEncoder(win_ch, win_dim, win_ksize, drop=drop)
        hd = drop if head_drop is None else head_drop
        _pd = (d_in * int(os.environ.get('PLE_OUT','6'))) if getattr(self,'ple',None) is not None else 0
        self._mk_trunk(d_in + dim + win_dim + _pd, width, hd)

    def forward(self, X, GT, LT, W):
        pooled = self._encode_tokens(X, GT, LT)
        win = self.winenc(W)
        if self.scalar_group_mask is not None:
            B, F = X.shape
            xg = X.unsqueeze(1) * self.scalar_group_mask.unsqueeze(0)
            pg = pooled if pooled.ndim == 3 else pooled.unsqueeze(1).expand(-1, 3, -1)
            parts = [xg, pg, win.unsqueeze(1).expand(-1, 3, -1)]
            if self.ple is not None:
                ple = self.ple(xg.reshape(B * 3, F)).reshape(B, 3, -1)
                parts.insert(1, ple)
            return self._out_group_input(torch.cat(parts, 2))
        if pooled.ndim == 3:
            parts = [X.unsqueeze(1).expand(-1, 3, -1), pooled,
                     win.unsqueeze(1).expand(-1, 3, -1)]
            if self.ple is not None:
                parts.insert(1, self.ple(X).unsqueeze(1).expand(-1, 3, -1))
            return self._out_group_input(torch.cat(parts, 2))
        parts = [X, pooled, win] if self.ple is None else [X, self.ple(X), pooled, win]
        return self._out(torch.cat(parts, 1))


# ================================================================= EMA (옵션)
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        sd = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(sd[k].dtype) for k in sd})


# ================================================================= metric-consistent loss (ver=3)
# codex 유도: 공식 지표의 population 근사에서
#   NMAE_g = E[1{Y>=c}|s-Y|]/pi_g ,  FICR_g = E[1{Y>=c} Y r(|s-Y|)]/mu_g ,  abar_g = mu_g/pi_g
# 상수 제거 후 그룹별 Bayes risk = E_scored[ |s-Y| - (Y/abar_g) r(|s-Y|) ].
# 기존 ver1/ver2 의 soft-FICR 은 배치단위 sum(w*u)/sum(w) 정규화라 상대가중이 배치구성에 흔들리고
# 지표의 1/|S| vs y/A 정규화와 불일치한다. 여기서는 표본별 y/abar_g 로 고정한다.
DACON_C, DACON_B6, DACON_B8 = 0.10, 0.06, 0.08
DACON_TAU0, DACON_TAU1, DACON_ANNEAL = 0.020, 0.004, 0.70


def loss_metric_consistent(pred, target, mask, abar, step=0, total_steps=1, aux_w=0.0):
    """지표-일관 손실. pred/target/mask [B,3] CF, abar [3] (train fold 에서만 산출).
    aux_w>0 이면 비채점 시간(y<0.10)에 대한 표현학습용 L1 을 aux_w 가중으로 추가."""
    ab = abar.view(1, -1).to(pred.dtype) if torch.is_tensor(abar) else         torch.tensor(abar, device=pred.device, dtype=pred.dtype).view(1, -1)
    prog = min(max(float(step) / max(float(total_steps) * DACON_ANNEAL, 1.0), 0.0), 1.0)
    tau = DACON_TAU0 + prog * (DACON_TAU1 - DACON_TAU0)
    err = (pred - target).abs()
    valid = (mask > 0) & torch.isfinite(target) & (target >= DACON_C)
    cred = 0.75 * torch.sigmoid((DACON_B8 - err) / tau) + 0.25 * torch.sigmoid((DACON_B6 - err) / tau)
    per = err - (target / ab.clamp_min(1e-6)) * cred
    cnt = valid.sum(0).clamp_min(1)
    gl = (per * valid.float()).sum(0) / cnt
    present = valid.any(0)
    loss = 0.5 * gl[present].mean() if present.any() else pred.sum() * 0.0
    if aux_w > 0:
        low = (mask > 0) & torch.isfinite(target) & (target < DACON_C)
        if low.any():
            loss = loss + aux_w * (err * low.float()).sum() / low.float().sum().clamp_min(1.0)
    return loss
