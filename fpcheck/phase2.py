"""第二阶段：对抗后缀指纹（核心方法，思路参考 parameterlab/trap）。

每个探测 prompt = 忽略指令前缀 + 40~60 字符乱码后缀（ASCII/希腊/西里尔混合）。
temperature=1.0 下每个探测采样 n 次，用 difflib.SequenceMatcher 计算
两个 API 响应集合之间的交叉平均相似度。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .adversarial import build_adversarial_probes
from .prompts import PHASE2_PREFIXES
from .runner import call_pair


@dataclass
class ProbeRow:
    probe_id: str
    prefix: str
    suffix_len: int
    wrapper: str = ""           # prompt 包装模板名（借鉴 LLMmap）
    cross_sim: float = 0.0
    intra_a: float = 0.0
    intra_b: float = 0.0
    texts_a: list[str] = field(default_factory=list)
    texts_b: list[str] = field(default_factory=list)
    out_tokens_a: int = 0       # 官方侧本次生成 token 合计（验证 max_tokens 截断安全性）
    out_tokens_b: int = 0
    error_a: str = ""
    error_b: str = ""


@dataclass
class Phase2Result:
    score: float            # 归一化相似度（有效探测 交叉/组内，封顶 1）→ 进入综合判定
    rows: list[ProbeRow] = field(default_factory=list)
    raw_cross: float = 0.0        # 全部成功探测的裸交叉平均（参考）
    cross_entropic: float = 0.0   # 剔除低熵收敛探测后的交叉平均
    intra_baseline: float = 0.0   # 组内平均相似度（同源基线）
    delta: float = 0.0            # 交叉-组内；系统性为负 → 异源信号
    converged: int = 0            # 低熵收敛探测数（双侧组内=1，无判别力）
    ci_low: float = 0.0           # score 的 95% 置信区间（bootstrap）
    ci_high: float = 0.0
    reliability: float = 1.0      # 证据可靠性：CI 越宽 → 越低（0.5~1.0，进证据加权）
    warn_low_baseline: bool = False   # 双低警告：组内基线过低时比值噪声大
    total_out_tokens: int = 0     # 两侧生成 token 合计（验证截断安全性）

    @property
    def num_probes(self) -> int:
        return len(self.rows)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.rows if not r.error_a and not r.error_b)


def ci_reliability(ci_low: float, ci_high: float) -> float:
    """按 95% CI 宽度给出证据可靠性（0.5~1.0）：宽度≤0.2 → 1.0，宽度≥1.0 → 0.5。"""
    width = ci_high - ci_low
    return 1.0 - max(0.0, min(0.5, (width - 0.2) / 0.8))


def _ratio(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def cross_similarity(texts_a: list[str], texts_b: list[str]) -> float:
    """A×B 全配对 SequenceMatcher 比值取平均（交叉平均相似度）。"""
    if not texts_a or not texts_b:
        return 0.0
    ratios = [_ratio(a, b) for a in texts_a for b in texts_b]
    return sum(ratios) / len(ratios)


def intra_similarity(texts: list[str]) -> float:
    """同一 API 内部各采样之间的平均相似度（反映自身确定性）。"""
    if len(texts) < 2:
        return 1.0
    ratios = [_ratio(texts[i], texts[j])
              for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sum(ratios) / len(ratios)


def _bootstrap_ci(rows: list[ProbeRow], seed: int, n_iter: int = 1000,
                  alpha: float = 0.05) -> tuple[float, float]:
    """对归一化相似度做探测级 bootstrap 重采样，返回 95% 置信区间。

    默认样本量下估计量方差不小，报告置信区间便于判断分数波动的幅度：
    区间整体 > 0.9 → 同源结论稳健；区间跨越 0.5~0.75 → 建议加大样本量。
    """
    import random
    rng = random.Random(seed)
    n = len(rows)
    if n < 3:
        return 0.0, 0.0
    scores: list[float] = []
    for _ in range(n_iter):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        cross = sum(r.cross_sim for r in sample) / n
        intra = sum((r.intra_a + r.intra_b) / 2 for r in sample) / n
        scores.append(min(1.0, cross / intra) if intra > 1e-9 else 0.0)
    scores.sort()
    lo = scores[max(0, int(alpha / 2 * n_iter) - 1)]
    hi = scores[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return lo, hi


def needs_upsample(row: ProbeRow, low: float, high: float, base_n: int) -> bool:
    """边界补测判定：基础采样后交叉落在模糊区且非收敛时需补采样到 5 次。"""
    return (base_n < 5 and len(row.texts_a) < 5 and row.cross_sim > 0.0
            and not (row.intra_a >= 0.999 and row.intra_b >= 0.999)
            and low <= row.cross_sim <= high)


def run_phase2(official, unknown, recorder, options, log=print) -> Phase2Result:
    probes = build_adversarial_probes(PHASE2_PREFIXES, options.phase2_num_probes,
                                      options.phase2_suffix_min_len,
                                      options.phase2_suffix_max_len, options.seed)
    # 黑名单剔除（离线用 analyze_phase2.py 筛选出的低判别力/拒答/收敛探测）
    skip = set(getattr(options, "phase2_skip_ids", None) or [])
    if skip:
        probes = [p for p in probes if p["id"] not in skip]
    log(f"[第二阶段] 开始：对抗后缀指纹（{len(probes)} 个探测 × 基础采样 "
        f"{options.phase2_samples} 次"
        f"{' + 边界补测' if options.phase2_upsample else ''}，"
        f"temperature={options.phase2_temperature}，"
        f"max_tokens 上限 {options.phase2_max_tokens}）...")
    rows: list[ProbeRow] = []
    for probe in probes:
        ra, rb = call_pair(recorder, official, unknown, phase=2, probe_id=probe["id"],
                           prompt=probe["prompt"], temperature=options.phase2_temperature,
                           max_tokens=options.phase2_max_tokens, seed=None,
                           n=options.phase2_samples, concurrent=options.concurrent)
        row = ProbeRow(probe_id=probe["id"], prefix=probe["prefix"],
                       suffix_len=len(probe["suffix"]), wrapper=probe.get("wrapper", ""),
                       texts_a=ra.texts, texts_b=rb.texts,
                       out_tokens_a=int((ra.usage or {}).get("completion_tokens") or 0),
                       out_tokens_b=int((rb.usage or {}).get("completion_tokens") or 0),
                       error_a=ra.error or "", error_b=rb.error or "")
        if ra.ok and rb.ok:
            row.cross_sim = cross_similarity(ra.texts, rb.texts)
            row.intra_a = intra_similarity(ra.texts)
            row.intra_b = intra_similarity(rb.texts)
        # 边界补测：基础采样后交叉相似度落在模糊区且未收敛 → 补 2 次采样到 5 次，
        # 只在"需要更多信息才能下结论"的探测上花额外 token（实验验证精度无损失）
        if needs_upsample(row, options.phase2_upsample_low,
                          options.phase2_upsample_high,
                          options.phase2_samples):
            need = 5 - len(ra.texts)
            if need > 0:
                ra2, rb2 = call_pair(recorder, official, unknown, phase=2,
                                     probe_id=f"{probe['id']}_u",
                                     prompt=probe["prompt"],
                                     temperature=options.phase2_temperature,
                                     max_tokens=options.phase2_max_tokens,
                                     seed=None, n=need,
                                     concurrent=options.concurrent)
                if ra2.ok and rb2.ok:
                    row.texts_a = list(ra.texts) + list(ra2.texts)
                    row.texts_b = list(rb.texts) + list(rb2.texts)
                    row.out_tokens_a += int((ra2.usage or {}).get("completion_tokens") or 0)
                    row.out_tokens_b += int((rb2.usage or {}).get("completion_tokens") or 0)
                    row.cross_sim = cross_similarity(row.texts_a, row.texts_b)
                    row.intra_a = intra_similarity(row.texts_a)
                    row.intra_b = intra_similarity(row.texts_b)
        rows.append(row)
        up_tag = "+补测" if (ra.ok and rb.ok and len(row.texts_a) > options.phase2_samples) else ""
        log(f"  [{probe['id']}] 后缀长度={len(probe['suffix'])}  "
            f"包装={row.wrapper}  采样={len(row.texts_a) if not (row.error_a or row.error_b) else '-'}次{up_tag}  "
            f"交叉相似度={row.cross_sim:.3f}"
            + (f"  错误: {row.error_a or row.error_b}" if row.error_a or row.error_b else ""))

    total_tokens = sum(r.out_tokens_a + r.out_tokens_b for r in rows)

    valid = [r for r in rows if not r.error_a and not r.error_b]
    raw_cross = (sum(r.cross_sim for r in valid) / len(valid)) if valid else 0.0
    # 剔除低熵收敛探测（双侧组内相似度=1：任何模型都给相同答案，无判别力），
    # 再以组内相似度为同源基线归一化：交叉≈组内 → ≈1（与同源一致）；
    # 交叉系统性低于组内 → 低分（行为分布不同，异源证据）。
    ent = [r for r in valid if not (r.intra_a >= 0.999 and r.intra_b >= 0.999)]
    converged = len(valid) - len(ent)
    if ent:
        cross_ent = sum(r.cross_sim for r in ent) / len(ent)
        intra_base = sum((r.intra_a + r.intra_b) / 2 for r in ent) / len(ent)
        delta = cross_ent - intra_base
        score = min(1.0, cross_ent / intra_base) if intra_base > 1e-9 else raw_cross
        ci_low, ci_high = _bootstrap_ci(ent, options.seed)
    else:
        # 全部收敛（如确定性端点/自测 mock）→ 退化为裸交叉
        cross_ent, intra_base, delta = raw_cross, 1.0, raw_cross - 1.0
        score = raw_cross
        ci_low = ci_high = score
    if valid:
        sims = [r.cross_sim for r in valid]
        warn = intra_base < 0.5 and cross_ent < 0.6
        warn_txt = "，警告: 组内基线过低，比值噪声大，本阶段证据弱" if warn else ""
        log(f"[第二阶段] 完成：归一化相似度 {score:.3f}"
            f"（95% CI [{ci_low:.3f}, {ci_high:.3f}]，"
            f"裸交叉 {raw_cross:.3f}，有效交叉 {cross_ent:.3f}，"
            f"组内基线 {intra_base:.3f}，交叉-组内 {delta:+.3f}，"
            f"剔除收敛 {converged}/{len(valid)}；"
            f"min={min(sims):.3f} median={statistics.median(sims):.3f} "
            f"max={max(sims):.3f}{warn_txt}）")
    else:
        log("[第二阶段] 失败：没有成功的探测样本")
    return Phase2Result(score=score, rows=rows, raw_cross=raw_cross,
                        cross_entropic=cross_ent, intra_baseline=intra_base,
                        delta=delta, converged=converged,
                        ci_low=ci_low, ci_high=ci_high,
                        reliability=ci_reliability(ci_low, ci_high),
                        warn_low_baseline=warn if valid else False,
                        total_out_tokens=total_tokens)
