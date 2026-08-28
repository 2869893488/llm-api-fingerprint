"""第五阶段：采样自洽性画像（借鉴 SelfCheckGPT 的 sample-and-compare 思想）。

SelfCheckGPT（Manakul et al., ACL 2023 / arXiv 2303.08896）用"对同一问题
多次采样、再比较采样间一致性"做黑盒幻觉检测；本阶段把同一操作反转为指纹：

  对一组开放任务 prompt，分别从两个 API 采样 n 次，得到每个 API 在每个
  prompt 上的"自洽度"（组内平均相似度）画像向量。

同一底层模型（同一采样分布）→ 两端画像高度相关（Spearman）且幅值相近；
不同模型 / 服务端改写 → 画像分叉。不依赖 logprobs，任何 OpenAI 兼容端点
都支持，与第二阶段互补：第二阶段测低熵对抗输入上的自洽度（多为组内基线
≈1），这里测高熵开放任务上的自洽度（每模型有各自的采样方差特征）。

退化处理（fail-soft）：端点完全确定性（画像恒为 1、无采样方差）时本阶段
无判别力，自动跳过、不参与判定——与 logprob 不可用时跳过第三阶段的语义
一致，避免把"无采样信息"误判为"异源证据"。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .phase2 import ci_reliability, intra_similarity
from .prompts import PHASE5_PROMPTS
from .runner import call_pair


@dataclass
class SelfAgreementRow:
    prompt: str
    agreement_a: float = 0.0    # 官方 API 在该 prompt 上的组内自洽度
    agreement_b: float = 0.0    # 未知 API 在该 prompt 上的组内自洽度
    texts_a: list[str] = field(default_factory=list)
    texts_b: list[str] = field(default_factory=list)
    error_a: str = ""
    error_b: str = ""


@dataclass
class Phase5Result:
    score: Optional[float] = None   # None = 已跳过
    amp: float = 0.0                # 幅值相似度 1-|ma-mb|/max(ma,mb)
    corr: Optional[float] = None    # 画像 Spearman 相关（截断到 [0,1]）
    rows: list[SelfAgreementRow] = field(default_factory=list)
    ci_low: float = 0.0             # score 的 95% 置信区间（bootstrap）
    ci_high: float = 0.0
    skipped: str = ""
    reliability: float = 1.0        # 证据可靠性：CI 越宽 → 越低（0.5~1.0，进证据加权）
    mean_a: float = 0.0             # 官方平均自洽度（参考）
    mean_b: float = 0.0             # 未知平均自洽度（参考）

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.rows if not r.error_a and not r.error_b)


def _rank(values: list[float]) -> list[float]:
    """平均秩（处理并列）；输入非空。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: list[float], b: list[float]) -> Optional[float]:
    """秩相关（Spearman）。任一向量为常向量或样本太少时返回 None。"""
    if len(a) < 3 or len(b) < 3:
        return None
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra)
    db = sum((y - mb) ** 2 for y in rb)
    if da == 0 or db == 0:
        return None
    return num / (da * db) ** 0.5


def _amplitude_sim(ma: float, mb: float) -> float:
    """画像幅值相似度：两端平均自洽度越接近 → 越接近 1。"""
    if ma <= 0 and mb <= 0:
        return 1.0
    return 1.0 - abs(ma - mb) / max(ma, mb)


def _corr_component(corr: Optional[float], ag_a: list[float],
                    ag_b: list[float]) -> float:
    """把 Spearman 相关折成 [0,1] 分量；常向量等不可比情形按保守规则处理。"""
    if corr is not None:
        return max(0.0, min(1.0, corr))
    if len(set(ag_a)) < 2 and len(set(ag_b)) < 2:
        # 双侧都无方差：画像无内部结构 → 直接比幅值（相等=吻合）
        return 1.0 if ag_a[0] == ag_b[0] else 0.0
    # 单侧无方差（一端确定性、一端有采样方差）→ 采样分布显著不同
    return 0.0


def _score_of(ag_a: list[float], ag_b: list[float]) -> tuple[float, float, Optional[float]]:
    ma = sum(ag_a) / len(ag_a)
    mb = sum(ag_b) / len(ag_b)
    amp = _amplitude_sim(ma, mb)
    corr = _spearman(ag_a, ag_b)
    return 0.5 * amp + 0.5 * _corr_component(corr, ag_a, ag_b), amp, corr


def _bootstrap_ci(rows: list[SelfAgreementRow], seed: int, n_iter: int = 300,
                  alpha: float = 0.05) -> tuple[float, float]:
    """对 prompt 做 bootstrap 重采样，返回 score 的 95% 置信区间。"""
    rng = random.Random(seed)
    n = len(rows)
    if n < 3:
        return 0.0, 0.0
    vals: list[float] = []
    for _ in range(n_iter):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        aa = [r.agreement_a for r in sample]
        bb = [r.agreement_b for r in sample]
        vals.append(_score_of(aa, bb)[0])
    vals.sort()
    lo = vals[max(0, int(alpha / 2 * n_iter) - 1)]
    hi = vals[min(n_iter - 1, int((1 - alpha / 2) * n_iter))]
    return lo, hi


def run_phase5(official, unknown, recorder, options, log=print) -> Phase5Result:
    n = options.phase5_samples
    log(f"[第五阶段] 开始：采样自洽性画像（{len(PHASE5_PROMPTS)} 个开放任务 × "
        f"每侧采样 {n} 次，temperature={options.phase5_temperature}）...")
    rows: list[SelfAgreementRow] = []
    for idx, prompt in enumerate(PHASE5_PROMPTS):
        probe_id = f"p5_{idx:02d}"
        ra, rb = call_pair(recorder, official, unknown, phase=5, probe_id=probe_id,
                           prompt=prompt, temperature=options.phase5_temperature,
                           max_tokens=options.phase5_max_tokens, seed=None,
                           n=n, concurrent=options.concurrent)
        err_a = ra.error or ("" if len(ra.texts) >= 2 else "采样数不足（n 被降级为 1）")
        err_b = rb.error or ("" if len(rb.texts) >= 2 else "采样数不足（n 被降级为 1）")
        row = SelfAgreementRow(
            prompt=prompt,
            agreement_a=intra_similarity(ra.texts) if not err_a else 0.0,
            agreement_b=intra_similarity(rb.texts) if not err_b else 0.0,
            texts_a=ra.texts, texts_b=rb.texts, error_a=err_a, error_b=err_b)
        rows.append(row)
        if not (err_a or err_b):
            log(f"  [{probe_id}] 官方自洽度={row.agreement_a:.3f}  "
                f"未知自洽度={row.agreement_b:.3f}")
        else:
            log(f"  [{probe_id}] 跳过（错误: {err_a or err_b}）")

    valid = [r for r in rows if not r.error_a and not r.error_b]
    if len(valid) < 2:
        log(f"[第五阶段] 跳过：有效开放任务不足（{len(valid)} < 2）")
        return Phase5Result(skipped=f"有效开放任务不足（{len(valid)}<2）", rows=rows)
    ag_a = [r.agreement_a for r in valid]
    ag_b = [r.agreement_b for r in valid]
    if all(x >= 0.999 for x in ag_a) and all(x >= 0.999 for x in ag_b):
        log("[第五阶段] 跳过：两端采样零方差（确定性端点/缓存），画像无判别力")
        return Phase5Result(skipped="两端采样零方差（确定性端点/缓存），画像无判别力",
                            rows=rows)

    score, amp, corr = _score_of(ag_a, ag_b)
    ci_low, ci_high = _bootstrap_ci(valid, options.seed)
    ma = sum(ag_a) / len(ag_a)
    mb = sum(ag_b) / len(ag_b)
    corr_txt = "N/A" if corr is None else f"{corr:.3f}"
    log(f"[第五阶段] 完成：画像相关(Spearman)={corr_txt}  幅值相似度={amp:.3f}  "
        f"综合评分={score:.3f}（95% CI [{ci_low:.3f}, {ci_high:.3f}]；"
        f"官方平均自洽度 {ma:.3f}，未知 {mb:.3f}）")
    return Phase5Result(score=score, amp=amp, corr=corr, rows=rows,
                        ci_low=ci_low, ci_high=ci_high,
                        reliability=ci_reliability(ci_low, ci_high),
                        mean_a=ma, mean_b=mb)