"""第六阶段已降级为诊断性阶段，综合判定由五个阶段构成（加权评分）。

权重设计以"普适性 + 抗单通道失效 + 证据等级"为原则：
  - 不依赖可选参数的行为阶段（二/五）是端点级证据主体：阶段二抗包装、
    样本量大但单条信息量低且高熵下有伪同源风险；阶段五（采样画像）是
    独立采样分布证据——各 0.30 并列主支柱；
  - 阶段一（行为对齐，已剔除低熵回声类别）0.25；阶段四（分词器行为）
    是**家族级证据**（同族不同模型也可能满分），0.10；
  - 阶段三（logprob）可用性最差，仅辅助（0.05）；
  - **第六阶段（单 Token 指纹）只作诊断**：分数不进入判定（推理型端点
    常无可见首 token，得分不可靠），但其"退化现象"（推理型/不可读、
    读取通道统计）保留在报告与 summary 中；
  - **证据加权**：权重 × 成功率（ok/total）再 × CI 证据可靠性
    （reliability，由阶段自带的 bootstrap 置信区间宽度折算），下限
    RELIABILITY_FLOOR，最后归一化——成功率低或 CI 宽（高权重阶段
    尤其如此）的阶段自动降权；
  - **证据冲突规则**：加权分 >0.75 但各阶段得分跨度 >0.45（≥3 个
    有效阶段时）→ 降级"中等怀疑"并标注冲突，强制"多通道须共同吻合"；
  - 某阶段被跳过时权重在剩余阶段间重新归一化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BASE_WEIGHTS = {"phase1": 0.25, "phase2": 0.30, "phase3": 0.05,
                "phase4": 0.10, "phase5": 0.30}
# 证据加权下限：单阶段成功率再低（<0.8 已触发 fail-closed）仍保留至少一半权重
RELIABILITY_FLOOR = 0.5
# 证据冲突：高分但阶段间跨度超过该值 → 判定降级
CONFLICT_SPREAD = 0.45
THRESHOLD_HIGH = 0.75
THRESHOLD_MEDIUM = 0.5


@dataclass
class Verdict:
    score: float
    level: str                      # high / medium / low
    level_cn: str
    summary: str
    weights_used: dict[str, float]
    skipped_phases: list[str]
    fail_closed: bool = False       # True = 证据不足，判定强制降级
    fail_reason: str = ""
    conflict: bool = False          # True = 各阶段证据分歧显著
    phase_spread: float = 0.0       # 有效阶段得分最大跨距


# fail-closed 证据规则（借鉴 llm-verify）：有效探测太少或某阶段成功率过低时，
# 不允许给出干净结论——探测失败/被限流时拒绝判定，而不是把"无证据"当"异源"。
MIN_EVIDENCE_PROBES = 8        # 全部启用阶段的有效探测总数下限
MIN_SUCCESS_RATIO = 0.8        # 单阶段成功率下限


def calibrate(scores: dict[str, Optional[float]],
              self_sim: Optional[float],
              phases=("phase1", "phase4")) \
        -> tuple[dict[str, Optional[float]], Optional[dict]]:
    """官方自比归一化校准。

    self_sim = 官方端点对自身的平均相似度（自比基线）：它度量官方一侧的
    噪声地板（GPU 浮点漂移/多副本负载均衡等），也是"同源可观测相似度"的
    上限。相似度类阶段（一/四）的天花板因此不是 1.0 而是 self_sim：
      校准后得分 = min(1.0, 原始得分 / self_sim)
    自比为 1.0 时校准无效果；未跑或过低（<0.5，官方侧本身极不稳定）时
    跳过校准（fail-soft，不放大噪声）。阶段二已用组内相似度自归一化、
    阶段五为画像相关/幅值结构，不做二次校准；阶段六为诊断性阶段，不参与
    判定，也不校准。

    返回 (校准后的 scores, 校准信息 dict 或 None)；不做原地修改。
    """
    if self_sim is None or self_sim < 0.5:
        return dict(scores), None
    out = dict(scores)
    applied = [k for k in phases if out.get(k) is not None]
    if not applied:
        return out, None
    for k in applied:
        out[k] = min(1.0, out[k] / self_sim)
    return out, {"self_sim": round(self_sim, 4),
                 "factor": round(self_sim, 4), "phases": applied}


def _effective_weights(avail: list[str],
                       evidence: Optional[dict[str, dict]]) -> dict[str, float]:
    """证据加权：base × 成功率 × CI 可靠性（下限 RELIABILITY_FLOOR），再归一化。"""
    raw: dict[str, float] = {}
    for k in avail:
        ev = (evidence or {}).get(k) or {}
        total = ev.get("total") or 0
        ok = ev.get("ok") or 0
        reliability = ev.get("reliability", 1.0)   # 0.5~1.0（阶段自带的 CI 折算）
        factor = (ok / total if total > 0 else 1.0) * max(0.0, min(1.0, reliability))
        factor = max(RELIABILITY_FLOOR, min(1.0, factor))
        raw[k] = BASE_WEIGHTS[k] * factor
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def combine(scores: dict[str, Optional[float]],
            evidence: Optional[dict[str, dict]] = None) -> Verdict:
    """scores: {"phase1": float|None, ...}，None 表示该阶段被跳过。

    evidence: {"phase1": {"ok": n, "total": m}, ...}，用于 fail-closed 检查
    与证据加权；缺失的启用阶段不参与检查。
    """
    avail = [k for k, v in scores.items() if v is not None]
    skipped = [k for k, v in scores.items() if v is None]
    if not avail:
        return Verdict(0.0, "low", "证据不足", "所有阶段均未产生有效数据，无法判定。",
                       {}, skipped)
    weights_used = _effective_weights(avail, evidence)
    score = sum(weights_used[k] * scores[k] for k in avail)

    fail_closed, fail_reason = False, ""
    if evidence:
        totals = {k: v for k, v in evidence.items()
                  if k in avail and (v or {}).get("total", 0) > 0}
        if totals:
            ok_sum = sum(v["ok"] for v in totals.values())
            worst = min(v["ok"] / v["total"] for v in totals.values())
            if ok_sum < MIN_EVIDENCE_PROBES:
                fail_closed, fail_reason = True, \
                    f"有效探测总数 {ok_sum} < {MIN_EVIDENCE_PROBES}"
            elif worst < MIN_SUCCESS_RATIO:
                fail_closed, fail_reason = True, \
                    f"某阶段成功率 {worst:.0%} < {MIN_SUCCESS_RATIO:.0%}"

    if score > THRESHOLD_HIGH:
        level, level_cn = "high", "高度确信同一底层模型"
        summary = ("综合评分 > 0.75：行为指纹高度吻合，统计上无法用巧合解释，"
                   "高度确信两个 API 背后是同一底层模型。")
    elif score >= THRESHOLD_MEDIUM:
        level, level_cn = "medium", "中等怀疑"
        summary = ("综合评分介于 0.5~0.75：存在中等程度怀疑，建议扩大高判别力样本量"
                   "（更多有效对抗探测、单 token 探测与开放画像任务）后重测以收敛结论。")
    else:
        level, level_cn = "low", "证据不足"
        summary = ("综合评分 < 0.5：行为指纹差异明显，大概率不是同一底层模型；"
                   "但请注意，若未知 API 在服务端做了改写/微调，仍可能掩盖同源关系。")

    # 证据冲突规则：有效阶段 ≥3 且得分跨度大 → 即使加权分高也降级，
    # 并在摘要中显式说明分歧（加权平均不应掩盖证据打架）。
    phase_spread = 0.0
    conflict = False
    if len(avail) >= 3:
        vals = [scores[k] for k in avail]
        phase_spread = max(vals) - min(vals)
        conflict = phase_spread > CONFLICT_SPREAD
        if conflict:
            if level == "high":
                level, level_cn = "medium", "中等怀疑"
                summary = (f"证据冲突：加权评分 {score:.3f} 虽 > 0.75，但各阶段"
                           f"得分跨度达 {phase_spread:.2f}（最强 {max(vals):.2f} vs "
                           f"最弱 {min(vals):.2f}），多通道未共同吻合，降级为中等怀疑，"
                           "建议扩大高判别力样本量重测后收敛。")
            else:
                summary = summary + (f"；另注：各阶段得分跨度 {phase_spread:.2f}，"
                                     "证据存在分歧，结论应谨慎解读")

    if fail_closed:
        level, level_cn = "low", "证据不足"
        summary = (f"fail-closed：{fail_reason}，现有证据不足以给出结论。"
                   "请检查网络/限流/端点可用性后重测，或扩大样本量。")
    return Verdict(score=score, level=level, level_cn=level_cn, summary=summary,
                   weights_used=weights_used, skipped_phases=skipped,
                   fail_closed=fail_closed, fail_reason=fail_reason,
                   conflict=conflict, phase_spread=phase_spread)
