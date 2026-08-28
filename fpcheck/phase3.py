"""第三阶段：logprob 数值对齐（最强证据形式）。

同一底层模型即使被外包一层提示工程，其 logprob 数值仍会高度吻合；
不同模型给出相同 top-k 候选分布的概率趋近于零。
未知 API 不支持 logprobs 参数时优雅跳过。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .prompts import PHASE3_PROMPTS
from .runner import call_pair


@dataclass
class PromptLogprob:
    prompt: str
    positions: int
    overlap: float          # top-k 候选集合重叠率（按位置平均）
    top1_rate: float        # top-1 token 一致率
    text_a: str
    text_b: str


@dataclass
class Phase3Result:
    score: Optional[float] = None   # None = 已跳过
    top1_rate: Optional[float] = None
    prompts: list[PromptLogprob] = field(default_factory=list)
    skipped: str = ""


def _position_overlap(tokens_a: list[dict], tokens_b: list[dict], top_k: int):
    """逐位置比较 top-k 候选集合，返回 (平均重叠率, top-1 一致率)；无法比较返回 None。"""
    if not tokens_a or not tokens_b:
        return None
    n = min(len(tokens_a), len(tokens_b))
    hits, top1s, count = 0.0, 0, 0
    for i in range(n):
        ca = tokens_a[i].get("top_logprobs") or [tokens_a[i]]
        cb = tokens_b[i].get("top_logprobs") or [tokens_b[i]]
        sa = {t["token"] for t in ca[:top_k]}
        sb = {t["token"] for t in cb[:top_k]}
        denom = max(len(sa), len(sb))
        if denom == 0:
            continue
        hits += len(sa & sb) / denom
        top1s += 1 if tokens_a[i]["token"] == tokens_b[i]["token"] else 0
        count += 1
    if count == 0:
        return None
    return hits / count, top1s / count


def run_phase3(official, unknown, recorder, options, log=print) -> Phase3Result:
    top_k = options.phase3_top_logprobs
    log(f"[第三阶段] 开始：logprob 数值对齐（top-{top_k} 候选重叠率）...")

    # 先用第一个 prompt 探测 logprobs 支持情况，避免整阶段做无用功
    probe_prompt = PHASE3_PROMPTS[0]
    ra, rb = call_pair(recorder, official, unknown, phase=3, probe_id="p3_probe",
                       prompt=probe_prompt, temperature=0.0,
                       max_tokens=8, seed=options.seed, n=1,
                       logprobs=True, top_logprobs=top_k, concurrent=options.concurrent)
    if not ra.ok or not ra.logprobs or not ra.logprobs[0]:
        return Phase3Result(skipped=f"官方 API 未返回 logprob 数据"
                                   f"（{ra.error or 'logprobs 为空'}）")
    if not rb.ok or not rb.logprobs or not rb.logprobs[0]:
        if unknown.capabilities.get("logprobs") is False:
            reason = "未知 API 拒绝了 logprobs 参数（已自动降级重试，本阶段无法比较）"
        else:
            reason = rb.error or "未知 API 未返回 logprob 数据（可能静默忽略了 logprobs 参数）"
        return Phase3Result(skipped=f"未知 API 不支持 logprobs，本阶段跳过：{reason}")
    log("  [p3_probe] 两个 API 均支持 logprobs，继续...")

    prompts_out: list[PromptLogprob] = []
    for idx, prompt in enumerate(PHASE3_PROMPTS):
        probe_id = f"p3_{idx:02d}"
        ra, rb = call_pair(recorder, official, unknown, phase=3, probe_id=probe_id,
                           prompt=prompt, temperature=0.0,
                           max_tokens=options.phase3_max_tokens, seed=options.seed, n=1,
                           logprobs=True, top_logprobs=top_k, concurrent=options.concurrent)
        if not (ra.ok and rb.ok and ra.logprobs and rb.logprobs
                and ra.logprobs[0] and rb.logprobs[0]):
            log(f"  [{probe_id}] 跳过（错误: {ra.error or rb.error or 'logprobs 为空'}）")
            continue
        pos = _position_overlap(ra.logprobs[0], rb.logprobs[0], top_k)
        if pos is None:
            log(f"  [{probe_id}] 跳过（无重叠位置）")
            continue
        overlap, top1 = pos
        prompts_out.append(PromptLogprob(
            prompt=prompt, positions=len(ra.logprobs[0]),
            overlap=overlap, top1_rate=top1,
            text_a=ra.texts[0] if ra.texts else "",
            text_b=rb.texts[0] if rb.texts else ""))
        log(f"  [{probe_id}] 位置数={len(ra.logprobs[0])}  "
            f"top-{top_k} 重叠率={overlap:.3f}  top-1 一致率={top1:.3f}")

    if not prompts_out:
        return Phase3Result(skipped="所有 logprob 探测均失败")
    score = sum(p.overlap for p in prompts_out) / len(prompts_out)
    top1_rate = sum(p.top1_rate for p in prompts_out) / len(prompts_out)
    log(f"[第三阶段] 完成：top-{top_k} 候选重叠率 {score:.3f}，"
        f"top-1 一致率 {top1_rate:.3f}")
    return Phase3Result(score=score, top1_rate=top1_rate, prompts=prompts_out)
