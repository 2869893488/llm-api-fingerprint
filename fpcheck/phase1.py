"""第一阶段：基础行为对齐。

对 9 类"性格测试" prompt 在配置温度（默认 temperature=0）下调用两个 API。
主指标为平均相似度（SequenceMatcher ratio）：服务端即使同源也常因
GPU 浮点非确定性/多副本负载均衡产生微小漂移，逐字比较对此过于敏感，
相似度评分能区分"微小漂移"与"系统性差异"。

评分对象**只取自由度类别**（自我认知/风格/系统提示泄漏/数值推理/拒答模式
——不同模型的回答必然分叉）；"低熵回声类别"（固定复读/分词泄漏/严格格式
遵循——任何模型都可能逐字一致，无判别力，可能拉高平均分）仅作参考展示、
不进入评分。
注：部分推理模型不支持自定义 temperature，会自动降级为不传该参数。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .prompts import PHASE1_CATEGORIES, PHASE1_REFERENCE_CATEGORIES
from .runner import call_pair


@dataclass
class PromptResult:
    category: str
    category_cn: str
    prompt: str
    text_a: str
    text_b: str
    similarity: float       # SequenceMatcher ratio（主指标）
    exact: bool
    normalized: bool
    latency_a: float = 0.0  # 官方 API 本次调用延迟（ms，红旗检测用）
    latency_b: float = 0.0  # 未知 API 本次调用延迟（ms）
    tokens_a: int = 0       # 官方输出 completion tokens（延迟按长度归一化用）
    tokens_b: int = 0
    error_a: str = ""
    error_b: str = ""


@dataclass
class Phase1Result:
    score: float            # 平均相似度（仅自由度类别，进入综合判定）
    exact_rate: float       # 逐字一致率（自由度类别，参考）
    normalized_rate: float  # 忽略大小写/空白后的一致率（自由度类别，参考）
    prompts: list[PromptResult] = field(default_factory=list)
    reference_rows: list[PromptResult] = field(default_factory=list)  # 低熵回声类别

    @property
    def total(self) -> int:
        return len(self.prompts) + len(self.reference_rows)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _similarity(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def run_phase1(official, unknown, recorder, options, log=print) -> Phase1Result:
    log(f"[第一阶段] 开始：基础行为对齐（temperature={options.phase1_temperature}，"
        f"相似度评分；低熵回声类别仅参考不评分）...")
    scored: list[PromptResult] = []
    reference: list[PromptResult] = []
    for cat_key, cat_cn, prompts in PHASE1_CATEGORIES:
        is_ref = cat_key in PHASE1_REFERENCE_CATEGORIES
        for idx, prompt in enumerate(prompts):
            probe_id = f"p1_{cat_key}_{idx:02d}"
            ra, rb = call_pair(recorder, official, unknown, phase=1, probe_id=probe_id,
                               prompt=prompt, temperature=options.phase1_temperature,
                               max_tokens=options.phase1_max_tokens, seed=options.seed,
                               n=1, concurrent=options.concurrent)
            ok_a, ok_b = ra.ok, rb.ok
            text_a = ra.texts[0] if ra.texts else ""
            text_b = rb.texts[0] if rb.texts else ""
            sim = _similarity(text_a, text_b) if ok_a and ok_b else 0.0
            exact = ok_a and ok_b and text_a == text_b
            normalized = ok_a and ok_b and _normalize(text_a) == _normalize(text_b)
            tokens_a = _usage_tokens(ra)
            tokens_b = _usage_tokens(rb)
            row = PromptResult(
                category=cat_key, category_cn=cat_cn, prompt=prompt,
                text_a=text_a, text_b=text_b,
                similarity=sim, exact=exact, normalized=normalized,
                latency_a=ra.latency_ms, latency_b=rb.latency_ms,
                tokens_a=tokens_a, tokens_b=tokens_b,
                error_a=ra.error or "", error_b=rb.error or "")
            (reference if is_ref else scored).append(row)
            status = "逐字一致" if exact else ("归一化一致" if normalized
                    else f"相似度={sim:.3f}")
            if not (ok_a and ok_b):
                status = "调用失败"   # 单侧报错不计入指标，避免误读为"不一致"
            tag = "参考" if is_ref else "评分"
            log(f"  [{probe_id}] [{tag}] {cat_cn}: {status}"
                + (f"  错误: {ra.error or rb.error}" if not (ok_a and ok_b) else ""))

    valid = [p for p in scored if not p.error_a and not p.error_b]
    score = sum(p.similarity for p in valid) / len(valid) if valid else 0.0
    exact_rate = sum(p.exact for p in valid) / len(valid) if valid else 0.0
    norm_rate = sum(p.normalized for p in valid) / len(valid) if valid else 0.0
    log(f"[第一阶段] 完成：共 {len(scored) + len(reference)} 个 prompt"
        f"（评分 {len(scored)} + 低熵参考 {len(reference)}），"
        f"平均相似度 {score:.3f}（主指标，不含回声类别），"
        f"逐字一致率 {exact_rate:.2%}，归一化一致率 {norm_rate:.2%}")
    return Phase1Result(score=score, exact_rate=exact_rate,
                        normalized_rate=norm_rate, prompts=scored,
                        reference_rows=reference)


def _usage_tokens(result) -> int:
    """从响应 usage 取 completion tokens；取不到返回 0。"""
    usage = getattr(result, "usage", None) or {}
    return int(usage.get("completion_tokens") or 0)
