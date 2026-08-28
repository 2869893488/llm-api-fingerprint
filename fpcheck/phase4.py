"""第四阶段：分词器行为指纹（不依赖 logprob 的核心证据）。

分词器是模型家族的硬指纹：同一 tokenizer 对同一输入的切分是确定的，
因此在数数/倒序/罕见字符复读/切分这类确定性任务上，同一底层模型的
答案（含其特有的错误方式）高度一致；不同分词器/不同模型几乎必然分叉。
本阶段只用最普通的 chat 参数（temperature=0，n=1），任何 OpenAI 兼容
端点都支持，解决了 logprob 阶段普遍不可用的问题。
主指标为平均相似度（SequenceMatcher ratio），逐字一致率为参考。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .prompts import PHASE4_CATEGORIES, PROMPT_WRAPPERS, wrap_prompt
from .runner import call_pair


@dataclass
class TokenizerProbeResult:
    category: str
    category_cn: str
    prompt: str
    text_a: str
    text_b: str
    wrapper: str = ""           # prompt 包装模板名（借鉴 LLMmap）
    similarity: float = 0.0     # SequenceMatcher ratio（主指标）
    exact: bool = False
    error_a: str = ""
    error_b: str = ""


@dataclass
class Phase4Result:
    score: float            # 平均相似度（主指标，进入综合判定）
    exact_rate: float       # 逐字一致率（参考）
    prompts: list[TokenizerProbeResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.prompts)


def _similarity(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def run_phase4(official, unknown, recorder, options, log=print) -> Phase4Result:
    log(f"[第四阶段] 开始：分词器行为指纹（temperature={options.phase4_temperature}，"
        f"不依赖 logprobs，随机 prompt 包装）...")
    results: list[TokenizerProbeResult] = []
    # 随机包装序列与对抗探测使用同一 seed 派生，保证两次运行完全可复现
    rng = random.Random(options.seed)
    wrappers = [w[0] for w in PROMPT_WRAPPERS]
    for cat_key, cat_cn, prompts in PHASE4_CATEGORIES:
        for idx, prompt in enumerate(prompts):
            probe_id = f"p4_{cat_key}_{idx:02d}"
            wrapper = rng.choice(wrappers)
            wrapped = wrap_prompt(prompt, wrapper)
            ra, rb = call_pair(recorder, official, unknown, phase=4, probe_id=probe_id,
                               prompt=wrapped, temperature=options.phase4_temperature,
                               max_tokens=options.phase4_max_tokens, seed=options.seed,
                               n=1, concurrent=options.concurrent)
            ok_a, ok_b = ra.ok, rb.ok
            text_a = ra.texts[0] if ra.texts else ""
            text_b = rb.texts[0] if rb.texts else ""
            sim = _similarity(text_a, text_b) if ok_a and ok_b else 0.0
            exact = ok_a and ok_b and text_a == text_b
            results.append(TokenizerProbeResult(
                category=cat_key, category_cn=cat_cn, prompt=prompt,
                text_a=text_a, text_b=text_b, wrapper=wrapper,
                similarity=sim, exact=exact,
                error_a=ra.error or "", error_b=rb.error or ""))
            status = "逐字一致" if exact else f"相似度={sim:.3f}"
            if not (ok_a and ok_b):
                status = "调用失败"   # 单侧报错不计入指标，避免误读为"不一致"
            log(f"  [{probe_id}] {cat_cn} 包装={wrapper}: {status}"
                + (f"  错误: {ra.error or rb.error}" if not (ok_a and ok_b) else ""))

    valid = [p for p in results if not p.error_a and not p.error_b]
    score = sum(p.similarity for p in valid) / len(valid) if valid else 0.0
    exact_rate = sum(p.exact for p in valid) / len(valid) if valid else 0.0
    log(f"[第四阶段] 完成：共 {len(results)} 个探测，"
        f"平均相似度 {score:.3f}（主指标），逐字一致率 {exact_rate:.2%}")
    return Phase4Result(score=score, exact_rate=exact_rate, prompts=results)
