"""第六阶段：单 Token 行为指纹（诊断性阶段，分数不参与综合判定）。

思路：对乱码短串/刁钻输入只保留第一个 token（max_tokens=1）。首 token 由
"分词器 + 模型自身的分布"决定——同源端点首 token 分布重叠度接近 1，
不同端点几乎必然分叉。**但推理型端点（思考过程消耗全部输出预算）无
可见首 token，得分不可靠**——因此本阶段的分数不进判定，只作为诊断层：

  - 保留"退化现象"诊断：推理型/不可读/读取通道统计（logprobs 补读、
    加大预算重试、整阶段退化）如实记录到报告与 summary；
  - **识别到推理型端点时立即跳过**（usage 显示 reasoning_tokens ≥
    output_tokens 且 content 为空 → 不再做 64/512 升级重试，直接跳过
    并说明原因）；
  - 平衡档：6 个固定短串 × 4 采样（时间成本与证据完整性折中）。

读取三级兜底（非推理型端点仍可用）：
  1) content 非空 → 直接作为首 token；
  2) content 为空/纯空白 → 经 logprobs 补读真实首 token（端点不支持
     时自动降级）；
  3) 仍不可读 → 加大预算重试（chat 用 2、Responses 用 64），再不读则
     剔除该探测（不给假证据）。
纯空白 token 统一归一为哨兵 SENTINEL。
"""
from __future__ import annotations

import random
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .phase2 import ci_reliability
from .prompts import PHASE6_FIXED_PROBES
from .runner import call_pair

GARBLE_POOL = string.ascii_letters + string.digits + "αβγδεζηθικλμνξοπρστυφχψω"
# 纯空白/空的首 token 归一化哨兵（真实 token 不可能长这样）
SENTINEL = "␣"
# Responses 通道退化重试的生成预算：max_output_tokens 有 16 下限，且推理型
# 模型可能把预算耗在思考上 → 用更大的预算保证能取到可见内容（64 足够，
# 若仍为空说明端点行为极端，用户可自行调大）
RESPONSES_RETRY_TOKENS = 64
# 升级预算：推理型端点（预算全被思考消耗）的"一次性验证级"预算——
# 只探测一次；仍无可见内容 → 整阶段自动跳过（fail-soft），不再逐个探测空耗
RESPONSES_ESCALATE_TOKENS = 512


def build_probes(num_garbled: int, seed: int) -> list[str]:
    """乱码短串（seed 派生，可复现）+ 固定刁钻串。"""
    rng = random.Random(seed)
    garbled = ["".join(rng.choice(GARBLE_POOL) for _ in range(rng.randint(3, 14)))
               for _ in range(num_garbled)]
    return garbled + PHASE6_FIXED_PROBES


def first_tokens_of(result) -> tuple[Optional[list[str]], bool]:
    """提取每个采样的首 token；返回 (tokens, 是否用过 logprobs 兜底)。

    content 为空/纯空白（官方端 max_tokens=1 的常见行为，API 把首 token
    归一化掉了）→ 从 logprobs 首位补读；连 logprobs 都没有 → 返回
    (None, _) 表示该侧首 token 不可读（调用方按"探测无判别力"剔除）。
    """
    texts = getattr(result, "texts", None) or []
    lps = getattr(result, "logprobs", None) or []
    used_lp = False
    tokens: list[str] = []
    for i, text in enumerate(texts):
        tok = text or ""
        if not tok.strip():
            lp = lps[i] if i < len(lps) else None
            if not lp:
                return None, used_lp          # 真首 token 不可读
            tok = (lp[0].get("token") or "") if lp else ""
            used_lp = True
        tokens.append(SENTINEL if not tok.strip() else tok)
    return tokens, used_lp


@dataclass
class OneTokenRow:
    probe: str
    overlap: float = 0.0        # 两端首 token 分布重叠系数（主指标）
    tokens_a: list[str] = field(default_factory=list)
    tokens_b: list[str] = field(default_factory=list)
    used_lp_a: bool = False     # 官方侧是否走了 logprobs 兜底通道
    used_lp_b: bool = False
    error_a: str = ""
    error_b: str = ""


@dataclass
class Phase6Result:
    score: Optional[float] = None   # None = 已跳过
    rows: list[OneTokenRow] = field(default_factory=list)
    ci_low: float = 0.0
    ci_high: float = 0.0
    skipped: str = ""
    reliability: float = 1.0        # 证据可靠性：CI 越宽 → 越低（0.5~1.0，进证据加权）
    lp_fallback: int = 0            # 走 logprobs 兜底补读的探测数
    retried: int = 0                # max_tokens=1 不可读、用更大预算重试的探测数
    unreadable: int = 0             # 首 token 彻底不可读而被剔除的探测数
    degraded: bool = False          # True = 整阶段退化到短前缀指纹（max_tokens>1）
    retry_tokens: int = 0           # 实际使用的重试预算（记录在案）

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.rows if not r.error_a and not r.error_b)


def distribution_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    """两端首 token 频次分布的重叠系数 Σ min(pA(t), pB(t))。

    等价于两个离散分布的"最小概率质量交集"：完全一致 → 1，完全不相交 → 0。
    """
    if not tokens_a or not tokens_b:
        return 0.0
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    na, nb = len(tokens_a), len(tokens_b)
    return sum(min(ca.get(t, 0) / na, cb.get(t, 0) / nb)
               for t in set(ca) | set(cb))


def _bootstrap_ci(rows: list[OneTokenRow], seed: int, n_iter: int = 300,
                  alpha: float = 0.05) -> tuple[float, float]:
    import random as _random
    rng = _random.Random(seed)
    if len(rows) < 3:
        return 0.0, 0.0
    vals = [rows[rng.randrange(len(rows))].overlap for _ in range(n_iter * len(rows))]
    vals.sort()
    scores = []
    for i in range(0, len(vals), len(rows)):
        chunk = vals[i:i + len(rows)]
        scores.append(sum(chunk) / len(chunk))
    scores.sort()
    lo = scores[max(0, int(alpha / 2 * len(scores)) - 1)]
    hi = scores[min(len(scores) - 1, int((1 - alpha / 2) * len(scores)))]
    return lo, hi


def _reasoning_hint(result) -> str:
    """从 usage 提取推理 token 信息（用于跳过原因说明）。"""
    try:
        usage = getattr(result, "usage", None) or {}
        det = usage.get("output_tokens_details") or {}
        rt = int(det.get("reasoning_tokens") or 0)
        ot = int(usage.get("output_tokens") or 0)
        if ot > 0 and rt >= ot:
            return "（输出预算全部被思考消耗，无可见内容）"
        if rt > 0:
            return f"（含 {rt} 个推理 token）"
    except Exception:  # noqa: BLE001
        pass
    return ""


def _reasoning_only(result) -> bool:
    """推理型端点判定：本次输出预算全部被思考消耗（且因此无可见内容）。"""
    try:
        usage = getattr(result, "usage", None) or {}
        det = usage.get("output_tokens_details") or {}
        rt = int(det.get("reasoning_tokens") or 0)
        ot = int(usage.get("output_tokens") or 0)
        return ot > 0 and rt >= ot
    except Exception:  # noqa: BLE001
        return False


def run_phase6(official, unknown, recorder, options, log=print) -> Phase6Result:
    probes = build_probes(options.phase6_num_probes, options.seed)
    n = options.phase6_samples
    style_responses = getattr(official, "api_style", "chat") == "responses"
    # 阶段专用短超时：慢网关/代理下单个探测快速失败，避免 90s×重试烧几分钟
    tmo = float(getattr(options, "phase6_timeout", 0) or 0)
    if tmo > 0:
        official = official.clone_with_timeout(tmo, max_retries=2)
        unknown = unknown.clone_with_timeout(tmo, max_retries=2)
    # Responses 通道硬性要求 max_output_tokens>=16：此时阶段六退化为
    # "短前缀指纹"（首个内容块），语义兼容，两端比较方式不变
    style_note = ("，Responses 通道 max_output_tokens>=16，将以短前缀指纹运行"
                  if style_responses else "")
    log(f"[第六阶段] 开始：单 Token 行为指纹（{len(probes)} 个短串 × "
        f"每侧采样 {n} 次，max_tokens=1, temperature={options.phase6_temperature}"
        f"{style_note}，阶段超时 {tmo if tmo else '继承'}s；"
        "content 为空时先经 logprobs 补读，仍不可读则加大预算重试；"
        "推理型端点无可见内容时自动跳过本阶段）...")
    rows: list[OneTokenRow] = []
    lp_fallback = unreadable = retried = 0
    degraded = False
    retry_tokens = 0

    def fetch(idx: int, probe: str, max_tok: int, want_lp: bool):
        """取一次成对响应并提取首 token，返回 (tokens_a, used_a, tokens_b, used_b, ra, rb)。"""
        probe_id = f"p6_{idx:02d}{'e' if max_tok > 1 else ''}"
        ra, rb = call_pair(recorder, official, unknown, phase=6, probe_id=probe_id,
                           prompt=probe, temperature=options.phase6_temperature,
                           max_tokens=max_tok, seed=options.seed,
                           n=n, logprobs=want_lp, top_logprobs=1,
                           concurrent=options.concurrent)
        ta, ua = first_tokens_of(ra)
        tb, ub = first_tokens_of(rb)
        return ta, ua, tb, ub, ra, rb

    for idx, probe in enumerate(probes):
        probe_id = f"p6_{idx:02d}"
        use_tok = retry_tokens if degraded else options.phase6_max_tokens
        ta, ua, tb, ub, ra, rb = fetch(idx, probe, use_tok, not degraded)
        retry_reason = ""
        if not ra.error and not rb.error and (ta is None or tb is None):
            # 推理型端点短路：usage 显示思考消耗全部输出预算 → 立即跳过本阶段，
            # 不再做 64/512 升级重试（省时间，退化现象本身就是要保留的诊断）
            if style_responses and (_reasoning_only(ra) or _reasoning_only(rb)):
                hint = _reasoning_hint(ra) or _reasoning_hint(rb)
                reason = (f"识别到推理型端点（思考过程消耗了全部输出预算）{hint}；"
                          "单 token 指纹无可见内容、得分不可靠，本阶段跳过"
                          "（诊断性阶段，不影响综合判定）")
                log(f"[第六阶段] 跳过：{reason}")
                return Phase6Result(skipped=reason, rows=rows,
                                    lp_fallback=lp_fallback, retried=retried,
                                    unreadable=unreadable, degraded=False,
                                    retry_tokens=0)
            # 兜底重试：任一侧首 token 不可读 → 双侧一起加大预算重试
            # （保持成对一致比较，避免一侧有 token 一侧空导致的假异源）
            tok = (2 if not style_responses else RESPONSES_RETRY_TOKENS)
            ta2, ua2, tb2, ub2, ra2, rb2 = fetch(idx, probe, tok, True)
            retried += 1
            if ta2 is not None and tb2 is not None:
                ta, ua, tb, ub, ra, rb = ta2, ua2, tb2, ub2, ra2, rb2
                retry_reason = f"（max_tokens=1 不可读，已用 max_tokens={tok} 重试）"
            elif style_responses:
                # Responses/推理型端点"永远返回空"：升级预算探测一次——
                # 有意义就整阶段切换；首探测仍空则整阶段跳过（fail-soft，
                # 不逐个探测空耗，不给假证据）
                ta3, ua3, tb3, ub3, ra3, rb3 = \
                    fetch(idx, probe, RESPONSES_ESCALATE_TOKENS, False)
                retried += 1
                if ta3 is not None and tb3 is not None:
                    ta, ua, tb, ub, ra, rb = ta3, ua3, tb3, ub3, ra3, rb3
                    degraded = True
                    retry_tokens = RESPONSES_ESCALATE_TOKENS
                    retry_reason = (f"（推理型端点内容恒空，整阶段切换到 "
                                    f"max_tokens={retry_tokens} 短前缀指纹）")
                    log(f"  [第六阶段] 升级预算 {retry_tokens} 后可见内容恢复，"
                        "整阶段按短前缀指纹继续……")
                else:
                    hint = _reasoning_hint(ra3) or _reasoning_hint(rb3) \
                        or _reasoning_hint(ra) or _reasoning_hint(rb)
                    if idx == 0:
                        reason = (f"官方/未知端点无可见首 token{hint}，"
                                  "首 token 指纹无法成对比较，本阶段自动跳过"
                                  "（fail-soft，权重重新归一化）")
                        log(f"[第六阶段] 跳过：{reason}")
                        return Phase6Result(skipped=reason, rows=rows,
                                            lp_fallback=lp_fallback,
                                            retried=retried,
                                            unreadable=unreadable,
                                            degraded=False, retry_tokens=0)
                    hint_txt = hint or ""
                    log(f"  [{probe_id}] 跳过（错误: 升级预算后仍无可见内容{hint_txt}）")
                    unreadable += 1
                    rows.append(OneTokenRow(probe=probe, error_a="不可读（升级预算后仍为空）",
                                            error_b="不可读（升级预算后仍为空）",
                                            used_lp_a=ua, used_lp_b=ub))
                    continue
            else:
                # chat 风格：个别探测不可读 → 剔除该探测（不给假证据）
                err_msg = "首 token 不可读：content 为空且无 logprobs，加大预算重试后仍无输出"
                rows.append(OneTokenRow(probe=probe, error_a=err_msg,
                                        error_b=err_msg, used_lp_a=ua, used_lp_b=ub))
                unreadable += 1
                log(f"  [{probe_id}] 跳过（错误: {err_msg}）")
                continue
        row = OneTokenRow(probe=probe,
                          tokens_a=ta or [], tokens_b=tb or [],
                          used_lp_a=ua, used_lp_b=ub,
                          error_a=ra.error or "", error_b=rb.error or "")
        if not row.error_a and not row.error_b:
            row.overlap = distribution_overlap(row.tokens_a, row.tokens_b)
        if ua or ub:
            lp_fallback += 1
        rows.append(row)
        if not (row.error_a or row.error_b):
            tag = " [lp兜底]" if (ua or ub) else ""
            show_a = f"{row.tokens_a[0]!r}{'·' if ua else ''}"
            show_b = f"{row.tokens_b[0]!r}{'·' if ub else ''}"
            log(f"  [{probe_id}] 短串={probe!r:24} 重叠率={row.overlap:.3f}{tag}"
                f"{retry_reason}  官方首token={show_a} 未知首token={show_b}")
        else:
            log(f"  [{probe_id}] 跳过（错误: {row.error_a or row.error_b}）")

    valid = [r for r in rows if not r.error_a and not r.error_b]
    if len(valid) < 2:
        log(f"[第六阶段] 跳过：有效探测不足（{len(valid)} < 2）")
        return Phase6Result(skipped=f"有效探测不足（{len(valid)}<2）", rows=rows,
                            lp_fallback=lp_fallback, retried=retried,
                            unreadable=unreadable, degraded=degraded,
                            retry_tokens=retry_tokens)
    score = sum(r.overlap for r in valid) / len(valid)
    ci_low, ci_high = _bootstrap_ci(valid, options.seed)
    log(f"[第六阶段] 完成：首 token 分布重叠率 {score:.3f}"
        f"（95% CI [{ci_low:.3f}, {ci_high:.3f}]，"
        f"logprobs 补读 {lp_fallback} 个、加大预算重试 {retried} 个"
        f"{'（整阶段退化到 ' + str(retry_tokens) + ' tokens）' if degraded else ''}、"
        f"剔除不可读 {unreadable} 个，"
        f"min={min(r.overlap for r in valid):.3f} "
        f"max={max(r.overlap for r in valid):.3f}）")
    return Phase6Result(score=score, rows=rows, ci_low=ci_low, ci_high=ci_high,
                        reliability=ci_reliability(ci_low, ci_high),
                        lp_fallback=lp_fallback, retried=retried,
                        unreadable=unreadable, degraded=degraded,
                        retry_tokens=retry_tokens)