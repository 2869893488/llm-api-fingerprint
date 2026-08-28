"""身份一致性红旗检测（借鉴 mintesnot-teshome/llm-verify 的思路）。

零额外请求：直接复用第一阶段或第二阶段的取证数据（自我认知类 prompt 的回答 + 延迟 + token usage），
检测八类红旗：
  1. 知识截止声明矛盾 —— 同一 API 在多个回答中给出不同的截止声明
     （语境限定提取：只认"截止/训练数据"措辞附近的年份，过滤无关年份；
       另做"同一品牌下的跨端系统性偏差"检查——不同模型的截止日期本就
       不同，因此绝不拿"两端年份相同"当同源证据）
  2. 自我身份声称与请求型号品牌不符 —— 请求 gpt-4o 却自称 Claude 等
  3. 延迟显著不对称 —— 一端平均延迟远高于另一端，提示存在转发/中继层
  4. 跨端自称品牌不一致 —— 官方自称品牌与未知自称品牌完全不同
     （同一底层模型应自称一致；此红旗同时覆盖"伪装层改写自称"的情形）
  5. 拒答模式不一致 —— 官方与未知在"拒答模式类"探测上的 refuse/comply
     行为不一致（思路参考模型替代审计工作：自我声称不可靠，
     行为画像（含拒答/护栏行为）是判别证据；不同护栏层会改变拒答模式）
  6. 代理/中继自我披露 —— 出现 proxy/relay 等词
  7. 失败率不对称 —— 未知 API 阶段二失败率明显偏高
  8. Token 用量不对称 —— prompt_tokens 比值 >1.3（疑似隐藏 system prompt 注入）、
     completion_tokens 比值 >1.5（疑似附加输出/水印）、或未知端存在 reasoning_tokens（推理型模型替换）

红旗是"辅助信号"而非证据：同源结论成立时红旗不影响判定，但会在报告中
明确标注，提醒用户复核。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

# 品牌强词：出现在自我认知回答中但不符合请求型号品牌时触发红旗。
# 故意不含 "gpt/openai/chatgpt" —— 其他模型也常提及这些词，判别力低。
BRAND_WORDS: dict[str, list[str]] = {
    "anthropic": ["claude", "anthropic"],
    "google": ["gemini", "google"],
    "meta": ["llama"],
    "deepseek": ["deepseek"],
    "qwen": ["qwen"],
    "mistral": ["mistral"],
    "other": ["文心", "豆包", "通义", "讯飞", "智谱", "kimi", "moonshot",
              "glm", "baichuan", "minimax", "yi-", "coze"],
}

# 期望品牌推断：model_name 命中这些词 → 期望该品牌
_EXPECT_BY_KEYWORD: list[tuple[str, str]] = [
    ("gpt", "openai"), ("chatgpt", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("llama", "meta"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("mistral", "mistral"),
]

# 年份提取（知识截止声明）。
# 设计要点（不同模型的截止日期本就不同，因此只做两种检查：
#   - 单端内部矛盾：同一 API 跨回答给出 ≥2 个不同声明 → 红旗；
#   - 跨端系统性偏差：两端各自单一声明但年份不同、且声称同一品牌 → 红旗；
#   绝不做"两端年份必须相同"的低判别力比较。
# 提取规则：
#   - 只统计"截止/训练数据"语境窗口内的年份，过滤"成立于 2015 年"等无关年份；
#   - 每个回答只取一个声明（语境年份的最大值），"从 2019 到 2023"这类
#     区间句不会把中间年份误判为矛盾；
#   - 跨回答出现 ≥2 个不同声明才触发红旗——同一模型只有一条截止元数据。
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
# STRONG：明确的截止/边界词；WEAK：训练/知识描述词（兜底）
_YEAR_CTX_STRONG = re.compile(r"cut\s*-?\s*off|截至|截止", re.IGNORECASE)
_YEAR_CTX_WEAK = re.compile(r"trained|training|knowledge|as of|up to|"
                            r"训练数据|知识|数据", re.IGNORECASE)
_CTX_BEFORE, _CTX_AFTER = 45, 25


def _year_context(text: str, pos: int) -> str:
    """年份所在语境窗口（前 45 字符/后 25 字符）命中的关键词等级。

    等级越强越像"截止声明"：strong = 明确的截止/边界措辞；
    weak = 训练/知识描述措辞（可能是"被训练于 2021 年"这类非边界信息）。
    """
    window = text[max(0, pos - _CTX_BEFORE): pos + _CTX_AFTER]
    if _YEAR_CTX_STRONG.search(window):
        return "strong"
    if _YEAR_CTX_WEAK.search(window):
        return "weak"
    return ""


def _extract_cutoff_claims(texts: list[str]) -> set[str]:
    """提取每个回答的"知识截止声明"，返回跨回答的声明集合。

    每个回答优先取 strong 语境年份的最大值（无 strong 时取 weak 的），
    只贡献一个声明——既过滤无关键语的年份，又避免区间句产生假矛盾。
    """
    claims: set[str] = set()
    for t in texts:
        strong: list[str] = []
        weak: list[str] = []
        for m in _YEAR_RE.finditer(t):
            ctx = _year_context(t, m.start())
            if ctx == "strong":
                strong.append(m.group())
            elif ctx == "weak":
                weak.append(m.group())
        pool = strong or weak
        if pool:
            claims.add(max(pool))
    return claims


@dataclass
class RedFlagsResult:
    flags: list[str] = field(default_factory=list)   # 红旗描述（空 = 无红旗）
    cutoffs_a: set[str] = field(default_factory=set)
    cutoffs_b: set[str] = field(default_factory=set)
    identity_mismatch: bool = False     # 自称品牌与请求型号品牌不符
    latency_ratio: float = 0.0          # 未知端平均延迟 / 官方端平均延迟
    self_claims_a: list[str] = field(default_factory=list)
    self_claims_b: list[str] = field(default_factory=list)
    identity_votes_a: dict[str, int] = field(default_factory=dict)  # 官方自称品牌票数
    identity_votes_b: dict[str, int] = field(default_factory=dict)  # 未知自称品牌票数
    refusal_match_rate: float | None = None   # 拒答模式一致性（0~1；None=无数据）
    prompt_tokens_ratio: float = 0.0          # 未知端 prompt_tokens / 官方端 prompt_tokens
    completion_tokens_ratio: float = 0.0      # 未知端 completion_tokens / 官方端 completion_tokens
    has_reasoning_asymmetry: bool = False     # 存在 reasoning_tokens 而官方没有

    @property
    def has_flags(self) -> bool:
        return bool(self.flags)


def _detect_brand(text: str) -> str:
    low = text.lower()
    for brand, words in BRAND_WORDS.items():
        for w in words:
            if w in low:
                return brand
    return ""


def _expected_brand(model_name: str) -> str:
    low = model_name.lower()
    for kw, brand in _EXPECT_BY_KEYWORD:
        if kw in low:
            return brand
    return ""


# 拒答/回避标记：出现任一标记即视为 refuse（其余非空回答视为 comply）。
# 刻意选择跨语言、跨模型常见表述；低判别力词（如 "not"）不收录。
REFUSE_MARKERS: list[str] = [
    "cannot", "can't", "sorry", "unable", "not able", "won't", "will not",
    "violat", "policy", "guideline", "safety", "ethical", "inappropriate",
    "harmful", "refuse", "拒绝", "无法", "不能", "对不起", "抱歉", "违反",
    "安全", "不提供", "不能提供", "帮不了", "做不到", "没法", "不讲", "不回答",
]

# 代理/中继自我披露词（llm-verify 的 relay provenance 思路）：
# 自我描述或系统提示泄漏回答中出现这些词 → 提示存在转发/网关层
PROXY_WORDS: list[str] = [
    "proxy", "relay", "gateway", "reverse proxy", "vllm", "litellm",
    "one-api", "new-api", "中转", "代理", "转发", "套壳", "api网关",
]

# 跨端自称票的附加品牌词（与 BRAND_WORDS 不同：这里收录 openai 系自称词，用于
# 比较"官方 vs 未知"的自称一致性；不用于"自称 vs 请求型号"的判别，那类判别低）
_OPENAI_VOTE_WORDS = ("openai", "chatgpt", "gpt-4", "gpt-4o", "gpt-3.5")


def _refusal_class(text: str) -> str:
    """把拒答模式类回答归为 refuse / comply / ""（空回答）。"""
    low = text.lower()
    if not low:
        return ""
    return "refuse" if any(m in low for m in REFUSE_MARKERS) else "comply"


def _vote_brands(claims: list[str]) -> dict[str, int]:
    """统计自称品牌票数（跨端一致性比较用）。"""
    votes: dict[str, int] = {}
    for c in claims:
        b = _detect_brand(c)
        if not b:
            low = c.lower()
            if any(w in low for w in _OPENAI_VOTE_WORDS):
                b = "openai"
        if b:
            votes[b] = votes.get(b, 0) + 1
    return votes


def _extract_usage_dict(val: Any) -> dict:
    """将 usage 对象（dict / Pydantic 模型 / SimpleNamespace）提取为标准 dict。"""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if hasattr(val, "model_dump"):
        try:
            return val.model_dump()
        except Exception:
            pass
    if hasattr(val, "__dict__"):
        return vars(val)
    return {}


def _get_prompt_tokens(usage: Any) -> int:
    """提取 prompt tokens（支持 prompt_tokens 与 input_tokens）。"""
    d = _extract_usage_dict(usage)
    val = d.get("prompt_tokens")
    if val is None:
        val = d.get("input_tokens")
    try:
        return int(val or 0)
    except (ValueError, TypeError):
        return 0


def _get_completion_tokens(usage: Any) -> int:
    """提取 completion tokens（支持 completion_tokens 与 output_tokens）。"""
    d = _extract_usage_dict(usage)
    val = d.get("completion_tokens")
    if val is None:
        val = d.get("output_tokens")
    try:
        return int(val or 0)
    except (ValueError, TypeError):
        return 0


def _get_reasoning_tokens(usage: Any) -> int:
    """提取 reasoning tokens（支持顶层与 details 内嵌字段）。"""
    d = _extract_usage_dict(usage)
    for k in ("reasoning_tokens", "reasoning_token_count"):
        if k in d and d[k] is not None:
            try:
                v = int(d[k])
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
    for det_key in ("completion_tokens_details", "output_tokens_details", "prompt_tokens_details"):
        det = _extract_usage_dict(d.get(det_key))
        for k in ("reasoning_tokens", "reasoning_token_count"):
            if k in det and det[k] is not None:
                try:
                    v = int(det[k])
                    if v > 0:
                        return v
                except (ValueError, TypeError):
                    pass
    return 0


def check_token_usage_asymmetry(
    official_usages: list[Any],
    unknown_usages: list[Any],
    prompt_ratio_threshold: float = 1.3,
    completion_ratio_threshold: float = 1.5,
) -> list[str]:
    """
    比较两端 usage 字段：
    - prompt_tokens 比值 > 1.3 → 疑似注入隐藏 system prompt
    - completion_tokens 比值 > 1.5 → 疑似附加输出/水印
    - 存在 reasoning_tokens 而官方没有 → 推理型模型替换

    参数:
        official_usages: 官方端 usage 记录列表
        unknown_usages: 未知端 usage 记录列表
        prompt_ratio_threshold: prompt_tokens 比值阈值（默认 1.3）
        completion_ratio_threshold: completion_tokens 比值阈值（默认 1.5）

    返回:
        触发的红旗描述列表（无异常返回空列表 []）
    """
    flags: list[str] = []
    if not official_usages or not unknown_usages:
        return flags

    valid_pairs = []
    for ua, ub in zip(official_usages, unknown_usages):
        if ua is None and ub is None:
            continue
        valid_pairs.append((ua or {}, ub or {}))

    if not valid_pairs:
        return flags

    total_pa = sum(_get_prompt_tokens(ua) for ua, _ in valid_pairs)
    total_pb = sum(_get_prompt_tokens(ub) for _, ub in valid_pairs)

    total_ca = sum(_get_completion_tokens(ua) for ua, _ in valid_pairs)
    total_cb = sum(_get_completion_tokens(ub) for _, ub in valid_pairs)

    total_ra = sum(_get_reasoning_tokens(ua) for ua, _ in valid_pairs)
    total_rb = sum(_get_reasoning_tokens(ub) for _, ub in valid_pairs)

    # 1) prompt_tokens 检查（疑似注入隐藏 system prompt）
    if total_pa > 0:
        p_ratio = total_pb / total_pa
        if p_ratio > prompt_ratio_threshold:
            flags.append(
                f"未知 API prompt_tokens 消耗显著高于官方（合计 {total_pb} vs {total_pa}，"
                f"比值 {p_ratio:.2f} > {prompt_ratio_threshold}），疑似服务端注入隐藏 system prompt"
            )

    # 2) completion_tokens 检查（疑似附加输出/水印）
    if total_ca > 0:
        c_ratio = total_cb / total_ca
        if c_ratio > completion_ratio_threshold:
            flags.append(
                f"未知 API completion_tokens 消耗显著高于官方（合计 {total_cb} vs {total_ca}，"
                f"比值 {c_ratio:.2f} > {completion_ratio_threshold}），疑似附加输出/水印或存在包装改写"
            )

    # 3) reasoning_tokens 检查（存在 reasoning_tokens 而官方没有 → 推理型模型替换）
    if total_rb > 0 and total_ra == 0:
        flags.append(
            f"未知 API 响应 usage 中检测到 reasoning_tokens（合计 {total_rb} 个）而官方端为 0，"
            f"疑似使用推理型模型（如 o1/DeepSeek-R1 等）进行替换"
        )

    return flags


def detect_redflags(phase1, official_model: str, unknown_model: str,
                    phase2=None,
                    latency_ratio_threshold: float = 3.0,
                    prompt_ratio_threshold: float = 1.3,
                    completion_ratio_threshold: float = 1.5) -> RedFlagsResult:
    """从第一阶段和第二阶段结果检测红旗。phase1 为 None（禁用）时返回空结果。
    phase2（可选）：提供对抗探测失败率及 usage 数据。"""
    res = RedFlagsResult()
    if phase1 is None or not getattr(phase1, "prompts", None):
        return res

    self_a = [p.text_a for p in phase1.prompts if p.category == "self_awareness"
              and not p.error_a and p.text_a]
    self_b = [p.text_b for p in phase1.prompts if p.category == "self_awareness"
              and not p.error_b and p.text_b]
    res.self_claims_a = self_a
    res.self_claims_b = self_b

    # 1) 知识截止声明矛盾（按 API 分别检测；语境限定提取，见 _extract_cutoff_claims）
    res.cutoffs_a = _extract_cutoff_claims(self_a)
    res.cutoffs_b = _extract_cutoff_claims(self_b)
    if len(res.cutoffs_a) >= 2:
        res.flags.append("官方 API 自我认知回答中检测到多个截止声明 "
                         f"({', '.join(sorted(res.cutoffs_a))})，"
                         "同一模型不应有多个知识截止日期")
    if len(res.cutoffs_b) >= 2:
        res.flags.append("未知 API 自我认知回答中检测到多个截止声明 "
                         f"({', '.join(sorted(res.cutoffs_b))})，"
                         "前后矛盾，疑似包装/转发层")

    # 1b) 跨端系统性偏差：两端各自只有一个稳定声明、年份却不同，
    #     且两端请求型号属于同一品牌 → 不是同一模型版本/快照。
    #     不比较"年份相同"（不同模型本就不同，无判别力），只抓系统性偏差。
    if len(res.cutoffs_a) == 1 and len(res.cutoffs_b) == 1:
        ca, cb = next(iter(res.cutoffs_a)), next(iter(res.cutoffs_b))
        ea, eb = _expected_brand(official_model), _expected_brand(unknown_model)
        if ca != cb and ea and ea == eb:
            res.flags.append(
                f"官方与未知均自称品牌 {ea} 且截止声明各自稳定，但年份系统性不同"
                f"（官方 {ca} vs 未知 {cb}）：两者并非同一模型版本/快照")

    # 2) 自我身份声称与请求型号品牌不符
    for label, model, claims in (("官方", official_model, self_a),
                                 ("未知", unknown_model, self_b)):
        if not claims:
            continue
        expect = _expected_brand(model)
        if not expect:
            continue
        mismatched = {_detect_brand(c) for c in claims}
        mismatched.discard("")
        mismatched.discard(expect)
        if mismatched:
            res.identity_mismatch = True
            res.flags.append(f"{label} API 请求型号 {model}（品牌 {expect}），"
                             f"但自我认知回答中出现其他品牌词 {sorted(mismatched)}")

    # 3) 延迟显著不对称（转发/中继层信号；按输出长度归一化——
    #    每 token 延迟才能反映真实处理开销，长回答不会虚高延迟）
    lat_a = [p.latency_a / max(1, getattr(p, "tokens_a", 0)) for p in phase1.prompts
             if not p.error_a and p.latency_a > 0]
    lat_b = [p.latency_b / max(1, getattr(p, "tokens_b", 0)) for p in phase1.prompts
             if not p.error_b and p.latency_b > 0]
    if lat_a and lat_b:
        mean_a, mean_b = mean(lat_a), mean(lat_b)
        ratio = mean_b / mean_a if mean_a > 0 else 0.0
        res.latency_ratio = ratio
        if ratio >= latency_ratio_threshold:
            res.flags.append(f"未知 API 每 token 延迟是官方的 {ratio:.1f} 倍"
                             f"（{mean_b:.1f}ms/tok vs {mean_a:.1f}ms/tok，"
                             "已按输出长度归一化），存在转发/中继层的可能")
        elif ratio <= 1 / latency_ratio_threshold:
            res.flags.append(f"官方 API 每 token 延迟是未知的 {1 / ratio:.1f} 倍"
                             f"（{mean_a:.1f}ms/tok vs {mean_b:.1f}ms/tok，"
                             "已按输出长度归一化），网络路径或服务端缓存差异较大")

    # 4) 跨端自称品牌不一致（同一底层模型应自称一致；
    #    若未知端确为"外包一层提示工程"的包装层，改写的自称正是红旗）
    res.identity_votes_a = _vote_brands(self_a)
    res.identity_votes_b = _vote_brands(self_b)
    if res.identity_votes_a and res.identity_votes_b \
            and not (set(res.identity_votes_a) & set(res.identity_votes_b)):
        res.flags.append(
            f"官方自称品牌 {sorted(res.identity_votes_a)} 与未知自称品牌 "
            f"{sorted(res.identity_votes_b)} 完全不同，同一底层模型应自称一致")

    # 5) 拒答模式一致性（复用第一阶段"拒答模式类"取证，零额外请求）
    refus = [p for p in phase1.prompts if p.category == "refusal_profile"]
    matched = total = 0
    for p in refus:
        ca = _refusal_class(p.text_a) if not p.error_a else ""
        cb = _refusal_class(p.text_b) if not p.error_b else ""
        if ca and cb:
            total += 1
            matched += 1 if ca == cb else 0
    if total > 0:
        rate = matched / total
        res.refusal_match_rate = rate
        if rate < 0.75:
            res.flags.append(
                f"未知 API 拒答模式与官方不一致（{matched}/{total} 个探测行为一致，"
                f"一致率 {rate:.0%}），提示不同的护栏/内容策略层")

    # 6) 代理/中继自我披露（llm-verify 的 relay provenance 思路，零额外请求）
    for label, claims in (("官方", [p.text_a for p in phase1.prompts
                                    if p.category in ("self_awareness",
                                                      "system_prompt_leak")
                                    and not p.error_a and p.text_a]),
                          ("未知", [p.text_b for p in phase1.prompts
                                    if p.category in ("self_awareness",
                                                      "system_prompt_leak")
                                    and not p.error_b and p.text_b])):
        hits = sorted({w for c in claims for w in PROXY_WORDS if w in c.lower()})
        if hits:
            res.flags.append(
                f"{label} API 自我描述/系统提示泄漏回答中出现代理/中继词 "
                f"{hits}，提示存在转发/网关层"
                "（若确为合法中转服务的自述，可忽略此红旗）")

    # 7) 失败率不对称（代理/网关不稳定信号，复用了阶段二对抗探测的成败）
    if phase2 is not None:
        rows = getattr(phase2, "rows", None)
        if rows:
            n = len(rows)
            fb = sum(1 for r in rows if r.error_b)
            fa = sum(1 for r in rows if r.error_a)
            if fb and fb / n >= 0.15 and fb > fa * 2 + 1:
                res.flags.append(
                    f"未知 API 阶段二失败率 {fb}/{n}（{fb / n:.0%}），"
                    f"明显高于官方（{fa}/{n}），常见于代理/网关不稳定或超时")

    # 8) Token 用量不对称（复用第一阶段与第二阶段取证数据）
    official_usages: list[Any] = []
    unknown_usages: list[Any] = []
    if phase1 is not None:
        p1_rows = getattr(phase1, "prompts", []) + getattr(phase1, "reference_rows", [])
        for p in p1_rows:
            if not p.error_a and not p.error_b:
                ua = getattr(p, "usage_a", None)
                ub = getattr(p, "usage_b", None)
                if ua is not None or ub is not None:
                    official_usages.append(ua)
                    unknown_usages.append(ub)
    if phase2 is not None:
        for r in getattr(phase2, "rows", []):
            if not r.error_a and not r.error_b:
                ua = getattr(r, "usage_a", None)
                ub = getattr(r, "usage_b", None)
                if ua is not None or ub is not None:
                    official_usages.append(ua)
                    unknown_usages.append(ub)

    if official_usages and unknown_usages:
        token_flags = check_token_usage_asymmetry(
            official_usages, unknown_usages,
            prompt_ratio_threshold=prompt_ratio_threshold,
            completion_ratio_threshold=completion_ratio_threshold,
        )
        res.flags.extend(token_flags)

        total_pa = sum(_get_prompt_tokens(u) for u in official_usages)
        total_pb = sum(_get_prompt_tokens(u) for u in unknown_usages)
        if total_pa > 0:
            res.prompt_tokens_ratio = total_pb / total_pa

        total_ca = sum(_get_completion_tokens(u) for u in official_usages)
        total_cb = sum(_get_completion_tokens(u) for u in unknown_usages)
        if total_ca > 0:
            res.completion_tokens_ratio = total_cb / total_ca

        total_ra = sum(_get_reasoning_tokens(u) for u in official_usages)
        total_rb = sum(_get_reasoning_tokens(u) for u in unknown_usages)
        if total_rb > 0 and total_ra == 0:
            res.has_reasoning_asymmetry = True

    return res
