"""统一的探测目标封装。

对官方 API 与未知 API 提供完全相同的 chat 接口（openai>=1.x 客户端风格：
先创建 OpenAI 客户端对象，再调用 chat.completions.create）。
未知 API 不支持 seed / logprobs / n 等参数时自动降级，不崩溃。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI, APIConnectionError, APITimeoutError
from openai import BadRequestError, RateLimitError, UnprocessableEntityError

# 参数类错误 → 走降级逻辑；网络/限流/鉴权类错误由 SDK 内置重试处理或向上抛出。
# 注: 新版 openai 不再导出 NotImplementedError，直接使用内置异常
# （SDK 对不支持参数的调用会抛内置 NotImplementedError）。
PARAM_ERRORS = (BadRequestError, UnprocessableEntityError, NotImplementedError)
# 连接层瞬时错误（如代理间歇性重置 TLS 连接）→ 本层额外重试，避免直接失败退出
CONN_ERRORS = (APIConnectionError, APITimeoutError)
CONN_RETRIES = 3


@dataclass
class ChatResult:
    """一次 chat 调用的标准化结果（已转为纯 dict，可直接 JSON 序列化）。"""

    prompt: str
    texts: list[str]
    finish_reasons: list[str]
    logprobs: Optional[list[list[dict]]] = None   # 每个采样 → 每个 token 的 top-k 候选
    usage: Optional[dict] = None
    raw: Optional[dict] = None                    # 完整原始响应（取证用）
    capabilities: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None
    resumed: bool = False              # True = 本次结果来自断点续跑缓存，未发网络请求

    @property
    def ok(self) -> bool:
        return self.error is None


class ProbeTarget:
    """一个可探测的 Chat API 端点。"""

    def __init__(self, name: str, base_url: str, api_key: str, model_name: str,
                 timeout: float = 90.0, max_retries: int = 4,
                 proxy: Optional[str] = None, api_style: str = "chat",
                 rate_limiter=None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy = proxy
        # 全局限速器（官方/未知共用一个实例）：每次真实 HTTP 请求都经过它，
        # 保证双 API 并行 + 多阶段并行时也不会突发到端点的并发限制。
        self._rate = rate_limiter
        # 输出 token 的滑动平均（EMA）：自动限速用它估算"下一次请求的体量"，
        # 把吞吐压在端点的 TPM 额度内（比按 max_tokens 上限估算更贴近实际）
        self._ema_out_tokens = 0.0
        # 调用风格: "chat" = /chat/completions；"responses" = /responses
        # （部分网关的 chat completions 通道故障时可用 responses 通道绕行）
        if api_style not in ("chat", "responses"):
            raise ValueError(f"api_style 只能是 chat 或 responses: {api_style}")
        self.api_style = api_style
        kwargs = dict(base_url=base_url, api_key=api_key,
                      timeout=timeout, max_retries=max_retries)
        if proxy:
            # 显式代理：所有请求（预检、三个阶段的全部探测）都走该代理，
            # 且关闭 trust_env，避免环境变量/系统代理干扰，保证"全程走代理"。
            import httpx
            kwargs["http_client"] = httpx.Client(
                proxy=proxy, trust_env=False, timeout=httpx.Timeout(timeout))
        self.client = OpenAI(**kwargs)
        # None=未知, True=支持, False=不支持（探测过程中逐步确认）
        self.capabilities: dict[str, Optional[bool]] = {
            "temperature": None, "seed": None, "logprobs": None, "n": None}
        if api_style == "responses":   # Responses API 无 seed/n 参数，n 由并行调用补齐
            self.capabilities["seed"] = False

    def clone_with_timeout(self, timeout: float,
                           max_retries: Optional[int] = None) -> "ProbeTarget":
        """按新的请求超时克隆目标（阶段专用短超时，如阶段六防慢网关卡死）。

        共享同一个全局限速器；新客户端、独立能力探测缓存。
        """
        return ProbeTarget(self.name, self.base_url, self.api_key,
                           self.model_name, timeout=timeout,
                           max_retries=self.max_retries
                           if max_retries is None else max_retries,
                           proxy=self.proxy, api_style=self.api_style,
                           rate_limiter=self._rate)

    # ------------------------------------------------------------------ 统一接口
    def _call(self, fn, est_tokens: float = 0.0):
        """在限速器槽位内执行一次 HTTP 请求；未配置限速器时原样执行。"""
        if self._rate is None:
            return fn()
        with self._rate.slot(est_tokens):
            try:
                return fn()
            except RateLimitError:
                self._rate.note_throttled()   # 429：自动限速加倍保守
                raise

    def _estimate_tokens(self, prompt: str, output_cap: int) -> float:
        """本次请求的 token 估算：prompt 按 ~3 字符/token，输出按 EMA 实际值。"""
        prompt_tokens = len(prompt) / 3.0
        return prompt_tokens + min(max(self._ema_out_tokens, 32.0),
                                   float(max(output_cap, 1)))

    def _observe_response(self, resp, raw_usage: Optional[dict]) -> None:
        """从响应反馈限速策略：真实额度头 + 实际 token 用量。"""
        headers = getattr(resp, "headers", None)
        if headers is None:
            headers = getattr(getattr(resp, "_response", None), "headers", None)
        if headers is not None and self._rate is not None:
            self._rate.observe_headers(self.name, headers)
        if raw_usage:
            done = raw_usage.get("completion_tokens") or 0
            if done > 0:
                self._ema_out_tokens = 0.7 * self._ema_out_tokens + 0.3 * done

    def chat(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512,
             seed: Optional[int] = None, n: int = 1,
             logprobs: bool = False, top_logprobs: Optional[int] = None) -> ChatResult:
        """对两个 API 完全一致的探测入口。

        参数不受支持时按 temperature -> logprobs -> seed -> n 的顺序逐级降级：
        - temperature 不支持 → 请求体去掉 temperature 后重试（部分网关/推理模型拒绝该参数）
        - logprobs/top_logprobs 不支持 → 去掉后重试
        - seed 不支持 → 置 None 后重试
        - n>1 不支持 → 改为串行调用 n 次
        已确认不支持的能力会被缓存，后续调用直接跳过，避免反复失败重试。
        """
        # 已确认不支持的能力 → 本次调用直接不带（缓存生效）
        if self.capabilities.get("temperature") is False:
            temperature = None  # type: ignore[assignment]
        if self.capabilities.get("logprobs") is False:
            logprobs, top_logprobs = False, None
        if self.capabilities.get("seed") is False:
            seed = None
        if self.capabilities.get("n") is False and self.api_style == "chat":
            n = 1
        if self.api_style == "responses":
            seed = None  # Responses API 不支持 seed

        first = {"logprobs": logprobs, "top_logprobs": top_logprobs if logprobs else None,
                 "seed": seed, "n": n, "temperature": temperature}
        ladder: list[dict[str, Any]] = [first]
        if temperature is not None:
            ladder.append({**first, "temperature": None})      # 去掉 temperature
        if logprobs:
            ladder.append({**ladder[-1], "logprobs": False, "top_logprobs": None})
        if seed is not None:
            ladder.append({**ladder[-1], "seed": None})
        if n > 1 and self.api_style == "chat":
            ladder.append({**ladder[-1], "n": 1})

        last_err: Optional[Exception] = None
        for step in ladder:
            try:
                res = self._execute(prompt, max_tokens, **step)
                if step["n"] < n:
                    res.capabilities["n"] = False
                if step["temperature"] is None:
                    res.capabilities["temperature"] = False
                self._update_caps(res, logprobs=logprobs, seed=seed, n=n,
                                  temperature=temperature)
                return res
            except PARAM_ERRORS as e:
                last_err = e
                continue  # 降级到下一步重试
        return ChatResult(prompt=prompt, texts=[], finish_reasons=[],
                          error=f"{type(last_err).__name__}: {last_err}")

    def _execute(self, prompt: str, max_tokens: int,
                 seed, n: int, logprobs: bool, top_logprobs, temperature) -> ChatResult:
        if self.api_style == "responses":
            return self._execute_responses(prompt, max_tokens,
                                           logprobs, top_logprobs, temperature, n)
        t0 = time.monotonic()
        params = dict(model=self.model_name,
                      messages=[{"role": "user", "content": prompt}],
                      max_tokens=max_tokens,
                      n=n, seed=seed, logprobs=logprobs, top_logprobs=top_logprobs)
        if temperature is not None:      # 部分网关/推理模型拒绝 temperature 参数
            params["temperature"] = temperature
        resp = self._call(lambda: self.client.chat.completions.create(**params))
        raw = resp.model_dump()
        first_n = len(raw["choices"])
        # 部分服务端会静默忽略 n，只返回 1 个 choice：补齐到 n 个
        if n > 1 and first_n < n:
            for _ in range(n - first_n):
                extra = self._call(
                    lambda: self.client.chat.completions.create(**{**params, "n": 1})
                ).model_dump()
                raw["choices"].extend(extra["choices"])
            raw["choices"] = raw["choices"][:n]
        elapsed = (time.monotonic() - t0) * 1000.0

        texts = [c["message"].get("content") or "" for c in raw["choices"]]
        finish_reasons = [c.get("finish_reason") for c in raw["choices"]]
        logprobs_data = None
        if logprobs:
            # 部分服务端返回 logprobs=None（静默忽略参数），不能直接 .get()
            logprobs_data = [(c.get("logprobs") or {}).get("content")
                             for c in raw["choices"]]
        caps = {
            "temperature": temperature is not None,
            "seed": seed is not None,
            "logprobs": logprobs and logprobs_data is not None
                       and any(x is not None for x in logprobs_data),
            "n": first_n >= n,
        }
        return ChatResult(prompt=prompt, texts=texts, finish_reasons=finish_reasons,
                          logprobs=logprobs_data, usage=raw.get("usage"), raw=raw,
                          capabilities=caps, latency_ms=elapsed)

    def _execute_responses(self, prompt: str, max_tokens: int,
                           logprobs: bool, top_logprobs, temperature, n: int) -> ChatResult:
        """Responses API (/responses) 执行路径：输出归一化为与 chat completions 一致的 ChatResult。

        Responses API 无 n 参数，n>1 时并行调用 n 次合并（并发受全局
        限速器约束，避免串行放大延迟）；无 seed 参数。
        finish_reason 由响应级 status 映射：completed→stop，incomplete→length。
        """
        t0 = time.monotonic()
        max_tokens = max(max_tokens, 16)   # 部分上游要求 max_output_tokens >= 16
        base_params = dict(model=self.model_name,
                           input=[{"role": "user", "content": prompt}],
                           max_output_tokens=max_tokens)
        if temperature is not None:      # 部分推理模型拒绝 temperature 参数
            base_params["temperature"] = temperature
        if logprobs:
            # 本版本 SDK 的 responses.create 未显式声明 logprobs 参数，
            # 通过 extra_body 透传；服务端不支持时会报错，由上层降级处理
            base_params["extra_body"] = {"logprobs": True,
                                         "top_logprobs": top_logprobs}

        def fetch_one() -> dict:
            """单次采样：连接级重试（代理重置连接时），限速器内执行。"""
            last_conn_err: Optional[Exception] = None
            for _try in range(CONN_RETRIES):
                try:
                    return self._call(
                        lambda: self.client.responses.create(**base_params)
                    ).model_dump()
                except CONN_ERRORS as e:
                    last_conn_err = e
                    time.sleep(1.0 * (_try + 1))
            raise last_conn_err

        if n <= 1:
            raws = [fetch_one()]
        else:
            # n>1 → 并行采样，把串行延迟摊平（并发仍被全局限速器 cap 住）
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
                raws = list(pool.map(lambda _: fetch_one(), range(n)))
        elapsed = (time.monotonic() - t0) * 1000.0

        texts, finish_reasons, logprobs_data = [], [], []
        for raw in raws:
            if raw.get("status") == "failed":
                err = (raw.get("error") or {}).get("message") or "response failed"
                raise RuntimeError(f"Responses API 返回 failed: {err}")
            parts_text, parts_lp = [], None
            for item in raw.get("output", []):
                if item.get("type") != "message":
                    continue
                for part in item.get("content", []):
                    if part.get("type") != "output_text":
                        continue
                    parts_text.append(part.get("text") or "")
                    if logprobs and parts_lp is None:
                        parts_lp = part.get("logprobs")
            texts.append("".join(parts_text))
            finish_reasons.append(
                "stop" if raw.get("status") == "completed" else "length")
            logprobs_data.append(parts_lp)

        caps = {
            "temperature": temperature is not None,
            "seed": False,
            "logprobs": logprobs and any(x is not None for x in logprobs_data),
            "n": True,   # 已由串行调用补齐 n 个采样
        }
        raw_merged = raws[0] if len(raws) == 1 else {"merged_responses": raws}
        return ChatResult(prompt=prompt, texts=texts, finish_reasons=finish_reasons,
                          logprobs=logprobs_data if logprobs else None,
                          usage=raws[0].get("usage"), raw=raw_merged,
                          capabilities=caps, latency_ms=elapsed)

    def _update_caps(self, res: ChatResult, logprobs: bool, seed, n: int,
                     temperature) -> None:
        """把本次调用确认的能力写回 self.capabilities。"""
        if not res.ok:
            return
        if temperature is not None:
            self.capabilities["temperature"] = res.capabilities.get("temperature", True)
        if logprobs:
            self.capabilities["logprobs"] = res.capabilities.get("logprobs", False)
        if seed is not None:
            self.capabilities["seed"] = res.capabilities.get("seed", False)
        if n > 1:
            self.capabilities["n"] = res.capabilities.get("n", False)

    # ------------------------------------------------------------------ 预检
    def ping(self) -> Optional[str]:
        """连通性预检：失败返回错误描述，成功返回 None。"""
        try:
            res = self.chat("Say OK.", temperature=0, max_tokens=8)
            return res.error
        except Exception as e:  # noqa: BLE001 —— 预检需要把任何异常转成文本
            return f"{type(e).__name__}: {e}"
