"""成对探测执行器：同一 probe 同时（或依次）发给两个目标并记录取证数据。

参考 praetorian-inc/julius 的多目标批量探测思路：每个 probe 固定为
"官方 + 未知" 的成对结构，保证两个 API 收到的请求逐字节一致。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .storage import Recorder
from .target import ChatResult, ProbeTarget

# 续跑匹配键：这些参数一致才复用缓存（seed 除外——两个端点实际都不支持，
# 取证记录里存的是降级后的值，与请求值不一致）
_MATCH_KEYS = ("prompt", "temperature", "max_tokens", "n", "logprobs", "top_logprobs")


def _request_matches(saved: Optional[dict], new: dict) -> bool:
    """取证记录中的请求参数与本次请求是否一致。"""
    if not saved:
        return False
    return all(saved.get(k) == new.get(k) for k in _MATCH_KEYS)


def _result_from_entry(entry: dict) -> ChatResult:
    """从取证记录重建 ChatResult（断点续跑用，不发网络请求）。"""
    resp = entry.get("response") or {}
    return ChatResult(prompt=(entry.get("request") or {}).get("prompt", ""),
                      texts=resp.get("texts") or [],
                      finish_reasons=resp.get("finish_reasons") or [],
                      logprobs=resp.get("logprobs"), usage=resp.get("usage"),
                      raw=resp.get("raw"), capabilities=entry.get("capabilities") or {},
                      latency_ms=entry.get("latency_ms") or 0.0, resumed=True)


def call_pair(recorder: Recorder, official: ProbeTarget, unknown: ProbeTarget,
              phase: int, probe_id: str, prompt: str, *,
              temperature: float, max_tokens: int,
              seed=None, n: int = 1, logprobs: bool = False,
              top_logprobs=None, concurrent: bool = False):
    """对两个 API 发起完全相同的探测，返回 (official_result, unknown_result)。"""
    request = {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens,
               "seed": seed, "n": n, "logprobs": logprobs, "top_logprobs": top_logprobs}

    def call(target: ProbeTarget) -> ChatResult:
        if recorder.should_reuse(target):
            entry = recorder.find(phase, probe_id, target.name)
            # 官方侧缓存复用也要求端点/模型一致（防止官方模型升级后误用旧数据）
            if (entry is not None and not entry.get("error")
                    and _request_matches(entry.get("request"), request)
                    and entry.get("endpoint") == target.base_url
                    and entry.get("model") == target.model_name):
                return _result_from_entry(entry)
        t0 = time.monotonic()
        try:
            res = target.chat(prompt, temperature=temperature, max_tokens=max_tokens,
                              seed=seed, n=n, logprobs=logprobs, top_logprobs=top_logprobs)
            if res.latency_ms <= 0.0:
                res.latency_ms = (time.monotonic() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001 —— 网络/鉴权等异常也记录，不中断整体流程
            res = ChatResult(prompt=prompt, texts=[], finish_reasons=[],
                             error=f"{type(exc).__name__}: {exc}")
        # 取证记录应反映"实际发送"的请求：按本次调用的能力降级结果修正参数
        sent = dict(request)
        if res.capabilities.get("temperature") is False:
            sent.pop("temperature", None)        # 未发送 temperature 字段
        if res.capabilities.get("seed") is False:
            sent["seed"] = None
        if res.capabilities.get("logprobs") is False:
            sent["logprobs"] = False
            sent["top_logprobs"] = None
        if res.capabilities.get("n") is False:
            sent["n"] = 1
        recorder.record(phase, probe_id, target, sent, res)
        return res

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(call, official)
            fb = pool.submit(call, unknown)
            return fa.result(), fb.result()
    return call(official), call(unknown)
