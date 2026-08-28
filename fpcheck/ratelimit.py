"""全局请求限速器（防止并行探测触发模型的并发限制）。

双维度并行（五个阶段并行 × 同一探测双 API 并行）会把瞬时请求率放大约
10 倍。本模块提供在所有目标（官方/未知）、所有阶段之间共享的限速器：

  - min_interval —— 两次请求开始之间的最小间隔（秒）：全局平均速率
    ≈ 1/min_interval，带锁串行分配起始时刻，任何时刻都不会突发；
  - max_inflight —— 同时进行的最大请求数（在飞信号量上限）；
  - AdaptivePolicy —— 自动限速模式：**不内置任何型号限额表**（模型档位/
    额度随时在变，内置表必然过时），只依据两类实时信号：
      (1) 端点响应头 x-ratelimit-*（OpenAI 官方按账号档位/模型实时下发）；
      (2) 429 反馈：触发限流后自动加倍保守（最多 ×8），宁可慢不触限；
    端点不返回额度头时，回退到用户配置里显式填写的值。

两种模式：
  手动模式  RateLimiter(min_interval=0.5, max_inflight=4, policy=None)
            → 固定间隔，配置里 rate_limit_rpm 是什么就是什么。
  自动模式  RateLimiter(min_interval=0.0, max_inflight=4, policy=AdaptivePolicy(...))
            → 间隔由策略按响应头/429/配置回退动态给出。
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

# 限速窗口（秒）：OpenAI 的 x-ratelimit-* 以 60s 为窗口
WINDOW_SEC = 60.0
# 安全系数：只用到端点额度的 80%
SAFETY = 0.8


class AdaptivePolicy:
    """自动限速策略：限制值由"端点实时响应头 > 配置回退"决定。

    每个目标（官方/未知）独立记录从响应头发现的结果，全局取其中最保守
    的限度；429 反馈把间隔加倍（最多 ×8），保证被限流后立刻退让。
    不内置型号限额表——额度随账号档位/模型版本实时变化，只信端点下发。
    """

    def __init__(self, fallback_rpm: int = 120, fallback_tpm: int = 30000):
        self._lock = threading.Lock()
        self._fallback_rpm = max(0, fallback_rpm)
        self._fallback_tpm = max(0, fallback_tpm)
        self._targets: dict[str, dict] = {}   # name -> {"rpm": int|None, "tpm": int|None}
        self._throttle = 1.0
        self._window = WINDOW_SEC

    # ------------------------------------------------------------ 信息输入
    def observe_headers(self, target_name: str, headers) -> None:
        """从端点真实响应头发现当前额度（官方按账号档位/模型实时下发）。"""
        try:
            lim_req = _h(headers, "x-ratelimit-limit-requests")
            reset_req = _h(headers, "x-ratelimit-reset-requests")
            lim_tok = _h(headers, "x-ratelimit-limit-tokens")
            reset_tok = _h(headers, "x-ratelimit-reset-tokens")
        except Exception:  # noqa: BLE001 —— 头格式异常视为没有
            return
        rpm = tpm = None
        if lim_req is not None and reset_req:
            rpm = max(1, int(float(lim_req) * (self._window / float(reset_req))))
        if lim_tok is not None and reset_tok:
            tpm = max(1, int(float(lim_tok) * (self._window / float(reset_tok))))
        if rpm is None and tpm is None:
            return
        with self._lock:
            t = self._targets.setdefault(target_name, {"rpm": None, "tpm": None})
            if rpm is not None:
                t["rpm"] = rpm
            if tpm is not None:
                t["tpm"] = tpm

    def note_throttled(self) -> None:
        """端点返回 429：加倍保守（最多 ×8），后续节奏立刻退让。"""
        with self._lock:
            self._throttle = min(8.0, self._throttle * 2.0)

    # ------------------------------------------------------------ 计算间隔
    def interval(self, est_tokens: float = 0.0) -> float:
        """当前应使用的最小请求间隔（秒）；0 = 不限速。

        est_tokens: 本次请求预估 token 数（prompt + 输出），用于把吞吐
        压在 TPM 额度内（比只限 RPM 更贴合真实约束）。
        """
        with self._lock:
            rpms = [t["rpm"] for t in self._targets.values() if t["rpm"]]
            tpms = [t["tpm"] for t in self._targets.values() if t["tpm"]]
            rpm = min(rpms) if rpms else self._fallback_rpm
            tpm = min(tpms) if tpms else self._fallback_tpm
            throttle = self._throttle
        base = (60.0 / (rpm * SAFETY)) if rpm > 0 else 0.0
        tok = (est_tokens * self._window / (tpm * SAFETY)) \
            if (tpm > 0 and est_tokens > 0) else 0.0
        return max(base, tok) * throttle

    def describe(self) -> str:
        """当前生效方式的描述（用于启动日志）。"""
        with self._lock:
            rpms = [t["rpm"] for t in self._targets.values() if t["rpm"]]
            tpms = [t["tpm"] for t in self._targets.values() if t["tpm"]]
            rpm = min(rpms) if rpms else self._fallback_rpm
            tpm = min(tpms) if tpms else self._fallback_tpm
        if rpm <= 0 and tpm <= 0:
            return "自动（端点未下发额度头，且未配置回退值，不限速）"
        return f"自动（响应头/429 感知；回退 {rpm} RPM / {tpm} TPM）"


def _h(headers, key: str) -> Optional[str]:
    """大小写不敏感取响应头；取不到返回 None。"""
    if headers is None:
        return None
    try:
        return headers.get(key)
    except Exception:  # noqa: BLE001
        return None


class RateLimiter:
    """跨目标共享的请求限速器。线程安全。

    policy 为 None 时使用固定 min_interval（手动模式）；
    有 policy 时每次请求的间隔由策略动态决定（自动模式）。
    """

    def __init__(self, min_interval: float = 0.0, max_inflight: int = 64,
                 policy: Optional[AdaptivePolicy] = None):
        self._min_interval = max(0.0, min_interval)
        self._inflight = threading.BoundedSemaphore(max(1, max_inflight))
        self._lock = threading.Lock()
        self._next_at = 0.0
        self.policy = policy

    def acquire(self, est_tokens: float = 0.0) -> None:
        """占一个请求槽：先扣在飞配额，再按当前间隔排队等待起始时刻。"""
        self._inflight.acquire()
        interval = self.policy.interval(est_tokens) if self.policy \
            else self._min_interval
        if interval <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_at:
                    self._next_at = now + interval
                    return
                wait = self._next_at - now
            time.sleep(wait)   # 不加锁等待，避免阻塞其他线程领号

    def release(self) -> None:
        self._inflight.release()

    @contextmanager
    def slot(self, est_tokens: float = 0.0) -> Iterator[None]:
        """with limiter.slot(est_tokens): 包住一次 HTTP 请求。"""
        self.acquire(est_tokens)
        try:
            yield
        finally:
            self.release()

    # ------------------------------------------------------------ 自动模式入口
    def observe_headers(self, target_name: str, headers) -> None:
        if self.policy is not None:
            self.policy.observe_headers(target_name, headers)

    def note_throttled(self) -> None:
        if self.policy is not None:
            self.policy.note_throttled()