"""原始数据取证存储。

每次探测调用都会落一个 JSON 文件，包含 prompt 原文、完整响应、
时间戳与 API 端点，供事后审计与复现。
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from . import __version__
from .target import ChatResult, ProbeTarget


class Recorder:
    def __init__(self, root: str, resume: bool = False,
                 reuse_names: Optional[set[str]] = None):
        self.root = root
        self.resume = resume
        # 官方侧缓存复用：只复用这些 target 的成功取证（跨运行省一半请求）。
        # resume=True 时全部按续跑逻辑复用；否则仅名字在 reuse_names 里的复用。
        self.reuse_names = set(reuse_names or [])
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, str, str], dict] = {}
        os.makedirs(root, exist_ok=True)   # 输出目录可能尚不存在（如首次运行）
        if resume or self.reuse_names:
            self._load_existing()

    def should_reuse(self, target: ProbeTarget) -> bool:
        """该 target 本次调用是否复用本地缓存（断点续跑或官方侧缓存）。"""
        if self.resume:
            return True                  # 断点续跑：全部 target 都复用
        return target.name in self.reuse_names

    # ---------------------------------------------------------------- 断点续跑
    def _load_existing(self) -> None:
        """扫描 raw/phase*/ 下取证文件，加载成功的记录作为续跑缓存。

        同一 (phase, probe_id, target) 存在多个文件时取 mtime 最新者；
        解析失败的文件（如中断时写了一半）跳过。
        """
        raw_root = os.path.join(self.root, "raw")
        if not os.path.isdir(raw_root):
            return
        latest: dict[tuple[int, str, str], tuple[float, dict]] = {}
        for dir_name in os.listdir(raw_root):
            m = re.match(r"phase(\d+)$", dir_name)
            if not m:
                continue
            phase = int(m.group(1))
            for fn in os.listdir(os.path.join(raw_root, dir_name)):
                if not fn.endswith(".json"):
                    continue
                path = os.path.join(raw_root, dir_name, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                key = (phase, entry.get("probe_id"), entry.get("target"))
                mtime = os.path.getmtime(path)
                if key not in latest or mtime > latest[key][0]:
                    latest[key] = (mtime, entry)
        self._cache = {k: v[1] for k, v in latest.items()}

    def find(self, phase: int, probe_id: str, target_name: str) -> Optional[dict]:
        """返回对应探测的续跑缓存记录；不存在返回 None。"""
        return self._cache.get((phase, probe_id, target_name))

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    def record(self, phase: int, probe_id: str, target: ProbeTarget,
               request: dict, result: ChatResult) -> str:
        entry = {
            "phase": phase,
            "probe_id": probe_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": target.name,
            "endpoint": target.base_url,
            "model": target.model_name,
            "request": request,
            "response": {
                "texts": result.texts,
                "finish_reasons": result.finish_reasons,
                "usage": result.usage,
                "logprobs": result.logprobs,
                "raw": result.raw,
            },
            "capabilities": result.capabilities,
            "latency_ms": round(result.latency_ms, 1),
            "error": result.error,
        }
        slug = re.sub(r"[^\w\-]+", "_", target.name)
        phase_dir = os.path.join(self.root, "raw", f"phase{phase}")
        os.makedirs(phase_dir, exist_ok=True)
        fname = f"{probe_id}__{slug}__{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        path = os.path.join(phase_dir, fname)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        return path

    def save_manifest(self, config) -> str:
        """保存运行配置快照（脱敏：不写入任何 api_key）。"""
        def redact(t):
            key = t.api_key
            shown = (key[:6] + "****" + key[-4:]) if len(key) > 12 else "****"
            return {"name": t.name, "base_url": t.base_url,
                    "model_name": t.model_name, "api_key": shown}

        manifest = {
            "tool_version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "official": redact(config.official),
            "unknown": redact(config.unknown),
            "options": vars(config.options),
        }
        path = os.path.join(self.root, "manifest.json")
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path
