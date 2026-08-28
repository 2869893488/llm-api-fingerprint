"""运行日志系统：每次运行落两份文件，便于事后维护与回溯。

  output/logs/run_<时间戳>.log   人类可读的全流程日志（环境/配置脱敏快照、
                                  进度、警告/错误、最终结论）
  output/logs/run_<时间戳>.json  结构化摘要（版本、参数脱敏、端点、判定、
                                  耗时、退出码），可程序化对比多次运行

设计约束：
  - 所有写入做容错：日志系统任何故障都不能影响主流程；
  - api_key / 命令行密钥类参数一律脱敏后才落盘；
  - raw() 同时按行内前缀统计"警告/错误"计数，方便事后 grep 定位问题运行。
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional

_WARN_MARKS = ("警告", "warn", "WARN")
_ERR_MARKS = ("错误", "异常", "失败:", "error", "Error", "Traceback")
_MAX_EVENTS = 500              # 结构化事件上限，防止几百探测把 json 撑爆


def redact_secret(text: str) -> str:
    """密钥脱敏：只保留首 6 尾 4。"""
    if len(text) <= 8:
        return "****"
    return text[:6] + "****" + text[-4:]


def redact_args(args: Optional[dict]) -> dict:
    """命令行参数脱敏：key/password/token/secret 类键的值打码。"""
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if v is None:
            continue
        if any(s in k.lower() for s in ("key", "password", "token", "secret")):
            out[k] = redact_secret(str(v)) if isinstance(v, str) else "****"
        else:
            out[k] = v
    return out


class RunLogger:
    """每次运行一个实例；所有方法线程安全、写故障静默降级。"""

    def __init__(self, output_dir: str, *, version: str,
                 args: Optional[dict] = None,
                 official=None, unknown=None, options=None):
        self._lock = threading.Lock()
        self._closed = False
        self._start = time.monotonic()
        self._stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.log_path = os.path.join(self.logs_dir, f"run_{self._stamp}.log")
        self.json_path = os.path.join(self.logs_dir, f"run_{self._stamp}.json")
        self._data: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "tool_version": version,
            "args": redact_args(args or {}),
            "official": self._target(official),
            "unknown": self._target(unknown),
            "options": options,
            "warnings": [],
            "errors": [],
            "note": "",
        }
        self._fh = None
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            self._fh = open(self.log_path, "a", encoding="utf-8")
        except OSError:
            self._fh = None
        self._write(f"===== 运行开始 {datetime.now().isoformat(timespec='seconds')} =====")
        self._write(f"工具版本: {version}")
        if official is not None:
            self._write(f"官方端点: {official.name} | {official.base_url} | "
                        f"{official.model_name} | key={redact_secret(official.api_key)}")
        if unknown is not None:
            self._write(f"未知端点: {unknown.name} | {unknown.base_url} | "
                        f"{unknown.model_name} | key={redact_secret(unknown.api_key)}")

    # ------------------------------------------------------------------
    @staticmethod
    def _target(t) -> Optional[dict]:
        if t is None:
            return None
        return {"name": t.name, "base_url": t.base_url,
                "model": t.model_name,
                "api_key": redact_secret(getattr(t, "api_key", "") or "")}

    def _write(self, line: str) -> None:
        if self._fh is not None:
            try:
                # .log 每行带运行时刻（HH:MM:SS），便于事后核对"哪一步在什么
                # 时间发生"（如定位超时与卡点）
                self._fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
                self._fh.flush()
            except OSError:
                pass

    def raw(self, line: str) -> None:
        """镜像控制台的原始行（进度/结果等）。"""
        with self._lock:
            if self._closed:
                return
            self._write(line)
            if any(line.startswith(m) or f"{m}:" in line[:12]
                   for m in _ERR_MARKS):
                self._append_event("errors", line)
            elif any(line.startswith(m) or f"{m}:" in line[:12]
                     for m in _WARN_MARKS):
                self._append_event("warnings", line)

    def _append_event(self, key: str, line: str) -> None:
        ev = self._data[key]
        if len(ev) < _MAX_EVENTS:
            ev.append(line)

    def note(self, msg: str) -> None:
        """关键节点备注（进入/离开某阶段等）。"""
        with self._lock:
            if not self._closed:
                self._write(f"# {msg}")

    def finish(self, rc: int = 0, *, note: str = "", verdict=None,
               scores: Optional[dict] = None,
               report_path: Optional[str] = None,
               summary_path: Optional[str] = None,
               phase_elapsed: Optional[dict] = None) -> None:
        """运行结束：写尾部信息 + 结构化 JSON 摘要，关闭句柄（幂等）。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._data["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self._data["note"] = note
            self._data["exit_code"] = rc
            self._data["elapsed_sec"] = round(time.monotonic() - self._start, 1)
            if phase_elapsed:
                self._data["phase_elapsed_sec"] = {
                    k: round(v, 2) for k, v in phase_elapsed.items()}
            if verdict is not None:
                self._data["verdict"] = {
                    "score": getattr(verdict, "score", None),
                    "level": getattr(verdict, "level", None),
                    "level_cn": getattr(verdict, "level_cn", None),
                    "fail_closed": getattr(verdict, "fail_closed", False),
                    "conflict": getattr(verdict, "conflict", False),
                    "skipped_phases": getattr(verdict, "skipped_phases", []),
                }
            if scores:
                self._data["phase_scores"] = {k: round(v, 4) if v is not None else None
                                              for k, v in scores.items()}
            if report_path:
                self._data["report"] = report_path
            if summary_path:
                self._data["summary"] = summary_path
            self._write(f"===== 运行结束 rc={rc} 耗时 "
                        f"{self._data['elapsed_sec']}s "
                        f"{datetime.now().isoformat(timespec='seconds')} =====")
            if note:
                self._write(f"备注: {note}")
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
            except OSError:
                pass
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None