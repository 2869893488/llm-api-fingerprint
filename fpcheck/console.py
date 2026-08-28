"""多阶段分区进度面板（并行阶段时控制台排版）。

并行模式下五个阶段同时推进，若各阶段直接 print，日志会互相交错、完全
不可读。本模块把终端纵向分成 N 块区域，每个阶段独占一块，只刷新自己的
内容；emit() 为线程安全追加（带节流重绘，避免每行都闪屏）；finish() 时
先重绘最终状态，再按阶段逐段打印全量日志（供回滚查看）。

纯 ANSI（CSI 光标控制 + 清屏），无第三方依赖：
  - draw() 由 "\x1b[H" 回到屏幕左上角 + 重新书写整个面板 + "\x1b[J" 清尾；
  - 首次进入面板模式时 "\x1b[2J" 整屏清空一次。

非终端（stdout 重定向/管道，如脚本调用、CI）自动退化为普通逐行输出，
行为与旧版完全一致；也可用 --no-panel 强制退化。
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from typing import Optional, TextIO

# ---------------------------------------------------------------------------
# 面板显示层精简：裁剪每行里的冗余字段（完整原文保存在 .log 与 finish 全量
# 日志里，精简只影响屏幕）。含错误/警告标记的行不做精简、原样显示。
_KEEP_FULL = ("错误", "失败", "异常", "警告", "Traceback")
_COMPACT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\[(\w+)\]\s+"), r"\1 "),          # [p6_00] → p6_00
    (re.compile(r"\[(评分|参考)\]\s*"), " "),
    (re.compile(r"后缀长度=\d+\s*"), " "),
    (re.compile(r"包装=[A-Za-z_]+\s*"), " "),
    (re.compile(r"采样=\S+\s*"), " "),
    (re.compile(r"位置数=\d+\s*"), " "),
    (re.compile(r"官方首token=\S+\s*"), " "),
    (re.compile(r"未知首token=\S+\s*"), " "),
    (re.compile(r"  +"), " "),
]


def compact_line(line: str) -> str:
    """面板显示用的精简行；错误/警告行保持原样。"""
    if any(m in line for m in _KEEP_FULL):
        return line
    for rx, rep in _COMPACT_RULES:
        line = rx.sub(rep, line)
    return line.strip()


def panels_supported(stream: Optional[TextIO] = None) -> bool:
    """分区进度面板是否应该启用。

    面板依赖"回到屏幕顶部 + 整屏重绘"的 ANSI 全屏刷新，只有现代终端可靠：
    - 非终端（重定向/管道/CI）→ False（退化为普通逐行日志）；
    - Windows 旧控制台（conhost / Windows PowerShell 5.1）对整屏刷新支持
      不稳定、会出乱屏 → 仅当检测到 Windows Terminal（WT_SESSION）或
      VSCode（TERM_PROGRAM）等现代终端才启用；
    - 类 Unix 终端（TERM 存在）→ True。
    """
    if stream is not None:
        if not bool(getattr(stream, "isatty", lambda: False)()):
            return False
    elif not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return bool(os.environ.get("WT_SESSION")
                    or os.environ.get("TERM_PROGRAM"))
    return bool(os.environ.get("TERM"))


class PhasePanels:
    """按阶段分区的进度面板。线程安全。"""

    def __init__(self, panels: list[tuple[str, str]], *,
                 out: Optional[TextIO] = None,
                 lines_per_panel: int = 6,
                 redraw_interval: float = 0.3,
                 force: bool = False,
                 sink=None):
        """
        panels: [(key, 标题), ...]，key 与 emit 的第一个参数对应。
        force: 强制启用面板（即使 out 不是终端）——用于测试。
        sink: 可选回调，finish() 逐段打印全量日志时同步镜像到运行日志。
        """
        self._panels = list(panels)
        self._buf: dict[str, list[str]] = {k: [] for k, _ in panels}
        self._out = out or sys.stdout
        self._sink = sink
        self._lines = max(2, lines_per_panel)
        self._interval = max(0.05, redraw_interval)
        is_tty = bool(getattr(self._out, "isatty", lambda: False)())
        self._enabled = bool(panels) and (force or is_tty)
        self._lock = threading.Lock()
        self._last_draw = 0.0
        self._finished = False
        self._cleared = False

    # ------------------------------------------------------------------ 入口
    def emit(self, key: str, line: str) -> None:
        """追加一行到指定阶段的面板；未启用面板时直接打印（退化行为）。"""
        line = line.rstrip("\n").rstrip("\r")
        with self._lock:
            if key not in self._buf:
                key = self._panels[0][0] if self._panels else ""
            self._buf[key].append(line)
        if not self._enabled:
            print(line, file=self._out, flush=True)
            return
        self._maybe_draw()

    def finish(self) -> None:
        """结束面板模式：重绘最终状态 → 离开面板区 → 逐段打印全量日志。"""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            snapshot = {k: list(v) for k, v in self._buf.items()}
        if self._enabled:
            self.draw()
            # 在面板区下方留出空白，之后的普通输出不再回卷覆盖面板
            self._out.write("\n" * (len(self._panels) * (self._lines + 1) + 1))
            self._out.flush()
        for key, title in self._panels:
            self._out.write(f"===== {title}（全量日志） =====\n")
            for line in snapshot.get(key, []):
                self._out.write(line + "\n")
                if self._sink is not None:
                    try:
                        self._sink(line)
                    except Exception:  # noqa: BLE001 —— 日志镜像失败不影响输出
                        pass
        self._out.flush()

    # ------------------------------------------------------------------ 渲染
    def _maybe_draw(self) -> None:
        now = time.monotonic()
        if now - self._last_draw < self._interval:
            return
        self.draw()

    def draw(self) -> None:
        """把整个面板屏幕重写到终端（回到左上角 + 清尾）。"""
        if not self._enabled:
            return
        with self._lock:
            self._last_draw = time.monotonic()
            text = self._compose_locked()
            if not self._cleared:
                self._out.write("\x1b[2J")
                self._cleared = True
            self._out.write("\x1b[H" + text + "\x1b[J")
            self._out.flush()

    def render(self) -> str:
        """返回当前面板屏幕文本（测试用），与终端宽高无关。"""
        with self._lock:
            return self._compose_locked()

    def _compose_locked(self) -> str:
        try:
            cols, rows = shutil.get_terminal_size((100, 24))
        except Exception:  # noqa: BLE001
            cols, rows = 100, 24
        # 每模块固定面积：标题 1 行 + 内容行数（全程恒定，不随运行内容变化）
        lines = max(2, min(self._lines,
                           max(2, (rows - 2) // max(1, len(self._panels)))))
        blocks = []
        for key, title in self._panels:
            hist = self._buf.get(key, [])
            hidden = len(hist) - lines
            if hidden > 0:
                # 滚动形式：顶部指示"已滚动 N 行"，窗口内展示最新行
                content = [f"…（{hidden} 行已滚动）"] + hist[-(lines - 1):]
            else:
                content = list(hist)
            content = [compact_line(l) for l in content]
            content += [""] * (lines - len(content))     # 补空行 = 面积恒定
            block_lines = [f"── {title} ──"]
            block_lines += [l[: cols - 1] for l in content]
            blocks.append("\n".join(block_lines))
        return "\n".join(blocks)