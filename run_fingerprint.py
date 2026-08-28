#!/usr/bin/env python3
"""LLM 行为指纹验证工具 —— 入口。

用法:
    python run_fingerprint.py --config config.json
    python run_fingerprint.py --config config.json --official-key sk-xxx --unknown-key sk-yyy
    python run_fingerprint.py --config config.json --no-phase3   # 跳过 logprob 阶段
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# 特殊环境（如无法用 pip 安装依赖的沙箱）可将依赖装在 .deps/ 下
# （参见 run_pip.py）；存在时自动加入 sys.path，普通环境不受影响。
_LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(_LOCAL_DEPS):
    sys.path.insert(0, _LOCAL_DEPS)

from fpcheck.config import Config
from fpcheck.console import PhasePanels, panels_supported
from fpcheck.ratelimit import AdaptivePolicy, RateLimiter
from fpcheck.redflags import detect_redflags
from fpcheck.report import write_report
from fpcheck.runlog import RunLogger
from fpcheck.runner import call_pair
from fpcheck.storage import Recorder
from fpcheck.target import ProbeTarget
from fpcheck.verdict import calibrate, combine
from fpcheck import __version__, phase1, phase2, phase3, phase4, phase5, phase6

# 运行日志（每次运行落 output/logs/run_<时间戳>.log + .json）
RLOG: RunLogger | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="未知 API vs 官方 API 行为指纹验证工具")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}（LLM 行为指纹验证工具）")
    p.add_argument("--config", default="config.json", help="配置文件路径（默认 config.json）")
    p.add_argument("--official-key", help="官方 API key（覆盖配置文件）")
    p.add_argument("--unknown-key", help="未知 API key（覆盖配置文件）")
    p.add_argument("--official-url", help="官方 base_url（覆盖配置文件）")
    p.add_argument("--unknown-url", help="未知 base_url（覆盖配置文件）")
    p.add_argument("--official-model", help="官方模型名（覆盖配置文件）")
    p.add_argument("--unknown-model", help="未知模型名（覆盖配置文件）")
    p.add_argument("--official-api-style", choices=["chat", "responses"],
                   help="官方调用风格：chat=/chat/completions，responses=/responses（覆盖配置文件）")
    p.add_argument("--unknown-api-style", choices=["chat", "responses"],
                   help="未知调用风格：chat=/chat/completions，responses=/responses（覆盖配置文件）")
    p.add_argument("--output-dir", help="输出目录（覆盖配置文件）")
    p.add_argument("--proxy", help="代理地址，如 http://127.0.0.1:7897（覆盖配置文件；传空串=直连）")
    p.add_argument("--no-phase1", action="store_true", help="跳过第一阶段")
    p.add_argument("--no-phase2", action="store_true", help="跳过第二阶段")
    p.add_argument("--no-phase3", action="store_true", help="跳过第三阶段")
    p.add_argument("--no-phase4", action="store_true", help="跳过第四阶段（分词器行为指纹）")
    p.add_argument("--no-phase5", action="store_true", help="跳过第五阶段（采样自洽性画像）")
    p.add_argument("--no-phase6", action="store_true", help="跳过第六阶段（单 Token 行为指纹）")
    p.add_argument("--parallel-phases", action="store_true",
                   help="显式开启并行阶段（默认已开启，此参数仅用于覆盖配置文件里的关闭设置）")
    p.add_argument("--serial", action="store_true",
                   help="强制串行执行阶段（覆盖配置里的 parallel_phases=true）")
    p.add_argument("--no-concurrent", action="store_true",
                   help="同一探测改为依次发给两个 API（覆盖配置里的 concurrent=true）")
    p.add_argument("--no-panel", action="store_true",
                   help="关闭分区进度面板，退化为普通逐行日志")
    p.add_argument("--no-cache-official", action="store_true",
                   help="关闭官方侧响应缓存（默认开：复用同目录下官方取证，只实时打未知侧）")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑：复用输出目录中参数匹配且成功的取证记录，跳过已完成探测")
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)
    if RLOG is not None:
        RLOG.raw(msg)


# 对照基线 prompt（覆盖低中高熵任务，各 6 个；官方自比/已知异源对照用）
BASELINE_PROMPTS: list[str] = [
    "Describe the sky in exactly 3 words.",
    "What is 0.1 + 0.2? Give the exact decimal value.",
    "Which company or organization developed you? Answer directly.",
    "Repeat exactly: 3.141592653589793238462643383279",
    "Explain in one sentence how a magnet works.",
    "你是由哪家公司开发的？请直接回答。",
]


def _early_fail(args: argparse.Namespace, msg: str) -> None:
    """早期失败（运行日志尚未初始化）：用默认 output/ 目录也落一次失败日志，
    便于维护时回溯"连配置都没加载出来"的阶段。"""
    global RLOG
    try:
        RLOG = RunLogger("output", version=__version__, args=vars(args))
        RLOG.raw(f"错误: {msg}")
        RLOG.finish(rc=1, note=msg)
    except Exception:  # noqa: BLE001 —— 日志故障不掩盖原始错误
        pass
    sys.exit(f"错误: {msg}")


def main() -> None:
    try:  # Windows 控制台默认 GBK，提前切到 UTF-8 防止希腊/西里尔字符报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        try:  # 启用 Windows 终端的 VT/ANSI 支持（分区面板需要光标控制）
            os.system("")
        except Exception:  # noqa: BLE001
            pass

    args = parse_args()
    if not os.path.exists(args.config):
        _early_fail(args, f"找不到配置文件 {args.config}"
                    f"（可复制 config.example.json 为 config.json 后填写）")
    overrides = {k: v for k, v in {
        "official_key": args.official_key, "unknown_key": args.unknown_key,
        "official_url": args.official_url, "unknown_url": args.unknown_url,
        "official_model": args.official_model, "unknown_model": args.unknown_model,
        "official_api_style": args.official_api_style,
        "unknown_api_style": args.unknown_api_style,
        "output_dir": args.output_dir,
    }.items() if v}
    try:
        cfg = Config.load(args.config, overrides)
    except Exception as exc:  # noqa: BLE001 —— 配置损坏/缺字段等都会在这里失败
        _early_fail(args, f"配置加载失败（{exc}）")
    if args.proxy is not None:           # --proxy 覆盖配置（含 --proxy "" 强制直连）
        cfg.options.proxy = args.proxy
    if args.no_phase1:
        cfg.options.phase1_enabled = False
    if args.no_phase2:
        cfg.options.phase2_enabled = False
    if args.no_phase3:
        cfg.options.phase3_enabled = False
    if args.no_phase4:
        cfg.options.phase4_enabled = False
    if args.no_phase5:
        cfg.options.phase5_enabled = False
    if args.no_phase6:
        cfg.options.phase6_enabled = False
    if args.parallel_phases:
        cfg.options.parallel_phases = True
    if args.serial:
        cfg.options.parallel_phases = False
    if args.no_concurrent:
        cfg.options.concurrent = False
    if args.no_cache_official:
        cfg.options.cache_official = False

    # 运行日志：记录环境、脱敏配置、全流程进度与结论（写故障不影响主流程）
    global RLOG
    try:
        RLOG = RunLogger(cfg.options.output_dir, version=__version__,
                         args=vars(args),
                         official=cfg.official, unknown=cfg.unknown,
                         options=vars(cfg.options))
    except Exception:  # noqa: BLE001
        RLOG = None
        log("警告: 运行日志初始化失败，本次运行不落日志。")

    if cfg.official.api_key == "EMPTY" or cfg.unknown.api_key == "EMPTY":
        log("警告: 某个目标未配置 api_key（已用占位符），若端点需要鉴权将预检失败。")

    log(_rule("="))
    log(f" LLM 行为指纹验证工具 v{__version__}")
    log(_rule("="))
    log(f"官方 API: {cfg.official.name}  model={cfg.official.model_name}  "
        f"端点={cfg.official.base_url}  风格={cfg.official.api_style}")
    log(f"未知 API: {cfg.unknown.name}  model={cfg.unknown.model_name}  "
        f"端点={cfg.unknown.base_url}  风格={cfg.unknown.api_style}")
    log(f"网络通道: {'代理 ' + cfg.options.proxy if cfg.options.proxy else '直连'}")
    rpm = cfg.options.rate_limit_rpm
    # 全局限速器：官方与未知共用同一实例，所有阶段共享，
    # 保证双向并行 + 多阶段并行不会突发到模型端点的并发限制。
    # 自动模式（默认）：不内置型号限额表，按端点实时响应头 x-ratelimit-*
    # 决定限度（无头时回退到 rate_limit_rpm / fallback_tpm），429 自适应加倍。
    if cfg.options.auto_ratelimit:
        policy = AdaptivePolicy(fallback_rpm=rpm,
                                fallback_tpm=cfg.options.fallback_tpm)
        limiter = RateLimiter(min_interval=0.0,
                              max_inflight=cfg.options.max_inflight,
                              policy=policy)
        limit_label = policy.describe()
    else:
        limiter = RateLimiter(min_interval=(60.0 / rpm) if rpm > 0 else 0.0,
                              max_inflight=cfg.options.max_inflight)
        limit_label = f"固定 {rpm} RPM" if rpm > 0 else "不限速"
    log(f"并发配置: 双API并行={'开' if cfg.options.concurrent else '关'}，"
        f"阶段并行={'开' if cfg.options.parallel_phases else '关'}，"
        f"全局限速={limit_label}（并发上限 {cfg.options.max_inflight}）")

    official = ProbeTarget(cfg.official.name, cfg.official.base_url,
                           cfg.official.api_key, cfg.official.model_name,
                           timeout=cfg.options.request_timeout,
                           proxy=cfg.options.proxy or None,
                           api_style=cfg.official.api_style,
                           rate_limiter=limiter)
    unknown = ProbeTarget(cfg.unknown.name, cfg.unknown.base_url,
                          cfg.unknown.api_key, cfg.unknown.model_name,
                          timeout=cfg.options.request_timeout,
                          proxy=cfg.options.proxy or None,
                          api_style=cfg.unknown.api_style,
                          rate_limiter=limiter)

    # 预检：连通性 / 鉴权 / 模型可用性，快速失败
    log("预检: 连通性/鉴权/模型可用性 ...")
    for t in (official, unknown):
        err = t.ping()
        if err:
            msg = f"错误: {t.name} 预检失败（{err}）。" \
                  f"请检查 base_url / api_key / model_name 后重试。"
            log(msg)
            if RLOG is not None:
                RLOG.finish(rc=1, note="预检失败")
            sys.exit(msg)
    log("预检通过。")

    # 官方侧缓存：同 output_dir 下参数匹配的官方取证直接复用（端点/模型一致才
    # 生效），只实时请求未知侧，网络请求减半；加 --no-cache-official 关闭。
    reuse_names = {cfg.official.name} if (cfg.options.cache_official
                                          and not args.no_cache_official) else None
    recorder = Recorder(cfg.options.output_dir, resume=args.resume,
                        reuse_names=reuse_names)
    if args.resume:
        log(f"断点续跑：已加载 {recorder.cached_count} 条取证记录，"
            f"参数匹配且成功的探测将复用本地结果、不重发请求。")
    elif reuse_names:
        log(f"官方侧缓存：已加载 {recorder.cached_count} 条取证记录，"
            f"官方 {cfg.official.name} 侧的匹配探测将复用本地结果（端点/模型"
            f"一致才生效），只实时请求未知侧；--no-cache-official 可关闭。")
    recorder.save_manifest(cfg)

    # ---- 对照基线（不参与判定，只作报告参照系）：官方自比 + 已知异源模型 ----
    baselines: dict = {"self_sim": None, "diff_sim": None,
                       "diff_model": cfg.options.baseline_diff_model}

    def run_baseline(target_a, target_b, label: str) -> Optional[float]:
        sims = []
        for idx, prompt in enumerate(BASELINE_PROMPTS):
            ra, rb = call_pair(recorder, target_a, target_b, phase=0,
                               probe_id=f"base_{label}_{idx:02d}",
                               prompt=prompt, temperature=0.0,
                               max_tokens=128, seed=None, n=1,
                               concurrent=False)
            if ra.ok and rb.ok and ra.texts and rb.texts:
                sims.append(phase1._similarity(ra.texts[0], rb.texts[0]))
        return sum(sims) / len(sims) if sims else None

    if cfg.options.baseline_self:
        ref = ProbeTarget("官方自比", cfg.official.base_url, cfg.official.api_key,
                          cfg.official.model_name,
                          timeout=cfg.options.request_timeout,
                          proxy=cfg.options.proxy or None,
                          api_style=cfg.official.api_style, rate_limiter=limiter)
        try:
            baselines["self_sim"] = run_baseline(official, ref, "self")
        except Exception as exc:  # noqa: BLE001
            log(f"对照基线: 官方自比失败（{exc}）")
        s = baselines["self_sim"]
        if s is not None:
            log(f"对照基线: 官方自比平均相似度 = {s:.3f}"
                "（同端点同模型应接近 1；显著偏低提示官方侧自身不稳定）")
    if cfg.options.baseline_diff_model:
        known = ProbeTarget("已知异源", cfg.official.base_url, cfg.official.api_key,
                            cfg.options.baseline_diff_model,
                            timeout=cfg.options.request_timeout,
                            proxy=cfg.options.proxy or None,
                            api_style=cfg.official.api_style, rate_limiter=limiter)
        d = run_baseline(official, known, "diff")
        baselines["diff_sim"] = d
        if d is not None:
            log(f"对照基线: 已知异源（{cfg.options.baseline_diff_model}）平均相似度 = {d:.3f}"
                "（同源对应显著低于自比基线，否则需复核指标体系）")
        else:
            log(f"对照基线: 已知异源（{cfg.options.baseline_diff_model}）无法调用，跳过")

    # ---- 分区进度面板：并行时每个阶段独占一块区域，互不干扰 ----
    # 非终端（重定向/管道，如脚本调用与 CI）或 --no-panel 时退化为普通日志。
    _PHASE_TITLE = {
        "phase1": "第一阶段 行为对齐",
        "phase2": "第二阶段 对抗后缀指纹",
        "phase3": "第三阶段 logprob 数值",
        "phase4": "第四阶段 分词器行为指纹",
        "phase5": "第五阶段 采样自洽性画像",
        "phase6": "第六阶段 单Token行为指纹",
    }
    _phase_flags = {
        "phase1": cfg.options.phase1_enabled,
        "phase2": cfg.options.phase2_enabled,
        "phase3": cfg.options.phase3_enabled,
        "phase4": cfg.options.phase4_enabled,
        "phase5": cfg.options.phase5_enabled,
        "phase6": cfg.options.phase6_enabled,
    }
    panels = None
    if cfg.options.parallel_phases and not args.no_panel \
            and panels_supported():
        panels = PhasePanels(
            [(k, _PHASE_TITLE[k]) for k, on in _phase_flags.items() if on],
            sink=None if RLOG is None else RLOG.raw)
    elif cfg.options.parallel_phases and not args.no_panel \
            and sys.stdout.isatty():
        log("提示: 当前终端不支持全屏刷新，分区进度面板未启用（使用普通"
            "逐行日志；Windows Terminal / VSCode 终端可开启；--no-panel 强制关闭）")

    # 阶段任务表：五个阶段互不依赖（红旗复用第一阶段取证，判定在全部阶段
    # 之后进行），因此可以并行执行。parallel_phases=true 时并发执行，
    # 总时间 ≈ 最慢阶段；进度输出按阶段进各自面板，互不干扰。
    tasks: list[tuple[str, str, object]] = []

    def make_logger(key: str):
        def pl(line: str) -> None:
            if panels is not None:
                panels.emit(key, line)
            else:
                log(line)
        return pl

    def add_task(key: str, label: str, enabled: bool, fn) -> None:
        if enabled:
            tasks.append((key, label, fn))
        else:
            log(f"[第{label}阶段] 已禁用，跳过。")

    add_task("phase1", "一", cfg.options.phase1_enabled,
             lambda: phase1.run_phase1(official, unknown, recorder, cfg.options,
                                       make_logger("phase1")))
    add_task("phase2", "二", cfg.options.phase2_enabled,
             lambda: phase2.run_phase2(official, unknown, recorder, cfg.options,
                                       make_logger("phase2")))
    add_task("phase3", "三", cfg.options.phase3_enabled,
             lambda: phase3.run_phase3(official, unknown, recorder, cfg.options,
                                       make_logger("phase3")))
    add_task("phase4", "四", cfg.options.phase4_enabled,
             lambda: phase4.run_phase4(official, unknown, recorder, cfg.options,
                                       make_logger("phase4")))
    add_task("phase5", "五", cfg.options.phase5_enabled,
             lambda: phase5.run_phase5(official, unknown, recorder, cfg.options,
                                       make_logger("phase5")))
    add_task("phase6", "六", cfg.options.phase6_enabled,
             lambda: phase6.run_phase6(official, unknown, recorder, cfg.options,
                                       make_logger("phase6")))

    results: dict[str, object] = {}
    phase_times: dict[str, float] = {}

    def _timed(key: str, fn):
        """记录每个阶段的墙钟耗时（维护：定位"哪个阶段变慢/卡住"）。"""
        t0 = time.monotonic()
        try:
            return fn()
        finally:
            phase_times[key] = time.monotonic() - t0

    if cfg.options.parallel_phases and len(tasks) >= 2:
        log(f"并行模式：{len(tasks)} 个阶段同时执行，总时间 ≈ 最慢阶段；"
            f"瞬时请求率由全局限速器压制（--serial 可关闭）。")
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(_timed, key, fn): key for key, _, fn in tasks}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as exc:  # noqa: BLE001 —— 单阶段异常不阻断其他阶段
                    results[key] = None
                    log(f"[阶段 {key}] 异常退出，该阶段按缺失处理: "
                        f"{type(exc).__name__}: {exc}")
    else:
        for key, _, fn in tasks:
            results[key] = _timed(key, fn)

    # 启用但未产生结果（异常退出）的阶段显式记为错误，便于回溯
    for key in _phase_flags:
        if _phase_flags.get(key) and results.get(key) is None:
            log(f"错误: 阶段 {key} 启用但未产生结果（运行异常），该阶段按缺失处理")

    if panels is not None:
        panels.finish()   # 重绘最终状态 + 逐段打印全量日志

    r1 = results.get("phase1")
    r2 = results.get("phase2")
    r3 = results.get("phase3")
    r4 = results.get("phase4")
    r5 = results.get("phase5")
    r6 = results.get("phase6")

    scores = {
        "phase1": r1.score if r1 is not None else None,
        "phase2": r2.score if r2 is not None else None,
        "phase3": r3.score if r3 is not None else None,
        "phase4": r4.score if r4 is not None else None,
        "phase5": r5.score if r5 is not None else None,
        # 阶段六为诊断性阶段：分数不进入综合判定（推理型端点下不可靠），
        # 但其退化现象（推理型/不可读/读取通道）保留在报告与 summary
    }
    # 官方自比归一化校准：相似度类阶段（一/四/六）得分 ÷ self_sim（≤1 截断）。
    # 自比 = 官方端点自身的噪声地板/同源相似度上限；未跑或过低时自动跳过。
    calibration = None
    self_sim = baselines.get("self_sim")
    if self_sim is not None and self_sim >= 0.5:
        scores, calibration = calibrate(scores, self_sim)
        if calibration is not None:
            log(f"官方自比校准: self_sim={self_sim:.3f}，"
                f"阶段 {'/'.join(calibration['phases'])} 得分 ÷{self_sim:.3f}（≤1 截断）")
    elif self_sim is not None:
        log(f"警告: 官方自比基线过低（{self_sim:.3f} < 0.5），跳过归一化校准"
            "（官方侧本身不稳定，不放大噪声）")
    # 身份一致性红旗（复用第一阶段取证，零额外请求）
    redflags = detect_redflags(r1, cfg.official.model_name, cfg.unknown.model_name,
                               phase2=r2)
    if redflags.has_flags:
        log(f"身份一致性红旗 {len(redflags.flags)} 项（详见报告）:")
        for flag in redflags.flags:
            log(f"  ⚑ {flag}")
    else:
        log("身份一致性检查: 未发现红旗。")

    evidence = {}
    if r1 is not None:
        evidence["phase1"] = {"ok": sum(1 for p in r1.prompts
                                        if not p.error_a and not p.error_b),
                              "total": len(r1.prompts)}   # 只计评分类别，参考行不计
    if r2 is not None:
        evidence["phase2"] = {"ok": r2.succeeded, "total": r2.num_probes,
                              "reliability": r2.reliability}
    if r3 is not None and r3.score is not None:
        evidence["phase3"] = {"ok": len(r3.prompts), "total": len(r3.prompts)}
    if r4 is not None:
        evidence["phase4"] = {"ok": sum(1 for p in r4.prompts
                                        if not p.error_a and not p.error_b),
                              "total": r4.total}
    if r5 is not None and r5.score is not None:
        evidence["phase5"] = {"ok": r5.succeeded, "total": r5.total,
                              "reliability": r5.reliability}
    # 阶段六为诊断性阶段，不进入证据计数/fail-closed/证据加权
    verdict = combine(scores, evidence)
    if verdict.conflict:
        log(f"警告: 各阶段证据分歧显著（得分跨度 {verdict.phase_spread:.2f}），"
            f"综合判定已降级/标注冲突。")
    text, path = write_report(cfg, r1, r2, r3, r4, r5, r6, redflags, verdict,
                              baselines, calibration)

    log("")
    log(_rule("="))
    log(" 验证完成")
    log(_rule("="))
    if scores["phase1"] is not None:
        log(f"第一阶段  行为对齐相似度   : {scores['phase1']:.3f}")
    else:
        log("第一阶段  行为对齐相似度   : 跳过")
    if scores["phase2"] is not None:
        log(f"第二阶段  对抗指纹相似度   : {scores['phase2']:.3f}")
    else:
        log("第二阶段  对抗指纹相似度   : 跳过")
    if scores["phase3"] is not None:
        log(f"第三阶段  logprob 重叠率   : {scores['phase3']:.3f}（辅助指标）")
    else:
        log("第三阶段  logprob 重叠率   : 跳过")
    if scores["phase4"] is not None:
        log(f"第四阶段  分词器行为指纹   : {scores['phase4']:.3f}")
    else:
        log("第四阶段  分词器行为指纹   : 跳过")
    if scores["phase5"] is not None:
        log(f"第五阶段  采样自洽性画像   : {scores['phase5']:.3f}")
    else:
        log("第五阶段  采样自洽性画像   : 跳过")
    if r6 is not None and r6.score is not None:
        log(f"第六阶段  单Token分布重叠（诊断参考）: {r6.score:.3f}"
            + (f"（{r6.skipped}）" if r6.skipped else ""))
    else:
        log("第六阶段  单Token分布重叠（诊断参考）: 跳过"
            + (f"（{r6.skipped}）" if r6 is not None and r6.skipped else ""))
    log(f"综合评分 = {verdict.score:.3f} → {verdict.level_cn}")
    log(f"判定结论: {verdict.summary}")
    log(f"报告已保存: {path}")
    log(f"原始取证数据: {os.path.join(cfg.options.output_dir, 'raw')}")
    if RLOG is not None:
        RLOG.finish(rc=0, verdict=verdict, scores=scores,
                    report_path=path,
                    summary_path=os.path.join(cfg.options.output_dir,
                                              "summary.json"),
                    phase_elapsed=phase_times)
        log(f"运行日志: {RLOG.log_path}（结构化摘要: {RLOG.json_path}）")


def _rule(char: str, width: int = 60) -> str:
    return char * width


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if RLOG is not None:
            RLOG.finish(rc=130, note="用户中断（Ctrl+C）")
        raise
    except SystemExit as e:
        # 正常路径已 finish；这里是 sys.exit 直达（如配置缺失等前置失败）
        if RLOG is not None:
            code = e.code if isinstance(e.code, int) else 1
            RLOG.finish(rc=code, note="异常退出（sys.exit）")
        raise
    except BaseException:
        if RLOG is not None:
            RLOG.finish(rc=2, note="未捕获异常（详见控制台 Traceback）")
        raise
