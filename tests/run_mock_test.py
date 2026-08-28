"""端到端自测脚本（无需真实 API key，全部走本地 mock 端点）。

五个探测阶段: 一 行为对齐相似度 / 二 对抗后缀指纹 / 三 logprob（辅助）/
四 分词器行为指纹 / 五 采样自洽性画像（确定性端点自动跳过）。

七个场景:
  A) 未知 API 与官方"同源"，但不支持 seed/logprobs 参数
     → 预期: 第一/二/四阶段得分 1.0，第三/五阶段跳过，判定"高度确信同一底层模型"
  B) 未知 API 与官方"异源"（不同响应逻辑）
     → 预期: 各阶段得分均低，综合判定"证据不足"
  C) 未知 API 与官方"同源"且支持全部参数
     → 预期: 一~四阶段得分均 1.0，仅阶段五跳过（确定性 mock），判定"高度确信"
  E) 未知 API 拒绝 temperature 参数（400），其余参数支持
     → 预期: 自动降级去掉 temperature，各阶段照常评分，capabilities 记录 temperature=False
  F) 双方"同源"且都有采样方差（noisy 端点，方差模式一致）
     → 预期: 第五阶段画像相关 ≈ 1.0，综合评分 1.0，判定"高度确信"
  G) 双方文本高度相似但采样方差模式不同（noisy label 端点）
     → 预期: 第五阶段画像相关显著 < 0.9（唯一能抓住该差异的通道）
  H) 场景 C 的配置但用默认配置（双维度并行默认同时开启 + 全局限速器）
     → 预期: 各阶段结果与全串行完全一致（判定 1.0、仅 phase5 跳过），
            日志/manifest 确认并发配置生效

用法: python tests/run_mock_test.py
"""
from __future__ import annotations

import glob
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_server  # noqa: E402

# Windows 控制台默认 GBK，提前切到 UTF-8（与 run_fingerprint.py 入口一致）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
failures: list[str] = []


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(port: int, mode: str, label: str, reject_seed=False,
          reject_logprobs=False, reject_n=False, reject_temperature=False,
          noise="shared", blank_on_max1=False):
    handler = mock_server.make_handler(mode, label, reject_seed,
                                       reject_logprobs, reject_n,
                                       reject_temperature, noise,
                                       blank_on_max1)
    srv = mock_server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def run_scenario(name: str, unknown_cfg: dict,
                 official_port: int | None = None,
                 extra_options: dict | None = None,
                 out_dir: str | None = None) -> tuple[int, dict, str]:
    """运行 run_fingerprint.py（子进程），返回 (returncode, summary, out_dir)。

    输出目录默认用 tempfile 创建；若设置了环境变量 FPCHECK_TEST_OUT_ROOT，
    则改用其下固定的 tests/.e2e/<name> 目录（部分沙箱环境不允许
    向 mkdtemp 创建的目录写入，需要预建目录）。传入 out_dir 时复用该目录
    （用于官方侧缓存等跨运行场景，不做清理）。
    """
    root = os.environ.get("FPCHECK_TEST_OUT_ROOT")
    if out_dir is not None:
        out = out_dir
    elif root:
        out = os.path.join(root, name)
        if os.path.exists(out):          # 预建目录模式下先清理，保证结果确定性
            import shutil
            shutil.rmtree(out)
        os.makedirs(out)
    else:
        out = tempfile.mkdtemp(prefix=f"fpcheck_{name}_")
    off_port = official_port if official_port is not None else PORT_OFFICIAL
    options = {"output_dir": os.path.join(out, "output"), "seed": 42,
               "phase2": {"num_probes": 6, "samples": 3},
               "rate_limit_rpm": 0, "fallback_tpm": 0}  # 自测默认不限速，避免拖慢整套测试
    if extra_options:
        options.update(extra_options)
    cfg = {
        "official": {"name": "官方GPT",
                     "base_url": f"http://127.0.0.1:{off_port}/v1",
                     "api_key": "sk-official", "model_name": "gpt-4o"},
        "unknown": unknown_cfg,
        "options": options,
    }
    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    logf = os.path.join(out, "run.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(logf, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "run_fingerprint.py"),
             "--config", cfg_path],
            stdout=f, stderr=subprocess.STDOUT, env=env)
    out_dir = cfg["options"]["output_dir"]
    summary_path = os.path.join(out_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    return proc.returncode, summary, out


def test_rate_limiter() -> None:
    """限速器单元自检：RPM 间隔与在飞并发上限。"""
    from fpcheck.ratelimit import RateLimiter
    lim = RateLimiter(min_interval=0.02, max_inflight=2)
    t0 = time.monotonic()
    for _ in range(5):          # 5 次应至少间隔 4×20ms
        with lim.slot():
            pass
    elapsed = time.monotonic() - t0
    check(elapsed >= 0.079,
          f"限速器: 5 次请求间隔 >= 4×20ms（实际 {elapsed * 1000:.0f}ms）")

    active: list[int] = [0]
    peak: list[int] = [0]
    lock = threading.Lock()

    def worker() -> None:
        with lim.slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(peak[0] <= 2, f"限速器: 在飞并发上限 2（实际峰值 {peak[0]}）")


def test_cutoff_redflags() -> None:
    """知识截止判定自检：语境过滤 / 区间句取单一声明 / 单端矛盾 / 跨端偏差。"""
    from types import SimpleNamespace as NS
    from fpcheck.redflags import _extract_cutoff_claims, detect_redflags

    # 语境过滤：成立于 2015 年不算截止声明；cutoff 2023 / 截止 2024 才算
    claims = _extract_cutoff_claims([
        "I was developed by OpenAI, founded in 2015. My knowledge cutoff is October 2023.",
        "我的知识截止于2024年4月。",
        "这家成立于 2012 年的公司还做搜索引擎。",
    ])
    check(claims == {"2023", "2024"},
          f"截止提取: 过滤无关年份（实际 {sorted(claims)}）")

    # 区间句每回答只取一个声明（最大值），不产生假矛盾
    claims_range = _extract_cutoff_claims([
        "我的训练数据从2019年一直到2023年12月。",
        "My training data covers 2020 up to 2023.",
    ])
    check(claims_range == {"2023"},
          f"区间句: 每回答只取一个声明（实际 {sorted(claims_range)}）")

    def mk(tag: str, ta: str, tb: str):
        return NS(category=tag, text_a=ta, text_b=tb, error_a="", error_b="",
                  latency_a=10.0, latency_b=10.0)

    # 单端矛盾：官方两个回答给出 2023/2024 → 红旗；未知声明一致 → 无
    rf = detect_redflags(NS(prompts=[
        mk("self_awareness", "My knowledge cutoff is October 2023.",
           "My knowledge cutoff is April 2024."),
        mk("self_awareness", "My knowledge cutoff is January 2024.",
           "My knowledge cutoff is April 2024."),
        mk("refusal_profile", "ok", "ok"),
    ]), "gpt-4o", "gpt-4o")
    check(any("官方" in f and "多个截止声明" in f for f in rf.flags),
          "单端矛盾: 官方多个截止声明触发红旗")
    check(not any("未知" in f and "多个截止声明" in f for f in rf.flags),
          "单端矛盾: 未知声明一致不触发")

    # 跨端系统性偏差：两端各自稳定但年份不同 + 同一品牌 → 红旗
    rf2 = detect_redflags(NS(prompts=[
        mk("self_awareness", "My knowledge cutoff is October 2023.",
           "My knowledge cutoff is April 2024."),
        mk("self_awareness", "My knowledge cutoff is October 2023.",
           "My knowledge cutoff is April 2024."),
    ]), "gpt-4o", "gpt-4o-fake")
    check(any("系统性不同" in f for f in rf2.flags), "跨端偏差: 红旗触发")

    # 不同品牌（官方 GPT vs 未知 Claude）即使年份不同也不触发跨端偏差
    rf3 = detect_redflags(NS(prompts=[
        mk("self_awareness", "My knowledge cutoff is October 2023.",
           "My knowledge cutoff is April 2024."),
    ]), "gpt-4o", "claude-3-5-sonnet")
    check(not any("系统性不同" in f for f in rf3.flags),
          "跨端偏差: 异品牌不触发")


def test_adaptive_policy() -> None:
    """自动限速策略自检：响应头换算、429 退让、无回退不限速。"""
    from fpcheck.ratelimit import AdaptivePolicy

    p = AdaptivePolicy(fallback_rpm=120, fallback_tpm=30000)
    i_before = p.interval(est_tokens=100)      # 回退: max(60/96, 100*60/24000)=0.625
    p.observe_headers("official", {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-reset-requests": "60",
        "x-ratelimit-limit-tokens": "100000",
        "x-ratelimit-reset-tokens": "60",
    })
    i_after = p.interval(est_tokens=100)       # 响应头: max(60/800, 100*60/80000)=0.075
    check(i_after > 0 and i_after < i_before,
          f"策略: 响应头额度生效后间隔收紧（{i_after:.3f}s < {i_before:.3f}s）")
    p.note_throttled()                          # 429 → 加倍保守
    check(abs(p.interval(est_tokens=100) - 2 * i_after) < 1e-9,
          "策略: 429 后间隔加倍")
    check(AdaptivePolicy(fallback_rpm=0, fallback_tpm=0).interval(0) == 0.0,
          "策略: 无回退值时不限速")


def test_verdict_weights() -> None:
    """综合判定自检：权重封顶、单通道不能独推、证据加权回调、证据冲突降级。"""
    from fpcheck.verdict import BASE_WEIGHTS, combine

    check(BASE_WEIGHTS["phase2"] == 0.30 and BASE_WEIGHTS["phase4"] == 0.10
          and BASE_WEIGHTS["phase5"] == 0.30
          and "phase6" not in BASE_WEIGHTS and max(BASE_WEIGHTS.values()) <= 0.30,
          "权重: 阶段二/五 0.30 封顶、阶段四家族级 0.10、阶段六不进判定")
    # 阶段二满分、其余各 0.5 → 只能是中等怀疑，不能"高度确信"
    v = combine(
        {"phase1": 0.5, "phase2": 1.0, "phase3": None, "phase4": 0.5,
         "phase5": 0.5},
        {"phase1": {"ok": 30, "total": 32}, "phase2": {"ok": 50, "total": 50},
         "phase4": {"ok": 16, "total": 16}, "phase5": {"ok": 16, "total": 16}})
    check(v.score < 0.75 and v.level == "medium",
          f"权重: 单通道满分不足以推高结论（score={v.score:.3f}）")
    # 证据加权：阶段二成功率 80% 时话语权被回调（对比 100% 情形）
    ev_hi = {"phase1": {"ok": 30, "total": 32}, "phase2": {"ok": 50, "total": 50}}
    ev_lo = {"phase1": {"ok": 30, "total": 32}, "phase2": {"ok": 40, "total": 50}}
    v_hi = combine({"phase1": 0.0, "phase2": 1.0, "phase3": None,
                    "phase4": None, "phase5": None}, ev_hi)
    v_lo = combine({"phase1": 0.0, "phase2": 1.0, "phase3": None,
                    "phase4": None, "phase5": None}, ev_lo)
    check(v_lo.score < v_hi.score,
          f"证据加权: 成功率低的阶段话语权被回调（{v_lo.score:.3f} < {v_hi.score:.3f}）")
    # CI 可靠性入权：阶段二 reliability 0.5 时话语权低于 1.0
    ev_r1 = {"phase1": {"ok": 30, "total": 32},
             "phase2": {"ok": 50, "total": 50, "reliability": 1.0}}
    ev_r5 = {"phase1": {"ok": 30, "total": 32},
             "phase2": {"ok": 50, "total": 50, "reliability": 0.5}}
    vr1 = combine({"phase1": 0.0, "phase2": 1.0, "phase3": None,
                   "phase4": None, "phase5": None}, ev_r1)
    vr5 = combine({"phase1": 0.0, "phase2": 1.0, "phase3": None,
                   "phase4": None, "phase5": None}, ev_r5)
    check(vr5.score < vr1.score,
          f"证据加权: CI 可靠性低的阶段话语权被回调（{vr5.score:.3f} < {vr1.score:.3f}）")
    # 证据冲突：加权分 >0.75 但阶段跨度大 → 降级中等怀疑并标注
    vc = combine(
        {"phase1": 1.0, "phase2": 0.45, "phase3": None, "phase4": 0.9,
         "phase5": 1.0},
        {"phase1": {"ok": 30, "total": 32},
         "phase2": {"ok": 45, "total": 50, "reliability": 1.0},
         "phase4": {"ok": 16, "total": 16},
         "phase5": {"ok": 16, "total": 16}})
    check(vc.score > 0.75 and vc.conflict and vc.level == "medium"
          and abs(vc.phase_spread - 0.55) < 1e-9,
          f"冲突: 高分但阶段跨度大 → 降级中等怀疑（score={vc.score:.3f}, "
          f"spread={vc.phase_spread:.2f}, level={vc.level}）")


def test_phase_panels() -> None:
    """分区进度面板自检：固定面积、滚动指示、行精简、区域隔离、退化行为。"""
    import io
    from fpcheck.console import PhasePanels, compact_line

    out = io.StringIO()
    p = PhasePanels([("p1", "第一阶段"), ("p2", "第二阶段")], out=out,
                    lines_per_panel=2, force=True)
    p.emit("p1", "alpha-1")
    p.emit("p1", "alpha-2")
    p.emit("p2", "beta-1")
    p.emit("p2", "beta-2")
    p.emit("p1", "alpha-3")
    screen = p.render()
    block1, _, block2 = screen.partition("\n── 第二阶段 ──\n")
    check("alpha-3" in block1 and "alpha-2" not in block1 and "beta" not in block1
          and "行已滚动" in block1,
          "面板: 阶段一起始即固定面积，溢出走滚动指示")
    check("beta-2" in block2 and "alpha" not in block2,
          "面板: 阶段二区域只含自己的内容")
    check(len(screen.splitlines()) == 6,   # 每模块 = 标题1 + 内容2 行，恒定
          f"面板: 固定面积 6 行（实际 {len(screen.splitlines())}）")
    check("\x1b[" in out.getvalue(), "面板: 使用 ANSI 控制序列重绘")
    # 行精简：冗余字段裁剪，错误行原样保留
    c1 = compact_line("  [p6_00] 短串='x'  重叠率=1.000  官方首token='π'  "
                      "未知首token='π'")
    check("p6_00" in c1 and "重叠率=1.000" in c1
          and "官方首token" not in c1 and "短串='x'" in c1,
          f"面板: 探测行精简（{c1!r}）")
    c2 = compact_line("  [p6_01] 跳过（错误: 首 token 不可读：content 为空且无 logprobs）")
    check("错误: 首 token 不可读" in c2, "面板: 错误行保持完整")
    c3 = compact_line("  [adv_000] 后缀长度=47  包装=sys_basic  交叉相似度=0.932")
    check("adv_000" in c3 and "交叉相似度=0.932" in c3
          and "后缀长度" not in c3 and "包装" not in c3,
          f"面板: 阶段二探测行精简（{c3!r}）")
    p.finish()
    full = out.getvalue()
    check("===== 第一阶段（全量日志） =====" in full
          and "alpha-1" in full and "beta-2" in full,
          "面板: finish 后按阶段逐段打印全量日志（含已滚动行）")
    out2 = io.StringIO()
    p2 = PhasePanels([("p1", "第一阶段")], out=out2)   # 非终端且未 force
    p2.emit("p1", "plain-line")
    check(out2.getvalue() == "plain-line\n", "面板: 非终端退化为普通逐行输出")


def test_phase2_upsample() -> None:
    """边界补测单测：模糊区触发、明确同源/异源/收敛不触发。"""
    deps = os.path.join(ROOT, ".deps")          # 沙箱环境依赖在 .deps/ 下
    if os.path.isdir(deps):
        sys.path.insert(0, deps)
    from fpcheck.phase2 import ProbeRow, needs_upsample

    def row(cross: float, intra_a: float, intra_b: float) -> ProbeRow:
        return ProbeRow(probe_id="x", prefix="", suffix_len=1,
                        cross_sim=cross, intra_a=intra_a, intra_b=intra_b,
                        texts_a=["a", "b", "c"], texts_b=["d", "e", "f"])

    check(needs_upsample(row(0.60, 0.8, 0.8), 0.35, 0.85, 3), "补测: 模糊区触发补测")
    check(not needs_upsample(row(1.00, 0.9, 0.9), 0.35, 0.85, 3), "补测: 明确同源不补测")
    check(not needs_upsample(row(0.10, 0.8, 0.8), 0.35, 0.85, 3), "补测: 明确异源不补测")
    check(not needs_upsample(row(0.60, 1.0, 1.0), 0.35, 0.85, 3), "补测: 收敛探测不补测")
    check(not needs_upsample(row(0.60, 0.8, 0.8), 0.35, 0.85, 5), "补测: 已达 5 次不重复补")


def test_phase6_first_token() -> None:
    """首 token 读取自检：content 直读 / 空 content 走 logprobs 兜底 / 不可读剔除。"""
    from types import SimpleNamespace
    deps = os.path.join(ROOT, ".deps")
    if os.path.isdir(deps):
        sys.path.insert(0, deps)
    from fpcheck.phase6 import SENTINEL, first_tokens_of

    # 1) content 非空 → 直接读取（mock/正常端点路径）
    res = SimpleNamespace(texts=["π"], logprobs=None)
    toks, used = first_tokens_of(res)
    check(toks == ["π"] and not used, "首token: content 直读")
    # 2) 官方端常见：content 为空 → logprobs 兜底读出真实首 token
    res = SimpleNamespace(texts=[""], logprobs=[[{"token": "π", "logprob": -0.1}]])
    toks, used = first_tokens_of(res)
    check(toks == ["π"] and used, "首token: 空 content 经 logprobs 补读")
    # 3) content 为空且无 logprobs → 不可读（该侧剔除）
    res = SimpleNamespace(texts=[""], logprobs=None)
    toks, used = first_tokens_of(res)
    check(toks is None, "首token: 无 logprobs 时判定不可读")
    # 4) 纯空白 token（换行/空格）→ 归一化为哨兵，不丢失"空白首 token"信号
    res = SimpleNamespace(texts=["\n"], logprobs=[[{"token": "\n", "logprob": -0.2}]])
    toks, used = first_tokens_of(res)
    check(toks == [SENTINEL] and used, "首token: 纯空白归一化为哨兵")
    # 5) 多采样混合：样本0 空(lp 补读) + 样本1 content 直读
    res = SimpleNamespace(texts=["", "a"],
                          logprobs=[[{"token": "x", "logprob": -0.1}], None])
    toks, used = first_tokens_of(res)
    check(toks == ["x", "a"] and used, "首token: 多采样混合通道")


def test_clone_timeout() -> None:
    """阶段专用短超时克隆自检。"""
    deps = os.path.join(ROOT, ".deps")
    if os.path.isdir(deps):
        sys.path.insert(0, deps)
    from fpcheck.target import ProbeTarget

    t = ProbeTarget("x", "http://127.0.0.1:9/v1", "k", "m",
                    timeout=90.0, max_retries=4, api_style="responses")
    c = t.clone_with_timeout(30.0, max_retries=2)
    check(c.timeout == 30.0 and c.max_retries == 2, "克隆: 短超时与重试次数生效")
    check(c.name == "x" and c.api_style == "responses" and c.model_name == "m"
          and c.base_url == "http://127.0.0.1:9/v1" and c.api_key == "k",
          "克隆: 端点属性保持一致")
    check(c._rate is t._rate, "克隆: 共享同一个全局限速器")


def test_run_logger() -> None:
    """运行日志自检：落盘、脱敏、时间戳、阶段耗时、警告/错误计数、结构化摘要。"""
    import re as _re
    import tempfile as _tf
    from fpcheck.runlog import RunLogger, redact_args, redact_secret

    check(redact_secret("sk-abcdefgh123456") == "sk-abc****3456",
          "日志: 密钥脱敏规则")
    out_dir = _tf.mkdtemp(prefix="fpcheck_rlog_")
    rl = RunLogger(out_dir, version="9.9.9",
                   args={"official_key": "sk-abcdefgh123456", "proxy": "http://x"},
                   official=None, unknown=None, options={"seed": 1})
    rl.raw("开始探测……")
    rl.raw("警告: 组内基线过低")
    rl.raw("错误: 未知 API 超时")
    rl.raw("普通进度行")
    rl.finish(rc=0, verdict=None, scores={"phase1": 0.9},
              phase_elapsed={"phase1": 1.25, "phase2": 3.5})
    check(os.path.exists(rl.log_path) and os.path.exists(rl.json_path),
          "日志: .log 与 .json 均已落盘")
    with open(rl.log_path, encoding="utf-8") as f:
        text = f.read()
    check("开始探测" in text and "警告: 组内基线过低" in text,
          "日志: .log 记录全流程行")
    check(_re.search(r"\[\d{2}:\d{2}:\d{2}\] 开始探测", text) is not None,
          "日志: .log 每行带运行时刻戳")
    check("sk-abcdefgh123456" not in text, "日志: .log 不含明文 key")
    with open(rl.json_path, encoding="utf-8") as f:
        data = json.load(f)
    check(data["tool_version"] == "9.9.9" and data["exit_code"] == 0
          and data["phase_scores"]["phase1"] == 0.9
          and data["args"]["official_key"] == "sk-abc****3456",
          "日志: .json 结构化字段与脱敏正确")
    check(data.get("started_at") and data.get("finished_at"),
          "日志: .json 含开始/结束时间")
    check(data["phase_elapsed_sec"]["phase1"] == 1.25
          and data["phase_elapsed_sec"]["phase2"] == 3.5,
          "日志: .json 含各阶段耗时")
    check(len(data["warnings"]) == 1 and len(data["errors"]) == 1,
          "日志: 警告/错误计数正确")
    rl.finish(rc=1, phase_elapsed={"phase1": 1.0})   # 幂等：重复 finish 不报错


def test_early_failure_logged() -> None:
    """早期失败（配置缺失，logger 初始化之前）也应落盘失败日志。"""
    import tempfile as _tf
    out_root = _tf.mkdtemp(prefix="fpcheck_efail_")
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "run_fingerprint.py"),
         "--config", "no_such_config.json"],
        cwd=out_root, capture_output=True)
    check(proc.returncode == 1, "早败: 缺配置时退出码为 1")
    logs = glob.glob(os.path.join(out_root, "output", "logs", "run_*.json"))
    check(len(logs) == 1, f"早败: 失败日志已落盘（实际 {len(logs)} 个）")
    with open(logs[0], encoding="utf-8") as f:
        data = json.load(f)
    check(data["exit_code"] == 1 and "找不到配置文件" in data["note"],
          f"早败: .json 记录退出码与失败原因（note={data['note']!r}）")
    check(data.get("started_at") is not None, "早败: .json 含启动时间")


def test_panels_supported() -> None:
    """面板启用条件自检：非终端关闭、Windows 旧控制台关闭、现代终端开启。"""
    import io
    from fpcheck.console import panels_supported

    class _Tty:
        def isatty(self):
            return True

    tty = _Tty()
    check(not panels_supported(io.StringIO()), "面板条件: 非终端（管道）关闭")
    saved = {k: os.environ.get(k) for k in ("WT_SESSION", "TERM_PROGRAM", "TERM")}
    try:
        for k in ("WT_SESSION", "TERM_PROGRAM", "TERM"):
            os.environ.pop(k, None)
        if os.name == "nt":
            check(not panels_supported(tty), "面板条件: Windows 旧控制台默认关闭")
            os.environ["WT_SESSION"] = "e2e-test"
            check(panels_supported(tty), "面板条件: Windows Terminal 环境开启")
            os.environ.pop("WT_SESSION", None)
            os.environ["TERM_PROGRAM"] = "vscode"
            check(panels_supported(tty), "面板条件: VSCode 终端环境开启")
        else:
            check(not panels_supported(tty), "面板条件: 无 TERM 时关闭")
            os.environ["TERM"] = "xterm-256color"
            check(panels_supported(tty), "面板条件: 类 Unix TTY 开启")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_calibration() -> None:
    """官方自比归一化校准自检：除法/截断/跳过规则（阶段六为诊断不参与）。"""
    from fpcheck.verdict import calibrate

    out, info = calibrate({"phase1": 0.85, "phase2": 0.9, "phase4": 0.95,
                           "phase6": 1.0, "phase3": None}, 0.92)
    check(abs(out["phase1"] - 0.85 / 0.92) < 1e-9, "校准: 阶段一按噪声地板归一")
    check(out["phase2"] == 0.9, "校准: 阶段二不参与二次归一")
    check(out["phase4"] == 1.0, "校准: 除后超 1 截断")
    check(out["phase6"] == 1.0 and out["phase3"] is None, "校准: 阶段六诊断不校准")
    check(info is not None and info["phases"] == ["phase1", "phase4"]
          and abs(info["factor"] - 0.92) < 1e-9, "校准: 校准信息字段")
    s2 = {"phase1": 0.8, "phase4": 0.9}
    out2, info2 = calibrate(s2, None)
    check(out2 == s2 and info2 is None, "校准: 未跑自比时跳过")
    out3, info3 = calibrate(s2, 0.4)
    check(out3 == s2 and info3 is None, "校准: 自比过低(<0.5)时跳过")
    out4, info4 = calibrate(s2, 1.0)
    check(out4 == s2 and info4 is not None, "校准: 自比=1 数值不变但记录校准")


def main() -> None:
    test_rate_limiter()
    test_adaptive_policy()
    test_verdict_weights()
    test_calibration()
    test_phase_panels()
    test_panels_supported()
    test_phase2_upsample()
    test_phase6_first_token()
    test_clone_timeout()
    test_run_logger()
    test_early_failure_logged()
    test_cutoff_redflags()
    global PORT_OFFICIAL, PORT_UNKNOWN_A, PORT_UNKNOWN_B, PORT_UNKNOWN_C
    global PORT_UNKNOWN_E, PORT_OFFICIAL_NOISY, PORT_UNKNOWN_F, PORT_UNKNOWN_G
    global PORT_OFFICIAL_BLANK
    print("=== 启动 mock 端点 ===", flush=True)
    PORT_OFFICIAL = free_port()
    PORT_UNKNOWN_A = free_port()   # 同源，拒绝 seed/logprobs
    PORT_UNKNOWN_B = free_port()   # 异源，全部支持
    PORT_UNKNOWN_C = free_port()   # 同源，全部支持
    PORT_UNKNOWN_E = free_port()   # 同源，拒绝 temperature
    PORT_OFFICIAL_NOISY = free_port()   # 带采样方差的官方（phase5 用）
    PORT_UNKNOWN_F = free_port()   # 同源 noisy（方差模式一致）
    PORT_UNKNOWN_G = free_port()   # 异源 noisy（方差模式绑定身份）
    PORT_OFFICIAL_BLANK = free_port()   # 官方：max_tokens=1 返回空 content（阶段六重试兜底用）
    serve(PORT_OFFICIAL, "same", "official")
    serve(PORT_UNKNOWN_A, "same", "unknownA", reject_seed=True, reject_logprobs=True)
    serve(PORT_UNKNOWN_B, "diff", "unknownB")
    serve(PORT_UNKNOWN_C, "same", "unknownC")
    serve(PORT_UNKNOWN_E, "same", "unknownE", reject_temperature=True)
    serve(PORT_OFFICIAL_NOISY, "noisy", "officialNoisy", noise="shared")
    serve(PORT_UNKNOWN_F, "noisy", "unknownF", noise="shared")
    serve(PORT_UNKNOWN_G, "noisy", "unknownG", noise="label")
    serve(PORT_OFFICIAL_BLANK, "same", "officialBlank", blank_on_max1=True)

    # ---------------- 场景 A：同源 + 参数降级
    print("\n=== 场景 A：同源模型，未知 API 不支持 seed/logprobs ===", flush=True)
    rc, s, out = run_scenario("A", {
        "name": "未知API-A",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_A}/v1",
        "api_key": "sk-unknownA", "model_name": "gpt-4o"})
    check(rc == 0, "场景A: 退出码为 0")
    if s:
        check(s["scores"]["phase1_similarity"] == 1.0, "场景A: 第一阶段平均相似度 = 1.0")
        check(s["scores"]["phase1_exact_rate"] == 1.0, "场景A: 第一阶段逐字一致率 = 1.0")
        check(s["scores"]["phase2_cross_sim"] == 1.0, "场景A: 第二阶段交叉相似度 = 1.0")
        check(s["scores"]["phase3_topk_overlap"] is None, "场景A: 第三阶段已跳过")
        check(s["scores"]["phase4_similarity"] == 1.0, "场景A: 第四阶段分词器指纹相似度 = 1.0")
        check(s["scores"]["phase5_similarity"] is None, "场景A: 第五阶段已跳过（确定性端点）")
        check(s["scores"]["phase6_similarity"] == 1.0, "场景A: 第六阶段单Token重叠率 = 1.0")
        check("phase3" in s["skipped_phases"], "场景A: skipped_phases 含 phase3")
        check("phase5" in s["skipped_phases"], "场景A: skipped_phases 含 phase5")
        check(s["scores"]["verdict_score"] == 1.0, "场景A: 综合评分 = 1.0")
        check(s["verdict"] == "高度确信同一底层模型", "场景A: 判定为高度确信")
    raw1 = glob.glob(os.path.join(out, "output", "raw", "phase1", "*.json"))
    # 36 个 prompt × 2 个 API = 72 个取证文件
    check(len(raw1) == 72, f"场景A: phase1 取证文件 72 个（实际 {len(raw1)}）")
    raw3 = glob.glob(os.path.join(out, "output", "raw", "phase3", "*.json"))
    check(len(raw3) >= 1, "场景A: phase3 探测调用已记录取证")
    manifest = os.path.join(out, "output", "manifest.json")
    check(os.path.exists(manifest), "场景A: manifest.json 已生成")
    with open(manifest, encoding="utf-8") as f:
        mtext = f.read()
    check("sk-official" not in mtext and "sk-unknownA" not in mtext,
          "场景A: manifest 不含明文 api_key")
    check(s["baselines"]["self_sim"] == 1.0, "场景A: 官方自比基线 = 1.0")
    check(s["baselines"]["diff_sim"] is None, "场景A: 未配置异源基线 → None")
    check(s.get("calibration") is not None and s["calibration"]["factor"] == 1.0,
          "场景A: 自比=1 时校准无效果但已记录")
    check(s["scores"].get("phase6_in_verdict") is False,
          "场景A: 阶段六为诊断性阶段（不进判定）")
    logs_dir = os.path.join(out, "output", "logs")
    logs = glob.glob(os.path.join(logs_dir, "run_*.json"))
    check(len(logs) == 1, f"场景A: 运行日志 .json 已生成（实际 {len(logs)}）")
    with open(logs[0], encoding="utf-8") as f:
        rjson = json.load(f)
    check(rjson.get("tool_version") and rjson.get("verdict", {}).get("level_cn")
          == "高度确信同一底层模型" and rjson["exit_code"] == 0,
          "场景A: 运行日志结构化摘要含版本/判定/退出码")
    check("sk-official" not in json.dumps(rjson, ensure_ascii=False)
          and "sk-unknownA" not in json.dumps(rjson, ensure_ascii=False),
          "场景A: 运行日志不含明文 api_key")

    # ---------------- 场景 B：异源模型
    print("\n=== 场景 B：不同模型（响应逻辑完全不同） ===", flush=True)
    rc, s, out = run_scenario("B", {
        "name": "未知API-B",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_B}/v1",
        "api_key": "sk-unknownB", "model_name": "gpt-4o"})
    check(rc == 0, "场景B: 退出码为 0")
    if s:
        check(s["scores"]["phase1_similarity"] < 0.6, "场景B: 第一阶段平均相似度 < 0.6")
        check(s["scores"]["phase1_exact_rate"] == 0.0, "场景B: 第一阶段逐字一致率 = 0.0")
        check(s["scores"]["phase2_cross_sim"] < 0.5, "场景B: 第二阶段交叉相似度 < 0.5")
        check(s["scores"]["phase3_topk_overlap"] is not None, "场景B: 第三阶段正常执行")
        check(s["scores"]["phase3_topk_overlap"] < 0.5, "场景B: 第三阶段重叠率 < 0.5")
        check(s["scores"]["phase4_similarity"] < 0.6, "场景B: 第四阶段分词器指纹相似度 < 0.6")
        check(s["scores"]["phase5_similarity"] is None, "场景B: 第五阶段已跳过（确定性端点）")
        check(s["scores"]["phase6_similarity"] < 0.5, "场景B: 第六阶段单Token重叠率 < 0.5")
        check(s["scores"]["verdict_score"] < 0.5, "场景B: 综合评分 < 0.5")
        check(s["verdict"] == "证据不足", "场景B: 判定为证据不足")

    # ---------------- 场景 C：同源 + 全参数支持（显式全串行，作为并行对照）
    print("\n=== 场景 C：同源模型，全部参数支持（双维度均关闭） ===", flush=True)
    rc, s, out = run_scenario("C", {
        "name": "未知API-C",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_C}/v1",
        "api_key": "sk-unknownC", "model_name": "gpt-4o"},
        extra_options={"parallel_phases": False, "concurrent": False})
    check(rc == 0, "场景C: 退出码为 0")
    if s:
        check(s["scores"]["phase3_topk_overlap"] == 1.0, "场景C: 第三阶段重叠率 = 1.0")
        check(s["scores"]["phase5_similarity"] is None, "场景C: 第五阶段跳过（确定性 mock）")
        check(s["scores"]["phase6_similarity"] == 1.0, "场景C: 第六阶段单Token重叠率 = 1.0")
        check(s["skipped_phases"] == ["phase5"], "场景C: 仅 phase5 跳过")
        check(s["scores"]["verdict_score"] == 1.0, "场景C: 综合评分 = 1.0")
        check(s["verdict"] == "高度确信同一底层模型", "场景C: 判定为高度确信")
    with open(os.path.join(out, "run.log"), encoding="utf-8") as f:
        log_text = f.read()
    check("双API并行=关" in log_text and "阶段并行=关" in log_text,
          "场景C: 日志确认双维度并行均已关闭")
    with open(os.path.join(out, "output", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    check(manifest["options"]["concurrent"] is False
          and manifest["options"]["parallel_phases"] is False,
          "场景C: manifest 记录 concurrent/parallel_phases 均为 false")

    # ---------------- 场景 E：未知 API 拒绝 temperature（自动降级）
    print("\n=== 场景 E：同源模型，未知 API 拒绝 temperature 参数 ===", flush=True)
    rc, s, out = run_scenario("E", {
        "name": "未知API-E",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_E}/v1",
        "api_key": "sk-unknownE", "model_name": "gpt-4o"})
    check(rc == 0, "场景E: 退出码为 0")
    if s:
        check(s["scores"]["phase1_similarity"] == 1.0, "场景E: 第一阶段平均相似度 = 1.0")
        check(s["scores"]["phase1_exact_rate"] == 1.0, "场景E: 第一阶段逐字一致率 = 1.0")
        check(s["scores"]["phase2_cross_sim"] == 1.0, "场景E: 第二阶段交叉相似度 = 1.0")
        check(s["scores"]["phase3_topk_overlap"] == 1.0, "场景E: 第三阶段重叠率 = 1.0")
        check(s["scores"]["phase4_similarity"] == 1.0, "场景E: 第四阶段分词器指纹相似度 = 1.0")
        check(s["scores"]["phase5_similarity"] is None, "场景E: 第五阶段跳过（确定性端点）")
        check(s["scores"]["phase6_similarity"] == 1.0, "场景E: 第六阶段单Token重叠率 = 1.0")
        check(s["scores"]["verdict_score"] == 1.0, "场景E: 综合评分 = 1.0")
    # 取证数据中应记录未知 API 的 temperature=False（已自动降级）
    raw_u = glob.glob(os.path.join(out, "output", "raw", "phase1", "*未知API-E*.json"))
    if raw_u:
        with open(raw_u[0], encoding="utf-8") as f:
            entry = json.load(f)
        check(entry["capabilities"].get("temperature") is False,
              "场景E: 取证记录 capabilities.temperature = False")
        check("temperature" not in entry["request"],
              "场景E: 降级后请求体不含 temperature 字段")
    else:
        check(False, "场景E: 未找到未知API的取证文件")

    # ---------------- 场景 F：同源 + 两端都有采样方差（noisy shared）
    print("\n=== 场景 F：同源模型，两端采样方差模式一致 ===", flush=True)
    rc, s, out = run_scenario("F", {
        "name": "未知API-F",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_F}/v1",
        "api_key": "sk-unknownF", "model_name": "gpt-4o"},
        official_port=PORT_OFFICIAL_NOISY)
    check(rc == 0, "场景F: 退出码为 0")
    if s:
        check(s["scores"]["phase5_similarity"] == 1.0, "场景F: 第五阶段综合评分 = 1.0")
        check(s["scores"]["phase5_corr"] == 1.0, "场景F: 第五阶段画像相关 = 1.0")
        check(s["scores"]["phase1_similarity"] == 1.0, "场景F: 第一阶段平均相似度 = 1.0")
        check(s["scores"]["phase2_cross_sim"] == 1.0, "场景F: 第二阶段交叉相似度 = 1.0")
        check(not s["skipped_phases"], "场景F: 无跳过阶段")
        check(s["scores"]["verdict_score"] == 1.0, "场景F: 综合评分 = 1.0")
        check(s["verdict"] == "高度确信同一底层模型", "场景F: 判定为高度确信")

    # ---------------- 场景 G：内容高度相似但采样方差模式不同（noisy label）
    print("\n=== 场景 G：两端文本高度相似，但采样方差模式不同 ===", flush=True)
    rc, s, out = run_scenario("G", {
        "name": "未知API-G",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_G}/v1",
        "api_key": "sk-unknownG", "model_name": "gpt-4o"},
        official_port=PORT_OFFICIAL_NOISY)
    check(rc == 0, "场景G: 退出码为 0")
    if s:
        check("phase5" not in s["skipped_phases"], "场景G: 第五阶段未被跳过")
        check(s["scores"]["phase5_similarity"] is not None, "场景G: 第五阶段有评分")
        check(s["scores"]["phase5_corr"] is not None, "场景G: 第五阶段画像相关存在")
        check(s["scores"]["phase5_similarity"] < 0.9,
              f"场景G: 第五阶段评分 < 0.9（实际 {s['scores']['phase5_similarity']}）")

    # ---------------- 场景 H：默认配置（双维度并行默认开启 + 限速器生效）
    print("\n=== 场景 H：同源全参数 + 默认双维度并行 ===", flush=True)
    rc, s, out = run_scenario("H", {
        "name": "未知API-H",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_C}/v1",
        "api_key": "sk-unknownH", "model_name": "gpt-4o"},
        extra_options={"rate_limit_rpm": 3600, "fallback_tpm": 0,
                       "max_inflight": 16})
    check(rc == 0, "场景H: 退出码为 0")
    if s:
        check(s["scores"]["phase3_topk_overlap"] == 1.0, "场景H: 第三阶段重叠率 = 1.0")
        check(s["scores"]["phase5_similarity"] is None, "场景H: 第五阶段跳过（与串行一致）")
        check(s["skipped_phases"] == ["phase5"], "场景H: skipped_phases 与串行一致")
        check(s["scores"]["phase1_similarity"] == 1.0, "场景H: 第一阶段 = 1.0")
        check(s["scores"]["phase2_cross_sim"] == 1.0, "场景H: 第二阶段 = 1.0")
        check(s["scores"]["phase4_similarity"] == 1.0, "场景H: 第四阶段 = 1.0")
        check(s["scores"]["phase6_similarity"] == 1.0, "场景H: 第六阶段 = 1.0")
        check(s["scores"]["verdict_score"] == 1.0, "场景H: 综合评分 = 1.0")
        check(s["verdict"] == "高度确信同一底层模型", "场景H: 判定为高度确信")
    with open(os.path.join(out, "run.log"), encoding="utf-8") as f:
        log_text = f.read()
    check("并行模式" in log_text, "场景H: 日志确认阶段并行已启用")
    check("双API并行=开" in log_text and "阶段并行=开" in log_text,
          "场景H: 日志确认双维度并行默认同时开启")
    check("全局限速=自动" in log_text and "3600 RPM" in log_text,
          "场景H: 日志确认自动限速模式与回退值生效")
    with open(os.path.join(out, "output", "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    check(manifest["options"]["concurrent"] is True
          and manifest["options"]["parallel_phases"] is True
          and manifest["options"]["rate_limit_rpm"] == 3600,
          "场景H: manifest 记录双维度并行开启且有限速配置")

    # ---------------- 场景 I：官方侧缓存（第二次运行复用官方取证，只打未知侧）
    print("\n=== 场景 I：官方侧缓存（跨运行复用官方取证） ===", flush=True)
    out_i = tempfile.mkdtemp(prefix="fpcheck_I_")
    cfg_i = {"name": "未知API-I",
             "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_A}/v1",
             "api_key": "sk-unknownI", "model_name": "gpt-4o"}
    rc1, s1, _ = run_scenario("I", cfg_i, out_dir=out_i)
    check(rc1 == 0, "场景I: 第一次运行退出码为 0")
    raw_dir = os.path.join(out_i, "output", "raw", "phase2")
    n_off_1 = len(glob.glob(os.path.join(raw_dir, "*官方GPT*.json")))
    n_unk_1 = len(glob.glob(os.path.join(raw_dir, "*未知API-I*.json")))
    check(n_off_1 == 6 and n_unk_1 == 6,
          f"场景I: 第一次运行 phase2 两侧各 6 条取证（实际 官方{n_off_1}/未知{n_unk_1}）")
    rc2, s2, _ = run_scenario("I2", cfg_i, out_dir=out_i)   # 同目录第二次运行
    check(rc2 == 0, "场景I: 第二次运行退出码为 0")
    n_off_2 = len(glob.glob(os.path.join(raw_dir, "*官方GPT*.json")))
    n_unk_2 = len(glob.glob(os.path.join(raw_dir, "*未知API-I*.json")))
    check(n_off_2 == n_off_1, f"场景I: 官方侧取证未新增（复用缓存，{n_off_1}→{n_off_2}）")
    check(n_unk_2 == n_unk_1 * 2,
          f"场景I: 未知侧取证翻倍（实时请求，{n_unk_1}→{n_unk_2}）")
    with open(os.path.join(out_i, "run.log"), encoding="utf-8") as f:
        log2 = f.read()
    check("官方侧缓存" in log2, "场景I: 第二次运行日志确认官方侧缓存启用")
    check(s2 is not None and s2["verdict"] == "高度确信同一底层模型"
          and s2["scores"]["verdict_score"] == 1.0,
          "场景I: 缓存复用时判定结果不变")

    # ---------------- 场景 J：官方 max_tokens=1 返回空 content（阶段六重试兜底）
    print("\n=== 场景 J：官方端空 content + 无 logprobs（阶段六重试兜底） ===", flush=True)
    rc, s, out = run_scenario("J", {
        "name": "未知API-J",
        "base_url": f"http://127.0.0.1:{PORT_UNKNOWN_C}/v1",
        "api_key": "sk-unknownJ", "model_name": "gpt-4o"},
        official_port=PORT_OFFICIAL_BLANK)
    check(rc == 0, "场景J: 退出码为 0")
    if s:
        check("phase6" not in s["skipped_phases"], "场景J: 阶段六未被跳过")
        check(s["scores"]["phase6_similarity"] == 1.0,
              "场景J: 阶段六重试兜底后重叠率 = 1.0")
        check(s["scores"]["phase6_lp_fallback"] == 0, "场景J: 未走 logprobs 通道")
        check(s["scores"]["phase6_retried"] is not None
              and s["scores"]["phase6_retried"] > 0,
              f"场景J: max_tokens=2 重试已触发（{s['scores']['phase6_retried']} 个探测）")
        check(s["scores"]["phase6_unreadable"] == 0, "场景J: 无不可读探测被剔除")
        check(s["scores"]["verdict_score"] == 1.0, "场景J: 综合评分 = 1.0")

    print("\n" + "=" * 50)
    print(f"共 9 个场景 + 全部自检跑完，失败 {len(failures)} 项")
    if failures:
        print(f"共 {len(failures)} 项失败:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("全部自测通过 ✔")


if __name__ == "__main__":
    main()
