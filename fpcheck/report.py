"""生成文本报告与 JSON 摘要（自动输出到 output/ 目录）。"""
from __future__ import annotations

import json
import os
from datetime import datetime

from . import __version__
from .verdict import BASE_WEIGHTS, Verdict


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def build_report(cfg, phase1, phase2, phase3, phase4, phase5, phase6,
                 redflags, verdict: Verdict, baselines: dict | None = None,
                 calibration: dict | None = None) -> str:
    baselines = baselines or {}
    L: list[str] = []
    L.append(_rule("="))
    L.append(" LLM 行为指纹验证报告 —— 未知 API vs 官方 API")
    L.append(_rule("="))
    L.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"工具版本: {__version__}")
    L.append(f"官方 API: {cfg.official.name}  model={cfg.official.model_name}")
    L.append(f"          端点: {cfg.official.base_url}")
    L.append(f"未知 API: {cfg.unknown.name}  model={cfg.unknown.model_name}")
    L.append(f"          端点: {cfg.unknown.base_url}")
    L.append("")

    # ---------------- 对照基线（不参与判定，只作参照系）
    L.append(_rule("-"))
    L.append("对照基线（不参与判定，只作参照系）")
    L.append(_rule("-"))
    bs = baselines.get("self_sim")
    if bs is not None:
        L.append(f"  官方自比基线: 平均相似度 {bs:.3f}"
                 "（同端点同模型应接近 1；明显偏低提示官方侧自身不稳定）")
        if calibration is not None:
            L.append(f"  归一化校准: 阶段 {'/'.join(calibration['phases'])} "
                     f"得分 ÷{calibration['factor']:.3f}（≤1 截断），"
                     "已反映在综合评分中")
        elif bs < 0.5:
            L.append("  ⚠ 自比基线过低（<0.5），跳过归一化校准"
                     "（不放大噪声；结论应谨慎解读）")
    else:
        L.append("  官方自比基线: 未执行（baseline_self=false）"
                 "→ 未做归一化校准")
    bd = baselines.get("diff_sim")
    bm = baselines.get("diff_model") or ""
    if bd is not None:
        L.append(f"  已知异源对照（{bm}）: 平均相似度 {bd:.3f}"
                 "（应显著低于自比基线与待判得分，否则需复核指标体系）")
    elif bm:
        L.append(f"  已知异源对照（{bm}）: 未执行（无法调用该模型）")
    else:
        L.append("  已知异源对照: 未配置 baseline_diff_model")
    L.append("")

    # ---------------- 第一阶段
    L.append(_rule("-"))
    L.append("第一阶段  基础行为对齐（权重 0.15）")
    L.append(_rule("-"))
    if phase1 is None:
        L.append("  未执行（配置禁用）。")
    else:
        L.append(f"  探测 prompt 数: {phase1.total}"
                 f"（评分 {len(phase1.prompts)} + 低熵回声参考 "
                 f"{len(phase1.reference_rows)}，temperature=0）")
        L.append(f"  平均相似度（主指标，进入判定；不含回声类别）: {phase1.score:.3f}")
        L.append(f"  逐字一致率（参考）: {phase1.exact_rate:.2%}")
        L.append(f"  归一化一致率（忽略大小写/空白，参考）: {phase1.normalized_rate:.2%}")
        L.append("  注: 同源端点也常因服务端非确定性产生微小漂移，"
                 "相似度评分能区分'微小漂移'与'系统性差异'。")
        L.append("  注: 固定复读/分词泄漏/严格格式遵循为低熵回声类别"
                 "（任何模型都可能逐字一致，无判别力），仅参考、不评分。")
        L.append("  评分类别明细（自由度类别）:")
        cats: dict[str, list] = {}
        for p in phase1.prompts:
            cats.setdefault(p.category_cn, []).append(p)
        for cn, items in cats.items():
            avg = sum(p.similarity for p in items) / len(items)
            L.append(f"    - {cn}: 平均相似度 {avg:.3f}")
        if phase1.reference_rows:
            L.append("  低熵参考类别明细（不评分）:")
            refs: dict[str, list] = {}
            for p in phase1.reference_rows:
                refs.setdefault(p.category_cn, []).append(p)
            for cn, items in refs.items():
                avg = sum(p.similarity for p in items) / len(items)
                L.append(f"    - {cn}: 平均相似度 {avg:.3f}（参考）")
        errs = [p for p in phase1.prompts + phase1.reference_rows
                if p.error_a or p.error_b]
        if errs:
            L.append(f"  警告: {len(errs)} 个 prompt 调用失败（详见 raw 取证数据）")
    L.append("")

    # ---------------- 第二阶段
    L.append(_rule("-"))
    L.append("第二阶段  对抗后缀指纹（权重 0.25，主支柱）")
    L.append(_rule("-"))
    if phase2 is None:
        L.append("  未执行（配置禁用）。")
    else:
        L.append(f"  对抗探测数: {phase2.num_probes}")
        L.append("  乱码后缀: 40~60 字符，ASCII/希腊/西里尔字符混合，"
                 "temperature=1.0 每探测采样 "
                 f"{cfg.options.phase2_samples} 次")
        L.append(f"  归一化相似度（有效交叉/组内基线，进入判定）: {phase2.score:.3f}")
        L.append(f"  证据可靠性（95% CI 宽度折算，进证据加权）: {phase2.reliability:.2f}")
        L.append(f"  95% 置信区间（bootstrap）: [{phase2.ci_low:.3f}, {phase2.ci_high:.3f}]"
                 + ("（区间整体 > 0.9 → 结论稳健）" if phase2.ci_low > 0.9 else "")
                 + ("（区间较宽 → 建议增大 num_probes/samples 重测）"
                    if 0 < phase2.ci_high - phase2.ci_low > 0.2 else ""))
        L.append(f"  裸交叉平均相似度（参考）: {phase2.raw_cross:.3f}")
        L.append(f"  低熵收敛探测（已剔除，无判别力）: {phase2.converged}/{phase2.succeeded}")
        L.append(f"  输出 token 合计（两侧）: {phase2.total_out_tokens}"
                 f"（max_tokens 上限 {cfg.options.phase2_max_tokens}；"
                 "若实际生成量 P95 远低于上限，可自行调低该上限加速；"
                 "默认不放小是避免截断改变尾部相似度分布）")
        if phase2.warn_low_baseline:
            L.append("  ⚠ 双低警告: 两端组内基线过低（<0.5），交叉/组内比值在此"
                     "区域噪声大，本阶段证据弱，建议增大 samples 重测")
        if phase2.succeeded:
            L.append(f"  有效交叉 {phase2.cross_entropic:.3f}  组内基线 "
                     f"{phase2.intra_baseline:.3f}  交叉-组内 {phase2.delta:+.3f}"
                     + ("（系统性为负 → 行为分布不同，异源信号）"
                        if phase2.delta < -0.15 else ""))
        if phase2.succeeded and phase2.rows:
            sims = sorted(r.cross_sim for r in phase2.rows
                          if not r.error_a and not r.error_b)
            L.append(f"  分布: min={sims[0]:.3f}  median={sims[len(sims)//2]:.3f}  "
                     f"max={sims[-1]:.3f}")
            strong = sum(1 for s in sims if s > 0.8)
            L.append(f"  相似度>0.8 的探测占比: {strong}/{len(sims)} "
                     f"({strong/len(sims):.0%})")
        errs2 = [r for r in phase2.rows if r.error_a or r.error_b]
        if errs2:
            L.append(f"  警告: {len(errs2)}/{len(phase2.rows)} 个探测调用失败")
    L.append("")

    # ---------------- 第三阶段
    L.append(_rule("-"))
    L.append("第三阶段  logprob 数值对齐（权重 0.05，辅助加分项）")
    L.append(_rule("-"))
    if phase3 is None:
        L.append("  未执行（配置禁用）。")
    elif phase3.skipped:
        L.append(f"  已跳过: {phase3.skipped}")
        L.append("  提示: 很多第三方端点不支持 logprobs，跳过不影响主结论；"
                 "该阶段权重将按剩余阶段重新归一化。")
    else:
        L.append(f"  top-{cfg.options.phase3_top_logprobs} 候选重叠率（主指标）: "
                 f"{phase3.score:.3f}")
        L.append(f"  top-1 token 一致率（参考）: {phase3.top1_rate:.3f}")
        for p in phase3.prompts:
            L.append(f"    - 位置数={p.positions}  重叠率={p.overlap:.3f}  "
                     f"top-1={p.top1_rate:.3f}")
            L.append(f"      官方: {p.text_a[:60]!r}")
            L.append(f"      未知: {p.text_b[:60]!r}")
    L.append("")

    # ---------------- 第四阶段
    L.append(_rule("-"))
    L.append("第四阶段  分词器行为指纹（权重 0.10，家族级证据）")
    L.append(_rule("-"))
    if phase4 is None:
        L.append("  未执行（配置禁用）。")
    else:
        L.append(f"  探测 prompt 数: {phase4.total}"
                 "（字符计数/字符串倒序/罕见字符复读/分词边界，temperature=0）")
        L.append(f"  平均相似度（主指标，进入判定）: {phase4.score:.3f}")
        L.append(f"  逐字一致率（参考）: {phase4.exact_rate:.2%}")
        L.append("  注: 分词器是模型家族的硬指纹，确定性任务上的行为"
                 "（含特有的错误方式）同源时高度一致，且任何端点都支持。")
        L.append("  注: 本阶段是**家族级证据**——同族不同模型也可能满分，"
                 "不单独指向同权重，判定权重已相应调低（0.10）。")
        L.append("  分类明细:")
        cats4: dict[str, list] = {}
        for p in phase4.prompts:
            cats4.setdefault(p.category_cn, []).append(p)
        for cn, items in cats4.items():
            avg = sum(p.similarity for p in items) / len(items)
            L.append(f"    - {cn}: 平均相似度 {avg:.3f}")
        errs4 = [p for p in phase4.prompts if p.error_a or p.error_b]
        if errs4:
            L.append(f"  警告: {len(errs4)} 个 prompt 调用失败（详见 raw 取证数据）")
    L.append("")

    # ---------------- 第五阶段
    L.append(_rule("-"))
    L.append("第五阶段  采样自洽性画像（权重 0.20，不依赖 logprobs）")
    L.append(_rule("-"))
    if phase5 is None:
        L.append("  未执行（配置禁用）。")
    elif phase5.score is None:
        L.append(f"  已跳过: {phase5.skipped}")
        L.append("  提示: 确定性端点没有采样方差，画像无判别力；"
                 "跳过时该阶段权重按剩余阶段重新归一化。")
    else:
        L.append(f"  开放任务数: {phase5.total}"
                 f"（10 个开放任务 × 每侧采样 {cfg.options.phase5_samples} 次，"
                 f"temperature={cfg.options.phase5_temperature}）")
        L.append("  思路: 借鉴 SelfCheckGPT 的 sample-and-compare——对同一 prompt"
                 "多次采样，以组内一致性表征每个模型的采样分布特征。")
        L.append(f"  画像 Spearman 相关（进入判定）: "
                 f"{phase5.corr if phase5.corr is not None else 'N/A'}")
        L.append(f"  画像幅值相似度（进入判定）: {phase5.amp:.3f}")
        L.append(f"  综合评分（0.5×相关 + 0.5×幅值）: {phase5.score:.3f}")
        L.append(f"  证据可靠性（95% CI 宽度折算，进证据加权）: {phase5.reliability:.2f}")
        L.append(f"  95% 置信区间（bootstrap）: [{phase5.ci_low:.3f}, {phase5.ci_high:.3f}]")
        L.append(f"  官方平均自洽度 {phase5.mean_a:.3f} vs 未知 {phase5.mean_b:.3f}")
        if phase5.corr is not None and phase5.corr < 0.5:
            L.append("  提示: 画像相关性偏低 → 两端采样分布不同，异源信号。")
        L.append("  分类明细:")
        for p in phase5.rows:
            st = "跳过" if (p.error_a or p.error_b) else \
                f"官方 {p.agreement_a:.3f} / 未知 {p.agreement_b:.3f}"
            L.append(f"    - [{p.prompt[:48]}...] {st}")
    L.append("")

    # ---------------- 第六阶段
    L.append(_rule("-"))
    L.append("第六阶段  单 Token 行为指纹（诊断性阶段，分数不参与判定）")
    L.append(_rule("-"))
    if phase6 is None:
        L.append("  未执行（配置禁用）。")
    elif phase6.score is None:
        L.append(f"  已跳过: {phase6.skipped}")
        L.append("  提示: 有效短串探测不足时跳过，权重按剩余阶段重新归一化。")
    else:
        L.append(f"  短串探测数: {phase6.total}"
                 f"（seed 派生的乱码短串 {cfg.options.phase6_num_probes} 个 + "
                 "固定刁钻串 6 个，max_tokens=1，每串采样 "
                 f"{cfg.options.phase6_samples} 次；诊断档）")
        L.append("  定位: 诊断性阶段——推理型端点（思考消耗全部输出预算）"
                 "下得分不可靠，分数不进综合判定；下面的「退化现象」"
                 "如实保留：读取通道、升级重试、推理型跳过等。")
        L.append("  思路: 借鉴 'One Token Is Enough'（arXiv:2607.10252）——"
                 "只取首 token，比较两端输出分布的频次重叠系数（参考值）。")
        L.append(f"  首 token 分布重叠率（主指标，进入判定）: {phase6.score:.3f}")
        L.append(f"  证据可靠性（95% CI 宽度折算，进证据加权）: {phase6.reliability:.2f}")
        if phase6.lp_fallback or phase6.unreadable or phase6.degraded:
            if phase6.degraded:
                L.append(f"  整阶段退化: 端点内容恒空，已切换 max_tokens="
                         f"{phase6.retry_tokens} 的短前缀指纹继续")
            if phase6.lp_fallback:
                L.append(f"  读取通道: {phase6.lp_fallback} 个探测 content 为空、"
                         "由 logprobs 补读真实首 token")
            if phase6.unreadable:
                L.append(f"  {phase6.unreadable} 个探测首 token 不可读"
                         "（已剔除，不给假证据）")
        L.append(f"  95% 置信区间（bootstrap）: "
                 f"[{phase6.ci_low:.3f}, {phase6.ci_high:.3f}]")
        L.append("  分类明细:")
        for r in phase6.rows:
            st = "跳过" if (r.error_a or r.error_b) else f"重叠率 {r.overlap:.3f}"
            L.append(f"    - 短串 {r.probe!r}  {st}")
    L.append("")

    # ---------------- 身份一致性红旗
    L.append(_rule("-"))
    L.append("身份一致性红旗（辅助信号，复用第一阶段取证）")
    L.append(_rule("-"))
    if redflags is None or not redflags.has_flags:
        L.append("  未发现红旗。")
    else:
        for flag in redflags.flags:
            L.append(f"  ⚑ {flag}")
        if redflags.latency_ratio > 0:
            L.append(f"  （未知/官方 平均延迟比 {redflags.latency_ratio:.2f}）")
        if redflags.refusal_match_rate is not None:
            L.append(f"  （拒答模式一致率 {redflags.refusal_match_rate:.0%}）"
                     + (" ≥75%，未触发" if redflags.refusal_match_rate >= 0.75 else ""))
        if redflags.prompt_tokens_ratio > 0:
            L.append(f"  （未知/官方 prompt_tokens 消耗比 {redflags.prompt_tokens_ratio:.2f}）")
        if redflags.completion_tokens_ratio > 0:
            L.append(f"  （未知/官方 completion_tokens 消耗比 {redflags.completion_tokens_ratio:.2f}）")
        L.append("  注: 红旗不直接改变判定，但提醒复核——真实模型不应有矛盾自述。")
    L.append("")

    # ---------------- 第七阶段
    L.append(_rule("-"))
    L.append("第七阶段  综合判定")
    L.append(_rule("-"))
    L.append("  原始权重: " + ", ".join(f"{k}={v:.2f}"
             for k, v in BASE_WEIGHTS.items()))
    L.append("  证据加权: 权重按各阶段成功率（ok/total，下限 0.5）回调后"
             "重新归一化——证据越少的阶段话语权越小。")
    L.append("  注: 第六阶段为诊断性阶段，其分数不参与本综合评分（详见该阶段小节）。")
    if verdict.conflict:
        L.append(f"  ⚠ 证据冲突: 各阶段得分跨度 {verdict.phase_spread:.2f}"
                 "（>0.45），多通道未共同吻合，判定已降级/标注",
                 )
    if verdict.fail_closed:
        L.append(f"  ⚠ fail-closed: {verdict.fail_reason}，判定强制降级为证据不足。")
    if verdict.skipped_phases:
        L.append(f"  跳过阶段: {', '.join(verdict.skipped_phases)} "
                 "→ 权重已重新归一化: "
                 + ", ".join(f"{k}={v:.3f}" for k, v in verdict.weights_used.items()))
    L.append(f"  综合评分: {verdict.score:.3f}")
    L.append(f"  判定等级: {verdict.level_cn}")
    L.append(f"  判定结论: {verdict.summary}")
    L.append("")

    L.append(_rule("-"))
    L.append(" 说明与建议")
    L.append(_rule("-"))
    L.append("  * 本工具通过行为指纹寻找'统计上无法用巧合解释的相似性'，"
             "结论是强证据而非绝对证明。")
    L.append("  * 判定以不依赖可选参数的行为阶段为主（第二/四/五阶段）；"
             "logprob 阶段仅辅助加分，被跳过不影响主结论。")
    L.append("  * 同一模型被不同包装层改写时，第一阶段得分可能偏低，"
             "此时应更看重第二、四阶段的统计证据。")
    L.append("  * 低熵任务（如数值计算）上不同模型也可能趋同，"
             "第二阶段置信区间整体偏高（>0.9）时结论更可信。")
    L.append("  * 原始取证数据（每次调用的 prompt、完整响应、时间戳、端点）"
             "保存在 raw/ 目录，可用于复核与复现。")
    L.append(_rule("="))
    return "\n".join(L)


def write_report(cfg, phase1, phase2, phase3, phase4, phase5, phase6,
                 redflags, verdict: Verdict, baselines: dict | None = None,
                 calibration: dict | None = None):
    """返回 (报告文本, 报告路径)；同时写出 summary.json 便于程序化消费。"""
    text = build_report(cfg, phase1, phase2, phase3, phase4, phase5, phase6,
                        redflags, verdict, baselines, calibration)
    os.makedirs(cfg.options.output_dir, exist_ok=True)
    fname = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(cfg.options.output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool_version": __version__,
        "official": {"name": cfg.official.name, "model": cfg.official.model_name,
                     "endpoint": cfg.official.base_url},
        "unknown": {"name": cfg.unknown.name, "model": cfg.unknown.model_name,
                    "endpoint": cfg.unknown.base_url},
        "scores": {
            "phase1_similarity": phase1.score if phase1 is not None else None,
            "phase1_exact_rate": phase1.exact_rate if phase1 is not None else None,
            "phase2_cross_sim": phase2.score if phase2 is not None else None,
            "phase2_ci_low": phase2.ci_low if phase2 is not None else None,
            "phase2_ci_high": phase2.ci_high if phase2 is not None else None,
            "phase2_raw_cross": phase2.raw_cross if phase2 is not None else None,
            "phase2_intra_baseline": phase2.intra_baseline if phase2 is not None else None,
            "phase2_delta": phase2.delta if phase2 is not None else None,
            "phase2_converged": phase2.converged if phase2 is not None else None,
            "phase3_topk_overlap": phase3.score if phase3 is not None else None,
            "phase4_similarity": phase4.score if phase4 is not None else None,
            "phase4_exact_rate": phase4.exact_rate if phase4 is not None else None,
            "phase5_similarity": phase5.score if phase5 is not None else None,
            "phase5_amp": phase5.amp if phase5 is not None else None,
            "phase5_corr": phase5.corr if phase5 is not None else None,
            "phase5_ci_low": phase5.ci_low if phase5 is not None else None,
            "phase5_ci_high": phase5.ci_high if phase5 is not None else None,
            "phase6_similarity": phase6.score if phase6 is not None else None,
            "phase6_ci_low": phase6.ci_low if phase6 is not None else None,
            "phase6_ci_high": phase6.ci_high if phase6 is not None else None,
            "phase6_lp_fallback": phase6.lp_fallback if phase6 is not None else None,
            "phase6_retried": phase6.retried if phase6 is not None else None,
            "phase6_unreadable": phase6.unreadable if phase6 is not None else None,
            "phase6_degraded": phase6.degraded if phase6 is not None else None,
            "phase6_retry_tokens": phase6.retry_tokens if phase6 is not None else None,
            "phase6_in_verdict": False,
            "verdict_score": verdict.score,
        },
        "verdict": verdict.level_cn,
        "skipped_phases": verdict.skipped_phases,
        "weights_used": verdict.weights_used,
        "fail_closed": verdict.fail_closed,
        "fail_reason": verdict.fail_reason,
        "conflict": verdict.conflict,
        "phase_spread": round(verdict.phase_spread, 3),
        "evidence_reliability": {
            "phase2": phase2.reliability if phase2 is not None else None,
            "phase5": phase5.reliability
            if phase5 is not None and phase5.score is not None else None,
            "phase6": phase6.reliability
            if phase6 is not None and phase6.score is not None else None,
        },
        "baselines": {
            "self_sim": baselines.get("self_sim") if baselines else None,
            "diff_sim": baselines.get("diff_sim") if baselines else None,
            "diff_model": baselines.get("diff_model") if baselines else None,
        },
        "calibration": calibration,
        "red_flags": redflags.flags if redflags is not None else [],
        "latency_ratio": round(redflags.latency_ratio, 3)
        if redflags is not None and redflags.latency_ratio > 0 else None,
        "refusal_match_rate": round(redflags.refusal_match_rate, 3)
        if redflags is not None and redflags.refusal_match_rate is not None else None,
        "prompt_tokens_ratio": round(redflags.prompt_tokens_ratio, 3)
        if redflags is not None and redflags.prompt_tokens_ratio > 0 else None,
        "completion_tokens_ratio": round(redflags.completion_tokens_ratio, 3)
        if redflags is not None and redflags.completion_tokens_ratio > 0 else None,
        "has_reasoning_asymmetry": redflags.has_reasoning_asymmetry
        if redflags is not None else False,
        "identity_votes": {
            "official": redflags.identity_votes_a if redflags is not None else {},
            "unknown": redflags.identity_votes_b if redflags is not None else {},
        },
    }
    with open(os.path.join(cfg.options.output_dir, "summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return text, path
