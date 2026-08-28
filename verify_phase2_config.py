"""第二阶段配置精度对照实验（离线模拟，零 API 成本）。

问题：精简探测集（15~20 个）× 降采样（每探测 3 次 + 边界补测到 5）×
短输出（max_tokens=96），会不会降低准确度？

方法：合成"同源对/异源对"的响应分布（每探测有各自的变异率/基底差异/长度，
20% 为确定性收敛探测），把响应文本喂给 **fpcheck.phase2 的真实评分代码**
（cross_similarity / intra_similarity / 收敛剔除 / 归一化 / bootstrap CI），
对四种配置分别跑 N 个试验对，比较：
  - 同源对: 平均分、≥0.75 判定率（漏判率 = 误判为异源）
  - 异源对: 平均分、<0.5 判定率（误判率 = 误判为同源）
  - 不确定率（0.5~0.75）、95% CI 平均宽度
  - 综合误判率 = (漏判 + 误判) / 2

用法: python verify_phase2_config.py [试验对数，默认 300]
"""
from __future__ import annotations

import os
import random
import sys
from statistics import mean

# 特殊环境（无法 pip 安装依赖的沙箱）：依赖装在 .deps/ 下，自动加入 sys.path
_LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(_LOCAL_DEPS):
    sys.path.insert(0, _LOCAL_DEPS)

from fpcheck.phase2 import (ProbeRow, _bootstrap_ci, cross_similarity,
                            intra_similarity)

ALPHA = "abcdefghijklmnopqrstuvwxyz 0123456789αβγδεζηθικλμνξοπρστυφχψω"
THRESH_HIGH = 0.75
THRESH_LOW = 0.5


def _mutate(rng: random.Random, s: str, rate: float) -> str:
    out = []
    for ch in s:
        out.append(rng.choice(ALPHA) if rng.random() < rate else ch)
    return "".join(out)


def _sample(rng, base: str, rate: float, n: int, cap: int) -> list[str]:
    texts = []
    for _ in range(n):
        t = _mutate(rng, base, rate)
        if len(t) > cap:
            t = t[:cap]          # 模拟 max_tokens 截断
        texts.append(t)
    return texts


def run_pair(rng: random.Random, same: bool, num_probes: int, base_n: int,
             upsample: bool, cap: int) -> list[ProbeRow]:
    rows: list[ProbeRow] = []
    for i in range(num_probes):
        length = rng.randint(40, 110)
        base = "".join(rng.choice(ALPHA) for _ in range(length))
        # 20% 确定性收敛探测（任何模型输出一致 → 无判别力，两种配置都会被剔除）
        if rng.random() < 0.20:
            text = base[: min(length, cap)]
            rows.append(ProbeRow(probe_id=f"p{i:03d}", prefix="", suffix_len=length,
                                 texts_a=[text] * base_n, texts_b=[text] * base_n))
            continue
        rate = rng.uniform(0.05, 0.40)                 # 每字符变异率 → 自洽度
        drift = rng.uniform(0.0, 0.10) if same else rng.uniform(0.30, 0.70)
        base_b = _mutate(rng, base, drift)             # 同源=轻微漂移；异源=大改
        ta = _sample(rng, base, rate, base_n, cap)
        tb = _sample(rng, base_b, rate, base_n, cap)
        row = ProbeRow(probe_id=f"p{i:03d}", prefix="", suffix_len=length,
                       texts_a=ta, texts_b=tb,
                       cross_sim=cross_similarity(ta, tb),
                       intra_a=intra_similarity(ta), intra_b=intra_similarity(tb))
        # 边界补测：3 次后仍在新探针的非收敛且交叉落在 (0.35, 0.85) → 补到 5 次
        if upsample and base_n < 5 \
                and not (row.intra_a >= 0.999 and row.intra_b >= 0.999) \
                and 0.35 < row.cross_sim < 0.85:
            extra = 5 - base_n
            row.texts_a += _sample(rng, base, rate, extra, cap)
            row.texts_b += _sample(rng, base_b, rate, extra, cap)
            row.cross_sim = cross_similarity(row.texts_a, row.texts_b)
            row.intra_a = intra_similarity(row.texts_a)
            row.intra_b = intra_similarity(row.texts_b)
        rows.append(row)
    return rows


def score_of(rows: list[ProbeRow], seed: int) -> tuple[float, float, float]:
    """与 run_phase2 完全一致的评分路径：收敛剔除 → 交叉/组内归一化 → CI。"""
    valid = [r for r in rows if not r.error_a and not r.error_b]
    ent = [r for r in valid if not (r.intra_a >= 0.999 and r.intra_b >= 0.999)]
    if not ent:
        return 0.5, 0.0, 0.0
    cross = sum(r.cross_sim for r in ent) / len(ent)
    intra = sum((r.intra_a + r.intra_b) / 2 for r in ent) / len(ent)
    score = min(1.0, cross / intra) if intra > 1e-9 else 0.0
    lo, hi = _bootstrap_ci(ent, seed)
    return score, lo, hi


def main(trials: int = 300) -> None:
    """按配置逐项对照。同源判定=得分≥0.75；异源判定=得分<0.5；中间=不确定。"""
    configs = [
        ("A 基线   50探测 × 5采样 × 256", 50, 5, False, 256),
        ("B 精简+补 18探测 × 3+补测 × 96", 18, 3, True, 96),
        ("C 只减探测 18探测 × 5采样 × 256", 18, 5, False, 256),
        ("D 只减采样 50探测 × 3采样 × 256", 50, 3, False, 256),
        ("E 只截断  18探测 × 3+补测 × 256", 18, 3, True, 256),
    ]
    print(f"试验对数（每配置×每类型）: {trials}\n")
    for label, np_, ns, up, cap in configs:
        same, diff = [], []          # (score, ci_low, ci_high)
        for t in range(trials):
            rng = random.Random(1000 + t * 101)
            s = run_pair(rng, True, np_, ns, up, cap)
            d = run_pair(rng, False, np_, ns, up, cap)
            same.append(score_of(s, seed=42))
            diff.append(score_of(d, seed=42))
        s_avg = mean(x[0] for x in same)
        d_avg = mean(x[0] for x in diff)
        fn = sum(1 for x in same if x[0] < THRESH_LOW) / trials      # 同源被判异源
        fp = sum(1 for x in diff if x[0] >= THRESH_HIGH) / trials    # 异源被判同源
        unsure_s = sum(1 for x in same if THRESH_LOW <= x[0] < THRESH_HIGH) / trials
        unsure_d = sum(1 for x in diff if THRESH_LOW <= x[0] < THRESH_HIGH) / trials
        ci_w = mean(x[2] - x[1] for x in same + diff)
        # 加权判错分（把"不确定"算半错，更贴合真实结论的可信度损失）
        err = (fn + fp) / 2
        warn = (unsure_s + unsure_d) / 2
        print(f"{label}")
        print(f"  同源对: 平均分 {s_avg:.3f}  漏判率 {fn:.1%}  不确定率 {unsure_s:.1%}")
        print(f"  异源对: 平均分 {d_avg:.3f}  误判率 {fp:.1%}  不确定率 {unsure_d:.1%}")
        print(f"  综合误判率 {err:.2%}  平均 95% CI 宽度 {ci_w:.3f}"
              f"  加权判错分({err*2+warn:.2%})\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)