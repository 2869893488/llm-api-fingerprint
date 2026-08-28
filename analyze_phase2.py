"""第二阶段取证数据深度分析：逐探测判别力排名 + 黑名单建议。

用法: python analyze_phase2.py [raw/phase2 目录，默认 output/raw/phase2]

对每个对抗探测计算：
  - 交叉相似度 / 组内相似度（同源基线）/ 判别力 = |交叉 - 组内|
  - 收敛标记（双侧组内=1，无判别力）
  - 错误标记（任一侧调用失败）

输出三张表：低判别力候选（含收敛/错误/双低），建议把它们加入配置
`options.phase2.skip_ids`（黑名单，见 config.example.json）；同时对
保留的探测给出判别力中位数，供判断"精简到 N 个"是否安全。
注意：判别力与"当前对比的模型对"相关，换端点后请重新分析再决定黑名单。
"""
import json
import os
import sys
from difflib import SequenceMatcher
from statistics import mean, median


def ratio(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def analyze(raw_dir: str) -> list[dict]:
    probes: dict[str, dict[str, dict]] = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fn), encoding="utf-8") as f:
            e = json.load(f)
        probes.setdefault(e["probe_id"], {})[e["target"]] = {
            "texts": e["response"]["texts"], "error": e.get("error")}

    rows: list[dict] = []
    for pid, sides in sorted(probes.items()):
        if len(sides) < 2:
            continue
        any_err = any(v["error"] for v in sides.values())
        texts = [v["texts"] for v in sides.values()]
        if any(not t for t in texts) or any_err:
            rows.append({"id": pid, "cross": None, "intra": None,
                         "delta": None, "converged": False, "error": True,
                         "tokens_a": 0, "tokens_b": 0})
            continue
        xa, xb = texts
        cross = mean(ratio(a, b) for a in xa for b in xb)
        intra_a = mean(ratio(xa[i], xa[j])
                       for i in range(len(xa)) for j in range(i + 1, len(xa))) \
            if len(xa) >= 2 else 1.0
        intra_b = mean(ratio(xb[i], xb[j])
                       for i in range(len(xb)) for j in range(i + 1, len(xb))) \
            if len(xb) >= 2 else 1.0
        intra = (intra_a + intra_b) / 2
        converged = intra_a >= 0.999 and intra_b >= 0.999
        rows.append({"id": pid, "cross": cross, "intra": intra,
                     "delta": (cross - intra) if not converged else None,
                     "converged": converged, "error": False,
                     "tokens_a": 0, "tokens_b": 0})
    return rows


def main(raw_dir: str = "output/raw/phase2") -> None:
    rows = analyze(raw_dir)
    if not rows:
        print(f"没有找到取证数据：{raw_dir}（先运行 run_fingerprint.py）")
        return
    eff = [r for r in rows if not r["error"] and not r["converged"]]
    bad = [r for r in rows if r["error"] or r["converged"]]

    print(f"总探测数: {len(rows)}  有效: {len(eff)}  "
          f"收敛(无判别力): {sum(1 for r in rows if r['converged'])}  "
          f"错误: {sum(1 for r in rows if r['error'])}")
    if eff:
        eff_sorted = sorted(eff, key=lambda r: abs(r["delta"]))
        print(f"\n有效探测判别力(交叉-组内)中位数: {median(abs(r['delta']) for r in eff_sorted):.3f}")
        print("\n===== 低判别力候选（|交叉-组内| 最小的 10 个，供黑名单参考）=====")
        for r in eff_sorted[:10]:
            print(f"  {r['id']}  交叉={r['cross']:.3f} 组内={r['intra']:.3f} "
                  f"差值={r['delta']:+.3f}")
        print(f"\n===== 收敛/错误探测（无条件剔除，可直接加入黑名单）=====")
        for r in bad:
            tag = "error" if r["error"] else "converged"
            print(f"  {r['id']}  [{tag}]")
        ids = ", ".join(f'"{r["id"]}"' for r in bad)
        print(f"\n建议配置: \"phase2\": {{\"skip_ids\": [{ids}]}}"
              if bad else "\n无收敛/错误探测，黑名单可留空")
    print("\n注意: 判别力与当前对比的模型对相关，更换端点/模型后请重新分析。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/raw/phase2")