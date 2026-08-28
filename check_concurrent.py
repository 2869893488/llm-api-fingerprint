"""验证并行模式是否生效：配置值 + manifest 快照 + 取证数据时间重叠分析。"""
import json
import os
from datetime import datetime

print("config.json concurrent =",
      json.load(open("config.json", encoding="utf-8"))["options"]["concurrent"])
print()

for d in ("output", "output_smoke", "output_parallel"):
    mf = os.path.join(d, "manifest.json")
    if os.path.exists(mf):
        m = json.load(open(mf, encoding="utf-8"))
        ts = m["timestamp"]
        cc = m["options"].get("concurrent")
        print(f"[{d}] manifest 运行时间={ts}  concurrent={cc}")

        # 分析取证数据：同一 probe 的两个目标调用时间区间是否重叠
        raw = os.path.join(d, "raw")
        pairs = {}
        for phase_dir in sorted(os.listdir(raw)) if os.path.isdir(raw) else []:
            pdir = os.path.join(raw, phase_dir)
            if not os.path.isdir(pdir):
                continue
            for fn in os.listdir(pdir):
                if not fn.endswith(".json"):
                    continue
                try:
                    e = json.load(open(os.path.join(pdir, fn), encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if e.get("error"):
                    continue
                key = (e["phase"], e["probe_id"])
                end = datetime.fromisoformat(e["timestamp"])
                start_ms = end.timestamp() - (e.get("latency_ms") or 0) / 1000.0
                pairs.setdefault(key, {})[e["target"]] = (start_ms, end.timestamp())
        overlap = total = 0
        for k, v in pairs.items():
            if len(v) == 2:
                (s1, e1), (s2, e2) = v.values()
                total += 1
                # 重叠时长 > 2s 才算并行（时间戳精度为秒，串行衔接处会有 <1s 的截断噪声）
                if min(e1, e2) - max(s1, s2) > 2.0:
                    overlap += 1
        if total:
            print(f"    取证分析: {total} 个成对探测中 {overlap} 个时间重叠 "
                  f"({overlap / total:.0%}) → {'并行生效' if overlap / total > 0.5 else '串行'}")
        else:
            print("    取证分析: 无完整成对数据")
        print()
