"""配置加载与校验。

支持 JSON 配置文件 + 命令行覆盖 + 环境变量兜底：
  - 官方 API key 缺省时尝试环境变量 OPENAI_API_KEY
  - 未知 API key 缺省时尝试环境变量 UNKNOWN_API_KEY
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

ENV_OFFICIAL_KEY = "OPENAI_API_KEY"
ENV_UNKNOWN_KEY = "UNKNOWN_API_KEY"


@dataclass
class TargetConfig:
    name: str
    base_url: str
    api_key: str
    model_name: str
    api_style: str = "chat"          # "chat" = /chat/completions；"responses" = /responses


@dataclass
class Options:
    # 通用
    output_dir: str = "output"
    seed: int = 20240517              # 全局随机种子（对抗后缀生成、可复现性）
    request_timeout: float = 90.0
    concurrent: bool = False          # 维度一：同一探测同时对两个 API 发起（默认关闭/串行）
    parallel_phases: bool = False     # 维度二：多个探测阶段并行执行（默认关闭/串行）
    auto_ratelimit: bool = True       # 自动限速：按端点响应头 x-ratelimit-* + 429 自适应
    rate_limit_rpm: int = 120         # 手动固定限速；自动模式下为"端点未下发额度头"时的回退值
    fallback_tpm: int = 30000         # 自动模式下无额度头时的 TPM 回退值（0 = 不限）
    max_inflight: int = 4             # 同时在飞请求数上限（防突发并发顶到模型限制）
    proxy: str = ""                   # 代理地址，如 "http://127.0.0.1:7897"；留空 = 直连
    # 第一阶段
    phase1_enabled: bool = True
    phase1_temperature: float = 0.0
    phase1_max_tokens: int = 512
    # 第二阶段（默认=验证过的精简档；高精度档可改回 50×5×256）
    phase2_enabled: bool = True
    phase2_temperature: float = 1.0
    phase2_samples: int = 3           # 基础采样次数；边界探测自动补测到 5（见 upsample）
    phase2_upsample: bool = True      # 边界补测：交叉相似度落在模糊区(0.35~0.85)且非收敛时补 2 次
    phase2_upsample_low: float = 0.35
    phase2_upsample_high: float = 0.85
    phase2_num_probes: int = 20       # 对抗探测总数（实验：18~20 个与 50 个误判率无差异）
    phase2_skip_ids: list = field(default_factory=list)   # 黑名单探测 id（用 analyze_phase2.py 筛选）
    phase2_suffix_min_len: int = 40   # 乱码后缀长度下限
    phase2_suffix_max_len: int = 60   # 乱码后缀长度上限
    phase2_max_tokens: int = 96       # 生成上限（实验：96 与 256 误判率无差异；真实数据若 P95 接近上限请调回）
    # 第三阶段
    phase3_enabled: bool = True
    phase3_max_tokens: int = 32
    phase3_top_logprobs: int = 5      # top-k 候选数
    # 第四阶段（分词器行为指纹，不依赖 logprobs）
    phase4_enabled: bool = True
    phase4_temperature: float = 0.0
    phase4_max_tokens: int = 256
    # 第五阶段（采样自洽性画像，不依赖 logprobs）
    phase5_enabled: bool = True
    phase5_temperature: float = 1.0
    phase5_samples: int = 8           # 每个开放任务的采样次数 n（10 个任务）
    phase5_max_tokens: int = 200
    # 第六阶段（单 Token 行为指纹——诊断性阶段：分数不参与判定，
    # 只保留"退化现象"诊断（推理型端点/不可读等），6 串 × 4 采样的平衡档）
    phase6_enabled: bool = True
    phase6_temperature: float = 0.0
    phase6_samples: int = 4           # 每个短串的采样次数 n（诊断用，够用即可）
    phase6_max_tokens: int = 1
    phase6_num_probes: int = 0        # 乱码短串数（诊断档=0，只跑 6 个固定刁钻串）
    phase6_timeout: float = 30.0      # 阶段专用请求超时（秒）；0 = 继承 request_timeout
    # 对照基线（不参与判定，只作报告参照系）：官方自比 + 可选已知异源模型
    baseline_self: bool = True        # 官方 vs 官方（自比基线，6 个 prompt）
    baseline_diff_model: str = ""     # 官方端点上的已知异源模型名（如 gpt-4o-mini），留空=关闭
    # 官方侧响应缓存（复用同 output_dir 下参数匹配的官方取证结果，只实时打未知侧；
    # 端点/模型不一致自动失效；--no-cache-official 可关闭）
    cache_official: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Options":
        """兼容两种写法：扁平键（phase2_samples）或嵌套节（phase2: {samples: ...}）。"""
        opts = cls()
        if not d:
            return opts
        for key, value in d.items():
            if isinstance(value, dict):
                for sub, v in value.items():
                    full = f"{key}_{sub}"
                    if hasattr(opts, full) and v is not None:
                        setattr(opts, full, v)
            elif hasattr(opts, key) and value is not None:
                setattr(opts, key, value)
        return opts


@dataclass
class Config:
    official: TargetConfig
    unknown: TargetConfig
    options: Options = field(default_factory=Options)

    @classmethod
    def load(cls, path: str, overrides: Optional[dict] = None) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw, overrides or {})

    @classmethod
    def from_dict(cls, raw: dict, overrides: Optional[dict] = None) -> "Config":
        overrides = overrides or {}
        official_raw = raw.get("official", {})
        unknown_raw = raw.get("unknown", {})

        def build(tag: str, r: dict, env_key: str) -> TargetConfig:
            name = overrides.get(f"{tag}_name") or r.get("name") or tag
            base_url = (overrides.get(f"{tag}_url") or r.get("base_url") or "").rstrip("/")
            api_key = (overrides.get(f"{tag}_key") or r.get("api_key")
                       or os.environ.get(env_key) or "EMPTY")
            model = overrides.get(f"{tag}_model") or r.get("model_name") or ""
            api_style = (overrides.get(f"{tag}_api_style")
                         or r.get("api_style") or "chat")
            if not base_url:
                raise ValueError(f"[{tag}] 缺少 base_url（或在命令行传 --{tag}-url）")
            if not base_url.startswith(("http://", "https://")):
                raise ValueError(f"[{tag}] base_url 必须以 http(s):// 开头: {base_url}")
            if not model:
                raise ValueError(f"[{tag}] 缺少 model_name（或在命令行传 --{tag}-model）")
            if api_style not in ("chat", "responses"):
                raise ValueError(f"[{tag}] api_style 只能是 chat 或 responses: {api_style}")
            return TargetConfig(name=name, base_url=base_url, api_key=api_key,
                                model_name=model, api_style=api_style)

        official = build("official", official_raw, ENV_OFFICIAL_KEY)
        unknown = build("unknown", unknown_raw, ENV_UNKNOWN_KEY)
        opts = Options.from_dict(raw.get("options"))
        if overrides.get("output_dir"):
            opts.output_dir = overrides["output_dir"]
        return cls(official=official, unknown=unknown, options=opts)
