# LLM 行为指纹验证工具（未知 API vs 官方 API）

对 **OpenAI 官方 API** 与一个**未知来源的第三方 API** 发送完全相同的探测请求，比较响应的行为特征，寻找"统计上无法用巧合解释的相似性"，用于判断第三方 API 背后是否运行着与官方相同的底层模型。全程不接触模型权重，仅通过黑盒行为取证。

## 工作原理：六阶段递进式验证

| 阶段 | 方法 | 权重 |
|------|------|------|
| 一 | **基础行为对齐**：9 类"性格测试"prompt（自我认知含受约束身份审讯 / 风格指纹 / 分词泄漏 / 系统提示泄漏 / 数值推理 / 固定字符串复读 / 拒答模式 / 严格格式遵循），temperature=0 比较响应，**主指标为平均相似度**（逐字比较对服务端非确定性过于敏感）。**评分只取 5 个自由度类别（24 条）**；固定复读/分词泄漏/严格格式遵循为低熵回声类别（12 条，任何模型都可能逐字一致），仅参考不评分 | 0.15 |
| 二 | **对抗后缀指纹**（参考 [parameterlab/trap](https://github.com/parameterlab/trap)）：每个探测 = 忽略指令前缀 + 40～60 字符乱码后缀（ASCII/希腊/西里尔混合），temperature=1.0 采样；指令前缀带释义扰动变体、探测随机套用 prompt 包装模板（借鉴 [LLMmap](https://github.com/pasquini-dario/LLMmap)，USENIX Security '25）。**默认精简档：20 个探测 × 基础 3 采样 + 边界探测补测到 5 次，max_tokens=96**（对照实验见 `verify_phase2_config.py`：18～20×3×96 与 50×5×256 的误判率无差异，仅 CI 变宽约 68%，已被 CI 降权兜住；高精度档可改回 50/5/256）。剔除"双侧组内相似度=1"的低熵收敛探测后，以**组内相似度为同源基线**归一化：交叉≈组内 → ≈1（同源一致）；交叉系统性低于组内 → 低分（异源证据）。报告附 bootstrap 95% 置信区间，**CI 越宽证据权重自动回调**；组内基线过低（<0.5）报告"双低警告" | 0.25 |
| 三 | **logprob 数值对齐**（辅助加分项）：对固定 prompt 获取每个 token 的 top-k 候选 logprob 分布，比较同一位置的候选集合重叠率。证据力强但**很多第三方端点不支持**，故只作辅助项（0.05），被跳过时不影响主结论（权重自动归一化） | 0.05 |
| 四 | **分词器行为指纹**（不依赖 logprobs）：字符计数 / 字符串倒序 / 罕见字符复读 / 分词边界共 16 个确定性探测，temperature=0。**注意：这是家族级证据**——同族不同模型（如共享分词器的兄弟模型）也可能满分，不单独指向同权重，故权重仅 0.10 | 0.10 |
| 五 | **采样自洽性画像**（借鉴 [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt) 的 sample-and-compare 思想，不依赖 logprobs）：**16 个开放任务** × 每侧采样 8 次，得到每个 API 在每个任务上的组内自洽度画像向量；同一采样分布 → 两端画像 Spearman 相关 ≈1 且幅值相近，不同模型/服务端改写 → 画像分叉。**任务数 10→16 收窄 CI（画像相关标准误 ≈1/√(m-1)，约 -25%）**；确定性端点（无采样方差）自动跳过；CI 越宽权重自动回调 | 0.30 |
| 六 | **单 Token 行为指纹（诊断性阶段，分数不进判定）**（借鉴 [One Token Is Enough](https://arxiv.org/abs/2607.10252) / [vivgrid/llm-fingerprinting](https://github.com/vivgrid/llm-fingerprinting)）：6 个固定短串 × 每串 4 采样（`max_tokens=1`），保留**退化现象诊断**——推理型端点（思考消耗全部输出预算，如部分 o 系/网关模型）**一经识别立即跳过**（usage 显示 reasoning_tokens ≥ output_tokens），读取通道/升级重试/不可读等如实入报告与 summary；其参考重叠率只作辅助参考（6 串 × 4 采样是时间与证据完整性的平衡点） | —（诊断） |
| 七 | **综合判定**：**证据加权求和**——权重 × 成功率（ok/total，下限 0.5）× **CI 可靠性**再归一化；相似度类阶段（一/四/六）先做**官方自比归一化校准**（得分 ÷ self_sim，≤1 截断，自比 <0.5 或未跑时跳过）；**证据冲突规则**：加权分 >0.75 但各阶段得分跨度 >0.45（≥3 个有效阶段）→ 降级"中等怀疑"并标注冲突，多通道必须共同吻合。>0.75 高度确信；0.5～0.75 中等怀疑；<0.5 证据不足。**fail-closed 规则**（借鉴 [llm-verify](https://github.com/asale-ai/llm-verify)，原 mintesnot-teshome/llm-verify）：有效探测总数 < 8 或某阶段成功率 < 80% 时强制判定"证据不足" | — |

另附**身份一致性红旗**（复用第一/二阶段取证，零额外请求）：知识截止声明前后矛盾（**语境限定提取**：只认"截止/训练数据"措辞附近的年份，过滤"成立于 2015 年"等无关年份；不同模型的截止日期本就不同，因此不做"两端年份必须相同"的比较，只抓同一品牌下两端截止年份的**系统性偏差**）、自我身份声称与请求型号品牌不符、两端延迟显著不对称（转发/中继层信号）、官方与未知**跨端自称品牌不一致**、**拒答模式不一致**（不同护栏/内容策略层）、**代理/中继自我披露**（自我描述或系统提示泄漏回答中出现 proxy/relay/中转/代理 等词，参考 llm-verify 的 relay provenance 检查）、**Token 用量不对称**（prompt_tokens 比值 >1.3 疑似注入隐藏 system prompt、completion_tokens 比值 >1.5 疑似附加输出/水印、未知端存在 reasoning_tokens 提示推理型模型替换）。红旗为辅助信号，在报告中标注但不改变判定。

Prompt 设计参考 [litemars/LLM-Fingerprinter](https://github.com/litemars/LLM-Fingerprinter) 的分层探测思路；成对批量探测架构参考 [praetorian-inc/julius](https://github.com/praetorian-inc/julius)（注意：julius 本身是 HTTP 层服务识别工具，本工具仅借鉴其成对探测架构）；随机 prompt 包装与指令释义扰动参考 [LLMmap](https://github.com/pasquini-dario/LLMmap)（[arXiv:2407.15847](https://arxiv.org/abs/2407.15847)，USENIX Security '25）；身份红旗、代理披露检查与 fail-closed 规则参考 [llm-verify](https://github.com/asale-ai/llm-verify)（原名 mintesnot-teshome/llm-verify：32 个取证探针，专查套壳红旗）；采样自洽性画像参考 [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt)（[arXiv:2303.08896](https://arxiv.org/abs/2303.08896)，ACL 2023）；单 Token 指纹参考 [One Token Is Enough](https://arxiv.org/abs/2607.10252)（[vivgrid/llm-fingerprinting](https://github.com/vivgrid/llm-fingerprinting)，另有 [ToseaAI 的 TS 参考实现](https://github.com/ToseaAI/llm-fingerprint-detector)）；严格格式遵循探针借鉴 [Chain & Hash（Russinovich, Cai, Salem，微软）](https://arxiv.org/abs/2407.10887)（[ICLR](https://openreview.net/pdf?id=UWi94bRsgm)）——注意其全套方案需微调模型记忆哈希链，黑盒端点无法复现，本工具只吸收其"严格格式指令"维度；受约束身份审讯（单 token 约束 / 补全句 / 释义变体）参考 Noto et al.《Identifiability of Large Language Models via Reverse Engineering》(ICLR 2025)，并参考了 [LessWrong 上对 190 个模型的单 token 身份扫描实践](https://www.lesswrong.com/posts/KK5pqtrfb8XnmqLka/some-models-don-t-identify-as-their-official-name)（很多模型不回自己的正式型号名，身份自称必须配合行为画像使用）；拒答/护栏行为画像参考模型替代审计类工作（如 [Auditing Model Substitution in LLM APIs](https://arxiv.org/abs/2504.04715)，NeurIPS 2025），其结论"自我身份声称不可靠、需行为证据"与本工具的红旗定位一致。

## 目录结构

```
api-fingerprint/
├── README.md                  # 本文件
├── requirements.txt           # 依赖清单（仅 openai）
├── config.example.json        # 配置模板（复制为 config.json 后填写）
├── run_fingerprint.py         # 入口脚本
├── analyze_phase2.py          # 阶段二取证深度分析（逐探测判别力排名 + 黑名单建议）
├── verify_phase2_config.py    # 阶段二配置精度对照实验（精简/降采样/截断是否伤精度）
├── fpcheck/                   # 核心包
│   ├── __init__.py
│   ├── config.py              # 配置加载与校验（文件 + CLI + 环境变量）
│   ├── console.py             # 分区进度面板（并行时每阶段一块区域，ANSI 无依赖）
│   ├── target.py              # ProbeTarget 统一 chat 接口 + 参数优雅降级
│   ├── prompts.py             # 六阶段探测 prompt 库 + 随机包装模板
│   ├── adversarial.py         # 对抗乱码后缀生成（trap 思路）+ prompt 包装
│   ├── runner.py              # 成对批量探测执行器 + 取证记录
│   ├── storage.py             # 原始请求/响应 JSON 取证存储
│   ├── phase1.py / phase2.py / phase3.py / phase4.py / phase5.py / phase6.py
│   ├── redflags.py            # 身份一致性与 Token 消耗红旗检测（复用第一/二阶段取证）
│   ├── verdict.py             # 加权综合判定 + fail-closed 规则
│   └── report.py              # 文本报告 + summary.json
├── tests/
│   ├── mock_server.py         # 本地 OpenAI 兼容 mock 端点
│   └── run_mock_test.py       # 端到端自测（无需真实 API key）
└── output/                    # 运行后自动生成（报告 + 取证数据 + 运行日志）
    ├── report_YYYYMMDD_HHMMSS.txt
    ├── summary.json
    ├── manifest.json          # 脱敏配置快照
    ├── logs/                  # 每次运行的日志（维护用）
    │   ├── run_YYYYMMDD_HHMMSS.log    # 全流程人类可读日志（版本/脱敏配置/进度/结论）
    │   └── run_YYYYMMDD_HHMMSS.json   # 结构化摘要（判定/耗时/退出码/警告错误计数）
    └── raw/phase{1,2,3,4,5,6}/    # 每次调用的完整取证 JSON
```

## 安装

```bash
# Python 3.10+（已在 3.13 验证）
pip install -r requirements.txt
```

## 配置 API 信息

复制 `config.example.json` 为 `config.json` 并填写：

```jsonc
{
  "official": {
    "name": "官方GPT",                              // 显示名称
    "base_url": "https://api.openai.com/v1",        // OpenAI 官方地址
    "api_key": "sk-...",                            // 官方 key（或环境变量 OPENAI_API_KEY）
    "model_name": "gpt-5.6-luna"                          // 你怀疑的官方模型型号
  },
  "unknown": {
    "name": "未知API",
    "base_url": "https://your-unknown-endpoint.example.com/v1",
    "api_key": "sk-...",                            // 或环境变量 UNKNOWN_API_KEY
    "model_name": "gpt-5.6-luna"                          // 对方声称的型号名
  },
  "options": {
    "output_dir": "output",
    "seed": 20240517,                               // 全局随机种子（保证两次运行可复现）
    "request_timeout": 90,
    "concurrent": true,                             // 维度一：同一探测并行打两个 API（默认开）
    "parallel_phases": true,                        // 维度二：六个阶段并行执行（默认开）
    "auto_ratelimit": true,                         // 自动限速：按端点响应头 x-ratelimit-* + 429 自适应
    "rate_limit_rpm": 120,                          // 手动固定限速；自动模式下=无额度头时的回退值
    "fallback_tpm": 30000,                          // 自动模式下无额度头时的 TPM 回退值（0 = 不限）
    "max_inflight": 4,                              // 同时在飞请求数上限（防突发并发顶到模型限制）
    "proxy": "",                                    // 代理地址；留空 = 直连
    "phase1": { "temperature": 0.0, "max_tokens": 512 },
    "phase2": { "temperature": 1.0, "samples": 3, "upsample": true,
                "num_probes": 20, "skip_ids": [],
                "suffix_min_len": 40, "suffix_max_len": 60, "max_tokens": 96 },
    "phase3": { "max_tokens": 32, "top_logprobs": 5 },
    "phase4": { "temperature": 0.0, "max_tokens": 256 },
    "phase5": { "temperature": 1.0, "samples": 8, "max_tokens": 200 },
    "phase6": { "temperature": 0.0, "samples": 4, "max_tokens": 1,
                "num_probes": 0, "timeout": 30 },
    "baseline_self": true,                          // 官方自比基线（6 个 prompt）
    "baseline_diff_model": ""                       // 已知异源对照模型名，如 "gpt-4o-mini"；留空=关闭
  }
}
```

key 也可以不写进文件：官方 key 用环境变量 `OPENAI_API_KEY`，未知 API 用 `UNKNOWN_API_KEY`。命令行可覆盖任意字段：`--official-key` / `--unknown-key` / `--official-url` / `--unknown-url` / `--official-model` / `--unknown-model` / `--output-dir` / `--proxy`。

### 全程走代理

在 `options.proxy` 里填代理地址，或运行时传 `--proxy http://127.0.0.1:    `，则**所有请求**（预检、五个阶段的每一次探测）都通过该代理发出，`ProbeTarget` 内部为每个端点创建一个挂载代理的 httpx 客户端。传 `--proxy ""` 或把配置留空则为直连（此时若环境变量/系统存在代理，httpx 的 trust_env 会按常规逻辑生效）。可用 `python tests\proxy_check.py` 本地验证代理确实生效（它会起一个记录请求的转发代理做对照）。

## 运行

```bash
python run_fingerprint.py --config config.json
# 跳过 logprob 阶段（很多端点不支持，主结论不受影响）：
python run_fingerprint.py --config config.json --no-phase3
# 跳过分词器行为指纹阶段：
python run_fingerprint.py --config config.json --no-phase4
# 跳过采样自洽性画像阶段（确定性端点本来也会自动跳过）：
python run_fingerprint.py --config config.json --no-phase5
# 跳过单 Token 行为指纹阶段：
python run_fingerprint.py --config config.json --no-phase6
# 并行执行五个阶段（默认开启；时间≈最慢阶段，瞬时请求率更高）：
python run_fingerprint.py --config config.json --parallel-phases
# 强制串行执行阶段（覆盖配置里的 parallel_phases=true，限流时用）：
python run_fingerprint.py --config config.json --serial
# 同一探测改为依次发给两个 API（覆盖配置里的 concurrent=true）：
python run_fingerprint.py --config config.json --no-concurrent
# 关闭分区进度面板（终端不支持 ANSI 时用，退化为普通逐行日志）：
python run_fingerprint.py --config config.json --no-panel
# 关闭官方侧响应缓存（默认开：复用同目录官方取证，只实时请求未知侧）：
python run_fingerprint.py --config config.json --no-cache-official
# 断点续跑（中断后继续）：
python run_fingerprint.py --config config.json --resume
```

运行流程：预检连通性/鉴权 → 逐阶段探测（进度打印到控制台）→ 身份一致性红旗检查 → 综合判定 → 自动生成 `output/report_*.txt` 与 `output/summary.json`。

### 双维度并行 + 限速保护（默认同时开启）

"并行"指两个维度，均**默认开启**：

1. **双 API 并行**（`concurrent: true`，`--no-concurrent` 关闭）：同一探测同时发给官方与未知 API，单探测耗时 ≈ 两者中较慢者；
2. **阶段并行**（`parallel_phases: true`，`--serial` 关闭）：一至六阶段同时推进，总时间 ≈ 最慢阶段。

**限速保护（防触发模型并发限制）**：两维度叠加会把瞬时请求率放大约 10 倍，因此内置**全局限速器**（官方与未知共用、所有阶段共享）：

- `auto_ratelimit: true`（默认）：**自动限速**。限度由两类实时信号决定：
  1. 端点响应头 `x-ratelimit-limit-requests/reset-requests/limit-tokens/reset-tokens`——OpenAI 官方按账号档位与模型实时下发，自动换算成 RPM/TPM 并取两端的保守值；
  2. 429 反馈——一旦触发限流，间隔自动加倍（最多 ×8），宁可慢不触限；
  端点不返回额度头时回退到配置里你显式填写的 `rate_limit_rpm` / `fallback_tpm`；
- `max_inflight: 4`（默认）：同时在飞请求数上限，杜绝并发尖峰；
- 想完全手动：`auto_ratelimit: false` 后 `rate_limit_rpm` 即固定限速（0 = 不限速）；第三方端点限制未知时建议先按 60～120 试跑，看有没有 429。

时间账（默认参数）：

| 模式 | 总时长 |
|------|--------|
| 默认（双维度并行 + 限速 120 RPM，阶段二精简档） | **≈6～8 分钟**（限速是主要地板：≈660 次请求 × 0.5s ≈ 5.5 分钟） |
| 同目录第二次运行（官方侧缓存生效，请求减半） | **≈3～4 分钟** |
| 双维度并行 + 不限速（`rate_limit_rpm: 0`） | ≈5～6 分钟 |
| 全串行（`--serial --no-concurrent`） | ≈11～13 分钟 |

并行只改变执行顺序，**不影响探测内容与判定结果**：同一探测仍逐字节相同、取证与报告与串行一致。

**提速与精度的取舍（默认=实验验证过的精简档，精度与速度兼顾）**：
- **阶段二默认精简**（20 探测 × 3 采样 + 边界补测 × 96 token）：对照实验（`python verify_phase2_config.py`，用合成同源/异源对 + 真实评分代码，150 对/配置）显示与高精度档（50×5×256）**误判率同为 0.0%**，唯一代价是 CI 变宽约 68%——已有 CI 降权兜底。**高精度档**：`phase2: {num_probes: 50, samples: 5, upsample: false, max_tokens: 256}`。
- **官方侧响应缓存**（默认开，`--no-cache-official` 关闭）：同 output_dir 下参数匹配且端点/模型一致的官方取证直接复用，第二次运行只实时请求未知侧，**网络请求减半**。官方模型若已升级，删除 output 目录或加 `--no-cache-official` 重跑。
- **可选黑名单** `phase2.skip_ids`：用 `python analyze_phase2.py` 分析 raw/phase2 取证数据，得到收敛/失败的探测 id（无条件无判别力）与低判别力排行，把要剔除的 id 填进配置。默认空 = 完全由数据决定，不预设裁剪。

**控制台排版（分区进度面板）**：并行时各阶段同时在跑，终端环境下自动切换到**分区进度面板**——屏幕纵向分为 N 块，**每个阶段固定一块面积**（从第一帧起恒定，不随运行内容增长而位移），区域内**滚动显示**最新进度（顶部提示"已滚动 N 行"）；显示行自动**精简**（裁剪后缀长度/包装/采样数/首 token 详情等冗余字段，错误与警告行保持完整原文）；节流重绘（默认 0.3s）、互不干扰；结束后按阶段逐段打印**全量日志**（含已滚动行）供回滚查看。非终端（脚本重定向/CI）或传 `--no-panel` 时退化为普通逐行日志，不受精简影响。

### 断点续跑（--resume）

运行被中断（Ctrl+C / 断网 / 关机）后，加 `--resume` 重跑即可从断点继续：启动时扫描输出目录 `raw/phase*/` 下的取证文件，**参数匹配且成功的探测直接复用本地结果、不重发请求**，只补跑缺失或之前失败的探测。匹配参数为 prompt / temperature / max_tokens / n / logprobs / top_logprobs（seed 除外，因端点实际不支持）；修改了这些配置或换了输出目录则自动视为不匹配、重新请求。不加 `--resume` 时行为与原来完全一致（全部重跑）。

### 无 API key 自测（推荐先跑一遍）

```bash
python tests/run_mock_test.py
```

七个场景全部走本地 mock：同源+参数降级、异源、同源+全参数、同源+拒绝 temperature、同源含采样方差（noisy）、采样方差模式不同（noisy）、并行阶段执行，验证六阶段全流程、随机 prompt 包装、降级逻辑与并行模式结果一致性。

## 结果解读

报告中的关键数字与判定标准：

| 指标 | 含义 |
|------|------|
| 第一阶段 平均相似度 | 相同 prompt 下两个 API 响应的 SequenceMatcher 平均比值（temperature=0）；对服务端非确定性容忍度高于逐字比较 |
| 第二阶段 归一化相似度 + 置信区间 | 剔除低熵收敛探测后，交叉相似度 / 组内相似度（同源基线）；≈1 与同源一致，系统性偏低为异源信号（0～1）。95% CI 整体 >0.9 → 结论稳健；区间宽 → 建议加大样本量 |
| 第三阶段 top-k 候选重叠率 | 同一位置 top-k logprob 候选集合的重叠比例（辅助加分项，端点不支持时跳过） |
| 第四阶段 分词器行为指纹 | 16 个确定性任务的响应平均相似度（字符计数/倒序/复读/切分），不依赖 logprobs |
| 第五阶段 采样自洽性画像 | 16 个开放任务 × 每侧采样 8 次的组内自洽度画像：Spearman 相关 ≈1 且幅值相近 = 同源；相关低 = 两端采样分布不同（异源信号）。确定性端点自动跳过（无采样方差即无判别力） |
| 第六阶段 单 Token 指纹（诊断参考） | 6 串 × 4 采样的重叠系数只作参考，**分数不进综合判定**；重点是保留"退化现象"：推理型端点（思考消耗全部输出预算）一经识别即跳过并注明 |
| 对照基线 + 归一化校准 | 官方自比（同端点同模型应 ≈1，即官方侧的"噪声地板"）：**自比 ≥0.5 时将阶段一/四的相似度得分 ÷ 自比（≤1 截断）后进综合评分**——把"官方侧漂移"从同源证据里扣除；自比过低（<0.5，官方侧不稳定）或未跑时跳过校准（fail-soft）。异源对照（`baseline_diff_model`）应显著更低，仅校验指标体系 |
| 证据可靠性 / 冲突 | 二/五/六阶段的 95% CI 越宽 → 权重自动回调（0.5～1.0）；加权分 >0.75 但阶段跨度 >0.45 → 强制降级"中等怀疑"并标注证据冲突 |
| 身份一致性红旗 | 截止声明矛盾或跨端系统性偏差 / 自称品牌不符 / 延迟显著不对称（**已按每 token 延迟归一化**）/ 跨端自称品牌不一致 / 拒答模式不一致 / 代理中继自我披露 / Token 用量不对称（辅助信号，不改变判定） |

综合评分 = 0.25×一阶段 + 0.30×二阶段 + 0.05×三阶段 + 0.10×四阶段 + 0.30×五阶段（**阶段六为诊断性阶段，分数不进评分**），各阶段权重先按**成功率（ok/total）× CI 可靠性**（下限 0.5）回调再归一化，跳过阶段自动归零重归一化：

- **> 0.75**：高度确信同一底层模型（但若各阶段跨度 >0.45，将被降级为中等怀疑）
- **0.5～0.75**：中等怀疑，建议**扩大高判别力样本量**（增大 `phase2.num_probes`、`phase6_samples`，或增加 logprob prompt）重测
- **< 0.5**：证据不足，大概率不是同一模型
- **fail-closed**：有效探测总数 < 8 或某阶段成功率 < 80% 时，无论分数多少一律判定"证据不足"（探测失败不等于异源证据）

注意事项：
- 第三/五阶段被跳过（如未知 API 不支持 logprobs、确定性端点无采样方差）时权重自动归一化，**主结论由第二、五、六阶段支撑，不受影响**；第四阶段是家族级证据，只作参考性支撑。
- **官方自比归一化校准**：官方自比分数（官方 vs 官方，默认 6 个 prompt）作为"噪声地板"，阶段一/四/六的相似度得分会 ÷ 自比（≤1 截断）再进综合评分——官方侧自身漂移越大，同等观测相似度被解读得越"同源"；自比 <0.5 或未跑时跳过（fail-soft，不放大噪声）。校准明细在报告的对照基线小节与 summary.json 的 `calibration` 字段。
- 权重会随**证据量（成功率）与 CI 宽度**自动回调：某阶段探测失败多或置信区间宽，它在结论里的话语权自动变小。
- 证据冲突：各阶段得分分歧大（跨度 >0.45）时，即使加权分 >0.75 也降级为"中等怀疑"——多通道必须共同吻合才算数。
- 报告含**对照基线**：官方自比（应 ≈1）与可选已知异源对照（应显著更低），用于校准对分数的预期。
- 结论是"强证据"而非绝对证明：同一模型可能被服务端改写/微调掩盖，不同模型也可能在低熵任务上趋同。
- 两个相同配置的 API 测出非满分是正常现象：服务端 GPU 浮点非确定性、多副本负载均衡会让 temperature=0 的输出也产生微小漂移（第一阶段用相似度评分吸收这部分噪声）；第二阶段 temperature=1.0 采样本身就有方差，看置信区间而非单点分数；第五阶段看画像相关而非逐任务自洽度；红旗里的延迟比较已按输出长度归一化（每 token 延迟）。

## 降级与异常处理

- 未知 API 不支持 `seed` / `logprobs` / `top_logprobs` / `n>1` 时，按 `logprobs → seed → n` 的顺序自动去掉参数重试，全程不崩溃；实际支持能力记录在取证 JSON 的 `capabilities` 字段。
- 网络抖动/限流由 SDK 自动重试；单次调用失败只影响该 probe，其余探测继续，失败详情写入取证数据并在报告中警告。
- 预检阶段即失败（连接不上 / 401 / 模型不存在）会快速退出并给出明确错误。

## 取证数据

`output/raw/phase{N}/` 下每个文件对应一次 API 调用，包含：probe 标识、prompt 原文、完整响应（含 logprobs）、时间戳（UTC）、API 端点、模型名、实际使用的能力、延迟与错误。`manifest.json` 保存脱敏配置快照（不含明文 key）。

## 运行日志

每次运行在 `output/logs/` 生成一份配对日志：

- **`run_<时间戳>.log`**：全流程人类可读——工具版本、官方/未知端点（key 脱敏）、命令行参数（密钥类脱敏）、预检、各阶段进度、警告/错误行、综合判定与报告路径；**每行带运行时刻戳（HH:MM:SS）**，便于核对"哪一步在什么时间发生"（如定位超时卡点）；分区进度面板模式下会在结束时把各阶段全量日志镜像进来。
- **`run_<时间戳>.json`**：结构化摘要——`started_at`/`finished_at`、`tool_version`、脱敏 args/端点、`phase_scores`、`phase_elapsed_sec`（各阶段耗时，定位变慢阶段）、`verdict`（含 fail_closed/conflict/skipped）、`exit_code`、`elapsed_sec`、`warnings/errors` 列表、报告与 summary 路径。

**失败也落盘**：所有退出路径都会收尾写日志——预检失败、Ctrl+C、未捕获异常（rc=2）、`sys.exit`、**配置缺失/配置损坏等早期失败**（此时用默认 `output/logs/` 目录落盘，note 里写明原因）；启用但运行异常的阶段会显式记入 `errors` 并在控制台标注。日志系统本身容错：写盘失败只降级为不落日志，不影响主流程；`api_key` 全程脱敏，不会出现在 .log/.json 中。

## 常见问题

- **401 / 403**：api_key 错误，或未知 API 的 key 不是 OpenAI 格式。
- **model not found**：`model_name` 与对方端点实际暴露的模型名不一致（有些代理需要 `gpt-4o-mini` 等具体型号）。
- **429 限流**：默认自动限速会自动加倍保守；仍频繁 429 就加 `--serial --no-concurrent`、把 `rate_limit_rpm`（自动模式下是回退值）调低到 60 或 30、降低 `phase2_samples` / `phase5_samples`，或增大 `request_timeout`。
- **第六阶段超时/过慢**：Responses 通道的 n 采样已并行化（并发由全局限速器约束），并默认 30s 阶段超时（`phase6.timeout`，0=继承）。仍慢就把 `samples` 降到 4、`num_probes` 降到 6。
- **第六阶段提示跳过**：诊断性阶段在**识别到推理型端点**（usage 显示思考消耗全部输出预算、无可见首 token）时立即跳过并注明原因——不再做升级预算重试；该阶段的参考重叠率与「退化现象」（读取通道/重试/不可读）仍如实写入报告与 summary.json（`phase6_*` 字段）。可直接 `--no-phase6` 或配置 `phase6.enabled: false` 移除。
- **第五阶段提示跳过**：确定性端点（忽略 temperature、采样零方差）或有效开放任务不足时自动跳过，与第三阶段"不支持 logprobs"同理——权重自动归一化，不影响主结论。
- **Windows 控制台乱码**：入口已自动将 stdout 切为 UTF-8；报告与 JSON 均为 UTF-8 编码。
- **进度面板乱屏/不刷新**：旧 Windows 控制台（conhost / PowerShell 5.1）对全屏刷新支持不稳。现在会自动检测：仅在 Windows Terminal / VSCode 终端（WT_SESSION/TERM_PROGRAM）启用分区面板，其余终端自动退化为普通逐行日志并提示；仍可 `--no-panel` 强制关闭。
- **超时**：把 `request_timeout` 调大；部分第三方端点响应很慢。

## 版本记录

- **1.2.0**（当前）—— 六阶段流水线：单 Token 指纹（阶段六，含"content 空 → logprobs 补读 → max_tokens=2 重试"三级读取兜底）、严格格式遵循与拒答画像（阶段一）、证据加权 + CI 可靠性降权 + 证据冲突降级、官方自比/异源对照基线、官方侧响应缓存、阶段二 20 探测 × 3 采样 + 边界补测 + max_tokens=96 精简档（对照实验 `verify_phase2_config.py` 验证误判率无差异）、分区进度面板、双维度并行 + 全局限速（响应头/429 感知自动限速）、代理披露/失败率红旗、调研与自测脚本（`analyze_phase2.py` / `tests/run_mock_test.py` 9 场景）
- **1.1.0** —— 五阶段：采样自洽画像（阶段五）、身份红旗扩展、fail-closed 证据规则
- **1.0.0** —— 初版：四阶段 + 断点续跑 + 取证存储

## 免责声明

本工具仅用于合法的 API 行为研究。请遵守目标服务的服务条款，勿用于规避鉴权或恶意探测。
