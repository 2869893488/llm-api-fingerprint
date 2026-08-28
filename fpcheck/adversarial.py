"""对抗后缀生成（核心方法，思路参考 parameterlab/trap）。

为每个探测请求附加一段随机乱码后缀（40~60 字符，混合 ASCII 字母、
希腊字母与西里尔字符），放大模型之间的"本能反应"差异。
正常 prompt 下各家模型行为趋同，面对对抗扰动时各自的偏好会暴露出来。
"""
from __future__ import annotations

import random
import string

from .prompts import PROMPT_WRAPPERS, wrap_prompt

GREEK = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
CYRILLIC = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
ASCII_POOL = string.ascii_letters + string.digits + ".,;:!?_"

# 各字符池权重：ASCII 40% / 希腊 30% / 西里尔 30%
_POOLS = [(ASCII_POOL, 0.40), (GREEK, 0.30), (CYRILLIC, 0.30)]


def generate_suffix(rng: random.Random, min_len: int, max_len: int) -> str:
    """生成一段 40~60 字符的乱码后缀。"""
    length = rng.randint(min_len, max_len)
    chars: list[str] = []
    for _ in range(length):
        if rng.random() < 0.08:          # 偶尔插入空格，制造更自然的"乱码感"
            chars.append(" ")
            continue
        pool = rng.choices(_POOLS, weights=[w for _, w in _POOLS])[0][0]
        chars.append(rng.choice(pool))
    return "".join(chars)


def build_adversarial_probes(prefixes: list[str], num_probes: int,
                             min_len: int, max_len: int, seed: int) -> list[dict]:
    """生成 num_probes 个对抗探测样本。

    同一批样本对两个 API 完全一致（RNG 以固定 seed 初始化），
    保证"完全相同的探测请求"这一前提。每个探测随机套用一个
    prompt 包装模板（借鉴 LLMmap），抵御服务端"外包提示工程"。
    返回形如：
    [{"id": "adv_000", "prefix": ..., "suffix": ..., "prompt": ...,
      "wrapper": "sys_basic"}, ...]
    """
    rng = random.Random(seed)
    wrappers = [w[0] for w in PROMPT_WRAPPERS]
    probes = []
    for i in range(num_probes):
        prefix = prefixes[i % len(prefixes)]
        suffix = generate_suffix(rng, min_len, max_len)
        wrapper = rng.choice(wrappers)
        probes.append({
            "id": f"adv_{i:03d}",
            "prefix": prefix,
            "suffix": suffix,
            "wrapper": wrapper,
            "prompt": wrap_prompt(f"{prefix}\n\n{suffix}", wrapper),
        })
    return probes
