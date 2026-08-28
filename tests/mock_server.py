"""本地模拟 OpenAI 兼容端点，用于端到端自测（不真实联网、不消耗 token）。

模式说明:
  same —— 与"官方"同源：对相同 prompt 返回相同文本，用于验证"同底层模型"场景
  diff —— 不同"模型"：返回与 prompt 无关的固定文本，用于验证"不同模型"场景
  noisy —— 每 API 有采样方差（不同 choice 索引累加波浪号，数量由 prompt 哈希
           决定），用于验证第五阶段"采样自洽性画像"：
           noise="shared" 时两端方差模式一致（同源画像）；
           noise="label"  时方差模式与端点身份绑定（异源画像）
  --reject-seed / --reject-logprobs / --reject-n / --reject-temperature ——
          模拟不支持这些参数的第三方服务，用于验证 ProbeTarget 的优雅降级逻辑

用法:
    python tests/mock_server.py --port 18080 --mode same
    python tests/mock_server.py --port 18081 --mode same --reject-seed --reject-logprobs
    python tests/mock_server.py --port 18082 --mode diff
    python tests/mock_server.py --port 18083 --mode noisy --noise shared
"""
from __future__ import annotations

import argparse
import json
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _noise_count(key: str, index: int) -> int:
    """按 (key, 采样下标) 加盐的确定性噪声计数 1..6。

    同一 key（noise=shared 两端 key 相同）→ 两端方差序列逐采样一致；
    不同 key（noise=label 含端点身份）→ 方差序列互不相关。
    """
    return zlib.crc32((key + f"|{index}").encode("utf-8")) % 6 + 1


def build_text(mode: str, label: str, prompt: str, noise: str = "shared",
               index: int = 0) -> str:
    if mode == "same":
        # 同源模式：两个 API 对相同 prompt 必须返回逐字节相同的文本
        # （因此不掺入各自的 label）
        return f"ANSWER[{prompt[:50]}]"
    if mode == "noisy":
        # 采样方差：每采样的波浪号数量由 (key, 采样下标) 的 crc 决定；
        # key 在 noise=shared 时为 prompt（两端一致），
        # 在 noise=label 时为 prompt+|label（两端分叉）。
        key = prompt if noise == "shared" else prompt + "|" + label
        base = f"NOISY[{prompt[:40]}]" if noise == "shared" \
            else f"NOISY[{label}][{prompt[:40]}]"
        return base + "~" * _noise_count(key, index)
    return f"DIFF[{label}]"


def make_handler(mode: str, label: str, reject_seed: bool = False,
                 reject_logprobs: bool = False, reject_n: bool = False,
                 reject_temperature: bool = False, noise: str = "shared",
                 blank_on_max1: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")

            if reject_seed and body.get("seed") is not None:
                return self._error(400, "Parameter 'seed' is not supported by this server")
            if reject_logprobs and body.get("logprobs"):
                return self._error(400, "Parameter 'logprobs' is not supported by this server")
            if reject_n and body.get("n", 1) != 1:
                return self._error(400, "Parameter 'n' is not supported by this server")
            if reject_temperature and "temperature" in body:
                return self._error(400, "Parameter 'temperature' is not supported by this server")

            # blank_on_max1：模拟"官方端 max_tokens=1 返回空 content 且无 logprobs"
            # （首 token 为空白类 token 被 API 归一化），用于验证阶段六的重试兜底；
            # max_tokens≠1 时行为与普通 same 模式一致。
            if blank_on_max1 and body.get("max_tokens") == 1:
                n = body.get("n", 1)
                resp = {"id": "chatcmpl-mock", "object": "chat.completion",
                        "created": 1234567890, "model": body.get("model", "mock"),
                        "choices": [{"index": i,
                                     "message": {"role": "assistant", "content": ""},
                                     "finish_reason": "length"}
                                    for i in range(n)],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 0,
                                  "total_tokens": 3}}
                j = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(j)))
                self.end_headers()
                self.wfile.write(j)
                return

            prompt = body["messages"][0]["content"]
            n = body.get("n", 1)
            choices = []
            for i in range(n):
                text = build_text(mode, label, prompt, noise, i)
                choice = {"index": i,
                          "message": {"role": "assistant", "content": text},
                          "finish_reason": "stop"}
                if body.get("logprobs"):
                    # 候选池：same 模式与 noisy-shared 两方一致（重叠率高），
                    # diff 模式与 noisy-label 不一致（重叠率低）
                    cands = ["α", "β", "γ", "δ", "ε"] \
                        if (mode == "same" or (mode == "noisy" and noise == "shared")) \
                        else ["1", "2", "3", "4", "5"]
                    content = []
                    for ch in text:
                        top = [{"token": ch, "logprob": -0.2,
                                "bytes": list(ch.encode("utf-8"))}]
                        top += [{"token": c, "logprob": -1.0 - j,
                                 "bytes": list(c.encode("utf-8"))}
                                for j, c in enumerate(cands) if c != ch]
                        content.append({"token": ch, "logprob": -0.2,
                                        "bytes": list(ch.encode("utf-8")),
                                        "top_logprobs": top[:5]})
                    choice["logprobs"] = {"content": content}
                choices.append(choice)

            resp = {"id": "chatcmpl-mock", "object": "chat.completion",
                    "created": 1234567890, "model": body.get("model", "mock"),
                    "choices": choices,
                    "usage": {"prompt_tokens": 10,
                              "completion_tokens": len(text),
                              "total_tokens": 10 + len(text)}}
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str) -> None:
            data = json.dumps({"error": {"message": message,
                                         "type": "invalid_request_error"}}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # 静默访问日志
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI 兼容 mock 端点")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=["same", "diff", "noisy"], default="same")
    ap.add_argument("--label", default="mock")
    ap.add_argument("--noise", choices=["shared", "label"], default="shared")
    ap.add_argument("--reject-seed", action="store_true")
    ap.add_argument("--reject-logprobs", action="store_true")
    ap.add_argument("--reject-n", action="store_true")
    ap.add_argument("--reject-temperature", action="store_true")
    ap.add_argument("--blank-on-max1", action="store_true",
                    help="max_tokens=1 时返回空 content（模拟官方端首 token 归一化）")
    args = ap.parse_args()

    handler = make_handler(args.mode, args.label, args.reject_seed,
                           args.reject_logprobs, args.reject_n,
                           args.reject_temperature, args.noise,
                           args.blank_on_max1)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"mock server [{args.mode}] label={args.label} noise={args.noise} "
          f"reject_seed={args.reject_seed} reject_logprobs={args.reject_logprobs} "
          f"reject_n={args.reject_n} reject_temperature={args.reject_temperature} "
          f"listening on {args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
