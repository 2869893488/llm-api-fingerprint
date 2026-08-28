"""代理功能验证：证明配置 proxy 后，ProbeTarget 的所有请求都经过代理。

原理：
  1. 起一个本地 mock OpenAI 端点（127.0.0.1:P1）；
  2. 起一个极简转发代理（127.0.0.1:P2），记录每一笔经过它的请求；
  3. ProbeTarget 配置 proxy=http://127.0.0.1:P2，发起 chat 调用：
     - 请求必须成功返回；
     - 代理必须记录到该请求（证明流量确实走了代理）。

用法: python tests/proxy_check.py
"""
from __future__ import annotations

import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 依赖可能装在 .deps/（沙箱环境）
_deps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".deps")
if os.path.isdir(_deps):
    sys.path.insert(0, _deps)

import mock_server  # noqa: E402
from fpcheck.target import ProbeTarget  # noqa: E402


# ---------------------------------------------------------------- 极简转发代理
class MiniProxy:
    """支持绝对 URI（http）与 CONNECT（https 隧道）的极简转发代理。"""

    def __init__(self, port: int):
        self.requests: list[str] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", port))
        self._srv.listen(16)
        self.port = port
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket):
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
                if len(data) > 1 << 20:
                    return
            head, _, rest = data.partition(b"\r\n\r\n")
            first_line = head.split(b"\r\n", 1)[0].decode("latin1", "replace")
            self.requests.append(first_line)
            parts = first_line.split()
            if len(parts) < 3:
                return
            method, target = parts[0], parts[1]
            if method == "CONNECT":
                host, _, port_s = target.partition(":")
                upstream = socket.create_connection((host, int(port_s or 443)), timeout=30)
                conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                self._relay(conn, upstream)
            else:  # 绝对 URI，如 http://127.0.0.1:P1/chat/completions
                from urllib.parse import urlsplit
                u = urlsplit(target)
                upstream = socket.create_connection((u.hostname, u.port or 80), timeout=30)
                origin = (u.path or "/") + (("?" + u.query) if u.query else "")
                new_head = head.replace(target.encode(), origin.encode(), 1)
                upstream.sendall(new_head + b"\r\n\r\n" + rest)
                self._relay(upstream, conn)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _relay(self, a: socket.socket, b: socket.socket):
        def pump(src, dst):
            try:
                while True:
                    chunk = src.recv(65536)
                    if not chunk:
                        break
                    dst.sendall(chunk)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        t1 = threading.Thread(target=pump, args=(a, b), daemon=True)
        t2 = threading.Thread(target=pump, args=(b, a), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> None:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
        if not cond:
            failures.append(msg)

    # 1) mock OpenAI 端点
    port1 = free_port()
    handler = mock_server.make_handler("same", "official")
    srv = mock_server.ThreadingHTTPServer(("127.0.0.1", port1), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 2) 极简代理
    port2 = free_port()
    proxy = MiniProxy(port2)

    base = f"http://127.0.0.1:{port1}/v1"

    # 3) 对照组：不配置代理 → 直连成功，代理无记录
    direct = ProbeTarget("direct", base, "sk-test", "mock", proxy=None)
    r = direct.chat("Say OK.", max_tokens=8)
    check(r.ok and r.texts and "OK" in r.texts[0], "直连（不配代理）调用成功")
    check(len(proxy.requests) == 0, "直连时代理未收到任何请求")

    # 4) 实验组：配置代理 → 调用成功，且代理记录到请求
    via_proxy = ProbeTarget("via-proxy", base, "sk-test", "mock",
                            proxy=f"http://127.0.0.1:{port2}")
    r2 = via_proxy.chat("Say OK.", max_tokens=8)
    check(r2.ok and r2.texts and "OK" in r2.texts[0], "走代理调用成功")
    check(len(proxy.requests) >= 1, f"代理记录了 {len(proxy.requests)} 笔请求（≥1）")
    if proxy.requests:
        check("chat/completions" in proxy.requests[0],
              f"代理首笔请求: {proxy.requests[0][:80]}")
        print("  → 证明请求确实经过了代理:", proxy.requests[0][:100])

    print()
    if failures:
        print(f"共 {len(failures)} 项失败:")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("代理功能验证全部通过 ✔")


if __name__ == "__main__":
    main()
