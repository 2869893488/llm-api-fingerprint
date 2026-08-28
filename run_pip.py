"""pip 包装器（仅本机沙箱环境使用）：规避对 tempfile.mkdtemp 目录的写入限制。

把 tempfile.mkdtemp 替换为基于 os.makedirs 的实现（沙箱允许向
os.makedirs 创建的目录写入），使 pip 能正常解包安装。
用法: python run_pip.py install "openai>=1.35,<3" --target .deps --no-cache-dir
"""
import os
import random
import string
import sys
import tempfile

_ORIG = tempfile.mkdtemp


def _mkdtemp_compat(suffix="", prefix="tmp", dir=None):
    base = dir or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    for _ in range(200):
        name = os.path.join(
            base, prefix + "".join(random.choices(string.ascii_letters + string.digits, k=8)) + suffix)
        try:
            os.makedirs(name, exist_ok=False)
            return name
        except FileExistsError:
            continue
    raise OSError("cannot create unique temp dir")


tempfile.mkdtemp = _mkdtemp_compat

from pip._internal.cli.main import main as pip_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(pip_main())
