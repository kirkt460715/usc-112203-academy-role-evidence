#!/usr/bin/env python3
"""命令行启动脚本。

用隔离模式运行（忽略环境变量与用户站点包，避免资料目录中的同名模块注入）：

    python3 -I run_analysis.py verify --records records
    python3 -I run_analysis.py analyze --records records --out output
"""
import sys
from pathlib import Path

# -I 隔离模式下脚本目录不会自动进入 sys.path，显式加入仓库根目录。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coaching.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
