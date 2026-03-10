#!/bin/bash
# 简单启动脚本：
#   ./owlbot.sh           -> 运行主程序 (main.py)
#   ./owlbot.sh wizard    -> 启动 CLI 配置向导
#   ./owlbot.sh wizard-web -> 启动 Web 配置向导

CMD="$1"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python}"

if [ "$CMD" = "wizard" ]; then
  exec "$PYTHON_BIN" wizard_cli.py "$@"
elif [ "$CMD" = "wizard-web" ]; then
  exec "$PYTHON_BIN" wizard_web.py "$@"
else
  exec "$PYTHON_BIN" main.py "$@"
fi

