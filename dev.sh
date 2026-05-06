#!/usr/bin/env bash
# XingClaw Linux/macOS 本地调试启动脚本
# 使用方式: ./dev.sh [--mode im|cli] [--transport webhook|longconn]
#
# 环境变量（在运行前设置，或创建 .env 文件）:
#   FEISHU_APP_ID=your_app_id
#   FEISHU_APP_SECRET=your_app_secret
#   FEISHU_VERIFY_TOKEN=your_verify_token       # 可选
#   ANTHROPIC_API_KEY=your_api_key               # 或在 env_api_keys.py 配置
#   OPENAI_API_KEY=your_openai_key               # 可选
#
# 模型配置:
#   XINGCLAW_PROVIDER=anthropic                  # 默认 anthropic
#   XINGCLAW_MODEL_ID=claude-sonnet-4-5          # 默认 claude-sonnet-4-5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 .env（如果存在）
if [ -f .env ]; then
    echo "[dev] Loading .env ..."
    set -a
    source .env
    set +a
fi

# 默认参数
MODE="${MODE:-im}"
TRANSPORT="${TRANSPORT:-webhook}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8787}"
WORKSPACE="${WORKSPACE:-.}"
LOG_LEVEL="${LOG_LEVEL:-debug}"
PROVIDER="${XINGCLAW_PROVIDER:-anthropic}"
MODEL_ID="${XINGCLAW_MODEL_ID:-claude-sonnet-4-5}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";      shift 2 ;;
        --transport)  TRANSPORT="$2";  shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --workspace)  WORKSPACE="$2";  shift 2 ;;
        --log-level)  LOG_LEVEL="$2";  shift 2 ;;
        *)            echo "[dev] Unknown arg: $1"; exit 1 ;;
    esac
done

# 确保已安装（开发模式）
if ! pip show xingclaw &>/dev/null; then
    echo "[dev] Installing xingclaw in editable mode ..."
    uv pip install -e ".[dev]"
fi

if [ "$MODE" = "im" ]; then
    # IM 模式
    if [ -z "${FEISHU_APP_ID:-}" ] || [ -z "${FEISHU_APP_SECRET:-}" ]; then
        echo "[dev] ERROR: FEISHU_APP_ID and FEISHU_APP_SECRET must be set."
        echo "[dev] Set them as env vars or in .env file"
        exit 1
    fi

    ARGS=(
        -m im
        --platform feishu
        --transport "$TRANSPORT"
        --workspace "$WORKSPACE"
        --host "$HOST"
        --port "$PORT"
        --provider "$PROVIDER"
        --model-id "$MODEL_ID"
        --feishu-app-id "$FEISHU_APP_ID"
        --feishu-app-secret "$FEISHU_APP_SECRET"
        --log-level "$LOG_LEVEL"
    )
    if [ -n "${FEISHU_VERIFY_TOKEN:-}" ]; then
        ARGS+=(--feishu-verify-token "$FEISHU_VERIFY_TOKEN")
    fi

    echo "[dev] Starting IM service ($TRANSPORT) on ${HOST}:${PORT} ..."
    echo "[dev] Provider: $PROVIDER | Model: $MODEL_ID"
    python "${ARGS[@]}"

else
    # CLI 交互模式
    echo "[dev] Starting CLI interactive mode ..."
    echo "[dev] Provider: $PROVIDER | Model: $MODEL_ID"
    python -m coding_agent \
        --mode interactive \
        --workspace "$WORKSPACE" \
        --provider "$PROVIDER" \
        --model-id "$MODEL_ID"
fi
