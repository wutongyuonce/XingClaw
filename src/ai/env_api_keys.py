from __future__ import annotations

"""
统一读取环境变量中的 API Key。
"""

import os


def get_env_api_key(provider: str) -> str | None:
    # Anthropic provider
    if provider == "anthropic":
        # return "9d96c1c9f4cb41d1aa6f55b0641478bc.stGL7zCVisZi0I68"
        return os.getenv("ANTHROPIC_API_KEY")
    # OpenAI 标准/兼容 provider
    if provider in {"openai", "openai-compatible", "openai-standard"}:
        return os.getenv("OPENAI_API_KEY")
    return None
