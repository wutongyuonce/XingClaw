from __future__ import annotations

"""
内置模型表（第一阶段只放 Anthropic + OpenAI 标准）。
"""

from .types import Model

_MODELS: dict[str, dict[str, Model]] = {
    "anthropic": {
        "claude-sonnet-4-5": Model(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            reasoning=True,
            input=["text", "image"],
            context_window=200_000,
            max_tokens=8192,
        ),
        "glm-4.7": Model(
            id="glm-4.7",
            name="GLM-4.7",
            api="anthropic-messages",
            provider="anthropic",
            base_url="https://open.bigmodel.cn/api/anthropic",
            reasoning=True,
            input=["text", "image"],
            context_window=200_000,
            max_tokens=8192,
        ),
    },
    "openai-standard": {
        "gpt-4o-mini": Model(
            id="gpt-4o-mini",
            name="GPT-4o mini",
            api="openai-standard",
            provider="openai-standard",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input=["text", "image"],
            context_window=128_000,
            max_tokens=16_384,
        )
    },
}


def get_model(provider: str, model_id: str) -> Model:
    """获取单个模型，找不到会抛 KeyError。"""
    try:
        return _MODELS[provider][model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {provider}/{model_id}") from exc


def get_models(provider: str) -> list[Model]:
    """获取某 provider 的全部模型。"""
    return list(_MODELS.get(provider, {}).values())


def get_providers() -> list[str]:
    """列出当前内置 provider。"""
    return list(_MODELS.keys())
