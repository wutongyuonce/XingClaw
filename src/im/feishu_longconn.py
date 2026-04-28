from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from .service import IMService
from .types import IMIncomingMessage

logger = logging.getLogger("xingclaw.im.feishu.longconn")


@dataclass
class FeishuLongConnOptions:
    app_id: str
    app_secret: str
    log_level: str = "info"


def run_feishu_long_connection(service: IMService, options: FeishuLongConnOptions) -> None:
    """使用飞书 SDK 长连接模式收事件。"""

    lark = _import_lark_sdk()
    _warn = getattr(lark.LogLevel, "WARN", None) or getattr(lark.LogLevel, "WARNING", None) or lark.LogLevel.INFO
    log_level_map = {
        "debug": lark.LogLevel.DEBUG,
        "info": lark.LogLevel.INFO,
        "warning": _warn,
        "error": lark.LogLevel.ERROR,
    }

    def _on_p2_im_message_receive_v1(data: Any) -> None:
        """lark-oapi EventDispatcher 回调，data 是 P2ImMessageReceiveV1 对象。"""
        try:
            raw = _to_dict(data)
            logger.debug("longconn received event payload keys=%s", list(raw.keys()) if raw else "empty")
            msg = _parse_event_object(data) or _parse_ws_message(raw)
            if msg is None:
                logger.debug("longconn event ignored (not a valid text message)")
                return
            logger.info(
                "longconn dispatching message chat_id=%s user=%s text=%r",
                msg.channel_id, msg.user_id, msg.text[:80],
            )
            _run_async(service.handle_incoming_message(msg))
        except Exception as exc:
            logger.exception("failed to process long connection event: %s", exc)

    logger.info("starting feishu long connection")

    event_handler = (
        lark.EventDispatcherHandler
        .builder("", "")
        .register_p2_im_message_receive_v1(_on_p2_im_message_receive_v1)
        .build()
    )

    client = lark.ws.Client(
        options.app_id,
        options.app_secret,
        event_handler=event_handler,
        log_level=log_level_map.get(options.log_level.lower(), lark.LogLevel.INFO),
    )
    client.start()


def _run_async(coro: Any) -> None:
    """在已有事件循环中调度协程，没有则新建。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import threading

        result_event = threading.Event()
        exception_holder: list[BaseException] = []

        def _thread_target() -> None:
            try:
                asyncio.run(coro)
            except Exception as exc:
                exception_holder.append(exc)
            finally:
                result_event.set()

        t = threading.Thread(target=_thread_target, daemon=True)
        t.start()
        result_event.wait(timeout=120)
        if exception_holder:
            logger.exception("async task failed: %s", exception_holder[0])
    else:
        asyncio.run(coro)


def _parse_event_object(data: Any) -> IMIncomingMessage | None:
    """尝试从 lark-oapi 的强类型事件对象中提取消息。"""
    event = getattr(data, "event", None)
    if event is None:
        return None

    message = getattr(event, "message", None)
    sender = getattr(event, "sender", None)
    if message is None or sender is None:
        return None

    sender_type = getattr(sender, "sender_type", "")
    if isinstance(sender_type, str) and sender_type.lower() in {"app", "bot"}:
        return None

    msg_type = getattr(message, "message_type", "")
    if msg_type != "text":
        logger.info("ignored non-text message_type=%s", msg_type)
        return None

    content_text = ""
    raw_content = getattr(message, "content", "")
    if isinstance(raw_content, str) and raw_content:
        try:
            content_obj = json.loads(raw_content)
            if isinstance(content_obj, dict):
                content_text = str(content_obj.get("text", ""))
        except Exception:
            content_text = raw_content

    mentions = getattr(message, "mentions", None) or []
    content_text = _strip_bot_mentions_from_objects(content_text, mentions)

    chat_id = getattr(message, "chat_id", "") or ""
    message_id = getattr(message, "message_id", "") or ""
    root_id = getattr(message, "root_id", None)
    thread_id = str(root_id) if isinstance(root_id, str) and root_id else None

    create_time_raw = getattr(message, "create_time", None)
    created_at: float | None = None
    if create_time_raw:
        try:
            ts = int(create_time_raw)
            created_at = ts / 1000.0 if ts > 1e12 else float(ts)
        except (ValueError, TypeError):
            pass

    sender_id_obj = getattr(sender, "sender_id", None)
    sender_id = ""
    if sender_id_obj is not None:
        for key in ("open_id", "user_id", "union_id"):
            value = getattr(sender_id_obj, key, None)
            if isinstance(value, str) and value:
                sender_id = value
                break

    if not chat_id or not content_text:
        return None

    return IMIncomingMessage(
        platform="feishu",
        channel_id=str(chat_id),
        user_id=sender_id or "unknown",
        text=content_text,
        thread_id=thread_id,
        message_id=str(message_id) if message_id else None,
        created_at=created_at,
        raw=_to_dict(data),
    )


def _parse_ws_message(payload: dict[str, Any]) -> IMIncomingMessage | None:
    """从原始 dict 解析消息（兼容旧版 SDK 或 webhook 透传格式）。"""
    header = payload.get("header")
    event_type = header.get("event_type") if isinstance(header, dict) else ""
    if event_type != "im.message.receive_v1":
        return None

    event = payload.get("event")
    if not isinstance(event, dict):
        return None
    message = event.get("message")
    sender = event.get("sender")
    if not isinstance(message, dict) or not isinstance(sender, dict):
        return None

    sender_type = sender.get("sender_type")
    if isinstance(sender_type, str) and sender_type.lower() in {"app", "bot"}:
        return None
    if message.get("message_type") != "text":
        return None

    content_text = ""
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        try:
            content_obj = json.loads(raw_content)
            if isinstance(content_obj, dict):
                content_text = str(content_obj.get("text", ""))
        except Exception:
            content_text = raw_content

    mentions = message.get("mentions") or []
    content_text = _strip_bot_mentions_from_dicts(content_text, mentions)

    chat_id = str(message.get("chat_id", ""))
    message_id = str(message.get("message_id", ""))
    root_id = message.get("root_id")
    thread_id = str(root_id) if isinstance(root_id, str) and root_id else None

    create_time_raw = message.get("create_time")
    created_at: float | None = None
    if create_time_raw:
        try:
            ts = int(create_time_raw)
            created_at = ts / 1000.0 if ts > 1e12 else float(ts)
        except (ValueError, TypeError):
            pass

    if not chat_id or not content_text:
        return None

    sender_id = ""
    sender_node = sender.get("sender_id")
    if isinstance(sender_node, dict):
        for key in ("open_id", "user_id", "union_id"):
            value = sender_node.get(key)
            if isinstance(value, str) and value:
                sender_id = value
                break

    return IMIncomingMessage(
        platform="feishu",
        channel_id=chat_id,
        user_id=sender_id or "unknown",
        text=content_text,
        thread_id=thread_id,
        message_id=message_id or None,
        created_at=created_at,
        raw=payload,
    )


def _strip_bot_mentions_from_objects(text: str, mentions: list[Any]) -> str:
    """从文本中移除 @机器人 占位符（lark-oapi 强类型对象）。"""
    for m in mentions:
        mentioned_type = getattr(m, "mentioned_type", "") or ""
        if mentioned_type.lower() != "bot":
            continue
        key = getattr(m, "key", "") or ""
        if key and key in text:
            text = text.replace(key, "")
    return text.strip()


def _strip_bot_mentions_from_dicts(text: str, mentions: list[dict[str, Any]]) -> str:
    """从文本中移除 @机器人 占位符（dict 格式）。"""
    for m in mentions:
        if not isinstance(m, dict):
            continue
        mentioned_type = m.get("mentioned_type", "") or ""
        if mentioned_type.lower() != "bot":
            continue
        key = m.get("key", "") or ""
        if key and key in text:
            text = text.replace(key, "")
    return text.strip()


def _to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(payload, attr, None)
        if callable(fn):
            try:
                obj = fn()
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    for attr in ("raw_body", "raw", "body"):
        value = getattr(payload, attr, None)
        if isinstance(value, (bytes, str)):
            try:
                obj = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return {}


def _import_lark_sdk():
    try:
        import lark_oapi as lark  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Feishu long connection requires `lark-oapi`. Install with: pip install lark-oapi"
        ) from exc
    return lark
