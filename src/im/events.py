from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from .types import IMIncomingMessage

logger = logging.getLogger("xingclaw.im.events")


@dataclass
class IMEventWatcherOptions:
    events_dir: str | Path
    poll_interval_sec: float = 1.0


class IMEventWatcher:
    """
    轮询事件目录，支持 immediate / one-shot / periodic。
    """

    def __init__(self, service, options: IMEventWatcherOptions) -> None:  # type: ignore[no-untyped-def]
        self.service = service
        self.options = options
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._events_dir = Path(options.events_dir)
        self._events_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, name="im-event-watcher", daemon=True)
        self._thread.start()
        logger.info("event watcher started dir=%s", self._events_dir)

    def stop(self) -> None:
        self._stopping.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("event watcher stopped")

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                self._scan_once()
            except Exception as exc:  # pragma: no cover
                logger.warning("event watcher scan failed: %s", exc)
            self._stopping.wait(self.options.poll_interval_sec)

    def _scan_once(self) -> None:
        for path in sorted(self._events_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("invalid event file %s: %s", path, exc)
                path.unlink(missing_ok=True)
                continue
            if not isinstance(payload, dict):
                path.unlink(missing_ok=True)
                continue
            event_type = str(payload.get("type") or "immediate")
            if event_type == "immediate":
                self._dispatch_payload(payload)
                path.unlink(missing_ok=True)
                continue
            if event_type == "one-shot":
                run_at = _parse_time(payload.get("run_at"))
                if run_at is None:
                    path.unlink(missing_ok=True)
                    continue
                if datetime.now(timezone.utc) >= run_at:
                    self._dispatch_payload(payload)
                    path.unlink(missing_ok=True)
                continue
            if event_type == "periodic":
                interval = payload.get("interval_sec")
                if not isinstance(interval, (int, float)) or interval <= 0:
                    path.unlink(missing_ok=True)
                    continue
                last_run = _parse_time(payload.get("last_run_at"))
                now = datetime.now(timezone.utc)
                if last_run is None or (now - last_run).total_seconds() >= float(interval):
                    self._dispatch_payload(payload)
                    payload["last_run_at"] = now.isoformat()
                    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            logger.warning("unknown event type=%s file=%s", event_type, path)
            path.unlink(missing_ok=True)

    def _dispatch_payload(self, payload: dict[str, Any]) -> None:
        platform = str(payload.get("platform") or "feishu")
        channel_id = str(payload.get("channel_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not channel_id or not text:
            logger.warning("skip invalid event payload missing channel/text: %s", payload)
            return
        message = IMIncomingMessage(
            platform=platform,
            channel_id=channel_id,
            user_id=str(payload.get("user_id") or "event"),
            text=text,
            thread_id=str(payload.get("thread_id")) if payload.get("thread_id") else None,
            message_id=str(payload.get("message_id") or f"evt-{uuid4().hex}"),
            raw={"event_payload": payload},
        )
        asyncio.run(self.service.handle_incoming_message(message))
        time.sleep(0)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
