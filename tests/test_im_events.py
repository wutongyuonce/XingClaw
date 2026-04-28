from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from im.events import IMEventWatcher, IMEventWatcherOptions


class _FakeService:
    def __init__(self) -> None:
        self.messages = []

    async def handle_incoming_message(self, message) -> None:
        self.messages.append(message)


class IMEventWatcherTests(unittest.TestCase):
    def test_scan_once_dispatches_immediate_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            events_dir = Path(tmp_dir) / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "type": "immediate",
                "platform": "feishu",
                "channel_id": "c1",
                "text": "hello from event",
            }
            event_file = events_dir / "e1.json"
            event_file.write_text(json.dumps(payload), encoding="utf-8")

            service = _FakeService()
            watcher = IMEventWatcher(service, IMEventWatcherOptions(events_dir=events_dir))
            watcher._scan_once()  # noqa: SLF001 - 测试内部扫描逻辑

            self.assertEqual(len(service.messages), 1)
            self.assertFalse(event_file.exists())


if __name__ == "__main__":
    unittest.main()
