from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from im.feishu_longconn import _parse_ws_message


class FeishuLongConnTests(unittest.TestCase):
    def test_parse_text_event(self) -> None:
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_type": "user", "sender_id": {"open_id": "ou_1"}},
            },
        }
        msg = _parse_ws_message(payload)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertEqual(msg.channel_id, "oc_1")
        self.assertEqual(msg.message_id, "om_1")
        self.assertEqual(msg.thread_id, None)

    def test_ignore_bot_message(self) -> None:
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_id": "oc_1",
                    "message_id": "om_1",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_type": "app", "sender_id": {"open_id": "ou_bot"}},
            },
        }
        self.assertIsNone(_parse_ws_message(payload))


if __name__ == "__main__":
    unittest.main()
