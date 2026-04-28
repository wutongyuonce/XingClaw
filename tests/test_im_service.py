from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai.types import AssistantMessage, TextContent
from im.service import IMService, IMServiceConfig
from im.types import IMIncomingMessage, IMOutgoingText, IMWebhookResult


class _FakeAdapter:
    def __init__(self, incoming: list[IMIncomingMessage]) -> None:
        self.incoming = incoming
        self.sent: list[IMOutgoingText] = []

    def handle_webhook(self, headers, body) -> IMWebhookResult:
        _ = headers, body
        return IMWebhookResult(ack={"code": 0}, messages=self.incoming)

    def send_text(self, message: IMOutgoingText) -> None:
        self.sent.append(message)


class _FakeSession:
    def __init__(self) -> None:
        self.messages = []
        self.extension_commands = {
            "ext_ping": type(
                "Cmd",
                (),
                {"name": "ext_ping", "description": "ext ping", "source": "extension", "handler": staticmethod(lambda ctx: "pong")},
            )()
        }

    async def prompt(self, text: str) -> None:
        self.messages.append(
            AssistantMessage(
                content=[TextContent(text=f"收到：{text}")],
                stop_reason="stop",
            )
        )

    def close(self) -> None:
        return None


class _SlowSession(_FakeSession):
    async def prompt(self, text: str) -> None:
        await asyncio.sleep(0.05)
        await super().prompt(text)


class IMServiceTests(unittest.TestCase):
    def test_handle_webhook_and_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )

            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                ack = asyncio.run(service.handle_webhook({}, b"{}"))

            self.assertEqual(ack, {"code": 0})
            create_mock.assert_called_once()
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("收到：你好", adapter.sent[0].text)

    def test_router_reuses_same_session_for_same_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(platform="feishu", channel_id="c1", user_id="u1", text="a", thread_id="t1"),
                    IMIncomingMessage(platform="feishu", channel_id="c1", user_id="u1", text="b", thread_id="t1"),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))

            calls = create_mock.call_args_list
            self.assertEqual(len(calls), 2)
            first_sid = calls[0].args[0].session_id
            second_sid = calls[1].args[0].session_id
            self.assertEqual(first_sid, second_sid)

    def test_duplicate_message_id_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m-dup-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="你好",
                        thread_id="t1",
                        message_id="m-dup-1",
                    ),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )

            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))

            # 同 message_id 只处理一次
            create_mock.assert_called_once()
            self.assertEqual(len(adapter.sent), 1)

    def test_session_command_does_not_call_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_not_called()
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("当前会话", adapter.sent[0].text)

    def test_clear_command_rotates_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-1",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/clear",
                        thread_id=None,
                        message_id="m-cmd-2",
                    ),
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/session",
                        thread_id=None,
                        message_id="m-cmd-3",
                    ),
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_not_called()
            self.assertEqual(len(adapter.sent), 3)
            first = adapter.sent[0].text
            third = adapter.sent[2].text
            self.assertIn("当前会话", first)
            self.assertIn("当前会话", third)
            self.assertNotEqual(first, third)

    def test_extension_slash_command_executes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/ext_ping a b",
                        thread_id=None,
                        message_id="m-ext-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                ),
            )
            fake_session = _FakeSession()
            fake_session.extension_commands = {
                "ext_ping": type(
                    "Cmd",
                    (),
                    {"handler": staticmethod(lambda ctx: f"ok:{ctx.name}:{len(ctx.args)}")},
                )()
            }
            with patch("im.service.create_agent_session", return_value=fake_session) as create_mock:
                asyncio.run(service.handle_webhook({}, b"{}"))
            create_mock.assert_called_once()
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("ok:ext_ping:2", adapter.sent[0].text)

    def test_help_command_includes_extension_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter(
                [
                    IMIncomingMessage(
                        platform="feishu",
                        channel_id="c1",
                        user_id="u1",
                        text="/help",
                        thread_id=None,
                        message_id="m-help-1",
                    )
                ]
            )
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(workspace_dir=tmp_dir, provider="openai-standard", model_id="gpt-4o-mini"),
            )
            with patch("im.service.create_agent_session", return_value=_FakeSession()):
                asyncio.run(service.handle_webhook({}, b"{}"))
            self.assertEqual(len(adapter.sent), 1)
            self.assertIn("/ext_ping", adapter.sent[0].text)

    def test_channel_queue_limit_drops_excess_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = _FakeAdapter([])
            service = IMService(
                adapter=adapter,
                config=IMServiceConfig(
                    workspace_dir=tmp_dir,
                    provider="openai-standard",
                    model_id="gpt-4o-mini",
                    channel_queue_limit=2,
                ),
            )
            messages = [
                IMIncomingMessage(
                    platform="feishu",
                    channel_id="c1",
                    user_id="u1",
                    text=f"m{i}",
                    thread_id=None,
                    message_id=f"m-{i}",
                )
                for i in range(5)
            ]
            with patch("im.service.create_agent_session", return_value=_SlowSession()) as create_mock:
                async def _run_all():
                    await asyncio.gather(*(service.handle_incoming_message(m) for m in messages))

                asyncio.run(_run_all())
            self.assertLess(create_mock.call_count, 5)


if __name__ == "__main__":
    unittest.main()
