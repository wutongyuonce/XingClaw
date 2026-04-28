from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coding_agent.cli import build_parser, main


class CodingAgentCliTests(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.mode, "interactive")
        self.assertEqual(args.workspace, ".")

    def test_main_print_mode_calls_runner(self) -> None:
        fake_session = Mock()
        fake_session.close = Mock()
        with patch("coding_agent.cli.create_agent_session", return_value=fake_session) as create_mock, patch(
            "coding_agent.cli.run", new_callable=AsyncMock
        ) as run_mock:
            code = main(["--mode", "print", "--prompt", "hello", "--provider", "anthropic", "--model-id", "glm-4.7"])

        self.assertEqual(code, 0)
        create_mock.assert_called_once()
        run_mock.assert_awaited_once()
        fake_session.close.assert_called_once()

    def test_main_print_mode_without_prompt_returns_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--mode", "print"])
        # argparse parser.error 会触发退出码 2
        self.assertEqual(ctx.exception.code, 2)

    def test_main_list_entries_short_circuit(self) -> None:
        fake_session = Mock()
        fake_session.session_id = "s1"
        fake_session.close = Mock()
        fake_session.list_entry_ids = Mock(return_value=["a1", "a2"])
        with patch("coding_agent.cli.create_agent_session", return_value=fake_session), patch(
            "coding_agent.cli.run", new_callable=AsyncMock
        ) as run_mock, patch("builtins.print") as print_mock:
            code = main(["--list-entries", "--provider", "anthropic", "--model-id", "glm-4.7"])
        self.assertEqual(code, 0)
        run_mock.assert_not_awaited()
        print_mock.assert_called()
        fake_session.close.assert_called_once()

    def test_main_fork_entry_short_circuit(self) -> None:
        fake_forked = Mock()
        fake_forked.session_id = "s2"
        fake_forked.close = Mock()

        fake_session = Mock()
        fake_session.session_id = "s1"
        fake_session.close = Mock()
        fake_session.fork_from_entry = Mock(return_value=fake_forked)

        with patch("coding_agent.cli.create_agent_session", return_value=fake_session), patch(
            "coding_agent.cli.run", new_callable=AsyncMock
        ) as run_mock, patch("builtins.print") as print_mock:
            code = main(["--fork-entry", "e1", "--provider", "anthropic", "--model-id", "glm-4.7"])
        self.assertEqual(code, 0)
        run_mock.assert_not_awaited()
        print_mock.assert_called()
        fake_forked.close.assert_called_once()
        fake_session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
