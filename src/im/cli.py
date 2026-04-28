from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .events import IMEventWatcher, IMEventWatcherOptions
from .feishu import FeishuAdapter, FeishuAdapterConfig
from .feishu_longconn import FeishuLongConnOptions, run_feishu_long_connection
from .server import IMServerOptions, run_http_server
from .service import IMService, IMServiceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XingClaw IM bridge service")
    parser.add_argument("--platform", choices=["feishu"], default="feishu", help="IM platform (current: feishu)")
    parser.add_argument(
        "--transport",
        choices=["webhook", "longconn"],
        default="webhook",
        help="IM transport mode: webhook or long connection",
    )
    parser.add_argument("--workspace", default=".", help="Workspace path")
    parser.add_argument("--host", default="127.0.0.1", help="Webhook server host")
    parser.add_argument("--port", type=int, default=8787, help="Webhook server port")
    parser.add_argument("--path", default="/feishu/events", help="Webhook path")
    parser.add_argument("--provider", default="openai-standard", help="coding_agent provider")
    parser.add_argument("--model-id", default="gpt-4o-mini", help="coding_agent model id")
    parser.add_argument("--read-only", action="store_true", help="Enable read-only tool mode")
    parser.add_argument("--channel-queue-limit", type=int, default=20, help="Per-channel in-memory queue limit")
    parser.add_argument(
        "--events-dir",
        default="",
        help="Optional event directory for immediate/one-shot/periodic IM messages",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level for IM bridge",
    )

    parser.add_argument("--feishu-app-id", default="", help="Feishu app_id")
    parser.add_argument("--feishu-app-secret", default="", help="Feishu app_secret")
    parser.add_argument("--feishu-verify-token", default="", help="Feishu event verify token (optional)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.platform == "feishu":
        if not args.feishu_app_id or not args.feishu_app_secret:
            parser.error("--feishu-app-id and --feishu-app-secret are required for feishu platform")
        adapter = FeishuAdapter(
            FeishuAdapterConfig(
                app_id=args.feishu_app_id,
                app_secret=args.feishu_app_secret,
                verify_token=args.feishu_verify_token or None,
            )
        )
    else:  # pragma: no cover
        parser.error(f"Unsupported platform: {args.platform}")
        return 2

    service = IMService(
        adapter=adapter,
        config=IMServiceConfig(
            workspace_dir=args.workspace,
            provider=args.provider,
            model_id=args.model_id,
            read_only_mode=bool(args.read_only),
            channel_queue_limit=max(1, int(args.channel_queue_limit)),
        ),
    )
    events_dir = Path(args.events_dir) if args.events_dir else Path(args.workspace) / ".xingclaw" / "im" / "events"
    watcher = IMEventWatcher(service, IMEventWatcherOptions(events_dir=events_dir))
    watcher.start()
    try:
        if args.transport == "longconn":
            run_feishu_long_connection(
                service,
                FeishuLongConnOptions(
                    app_id=args.feishu_app_id,
                    app_secret=args.feishu_app_secret,
                    log_level=args.log_level,
                ),
            )
        else:
            server_options = IMServerOptions(host=args.host, port=args.port, path=args.path)
            run_http_server(service, server_options)
    except KeyboardInterrupt:
        print("\n[im] stopped")
    finally:
        watcher.stop()
    return 0
