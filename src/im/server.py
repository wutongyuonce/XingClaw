from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import IMService

logger = logging.getLogger("xingclaw.im.server")


@dataclass
class IMServerOptions:
    host: str = "127.0.0.1"
    port: int = 8787
    path: str = "/feishu/events"


def run_http_server(service: IMService, options: IMServerOptions) -> None:
    """
    启动最小 HTTP webhook 服务。
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 固定命名
            logger.info("incoming request path=%s content_length=%s", self.path, self.headers.get("Content-Length", "0"))
            if self.path != options.path:
                logger.warning("request path not matched expected_path=%s actual_path=%s", options.path, self.path)
                self._send_json(404, {"error": "not found"})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            headers = {k.lower(): v for k, v in self.headers.items()}
            try:
                ack = asyncio.run(service.handle_webhook(headers, body))
                logger.info("webhook handled successfully ack=%s", ack)
                self._send_json(200, ack)
            except Exception as exc:
                logger.exception("webhook handling failed: %s", exc)
                self._send_json(500, {"code": 500, "msg": str(exc)})

        def _send_json(self, status: int, payload_obj: dict) -> None:
            payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError) as exc:
                # 对端提前断开连接在 webhook 场景较常见（例如隧道转发重试），记录为 warning，避免误判为业务失败。
                logger.warning("client disconnected before response was sent status=%s err=%s", status, exc)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            _ = format, args
            # 默认静默，避免刷屏；如需日志可在上层接入日志器。
            return

    server = ThreadingHTTPServer((options.host, options.port), _Handler)
    print(f"[im] listening http://{options.host}:{options.port}{options.path}")
    server.serve_forever()
