from __future__ import annotations

import json
import uuid
from pathlib import Path


class SessionRouter:
    """
    将 IM 会话（平台/会话维度）映射到 coding_agent 的 session_id。
    """

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.state_file = self.workspace_dir / ".xingclaw" / "im" / "session_map.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def get_or_create_session_id(self, *, platform: str, channel_id: str, thread_id: str | None = None) -> str:
        key = self._build_key(platform=platform, channel_id=channel_id, thread_id=thread_id)
        state = self._read_state()
        existing = state.get(key)
        if isinstance(existing, str) and existing:
            return existing
        new_id = f"im_{uuid.uuid4().hex[:12]}"
        state[key] = new_id
        self._write_state(state)
        return new_id

    def get_session_id(self, *, platform: str, channel_id: str, thread_id: str | None = None) -> str | None:
        key = self._build_key(platform=platform, channel_id=channel_id, thread_id=thread_id)
        state = self._read_state()
        value = state.get(key)
        if isinstance(value, str) and value:
            return value
        return None

    def rotate_session_id(self, *, platform: str, channel_id: str, thread_id: str | None = None) -> str:
        key = self._build_key(platform=platform, channel_id=channel_id, thread_id=thread_id)
        state = self._read_state()
        new_id = f"im_{uuid.uuid4().hex[:12]}"
        state[key] = new_id
        self._write_state(state)
        return new_id

    @staticmethod
    def _build_key(*, platform: str, channel_id: str, thread_id: str | None) -> str:
        if thread_id:
            return f"{platform}:{channel_id}:{thread_id}"
        return f"{platform}:{channel_id}"

    def _read_state(self) -> dict[str, str]:
        if not self.state_file.exists():
            return {}
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        result: dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                result[k] = v
        return result

    def _write_state(self, state: dict[str, str]) -> None:
        self.state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
