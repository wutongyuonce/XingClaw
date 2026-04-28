from __future__ import annotations

"""
会话持久化存储。

默认目录结构：
.xingclaw/sessions/<session_id>/
  - meta.json
  - context.jsonl
  - events.jsonl
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.types import Message

from .serde import message_from_dict, message_to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


class SessionStore:
    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.root = self.workspace_dir / ".xingclaw" / "sessions" / session_id
        self.meta_file = self.root / "meta.json"
        self.session_file = self.root / "session.jsonl"
        self.context_file = self.root / "context.jsonl"
        self.events_file = self.root / "events.jsonl"

    def ensure_initialized(self, *, model_id: str, provider: str, system_prompt: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.meta_file.exists():
            return
        meta = {
            "session_id": self.session_id,
            "model_id": model_id,
            "provider": provider,
            "system_prompt": system_prompt,
            "leaf_id": None,
            "parent_session_id": None,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.session_file.exists():
            header = {
                "type": "session",
                "version": 1,
                "id": self.session_id,
                "timestamp": _utc_now_iso(),
                "cwd": str(self.workspace_dir.resolve()),
                "parent_session": None,
            }
            self.session_file.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        if not self.context_file.exists():
            self.context_file.write_text("", encoding="utf-8")
        if not self.events_file.exists():
            self.events_file.write_text("", encoding="utf-8")

    def touch_updated_at(self) -> None:
        if not self.meta_file.exists():
            return
        meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        meta["updated_at"] = _utc_now_iso()
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_meta(self) -> dict[str, Any] | None:
        if not self.meta_file.exists():
            return None
        return json.loads(self.meta_file.read_text(encoding="utf-8"))

    def append_context_message(self, message: Message) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "message": message_to_dict(message),
        }
        with self.context_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.touch_updated_at()
        self.append_session_message(message)

    def append_event(self, event: dict[str, Any]) -> None:
        with self.events_file.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.touch_updated_at()

    def rewrite_context_messages(self, messages: list[Message]) -> None:
        lines = []
        for msg in messages:
            lines.append(json.dumps({"ts": _utc_now_iso(), "message": message_to_dict(msg)}, ensure_ascii=False))
        self.context_file.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
        self.touch_updated_at()
        self.rewrite_session_messages(messages)

    def load_context_messages(self) -> list[Message]:
        if not self.context_file.exists():
            return []

        out: list[Message] = []
        for line in self.context_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            message_data = payload.get("message", {})
            if isinstance(message_data, dict):
                out.append(message_from_dict(message_data))
        return out

    def _new_entry_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _read_session_lines(self) -> list[dict[str, Any]]:
        if not self.session_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.session_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                out.append(data)
        return out

    def _write_session_lines(self, lines: list[dict[str, Any]]) -> None:
        text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        self.session_file.write_text((text + ("\n" if text else "")), encoding="utf-8")
        self.touch_updated_at()

    def append_session_message(self, message: Message) -> str:
        """
        将消息写入 session tree（线性 parent 链），便于后续 fork/switch/branch。
        """

        lines = self._read_session_lines()
        header = lines[0] if lines and lines[0].get("type") == "session" else None
        entries = lines[1:] if header else lines
        meta = self.read_meta() or {}
        parent_id = meta.get("leaf_id")
        entry_id = self._new_entry_id()
        entry = {
            "type": "message",
            "id": entry_id,
            "parent_id": parent_id,
            "timestamp": _utc_now_iso(),
            "message": message_to_dict(message),
        }
        new_lines = ([header] if header else []) + [*entries, entry]
        self._write_session_lines(new_lines)

        meta["leaf_id"] = entry_id
        if "session_id" not in meta:
            meta["session_id"] = self.session_id
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return entry_id

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        lines = self._read_session_lines()
        header = lines[0] if lines and lines[0].get("type") == "session" else {
            "type": "session",
            "version": 1,
            "id": self.session_id,
            "timestamp": _utc_now_iso(),
            "cwd": str(self.workspace_dir.resolve()),
            "parent_session": None,
        }
        rebuilt: list[dict[str, Any]] = [header]
        parent_id: str | None = None
        for message in messages:
            entry_id = self._new_entry_id()
            rebuilt.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent_id,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent_id = entry_id
        self._write_session_lines(rebuilt)
        meta = self.read_meta() or {}
        meta["leaf_id"] = parent_id
        if "session_id" not in meta:
            meta["session_id"] = self.session_id
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        """
        从 session tree 恢复当前分支消息（默认使用 meta.leaf_id）。
        """

        lines = self._read_session_lines()
        if not lines:
            return []
        entries = [line for line in lines if line.get("type") == "message"]
        if not entries:
            return []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e.get("id"), str)}
        meta = self.read_meta() or {}
        current = leaf_id or meta.get("leaf_id")
        if not isinstance(current, str) or current not in by_id:
            # 回退到最后一条，兼容旧数据
            current = str(entries[-1].get("id"))

        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            chain.append(entry)
            parent_id = entry.get("parent_id")
            current = parent_id if isinstance(parent_id, str) else None

        chain.reverse()
        messages: list[Message] = []
        for entry in chain:
            msg_data = entry.get("message")
            if isinstance(msg_data, dict):
                messages.append(message_from_dict(msg_data))
        return messages

    def list_entry_ids(self) -> list[str]:
        lines = self._read_session_lines()
        return [str(line.get("id")) for line in lines if line.get("type") == "message" and isinstance(line.get("id"), str)]

    def get_leaf_id(self) -> str | None:
        meta = self.read_meta() or {}
        leaf = meta.get("leaf_id")
        return leaf if isinstance(leaf, str) else None

    def list_entries(self) -> list[dict[str, Any]]:
        """
        返回扁平 entry 列表，并补充导航信息：
        - depth: 根深度为 0
        - is_leaf: 是否当前叶子
        - preview: 文本摘要
        """

        lines = self._read_session_lines()
        entries = [line for line in lines if line.get("type") == "message"]
        by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            eid = entry.get("id")
            if isinstance(eid, str):
                by_id[eid] = entry

        leaf_id = self.get_leaf_id()
        result: list[dict[str, Any]] = []
        for eid, entry in by_id.items():
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            depth = len(self.get_entry_path(eid)) - 1
            result.append(
                {
                    "id": eid,
                    "parent_id": entry.get("parent_id"),
                    "timestamp": entry.get("timestamp"),
                    "role": role,
                    "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                    "depth": max(depth, 0),
                    "is_leaf": eid == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_entry_path(self, entry_id: str) -> list[str]:
        """
        返回从根到指定 entry 的 id 路径。
        """

        lines = self._read_session_lines()
        by_id = {
            str(line.get("id")): line
            for line in lines
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        }
        if entry_id not in by_id:
            raise ValueError(f"Entry not found: {entry_id}")

        path: list[str] = []
        current: str | None = entry_id
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            path.append(current)
            parent = by_id[current].get("parent_id")
            current = parent if isinstance(parent, str) else None
        path.reverse()
        return path

    def set_leaf(self, entry_id: str) -> None:
        lines = self._read_session_lines()
        ids = {str(line.get("id")) for line in lines if line.get("type") == "message" and isinstance(line.get("id"), str)}
        if entry_id not in ids:
            raise ValueError(f"Entry not found: {entry_id}")
        meta = self.read_meta() or {}
        meta["leaf_id"] = entry_id
        self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_session_tree(self) -> list[dict[str, Any]]:
        """
        返回 session 树结构（按 parent_id 组织）。
        每个节点形如：
        {
          "id": "...",
          "parent_id": "...|None",
          "timestamp": "...",
          "role": "user|assistant|toolResult",
          "preview": "...",
          "children": [...]
        }
        """

        lines = self._read_session_lines()
        entries = [line for line in lines if line.get("type") == "message"]
        node_by_id: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []

        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            preview = self._preview_message(msg if isinstance(msg, dict) else {})
            node_by_id[eid] = {
                "id": eid,
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp"),
                "role": role,
                "preview": preview,
                "children": [],
            }

        for node in node_by_id.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in node_by_id:
                node_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _preview_message(message: dict[str, Any]) -> str:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            if isinstance(content, str):
                return content[:80]
            if isinstance(content, list):
                text = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += str(block.get("text", ""))
                return text[:80]
        if role == "assistant" and isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
            return text[:80]
        if role == "toolResult" and isinstance(content, list):
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += str(block.get("text", ""))
            return text[:80]
        return ""

    def fork_to(
        self,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> "SessionStore":
        """
        基于当前会话分支创建新会话（fork）。
        """

        target = SessionStore(self.workspace_dir, new_session_id)
        meta = self.read_meta() or {}
        target.ensure_initialized(
            model_id=str(meta.get("model_id", "")),
            provider=str(meta.get("provider", "")),
            system_prompt=str(meta.get("system_prompt", "")),
        )

        messages = self.load_session_messages(leaf_id=from_entry_id)
        target.rewrite_context_messages(messages)

        tmeta = target.read_meta() or {}
        tmeta["parent_session_id"] = self.session_id
        target.meta_file.write_text(json.dumps(tmeta, ensure_ascii=False, indent=2), encoding="utf-8")
        target.append_event(
            {
                "type": "session_forked",
                "from_session_id": self.session_id,
                "from_entry_id": from_entry_id,
                "to_session_id": new_session_id,
            }
        )
        return target
