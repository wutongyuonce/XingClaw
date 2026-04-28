from __future__ import annotations

import re
from pathlib import Path

from .types import LoadedExtensions, RegisteredCommand, SkillSpec


def discover_skill_paths(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> list[Path]:
    workspace = Path(workspace_dir)
    paths: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        paths.append(resolved)

    default_dir = workspace / ".xingclaw" / "skills"
    if default_dir.exists() and default_dir.is_dir():
        for path in sorted(default_dir.glob("*.md")):
            _add(path)

    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / raw
        if target.exists() and target.is_dir():
            for path in sorted(target.glob("*.md")):
                _add(path)
        elif target.exists() and target.is_file() and target.suffix.lower() == ".md":
            _add(target)

    return paths


def load_skills(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> LoadedExtensions:
    result = LoadedExtensions()
    seen_cmds: dict[str, str] = {}
    for path in discover_skill_paths(workspace_dir, configured_paths=configured_paths):
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
            if not raw_text:
                continue
            meta, text = _parse_skill_frontmatter(raw_text)
            title = str(meta.get("name") or _extract_title(text) or path.stem).strip()
            if not title:
                title = path.stem
            cmd = str(meta.get("command") or f"skill:{_slugify(title)}").strip().lstrip("/")
            if not cmd:
                cmd = f"skill:{_slugify(path.stem)}"
            desc = str(meta.get("description") or f"执行技能：{title}").strip()
            skill = SkillSpec(
                name=title,
                command_name=cmd,
                description=desc,
                content=text,
                source_path=str(path),
            )
            if cmd in seen_cmds:
                result.diagnostics.append(f"skill command conflict: /{cmd} from {path} overrides {seen_cmds[cmd]}")
            seen_cmds[cmd] = str(path)

            result.skills.append(skill)
            result.prompt_guidelines.append(f"技能约束（{title}）：按该技能流程执行。")
            result.append_prompts.append(f"## Skill: {title}\n{text}")
            result.commands[cmd] = RegisteredCommand(
                name=cmd,
                description=desc,
                source="skill",
                handler=lambda ctx, _skill=skill: _render_skill_prompt(_skill, ctx.raw_text),
            )
            result.loaded_paths.append(str(path))
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
    return result


def _extract_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return None


def _parse_skill_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower()
        val = v.strip().strip("'").strip('"')
        if key and val:
            meta[key] = val
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "skill"


def _render_skill_prompt(skill: SkillSpec, raw_text: str) -> str:
    cmd_text = raw_text.strip() if raw_text else f"/{skill.command_name}"
    return (
        f"已应用技能 `{skill.name}`（命令：`{cmd_text}`）。\n"
        "请严格按照下述技能内容执行，并给出可执行结果：\n\n"
        f"{skill.content}"
    )
