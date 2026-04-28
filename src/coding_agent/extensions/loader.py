from __future__ import annotations

import importlib.util
from pathlib import Path

from .api import ExtensionAPI
from .types import LoadedExtensions


def discover_extension_paths(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> list[Path]:
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

    default_dir = workspace / ".xingclaw" / "extensions"
    if default_dir.exists() and default_dir.is_dir():
        for path in sorted(default_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            _add(path)

    for raw in configured_paths or []:
        target = Path(raw)
        if not target.is_absolute():
            target = workspace / raw
        if target.exists() and target.is_dir():
            for path in sorted(target.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                _add(path)
        elif target.exists() and target.is_file() and target.suffix == ".py":
            _add(target)

    return paths


def load_extensions(workspace_dir: str | Path, configured_paths: list[str] | None = None) -> LoadedExtensions:
    result = LoadedExtensions()
    for path in discover_extension_paths(workspace_dir, configured_paths=configured_paths):
        api = ExtensionAPI()
        try:
            module = _load_module_from_file(path)
            register = getattr(module, "register", None) or getattr(module, "setup", None)
            if not callable(register):
                result.errors.append(f"{path}: missing register(api) function")
                continue
            register(api)
            snapshot = api.snapshot()
            result.tools.extend(snapshot.tools)
            result.before_tool_hooks.extend(snapshot.before_tool_hooks)
            result.after_tool_hooks.extend(snapshot.after_tool_hooks)
            result.prompt_guidelines.extend(snapshot.prompt_guidelines)
            result.append_prompts.extend(snapshot.append_prompts)
            result.commands.update(snapshot.commands)
            result.before_prompt_hooks.extend(snapshot.before_prompt_hooks)
            result.after_prompt_hooks.extend(snapshot.after_prompt_hooks)
            result.loaded_paths.append(str(path))
        except Exception as exc:
            result.errors.append(f"{path}: {exc}")
    return result


def _load_module_from_file(path: Path):
    module_name = f"xingclaw_extension_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
