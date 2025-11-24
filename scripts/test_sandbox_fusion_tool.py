"""Smoke test for Sandbox Fusion code execution.

Usage:
    python scripts/test_sandbox_fusion_tool.py \
        --config recipe/retool/sandbox_fusion_tool_config.yaml \
        --code "print(1 + 1)" \
        [--mode direct|tool]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

import ray
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recipe.retool.retool import CustomSandboxFusionTool
from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case


def _maybe_patch_resource_tracker() -> None:
    """Work around multiprocess bug on Python 3.12."""
    try:
        from multiprocess import resource_tracker  # type: ignore

        resource_tracker._resource_tracker = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _prepare_code(raw_code: str) -> str:
    """Mimic CustomSandboxFusionTool logic to ensure code prints a result."""
    pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    matches = pattern.findall(raw_code)
    if matches:
        raw_code = matches[0].strip()

    lines = raw_code.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "":
            continue
        if not lines[i].lstrip().startswith("print"):
            lines[i] = f"print({lines[i]})"
        break
    return "\n".join(lines)


def _load_tool_config(config_path: Path) -> tuple[dict, OpenAIFunctionToolSchema]:
    config_dict = yaml.safe_load(config_path.read_text())
    tool_entry = config_dict["tools"][0]
    tool_config = tool_entry["config"]
    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_entry["tool_schema"])
    return tool_config, tool_schema


async def _run_via_tool(tool: CustomSandboxFusionTool, code: str) -> None:
    prepared_code = _prepare_code(code)
    instance_id, _ = await tool.create()
    try:
        response, _, _ = await tool.execute(instance_id, {"code": prepared_code})
    finally:
        await tool.release(instance_id)

    print("=== Sandbox Fusion output (tool mode) ===")
    print(response.text)


def _run_direct(tool_config: dict, code: str) -> None:
    code_to_run = _prepare_code(code)
    result_status, metadata = _process_single_case(
        case_index=0,
        stdin_data=None,
        expected_output=None,
        sandbox_fusion_url=tool_config["sandbox_fusion_url"],
        generation=code_to_run,
        timeout=tool_config.get("default_timeout", 30),
        memory_limit_mb=tool_config.get("memory_limit_mb", 1024),
        language=tool_config.get("default_language", "python"),
    )
    metadata = metadata or {}
    print("=== Sandbox Fusion output (direct mode) ===")
    print(f"Run status : {metadata.get('run_status')}")
    stdout = metadata.get("stdout")
    stderr = metadata.get("stderr")
    print(f"stdout     : {(stdout or '').strip()}")
    print(f"stderr     : {(stderr or '').strip()}")
    if metadata.get("run_status") != "Finished":
        print(f"[warn] Non-finished status ({result_status}). Check sandbox logs or code snippet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Sandbox Fusion code execution.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("recipe/retool/sandbox_fusion_tool_config.yaml"),
        help="Path to the sandbox fusion tool config.",
    )
    parser.add_argument(
        "--code",
        type=str,
        default="print(1 + 1)",
        help="Python code snippet to execute inside sandbox fusion.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Optional Ray address. Leave empty for local Ray.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "tool"],
        default="direct",
        help="direct: call sandbox API without Ray. tool: full CustomSandboxFusionTool + Ray.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        print(f"Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    tool_config, tool_schema = _load_tool_config(args.config)
    code = args.code if args.code.strip() else "print('hello from sandbox fusion')"

    if args.mode == "direct":
        _run_direct(tool_config, code)
        return

    _maybe_patch_resource_tracker()
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    try:
        tool = CustomSandboxFusionTool(tool_config, tool_schema)
        asyncio.run(_run_via_tool(tool, code))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
