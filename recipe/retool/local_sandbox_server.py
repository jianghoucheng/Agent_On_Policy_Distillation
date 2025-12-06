"""Simple Flask sandbox to execute Python snippets locally.

Run this script to expose an HTTP endpoint that mimics the Sandbox Fusion
interface expected by ``CustomSandboxFusionTool`` when the
``use_local_flask`` flag is enabled. The response structure matches what the
tool expects, so you can switch between the remote Sandbox Fusion backend and
this local runner by modifying the YAML config.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from flask import Flask, jsonify, request


app = Flask(__name__)


@app.route("/")
def home():
    return "Sandbox is running"


@app.route("/execute", methods=["POST"])
def execute():
    data = request.json or {}
    code = data.get("code", "")
    language = data.get("language", "python")

    if language != "python":
        return jsonify({"error": f"Unsupported language: {language}"}), 400

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        temp_path = tmp_file.name

    try:
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=int(os.getenv("LOCAL_SANDBOX_TIMEOUT", "5")),
        )
        return jsonify(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout"}), 408
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    host = os.getenv("LOCAL_SANDBOX_HOST", "0.0.0.0")
    port = int(os.getenv("LOCAL_SANDBOX_PORT", "8080"))
    app.run(host=host, port=port)
