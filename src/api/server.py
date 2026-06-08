"""
Minimal pipeline trigger API.

Exposes two endpoints so the browser can start a pipeline run and
poll for completion without the user touching a terminal.

GET  /api/status           → {"status": "idle"|"running"|"done"|"error", "error": str|null}
POST /api/run-pipeline     → starts a background run; returns current status immediately
"""

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ROOT / "configs" / "kitti.yaml"
OUTPUT_DIR = ROOT / "data" / "outputs"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_status: dict = {"status": "idle", "error": None}


@app.get("/api/status")
def get_status() -> dict:
    return _status


@app.post("/api/run-pipeline")
async def run_pipeline() -> dict:
    if _status["status"] == "running":
        return _status
    _status["status"] = "running"
    _status["error"] = None
    asyncio.create_task(_execute())
    return _status


async def _execute() -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "scripts" / "run_pipeline.py"),
            "--config", str(CONFIG),
            "--stage", "full",
            "--output", str(OUTPUT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            _status["status"] = "done"
        else:
            _status["status"] = "error"
            _status["error"] = stderr.decode()[-300:] if stderr else f"exit code {proc.returncode}"
    except Exception as exc:
        _status["status"] = "error"
        _status["error"] = str(exc)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
