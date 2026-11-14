#!/usr/bin/env python3
"""Serve the project overview alongside the genuine offline SwanLab dashboard."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    p.add_argument("--port", type=int, default=16007)
    p.add_argument("--log-dir", default=str(ROOT / "tracking/swanlab/logs"))
    args = p.parse_args()
    assert Path(args.log_dir).resolve() == ROOT / "tracking/swanlab/logs", "This service is scoped to CC records"
    os.environ.update(CUDA_VISIBLE_DEVICES="", SWANLAB_MODE="local",
                      SWANLAB_LOG_DIR=args.log_dir, SWANLAB_LOGDIR=args.log_dir,
                      SWANLAB_ROOT=str(ROOT / "tracking/swanlab/root"),
                      SWANLAB_PROBE_MONITOR="false", SWANLAB_PROBE_SWANLAB="false")
    sys.path.insert(0, str(ROOT / "src"))
    from cc_dashboard import snapshot, SOURCES
    from swanboard.db import connect
    connect(path=args.log_dir)
    from swanboard.app import app
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    @app.middleware("http")
    async def overview(request, call_next):
        if request.url.path in ("/", "/cc") and request.query_params.get("native") != "1":
            return HTMLResponse((ROOT / "src/cc_dashboard.html").read_text(),
                                headers={"Cache-Control": "no-store"})
        if request.url.path == "/cc/data.json":
            try:
                payload = await asyncio.to_thread(snapshot)
                return JSONResponse(payload, headers={"Cache-Control": "no-store"})
            except Exception as error:
                return JSONResponse({"error": str(error)}, status_code=503)
        if request.url.path.startswith("/cc/source/"):
            key = request.url.path.removeprefix("/cc/source/")
            if key not in SOURCES:
                return JSONResponse({"error": "Unknown source"}, status_code=404)
            return FileResponse(ROOT / SOURCES[key], media_type="application/json",
                                headers={"Cache-Control": "no-store"})
        if (request.url.path.startswith("/experiment/") or request.url.path == "/charts" or
                (request.url.path == "/" and request.query_params.get("native") == "1")):
            from swanboard.settings import INDEX
            html = Path(INDEX).read_text()
            backlink = '<a href="/" id="cc-overview-home" style="position:fixed;right:18px;bottom:18px;z-index:1000;background:#fff;color:#267760;border:1px solid #dce5df;border-radius:8px;padding:8px 12px;font:12px sans-serif;box-shadow:0 3px 15px #0001;text-decoration:none">← 研究总览</a>'
            return HTMLResponse(html.replace("</body>", backlink + "</body>"), headers={"Cache-Control": "no-store"})
        return await call_next(request)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
if __name__ == "__main__":
    main()
