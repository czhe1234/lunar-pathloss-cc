#!/usr/bin/env python3
"""Start CC's loopback-only, offline SwanLab dashboard. Never launches experiments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "tracking" / "swanlab"

def matches(pid, port):
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        return (str(ROOT / "scripts/serve_tracking.py") in argv and str(STATE / "logs") in argv and
                "--port" in argv and argv[argv.index("--port") + 1] == str(port))
    except (OSError, ValueError, IndexError):
        return False

def dashboard(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            page = response.read(32768).decode(errors="replace")
            return response.status == 200 and 'data-cc-dashboard="1"' in page
    except (OSError, urllib.error.URLError):
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=16007)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    STATE.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ROOT / ".venv-swanlab/bin/python"),
                    str(ROOT / "scripts/configure_swanlab_view.py")],
                   check=True, stdout=subprocess.DEVNULL,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    for name in ("logs", "root", "tmp", "cache"):
        (STATE / name).mkdir(exist_ok=True)
    suffix = "" if args.port == 16007 else f"-{args.port}"
    pidfile = STATE / f"server{suffix}.pid"
    logfile = STATE / f"server{suffix}.log"
    url = f"http://127.0.0.1:{args.port}"
    with (STATE / f"server{suffix}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pid = int(pidfile.read_text().strip()) if pidfile.exists() else 0
        if not matches(pid, args.port):
            if dashboard(url):
                raise RuntimeError("A SwanLab service already owns the port; verify its PID before adopting it.")
            env = os.environ.copy()
            for key in ("SWANLAB_API_KEY", "SWANLAB_WORKSPACE",
                        "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID"):
                env.pop(key, None)
            env.update(
                SWANLAB_MODE="local", SWANLAB_ROOT=str(STATE / "root"),
                SWANLAB_LOGDIR=str(STATE / "logs"), SWANLAB_PROBE_MONITOR="false",
                SWANLAB_PROBE_SWANLAB="false", CUDA_VISIBLE_DEVICES="",
                SWANLAB_API_HOST=url, DO_NOT_TRACK="true",
                TMPDIR=str(STATE / "tmp"), XDG_CACHE_HOME=str(STATE / "cache"),
                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            )
            cmd = [str(ROOT / ".venv-swanlab/bin/python"), "-u",
                   str(ROOT / "scripts/serve_tracking.py"), "--log-dir", str(STATE / "logs"),
                   "--host", "127.0.0.1", "--port", str(args.port)]
            with logfile.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = process.pid
            pidfile.write_text(str(pid) + "\n")
        for _ in range(60):
            if dashboard(url):
                print(json.dumps({"backend": "swanlab", "mode": "local",
                                  "pid": pid, "url": url, "log_dir": str(STATE / "logs")}))
                return
            if not matches(pid, args.port):
                raise RuntimeError(f"SwanLab stopped; inspect {logfile}")
            time.sleep(0.5)
        raise RuntimeError(f"SwanLab startup timed out; inspect {logfile}")

if __name__ == "__main__":
    main()
