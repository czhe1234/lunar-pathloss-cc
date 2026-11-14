"""Run the authorized C145 phases when an approved GPU is genuinely idle."""
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import struct
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit/c145_capacity"
RUN = ROOT / "runs/c145_capacity_native"
DEVICES = {
    1: "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    3: "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    4: "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
}


def status(record):
    record["updated"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    record["controller_pid"] = os.getpid()
    path = AUDIT / "controller_status.json"
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(path)
    print(json.dumps(record), flush=True)


def idle(card):
    identifier = DEVICES[card]
    try:
        jobs = subprocess.check_output(
            ["nvidia-smi", "-i", identifier, "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"], text=True, timeout=10
        ).strip()
        stats = subprocess.check_output(
            ["nvidia-smi", "-i", identifier, "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10
        ).strip().split(",")
        return not jobs and int(stats[0]) <= 256 and int(stats[1]) == 0
    except (subprocess.SubprocessError, ValueError, IndexError):
        return False



def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exports_complete(result):
    """Validate final export receipts before skipping GPU export recovery."""
    try:
        config = json.loads((RUN / "config.json").read_text())
        state = result["training_state"]
        expected_epochs = {
            "selected": state["best_monitor_epoch"],
            "fit_best": state["best_fit_epoch"],
            "last": state["additional_epoch"],
        }
        for tag, epoch in expected_epochs.items():
            metadata = json.loads((RUN / f"{tag}_export.json").read_text())
            if metadata["training_state"]["additional_epoch"] != epoch:
                return False
            for name in ("fit", "monitor"):
                path = RUN / f"{tag}_{name}_prediction.npy"
                sidecar = json.loads(path.with_suffix(".json").read_text())
                ids = config["panels"][name]
                id_hash = hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
                expected = {
                    "model_sha256": metadata["model_sha256"],
                    "panel_ids_sha256": id_hash,
                    "shape": [len(ids), 2, 256, 256],
                    "prediction_dtype": "float32",
                    "forward_precision": "BF16 autocast",
                }
                if any(sidecar.get(key) != value for key, value in expected.items()):
                    return False
                if any(metadata["panels"][name].get(key) != value for key, value in sidecar.items()):
                    return False
                if sidecar["file_size_bytes"] != path.stat().st_size:
                    return False
                if sidecar["prediction_sha256"] != file_hash(path):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def score_complete(tag):
    try:
        scored = json.loads((RUN / f"{tag}_verified.json").read_text())
        exported = json.loads((RUN / f"{tag}_export.json").read_text())
        return (
            scored["status"] == "verified"
            and scored["config_sha256"] == file_hash(RUN / "config.json")
            and scored["verification_source_sha256"] == file_hash(ROOT / "src/c145_verify.py")
            and all(scored["panels"][name]["prediction_sha256"] ==
                    exported["panels"][name]["prediction_sha256"] for name in ("fit", "monitor"))
            and (RUN / f"{tag}_terrain_sse.npz").is_file()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    lock = (ROOT / ".c145.controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cap = 128
    while True:
        result_path = RUN / "training_result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior["status"] == "training_complete" and exports_complete(prior):
                break
            last_epoch = prior["training_state"]["additional_epoch"]
            if cap <= last_epoch:
                cap = last_epoch+64
        choices = [card for card in DEVICES if idle(card)]
        if not choices:
            status({"stage": "waiting_for_idle_approved_GPU", "phase_cap": cap})
            time.sleep(30)
            continue
        card = choices[0]
        command = [str(ROOT / "scripts/c145_run.sh"), str(card), "--max-epochs", str(cap)]
        if (RUN / "last.pt").exists():
            command.append("--resume")
        elif RUN.exists():
            raise RuntimeError("Partial run directory exists without a recoverable checkpoint")
        log_path = AUDIT / f"train_to_{cap}.log"
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status({"stage": "training", "phase_cap": cap, "card": card,
                    "uuid": DEVICES[card], "child_pid": child.pid, "log": str(log_path)})
            code = child.wait()
        if code == 75:
            status({"stage": "waiting_for_idle_approved_GPU", "phase_cap": cap,
                    "last_launch_rejected_busy_card": card})
            time.sleep(30)
            continue
        if code:
            status({"stage": "failed", "exit_code": code, "log": str(log_path)})
            raise SystemExit(code)
        if not result_path.exists():
            raise RuntimeError("Training returned without a phase result")
    for tag in ("selected", "fit_best", "last"):
        verified = RUN / f"{tag}_verified.json"
        if score_complete(tag):
            continue
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        command = [str(ROOT / ".venv/bin/python"), "-B", str(ROOT / "src/c145_verify.py"),
                   "--run", str(RUN), "--tag", tag]
        status({"stage": "independent_CPU_scoring", "tag": tag})
        with (AUDIT / f"score_{tag}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    status({"stage": "writing_comparison_report", "run": str(RUN)})
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with (AUDIT / "report.log").open("w") as log:
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-B", str(ROOT / "src/c145_report.py"),
             "--run", str(RUN)],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    status({"stage": "complete_training_scoring_and_comparison_report",
            "run": str(RUN), "summary": str(AUDIT / "comparison_summary.json")})


if __name__ == "__main__":
    main()
