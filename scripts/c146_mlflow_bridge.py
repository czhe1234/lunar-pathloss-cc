#!/usr/bin/env python3
"""Read committed C146 JSON logs. Never import training code or use a GPU."""
import argparse, fcntl, hashlib, json, math, os, time, traceback
from datetime import datetime
from pathlib import Path

os.environ.update(MLFLOW_DISABLE_TELEMETRY="true", DO_NOT_TRACK="true",
                  MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="false",
                  MLFLOW_ENABLE_AI_GATEWAY="false", CUDA_VISIBLE_DEVICES="")
from mlflow.entities import Metric, Param, RunTag
from mlflow.telemetry import get_telemetry_client
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/c146_solver_ablation"
AUDIT = ROOT / "audit/c146_solver_ablation"
STATE = ROOT / "tracking"
URI = "http://127.0.0.1:16007"
EXPERIMENT = "C146_solver_ablation"
ARMS = ("with_solver", "without_solver")
BANDS = ("415MHz", "5p775GHz")

def read(path):
    return json.loads(path.read_text())

def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

def metrics_for(arm, initial, history):
    # Each step is a completed epoch. No synthetic interpolation or missing evaluations.
    points = []
    def add(key, value, epoch, seconds=0):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite metric: {key}")
        points.append((key, value, int(epoch), float(seconds)))
    for band, value in zip(BANDS, initial["both_arms"]["rmse"]):
        add(f"monitor/RMSE_{band}_dB", value, 0)
    for row in history:
        ep, seconds = row["epoch"], row["seconds"]
        for band, value in zip(BANDS, row["running_fit_rmse"][arm]):
            add(f"train/running_weighted_RMSE_{band}_dB", value, ep, seconds)
        for panel in ("monitor", "fit"):
            if panel in row:
                for band, value in zip(BANDS, row[panel][arm]["rmse"]):
                    add(f"{panel}/RMSE_{band}_dB", value, ep, seconds)
        for key, value in [("optim/learning_rate", row["lr"]),
                           ("progress/updates_per_arm", row["updates_per_arm"]),
                           ("progress/epoch", ep), ("time/elapsed_seconds", seconds),
                           ("best/monitor_mean_dB", row["best"][arm]["mean_rmse"]),
                           ("best/epoch", row["best"][arm]["epoch"])]:
            add(key, value, ep, seconds)
    return points

def final_metrics(arm, summary, history):
    points = []
    last_seconds = history[-1]["seconds"]
    for panel, result in summary["panels"].items():
        for checkpoint in ("selected", "final"):
            model = result["models"][f"{arm}_{checkpoint}"]
            prefix = f"verified/{checkpoint}/{panel}"
            # Final verification is a terminal evaluation, not a repeated training curve.
            ep = model["epoch"]
            for group, values in model["RMSE"].items():
                for band, value in zip(BANDS, values):
                    points.append((f"{prefix}/{group}/RMSE_{band}_dB", value, ep, last_seconds))
            for key in ("own_error_ge5_fraction", "own_error_ge5_SSE_share",
                        "tail_correction_direction_accuracy"):
                for band, value in zip(BANDS, model[key]):
                    points.append((f"{prefix}/{key}_{band}", value, ep, last_seconds))
    return points

class Bridge:
    def __init__(self):
        assert get_telemetry_client() is None
        self.client = MlflowClient(tracking_uri=URI)
        self.protocol = read(AUDIT / "protocol.json")
        self.digest = hashlib.sha256((AUDIT / "protocol.json").read_bytes()).hexdigest()
        self.start_ms = int(datetime.fromisoformat(self.protocol["created"]).timestamp() * 1000)
        exp = self.client.get_experiment_by_name(EXPERIMENT)
        self.exp_id = exp.experiment_id if exp else self.client.create_experiment(EXPERIMENT)
        self.ids, self.known = {}, {}
        for arm in ARMS:
            identity = f"C146/{self.digest}/{arm}"
            found = self.client.search_runs([self.exp_id],
                    filter_string=f"tags.`cc.identity` = '{identity}'")
            if len(found) > 1:
                raise RuntimeError("Duplicate run identity; refusing to merge")
            tags = {"cc.identity": identity, "cc.campaign": "C146",
                    "cc.arm": arm, "cc.storage": "server_only",
                    "cc.metric_step": "completed_epoch",
                    "cc.source": "existing committed JSON; no training changes",
                    "mlflow.note.content": (
                      "两组同网络、同划分、同128轮预算。仅45个求解器特征不同。"
                      "monitor为未参与训练的开发地形；不是官方VAL。"
                      "train/running_weighted与完整fit/RMSE含义不同。"
                      "大误差区与确认集指标在原计划的最终CPU核验后写入。"
                      "时间戳由协议创建时间与日志elapsed_seconds近似恢复。")}
            run = found[0] if found else self.client.create_run(
                    self.exp_id, start_time=self.start_ms, tags=tags,
                    run_name=arm)
            rid = self.ids[arm] = run.info.run_id
            if not found:
                params = {k: v for k, v in self.protocol.items()
                          if not isinstance(v, (dict, list))}
                params.update(arm=arm, solver_features=(arm == "with_solver"),
                              protocol_sha256=self.digest, C82_learning_paths=False)
                for panel, counts in self.protocol["panel_counts"].items():
                    params.update({f"{panel}_{k}": v for k, v in counts.items()})
                self.client.log_batch(rid, params=[Param(k, str(v)) for k, v in params.items()])
            if run.data.tags.get("cc.provenance_logged") != "true":
                for path in (AUDIT / "protocol.json", RUN / "config.json", RUN / "initial.json"):
                    self.client.log_artifact(rid, str(path), artifact_path="provenance")
                self.client.set_tag(rid, "cc.provenance_logged", "true")
            keys = self.client.get_run(rid).data.metrics
            self.known[arm] = {}
            for key in keys:
                for m in self.client.get_metric_history(rid, key):
                    identity = (key, m.step)
                    old = self.known[arm].get(identity)
                    if old is not None and old != m.value:
                        raise RuntimeError(f"Conflicting saved metric: {identity}")
                    self.known[arm][identity] = m.value

    def log_missing(self, arm, points):
        pending = []
        for key, value, step, seconds in points:
            identity = (key, step)
            old = self.known[arm].get(identity)
            if old is not None:
                if old != float(value):
                    raise RuntimeError(f"Committed source metric changed: {arm} {identity}")
                continue
            pending.append(Metric(key, float(value), self.start_ms + int(seconds * 1000), step))
        for start in range(0, len(pending), 500):
            batch = pending[start:start + 500]
            self.client.log_batch(self.ids[arm], metrics=batch)
            self.known[arm].update({(m.key, m.step): m.value for m in batch})
        return len(pending)

    def sync(self):
        history = read(RUN / "history.json")
        initial = read(RUN / "initial.json")
        controller = read(AUDIT / "controller_status.json")
        progress = read(RUN / "progress.json")
        summary_path = AUDIT / "comparison_summary.json"
        summary = read(summary_path) if summary_path.exists() else None
        complete = bool(summary and summary.get("status") == "verified"
                        and summary.get("protocol_sha256") == self.digest)
        added = {}
        for arm in ARMS:
            added[arm] = self.log_missing(arm, metrics_for(arm, initial, history))
            rid = self.ids[arm]
            stage = "verified" if complete else str(progress.get("stage", controller.get("stage")))
            self.client.log_batch(rid, tags=[
                RunTag("cc.stage", stage),
                RunTag("cc.current_epoch", str(progress.get("epoch", ""))),
                RunTag("cc.current_updates", str(progress.get("updates_per_arm", ""))),
                RunTag("cc.last_sync_utc", datetime.now().astimezone().isoformat())])
            if complete:
                added[arm] += self.log_missing(arm, final_metrics(arm, summary, history))
                for path in (summary_path, RUN / "history.json"):
                    self.client.log_artifact(rid, str(path), artifact_path="final")
                self.client.set_tag(rid, "cc.verdict", summary["verdict"])
                self.client.set_terminated(rid, status="FINISHED")
        record = {"pid": os.getpid(), "updated": datetime.now().astimezone().isoformat(),
                  "experiment_id": self.exp_id, "run_ids": self.ids,
                  "last_committed_epoch": history[-1]["epoch"] if history else 0,
                  "current_epoch": progress.get("epoch"), "updates_per_arm": progress.get("updates_per_arm"),
                  "added": added, "metric_points": {a: len(v) for a, v in self.known.items()},
                  "verified_complete": complete, "telemetry_disabled": True,
                  "tracking_uri": URI}
        atomic(STATE / "c146_bridge_status.json", record)
        if any(added.values()) or complete:
            print(json.dumps(record), flush=True)
        return complete

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    with (STATE / "c146_bridge.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bridge = Bridge()
        while True:
            try:
                complete = bridge.sync()
                if args.once or complete:
                    break
            except (OSError, json.JSONDecodeError):
                if args.once:
                    raise
                traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()
