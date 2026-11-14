"""CC's default SwanLab entry point. All records stay inside CC on transformer."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "tracking" / "swanlab"
PROJECT = "Lunar_CC"
PROBES = ("hardware", "runtime", "requirements", "conda", "git", "swanlab", "monitor")

def _sdk():
    # Set before SDK import and on every initialization. Never use cloud fallback.
    os.environ.update(
        CC_TRACKING_BACKEND="swanlab", SWANLAB_MODE="local",
        SWANLAB_LOGDIR=str(STATE / "logs"), SWANLAB_ROOT=str(STATE / "root"),
        SWANLAB_API_HOST="http://127.0.0.1:16007",
        SWANLAB_PROBE_MONITOR="false", SWANLAB_PROBE_SWANLAB="false",
    )
    for key in ("SWANLAB_API_KEY", "SWANLAB_WORKSPACE", "MLFLOW_TRACKING_URI",
                "MLFLOW_EXPERIMENT_ID"):
        os.environ.pop(key, None)
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    import swanlab
    return swanlab

def init(*, name, config, campaign, run_id=None, resume="never",
         parent_run_id=None, parent_checkpoint=None, start_step=None,
         description=None, tags=None):
    """Create one local run or an explicitly linked continuation segment.

    SwanLab 0.10.5 local mode does not append to the previous directory on resume.
    Restore/verify model, optimizer, RNG and true global step in the training code;
    then start a NEW segment with parent_run_id and the recovered start_step.
    """
    if resume != "never":
        raise ValueError("Local resume does not append. Create a new linked segment "
                         "using parent_run_id, parent_checkpoint and start_step.")
    if parent_run_id is not None:
        if not parent_checkpoint or start_step is None:
            raise ValueError("A continuation segment requires checkpoint and recovered start_step.")
        if run_id == parent_run_id:
            raise ValueError("A continuation segment must have a new run_id.")
    sdk = _sdk()
    metadata = {
        "backend": "swanlab", "mode": "local", "project": PROJECT,
        "campaign": campaign, "sdk_version": sdk.__version__,
        "parent_run_id": parent_run_id, "parent_checkpoint": parent_checkpoint,
        "recovered_start_step": start_step,
        "continuation_policy": "new linked segment; log true global steps",
    }
    return sdk.init(
        project=PROJECT, name=name, id=run_id, resume="never", mode="local",
        log_dir=str(STATE / "logs"),
        config={**config, "_cc_tracking": metadata}, description=description,
        tags=[campaign, *(tags or [])],
        settings=sdk.Settings(probe={key: False for key in PROBES}),
    )

def log(data, *, step):
    """Record the caller's real update/epoch index, without recalculation."""
    import swanlab
    return swanlab.log(data, step=step)

def finish():
    import swanlab
    return swanlab.finish()
