# CC experiment logging: server-local SwanLab only.
export CC_TRACKING_BACKEND=swanlab
export CC_TRACKING_ROOT=/home/chenzhe/work/lunar-pathloss-rm-challenge/CC/tracking/swanlab
export SWANLAB_MODE=local
export SWANLAB_LOGDIR="$CC_TRACKING_ROOT/logs"
export SWANLAB_LOG_DIR="$SWANLAB_LOGDIR"
export SWANLAB_ROOT="$CC_TRACKING_ROOT/root"
export SWANLAB_API_HOST=http://127.0.0.1:16007
export SWANLAB_PROBE_MONITOR=false
export SWANLAB_PROBE_SWANLAB=false
export CC_SWANLAB_PROJECT=Lunar_CC
export CC_SWANLAB_HOST=127.0.0.1
export CC_SWANLAB_PORT=16007
export CC_SWANLAB_DASHBOARD_URL=http://127.0.0.1:16007/
unset SWANLAB_API_KEY SWANLAB_WORKSPACE MLFLOW_TRACKING_URI MLFLOW_EXPERIMENT_ID
# Historical readers remain local and quiet.
export MLFLOW_DISABLE_TELEMETRY=true
export DO_NOT_TRACK=true
