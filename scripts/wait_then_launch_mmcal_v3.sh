#!/bin/bash
set -euo pipefail

LOG_PATH="logs/mmcal_v3_autostart.log"
mkdir -pv logs >/dev/null 2>&1 || true

echo "[$(date)] wait_then_launch_mmcal_v3: started" >> "${LOG_PATH}"

while true; do
  if pgrep -f "launch_prefix_calib_sweep_v2.sh" >/dev/null; then
    echo "[$(date)] wait_then_launch_mmcal_v3: waiting for sweep v2 launcher to finish" >> "${LOG_PATH}"
    sleep 60
    continue
  fi
  if pgrep -f "runmm.sh mmcal2_" >/dev/null; then
    echo "[$(date)] wait_then_launch_mmcal_v3: waiting for mmcal2 run to finish" >> "${LOG_PATH}"
    sleep 60
    continue
  fi
  break
done

echo "[$(date)] wait_then_launch_mmcal_v3: launching v3 accumulation sweep" >> "${LOG_PATH}"
./scripts/launch_prefix_calib_sweep_v3_accum.sh >> "${LOG_PATH}" 2>&1
echo "[$(date)] wait_then_launch_mmcal_v3: done" >> "${LOG_PATH}"
