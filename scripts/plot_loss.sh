#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <run_name> [refresh_seconds]" >&2
  echo "Example: $0 vm_dinovit_v2" >&2
  exit 1
fi

run_name="$1"
refresh_seconds="${2:-2}"
logfile="logs/${run_name}/logfile.txt"

if ! command -v gnuplot >/dev/null 2>&1; then
  echo "gnuplot is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -f "$logfile" ]]; then
  echo "logfile not found: $logfile" >&2
  exit 1
fi

if [[ ! "$refresh_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "refresh_seconds must be a positive number" >&2
  exit 1
fi

gnuplot_logfile="${logfile//\\/\\\\}"
gnuplot_logfile="${gnuplot_logfile//\"/\\\"}"
gnuplot_title="${run_name//\\/\\\\}"
gnuplot_title="${gnuplot_title//\"/\\\"}"

gnuplot -persist <<EOF
logfile = "${gnuplot_logfile}"
refresh = ${refresh_seconds}
set title "${gnuplot_title} loss"
set xlabel "step"
set ylabel "loss"
set grid
while (1) {
  plot "< awk '{step=\"\"; loss=\"\"; for(i=1;i<=NF;i++){ if(\$i ~ /^step=/){split(\$i,a,\"[=/]\"); step=a[2]} if(\$i ~ /^loss=/){split(\$i,b,\"=\"); loss=b[2]} } if(step != \"\" && loss != \"\") print step, loss}' " . logfile using 1:2 with lines lw 2 title "loss";
  pause refresh;
}
EOF
