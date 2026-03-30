#!/usr/bin/env bash

bundle_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

bundle_refresh_experiment_db() {
  local repo_root
  repo_root="$(bundle_repo_root)"
  (
    cd "${repo_root}" && \
    python3 scripts/build_experiment_db.py >/dev/null 2>&1
  ) || true
}

bundle_init() {
  local bundle_id="$1"
  local latest_link="$2"
  BUNDLE_DIR="logs/${bundle_id}"
  TIMELINE="${BUNDLE_DIR}/timeline.log"
  mkdir -p "${BUNDLE_DIR}"
  if [[ ! -f "${TIMELINE}" ]]; then
    : > "${TIMELINE}"
  fi
  ln -sfn "${bundle_id}" "${latest_link}"
  bundle_refresh_experiment_db
}

bundle_log_line() {
  local line="[$(date)] $*"
  echo "${line}" | tee -a "${TIMELINE}"
}

bundle_mark_start() {
  local run_id="$1"
  shift || true
  bundle_log_line "START ${run_id}${*:+ $*}"
  bundle_refresh_experiment_db
}

bundle_mark_end() {
  local run_id="$1"
  shift || true
  bundle_log_line "END   ${run_id}${*:+ $*}"
  bundle_refresh_experiment_db
}

bundle_mark_fail() {
  local run_id="$1"
  shift || true
  bundle_log_line "FAIL  ${run_id}${*:+ $*}"
  bundle_refresh_experiment_db
}
