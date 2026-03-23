#!/bin/bash

runtime_resolve_mode() {
  local requested="${EXEC_MODE:-auto}"
  case "${requested}" in
    docker|local)
      printf '%s\n' "${requested}"
      return 0
      ;;
    auto)
      if runtime_python_has_torch >/dev/null 2>&1; then
        printf 'local\n'
      elif runtime_docker_usable >/dev/null 2>&1; then
        printf 'docker\n'
      else
        printf 'local\n'
      fi
      return 0
      ;;
    *)
      echo "Unsupported EXEC_MODE=${requested}. Use docker, local, or auto." >&2
      return 1
      ;;
  esac
}

runtime_docker_usable() {
  command -v docker >/dev/null 2>&1 || return 1
  docker version >/dev/null 2>&1 || return 1
}

runtime_resolve_python() {
  if [[ -n "${LOCAL_PYTHON:-}" ]]; then
    printf '%s\n' "${LOCAL_PYTHON}"
    return 0
  fi
  if [[ -x ".venv_local/bin/python" ]]; then
    printf '%s\n' ".venv_local/bin/python"
    return 0
  fi
  if [[ -x ".venv/bin/python" ]]; then
    printf '%s\n' ".venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "Could not find a local Python interpreter. Set LOCAL_PYTHON=/path/to/python." >&2
  return 1
}

runtime_python_has_torch() {
  local py_bin
  py_bin="$(runtime_resolve_python)" || return 1
  "${py_bin}" -c "import torch" >/dev/null 2>&1
}

runtime_log_env_summary() {
  local mode="$1"
  if [[ "${mode}" == "docker" ]]; then
    echo "exec_mode=docker image=${DOCKER_IMAGE:-myrepo:gpu}"
    return 0
  fi
  local py_bin
  py_bin="$(runtime_resolve_python)" || return 1
  echo "exec_mode=local python=${py_bin}"
}

runtime_exec_python() {
  local mode
  mode="$(runtime_resolve_mode)" || return 1
  if [[ "${mode}" == "docker" ]]; then
    local image="${DOCKER_IMAGE:-myrepo:gpu}"
    docker run --rm --gpus all --ipc=host \
      -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
      -v "$(pwd)":/app -w /app "${image}" \
      "$@"
    return $?
  fi
  local py_bin
  py_bin="$(runtime_resolve_python)" || return 1
  if ! "${py_bin}" -c "import torch" >/dev/null 2>&1; then
    echo "Local Python (${py_bin}) does not have torch installed." >&2
    echo "Set LOCAL_PYTHON to a torch-enabled interpreter, or use EXEC_MODE=docker." >&2
    return 1
  fi
  PYTORCH_ENABLE_MPS_FALLBACK=1 "${py_bin}" "$@"
}
