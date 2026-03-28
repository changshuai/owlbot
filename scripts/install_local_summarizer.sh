#!/usr/bin/env bash
set -euo pipefail

# One-click local summarizer setup for macOS:
# - install Ollama if missing
# - start Ollama service if not running
# - pull a default lightweight model
# - run a quick smoke test

MODEL="${1:-qwen2.5:3b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

log() { printf "\033[1;34m[install]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

require_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This script currently supports macOS only."
    err "Install Ollama manually for your OS, then run: ollama pull ${MODEL}"
    exit 1
  fi
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

ensure_ollama_installed() {
  if has_cmd ollama; then
    log "Ollama already installed."
    return
  fi

  log "Ollama not found. Installing via Homebrew..."
  if ! has_cmd brew; then
    err "Homebrew not found. Please install Homebrew first: https://brew.sh"
    exit 1
  fi

  brew install ollama
  log "Ollama installed."
}

ensure_ollama_running() {
  if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    log "Ollama service is already running."
    return
  fi

  log "Starting Ollama service..."
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &

  for _ in {1..20}; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      log "Ollama service is up."
      return
    fi
    sleep 1
  done

  err "Ollama service failed to start in time."
  err "Check logs: /tmp/ollama-serve.log"
  exit 1
}

pull_model() {
  log "Pulling model: ${MODEL}"
  ollama pull "${MODEL}"
}

smoke_test() {
  log "Running smoke test..."
  local output
  output="$(ollama run "${MODEL}" "Summarize in one short sentence: tool execution succeeded with exit code 0." 2>/dev/null || true)"

  if [[ -z "${output// }" ]]; then
    warn "Smoke test returned empty output, but installation may still be OK."
  else
    log "Smoke test output:"
    printf "%s\n" "${output}"
  fi
}

print_next_steps() {
  cat <<EOF

Done.
- Model: ${MODEL}
- Ollama host: ${OLLAMA_HOST}

Quick usage examples:
  ollama run ${MODEL} "Summarize this log in 3 bullet points: ..."
  curl ${OLLAMA_HOST}/api/tags

Tip:
  To install another model:
    ./scripts/install_local_summarizer.sh qwen2.5:7b
EOF
}

main() {
  require_macos
  ensure_ollama_installed
  ensure_ollama_running
  pull_model
  smoke_test
  print_next_steps
}

main "$@"
