#!/usr/bin/env bash
# Run the stitaP tool harness against rie-server inside a sandbox (Linux, root or user
# namespaces):
#   * private network namespace with loopback only: no internet; rie-server runs inside it
#   * private mount namespace: the harness checkout is bind-mounted read-only
#   * throwaway HOME / STITAP_HOME, proxy variables removed, approvals: deny
#
# usage: scripts/sandbox-harness.sh HARNESS_DIR MODEL.gguf [harness args...]
#   e.g. scripts/sandbox-harness.sh ~/tool-harness model.gguf doctor
#        scripts/sandbox-harness.sh ~/tool-harness model.gguf batch prompts.jsonl --out traj.jsonl
# env:   SERVER_ARGS="--tool-choice required"   extra rie-server flags
#        WORK=/tmp/rie-sandbox                  where logs, adapter and HOME go
set -euo pipefail
HARNESS=$(realpath "$1"); MODEL=$(realpath "$2"); shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SERVER="$ROOT/target/release/rie-server"
[[ -x "$SERVER" ]] || cargo build --release -p rie-server --manifest-path "$ROOT/Cargo.toml"
WORK=${WORK:-/tmp/rie-sandbox}
mkdir -p "$WORK"
rm -rf "$WORK/home"
export HARNESS MODEL SERVER WORK SERVER_ARGS="${SERVER_ARGS:-}"
exec unshare --net --mount --pid --fork bash -c '
set -e
python3 -I -c "
import fcntl, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ifr = struct.pack(\"16sH\", b\"lo\", 0)
flags = struct.unpack(\"16sH\", fcntl.ioctl(s, 0x8913, ifr))[1]
fcntl.ioctl(s, 0x8914, struct.pack(\"16sH\", b\"lo\", flags | 1))"
mount --make-rprivate /
mount --bind "$HARNESS" "$HARNESS" && mount -o remount,ro,bind "$HARNESS"
export HOME="$WORK/home" STITAP_HOME="$WORK/home/.stitap"
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY NO_PROXY no_proxy NODE_EXTRA_CA_CERTS
mkdir -p "$STITAP_HOME"
cat > "$STITAP_HOME/config.yaml" <<YAML
model:
  provider: openai
  base_url: http://127.0.0.1:8091/v1
  name: rie-local
  tool_mode: native
  context_window: 32768
agent:
  tool_profile: slm
  max_iterations: 4
approvals:
  mode: deny
YAML
RIE_SERVER_LOG="$WORK/requests.jsonl" "$SERVER" -m "$MODEL" -c 32768 --port 8091 --alias rie-local \
    --rl --auto-reward --adapter "$HOME/adapter.riea" $SERVER_ARGS > "$WORK/server.log" 2>&1 &
for i in $(seq 60); do curl -sf http://127.0.0.1:8091/health >/dev/null && break; sleep 0.5; done
echo "sandbox: internet blocked: $(curl -s -m 3 -o /dev/null https://example.com && echo NO || echo yes); harness read-only: $(touch "$HARNESS/.w" 2>/dev/null && echo NO || echo yes)"
cd "$HARNESS"
node agent/bin/harness.mjs "$@" || echo "harness exited with $?"
echo "rie-server learning stats: $(curl -s http://127.0.0.1:8091/v1/rl)"
' bash "$@"
