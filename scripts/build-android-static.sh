#!/usr/bin/env bash
# Build a fully static ARM64 `rie` binary that runs on any Android phone via `adb shell`
# (or in Termux) without the NDK. Needs: rustup target aarch64-unknown-linux-musl and an
# aarch64 cross linker (Debian/Ubuntu: apt install gcc-aarch64-linux-gnu).
set -euo pipefail
cd "$(dirname "$0")/.."
rustup target add aarch64-unknown-linux-musl >/dev/null
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER="${LINKER:-aarch64-linux-gnu-gcc}" \
RUSTFLAGS="-C target-feature=+crt-static" \
  cargo build --release -p rie-cli -p rie-server --target aarch64-unknown-linux-musl
OUT=target/aarch64-unknown-linux-musl/release/rie
echo "Built $OUT and $OUT-server ($(du -h "$OUT" | cut -f1)). Next: scripts/phone-bench.sh"
