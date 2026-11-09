#!/usr/bin/env bash
# Build librie.so for Android and copy it into the Kotlin library module.
# Requirements: Android NDK r26+ ($ANDROID_NDK_HOME), `cargo install cargo-ndk`,
#               `rustup target add aarch64-linux-android x86_64-linux-android`.
# Usage: scripts/build-android.sh [--gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES="jni"
[[ "${1:-}" == "--gpu" ]] && FEATURES="jni,gpu"
OUT=bindings/android/rie/src/main/jniLibs
# API 28: first level with dependable Vulkan 1.1 compute; the CPU path runs on API 26+.
cargo ndk -t arm64-v8a -t x86_64 -P 28 -o "$OUT" build --release -p rie-ffi --features "$FEATURES"
# Also build the CLI for on-device benchmarking via `adb shell`.
cargo ndk -t arm64-v8a -P 28 build --release -p rie-cli ${1:+--features gpu}
echo "Libraries in $OUT; CLI at target/aarch64-linux-android/release/rie"
echo "Benchmark on a phone:"
echo "  adb push target/aarch64-linux-android/release/rie /data/local/tmp/ && adb push model.gguf /data/local/tmp/"
echo "  adb shell /data/local/tmp/rie bench -m /data/local/tmp/model.gguf"
