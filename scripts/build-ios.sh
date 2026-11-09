#!/usr/bin/env bash
# Build RIE.xcframework (device + simulator) for the Swift package. Run on macOS with Xcode.
# Usage: scripts/build-ios.sh [--gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES=""
[[ "${1:-}" == "--gpu" ]] && FEATURES="--features gpu"
rustup target add aarch64-apple-ios aarch64-apple-ios-sim >/dev/null
for t in aarch64-apple-ios aarch64-apple-ios-sim; do
  cargo build --release -p rie-ffi --no-default-features $FEATURES --target "$t"
done
OUT=bindings/ios/build
rm -rf "$OUT/RIE.xcframework" && mkdir -p "$OUT/headers"
cp crates/rie-ffi/include/rie.h "$OUT/headers/"
xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/release/librie.a -headers "$OUT/headers" \
  -library target/aarch64-apple-ios-sim/release/librie.a -headers "$OUT/headers" \
  -output "$OUT/RIE.xcframework"
echo "Built $OUT/RIE.xcframework"
