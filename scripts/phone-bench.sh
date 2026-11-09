#!/usr/bin/env bash
# Benchmark RIE on a connected Android phone and print a markdown report.
#
#   scripts/build-android-static.sh          # once
#   scripts/phone-bench.sh                   # synthetic models, no download needed
#   scripts/phone-bench.sh model.gguf ...    # also benchmark real GGUF files
#
# Environment:
#   PRESETS="qwen2.5-0.5b llama-3.2-1b qwen3-1.7b"   synthetic shapes to test
#   WTYPES="Q4_0 Q4_K"                               weight formats for synthetic models
#   THREADS="auto 2 4 6"                             thread counts to sweep
#   PP=128 TG=64                                     prompt / generated tokens
#   ADB=adb  (set ADB=local to run on the device itself, e.g. inside Termux)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${BIN:-target/aarch64-unknown-linux-musl/release/rie}
ADB=${ADB:-adb}
DIR=${DIR:-/data/local/tmp/rie}
PRESETS=${PRESETS:-"qwen2.5-0.5b llama-3.2-1b qwen3-1.7b"}
WTYPES=${WTYPES:-"Q4_0 Q4_K"}
THREADS=${THREADS:-"auto 2 4 6"}
PP=${PP:-128}
TG=${TG:-64}

if [[ "$ADB" == local ]]; then
  run() { sh -c "$*"; }
  DIR=${DIR_LOCAL:-$PWD/rie-bench}; mkdir -p "$DIR"; cp "$BIN" "$DIR/rie"
else
  run() { "$ADB" shell "$*"; }
  "$ADB" get-state >/dev/null || { echo "no device: connect a phone with USB debugging enabled" >&2; exit 1; }
  run "mkdir -p $DIR"
  "$ADB" push "$BIN" "$DIR/rie" >/dev/null
  run "chmod 755 $DIR/rie"
fi

echo "# RIE phone benchmark"
echo
if [[ "$ADB" != local ]]; then
  echo "Device: $(run getprop ro.product.manufacturer | tr -d '\r') $(run getprop ro.product.model | tr -d '\r'), SoC $(run getprop ro.soc.model | tr -d '\r'), Android $(run getprop ro.build.version.release | tr -d '\r')"
fi
echo
echo '```'
run "$DIR/rie device" | tr -d '\r'
echo '```'
echo

bench() { # label, model path, extra args
  local label=$1 model=$2; shift 2
  for t in $THREADS; do
    local targ=""; [[ "$t" != auto ]] && targ="-t $t"
    run "$DIR/rie bench -m $model -p $PP -n $TG $targ $*" 2>/dev/null | tr -d '\r' \
      | grep '^| ' | grep -v '^| model' | sed "s/^| /| $label | t=$t | /"
  done
}

echo "| case | threads | model | backend | test | tok/s |"
echo "|---|---|---|---|---|---|"
for p in $PRESETS; do
  for w in $WTYPES; do
    f="$DIR/synth-$p-$w.gguf"
    run "[ -f $f ] || $DIR/rie synth $p --wtype $w -o $f >/dev/null"
    bench "$p" "$f"
  done
done
# kernel tiers and 8-bit KV cache on the first synthetic model
first="$DIR/synth-$(echo $PRESETS | cut -d' ' -f1)-$(echo $WTYPES | cut -d' ' -f1).gguf"

for v in "RIE_DISABLE=i8mm" "RIE_DISABLE=i8mm,dotprod"; do
  run "$v $DIR/rie bench -m $first -p $PP -n $TG" 2>/dev/null | tr -d '\r' | grep '^| ' | grep -v '^| model' | sed "s/^| /| $v | auto | /"
done
run "$DIR/rie bench -m $first -p $PP -n $TG --kv q8_0" 2>/dev/null | tr -d '\r' | grep '^| ' | grep -v '^| model' | sed "s/^| /| kv=q8_0 | auto | /"

for m in "$@"; do
  name=$(basename "$m")
  if [[ "$ADB" != local ]]; then "$ADB" push "$m" "$DIR/$name" >/dev/null; else cp "$m" "$DIR/$name"; fi
  bench "$name" "$DIR/$name"
done
echo
echo "_Synthetic models have real shapes and random weights: speed is representative, output text is not._"
