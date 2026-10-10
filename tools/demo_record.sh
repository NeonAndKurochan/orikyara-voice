#!/bin/sh
# Records a demo of the chat screen for SNS: starts the conversation over, ふぇれ子 greets, the given
# lines are sent as the user's messages (real Claude turns), the screen is recorded and her voice is
# laid over it on the PC (Android's screenrecord has no sound).
#   tools/demo_record.sh <seconds (max 180)> "line|line|…" [out.mp4]
cd "$(dirname "$0")/.." || exit 1
# screenrecord starts recording ~0.45 s after the date it prints (measured 2026-10-10): the voice is shifted by that
LAG_MS=${LAG_MS:-450}
SECS=${1:-120}
SCRIPT=$2
OUT=${3:-out/demo/fereko_demo.mp4}
EXT=/sdcard/Android/data/local.fereko.probe/files
mkdir -p out/demo/rec
rm -f out/demo/rec/*.f32
sh tools/phone.sh shell "rm -rf $EXT/rec; rm -f /sdcard/fereko_demo.mp4"
sh tools/phone.sh shell am force-stop local.fereko.probe
# screen recording in the background, with the device time it started at
sh tools/phone.sh shell "date +%s%3N > /sdcard/fereko_demo_start.txt; screenrecord --time-limit $SECS --bit-rate 12000000 /sdcard/fereko_demo.mp4" &
REC=$!
sleep 1
sh tools/phone.sh shell "am start -n local.fereko.probe/.ChatAuto --ez rec true --ez reset true --ez greet true --es script '$SCRIPT'" > /dev/null
wait $REC
sleep 2
MSYS_NO_PATHCONV=1 "C:/Program Files/Unity 6000.3.24f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe" pull /sdcard/fereko_demo.mp4 out/demo/screen.mp4
START=$(sh tools/phone.sh shell cat /sdcard/fereko_demo_start.txt | tr -dc 0-9)
for f in $(sh tools/phone.sh shell ls $EXT/rec 2>/dev/null | tr -d '\r'); do
  sh tools/phone.sh pull rec/$f out/demo/rec/$f > /dev/null
done
/c/StabilityMatrix/Data/Packages/ComfyUI/venv/Scripts/python.exe -I tools/mux_demo.py out/demo/screen.mp4 "$((START + LAG_MS))" out/demo/rec "$OUT" ${FROM:-0} ${TO:-9999} ${CROP_TOP:-130}
