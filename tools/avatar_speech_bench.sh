#!/bin/sh
# 3D frame rate while ふぇれ子 speaks, per decoder setting (the chat screen's ChatAuto test hook).
#   tools/avatar_speech_bench.sh gpu16 gpu16@36 gpu16@28
cd "$(dirname "$0")/.." || exit 1
LINE='ふーん、そうなんだ。まあ、キミがそう言うなら、ふぇれ子もちょっとくらいは付き合ってあげてもいいけど。'
for DEC in "$@"; do
  start=$(sh tools/phone.sh shell "wc -l < /sdcard/Android/data/local.fereko.probe/files/runs.jsonl" | tr -dc 0-9)
  sh tools/phone.sh shell am force-stop local.fereko.probe
  sh tools/phone.sh shell "am start -n local.fereko.probe/.ChatAuto --es say '$LINE' --ei times 3 --es dec '$DEC'" > /dev/null
  sleep ${WAIT:-95}
  echo "== $DEC"
  sh tools/phone.sh shell "tail -n +$((start + 1)) /sdcard/Android/data/local.fereko.probe/files/runs.jsonl" | grep '"avatar_perf"\|"say_test"' | \
    .venv/Scripts/python.exe -I -c "
import sys, json
for l in sys.stdin:
    o = json.loads(l)
    if o['kind'] == 'avatar_perf':
        if o['talking_share'] > 0.3: print('  perf fps', o['fps'], 'p99', o['p99_ms'], 'max', o['max_ms'], 'talk', o['talking_share'])
    else:
        print('  say rtf', round(o['rtf'], 2), 'first', o['send_to_first_sound_ms'], 'underruns', o['underruns'], 'synth', o['synth_ms'])
"
done
