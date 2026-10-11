#!/bin/sh
# Helpers for the phone over adb (USB debugging). Usage:
#   tools/phone.sh install <apk>          install / update the probe app (keeps its data)
#   tools/phone.sh auto "<steps>"         run NpuActivity automation (see NpuActivity.runAuto)
#   tools/phone.sh runs [n]               last n records of runs.jsonl (default 20)
#   tools/phone.sh wait                   wait until the running automation writes auto_done (max 30 min)
#   tools/phone.sh pull <file> <dest>     copy a file from the app's external files dir
#   tools/phone.sh push <file>            put a file into the app's model pack (files/pack)
#   tools/phone.sh pushchar <dir>         a character folder into files/chars/<dir name> (see chars/template)
#   tools/phone.sh pushflat <dir> [id]    2D parts (rig.json, layers_flat.json, PNGs; tools/flat_bake.py) into
#                                         files/pack/flat, or files/chars/<id>/flat; replaces what was there
export MSYS_NO_PATHCONV=1
ADB="C:/Program Files/Unity 6000.3.24f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe"
PKG=local.fereko.probe
EXT=/sdcard/Android/data/$PKG/files
case "$1" in
  install) "$ADB" install -r "$2" ;;
  auto)
    "$ADB" shell wc -l < /dev/null >/dev/null 2>&1
    "$ADB" shell "wc -l < $EXT/runs.jsonl" > /tmp/fereko_runs_start 2>/dev/null
    "$ADB" shell am force-stop $PKG >/dev/null 2>&1
    "$ADB" shell am start -n $PKG/.NpuAuto --es auto "'$2'" ;;
  runs) "$ADB" shell tail -n "${2:-20}" $EXT/runs.jsonl ;;
  wait)
    start=$(date +%s)
    from=$(( $(cat /tmp/fereko_runs_start 2>/dev/null | tr -dc 0-9) + 1 ))
    while :; do
      last=$("$ADB" shell "tail -n +$from $EXT/runs.jsonl" | grep '"auto_done"' | tail -n 1)
      [ -n "$last" ] && break
      [ $(( $(date +%s) - start )) -gt 1800 ] && { echo "timeout"; break; }
      sleep 10
    done
    echo "$last" ;;
  pull) "$ADB" pull "$EXT/$2" "$3" ;;
  push)
    name=$(basename "$2")
    "$ADB" push "$2" /data/local/tmp/"$name" && "$ADB" shell chmod 644 /data/local/tmp/"$name" &&
    "$ADB" shell run-as $PKG cp /data/local/tmp/"$name" files/pack/"$name" && "$ADB" shell rm /data/local/tmp/"$name" ;;
  pushchar)
    # a character folder (character.json, persona.md, optional speaker.f32/json, *.vrm, *.vrma, stand.png)
    # into files/chars/<folder name>; give a Windows path (C:/...)
    id=$(basename "$2")
    "$ADB" shell run-as $PKG mkdir -p files/chars/"$id"
    for f in "$2"/*; do
      [ -f "$f" ] || continue
      name=$(basename "$f")
      "$ADB" push "$f" /data/local/tmp/"$name" > /dev/null && "$ADB" shell chmod 644 /data/local/tmp/"$name" &&
      "$ADB" shell run-as $PKG cp /data/local/tmp/"$name" files/chars/"$id"/"$name" && "$ADB" shell rm /data/local/tmp/"$name" &&
      echo "  $id/$name"
    done ;;
  pushflat)
    dest=files/pack/flat
    [ -n "$3" ] && dest=files/chars/"$3"/flat
    "$ADB" shell run-as $PKG rm -rf "$dest" && "$ADB" shell run-as $PKG mkdir -p "$dest"
    n=0
    for f in "$2"/*.png "$2"/*.json; do
      [ -f "$f" ] || continue
      name=$(basename "$f")
      "$ADB" push "$f" /data/local/tmp/"$name" > /dev/null && "$ADB" shell chmod 644 /data/local/tmp/"$name" &&
      "$ADB" shell run-as $PKG cp /data/local/tmp/"$name" "$dest"/"$name" && "$ADB" shell rm /data/local/tmp/"$name" &&
      n=$((n + 1))
    done
    echo "$n files -> $dest" ;;
  *) "$ADB" "$@" ;;
esac
