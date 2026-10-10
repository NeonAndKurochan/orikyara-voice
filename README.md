# ふぇれ子 voice chat

デモ（声あり）：会話をリセットして、あいさつと気軽なやりとりをしたところ。好感度がまだ低いので、しっぽは見せてもらえない。

https://github.com/user-attachments/assets/c14f24b3-0dd0-4260-bb99-ddfba566ab5b

スマホ（Android）で、キャラクターと声で会話するアプリ。会話は Claude（Cloudflare Worker の中継経由）、声の合成（Irodori-TTS）と音声認識（ReazonSpeech）はスマホの中で動く。3D（VRM）か立ち絵で表示する。

## このリポジトリについて

- **参考にしたもの・使っているもの：`CREDITS.md`**（Irodori-TTS と、スマホでの実行の手本にした Onsei 用の ONNX 変換版、ReazonSpeech、sherpa-onnx など）
- コードは MIT ライセンス（`LICENSE`）。`web/vendor` の three.js と three-vrm は MIT で、それぞれのライセンスファイルを同梱
- **入っていないもの**：キャラクターの絵・キャラ設定・声の説明・アイコン（`chat/`）、会話の記録、3D モデルとモーション（VRM / VRMA。作者専用）、音声合成と音声認識のモデル（`models/`、`fereko-pack*`）、テスト結果（`out/`）、Python 環境、中継の秘密の設定
- `chat/` がないときは、見本（`chat-example/`：テンプレートのキャラ設定と声の説明）と仮のアイコン（`android-probe/app/icon-placeholder/`）でビルドされる。自分のキャラは `chat/persona.md`、`chat/voice.txt`、`chat/icon/res/`（アダプティブアイコン）に置く。ほかのキャラは `chars/template` を見本に追加できる
- モデル：Irodori-TTS v4.1-Small-MF の ONNX 版（ライセンスは配布元を確認）、ReazonSpeech k2 v2（Apache-2.0、`reazon-research/reazonspeech-k2-v2`）。PC での確認用の `tools/lib/onnxruntime-1.30.0.jar` は Maven Central から取る
- ビルド：`sh tools/gradle.sh assembleRelease`（Unity 6000.3 の Android モジュールに入っている Gradle・JDK・SDK を使う。`android-probe/local.properties` に `sdk.dir` を書く）
- 中継：`relay/`。`relay/wrangler.example.toml` を `wrangler.toml` にコピーして KV の ID を自分のものにし、API キーと合言葉は `wrangler secret put` で入れる
- 中継の住所と合言葉はコードに書かない。アプリは設定の「合言葉」で、`https://…` の住所の行と合言葉の行を書いた小さなテキストファイルを取り込む。PC は `~/.fereko/relay-url.txt` と `~/.fereko/relay-token.txt`（または環境変数 `FEREKO_RELAY_URL` / `FEREKO_RELAY_TOKEN`）

以下は開発の記録。

# ふぇれ子 音声会話

ふぇれ子と声で話すアプリ（Windows と Android、あとで 3D）。今は 2 本を並行して進めている。

1. **スマホ単体で Irodori を動かす試験**（Nothing Phone (3)、ONNX Runtime の CPU 実行）
2. **PC で文字で話す**（返事は Claude、声は ONNX 版 Irodori）

## フォルダ

| 場所 | 中身 |
|---|---|
| `models/irodori-mf-a7542a4/` | `raratu/Onsei-Irodori-v4.1-Small-MF`（タグ `ios-mf-v2` = コミット `a7542a4`）。`manifest.json` の SHA-256 で全ファイルを確認済み |
| `fereko-pack/` | スマホに渡す一式（3.12GB）。参考音声なしで使うモデル 5 つ、トークナイザー、話者の定数、自己テスト用の正解データ。`tools/prep_phone_pack.py` で作る |
| `tools/pc_synth.py` | 基準になる推論（Python + onnxruntime）。Irodori-TTS 本体の `89f9d8f` と同じ手順 |
| `tools/fereko_chat.py` | PC で話すプログラム |
| `chat/persona.md` | ふぇれ子の性格と話し方。自由に書き換えてよい |
| `android-probe/` | Android の試験アプリ（Java、AndroidX なし）。推論の中身は `app/src/main/java/local/fereko/probe/core/` |
| `tools/java/PcCheck.java` | Android と同じ Java の中身を PC で動かして確かめる |
| `out/apk/` | ビルドした APK |

## これまでに確かめたこと（PC）

- トークン ID は本体と一致（28 文）。BOS=1、PAD=3
- 文字を実際の長さだけ渡しても、256 個分まで埋めたときと結果がビット単位で同じ。テキストエンコーダーは 0.9 秒 → 0.05 秒
- Java 版（スマホと同じコード）は Python 版とまったく同じ結果（潜在変数・波形とも差 0）
- PC の CPU で RTF 約 0.6。時間の 75% は最後の VAE デコーダー

## スマホで試す

1. USB でスマホをつないで「ファイル転送」にして、`out\apk\fereko-tts-probe-0.1.0.apk` と `fereko-pack` フォルダをまるごと、スマホの `Download` にコピーする
2. スマホのファイルアプリで APK を開いてインストールする（「不明なアプリのインストール」の許可を聞かれたら、自分で判断して許可する）
3. アプリを開いて **モデル取り込み** → `fereko-pack` フォルダを選ぶ（コピーとハッシュ確認で数分）
4. **読み込み** → **自己テスト**。`tokens: 28/28 ok` と `PC との一致 OK` が出れば正しく動いている
5. スマホが冷えた状態で **生成して再生** → **5回** → **10回連続**。充電中かどうかも覚えておく
6. **結果をコピー**（または共有）して、クロちゃんに貼る

このアプリにはネットとマイクの権限がない。結果はアプリの中（`runs.jsonl`）にだけ残る。

## 中継（Cloudflare Worker）

API キーは中継にだけ置く。PC もスマホも、中継の合言葉で中継を呼ぶ。

- 住所：`https://<Worker の名前>.<アカウントのサブドメイン>.workers.dev`（コードは `relay/`）
- `POST /v1/chat`（`{system, messages}` → Claude の返事をそのまま流す）、`GET /v1/usage`（今日の使用額）
- モデル（Claude Opus 5.5）、effort（`low`）、返事の長さの上限（2000 トークン）、断られたときの代わりのモデル（`fallbacks: "default"`）は中継が決める
- 1 日の上限は $3（日本時間で日が変わるとリセット）。`relay/wrangler.toml` の `DAILY_USD` で変える
- 使った額は KV `fereko-relay-usage` に記録する
- 合言葉は `%USERPROFILE%\.fereko\relay-token.txt`（あなたしか読めない設定）
- 無料プランの CPU 上限（1 回 10ms）に収めるため、中継の中では SDK を使わずに HTTP で呼んで、返事の流れをそのまま渡している

API キーの入れ方（キーは画面に出ない。チャットにも書かない）：

```powershell
cd relay
$env:PATH = "$env:LOCALAPPDATA\Programs\node-v24.21.0-win-x64;$env:PATH"
node C:\WebProjects\WatteHappy\deploy\node_modules\wrangler\bin\wrangler.js secret put ANTHROPIC_API_KEY
```

PowerShell から `|` で値を渡すと、末尾に改行が付いて合わなくなる。必ず上の聞き返しに貼り付けるか、`secret bulk` を使う。

ローカルで試す（本物の API にはつながない）：`relay/test/mock-anthropic.mjs` を 8799 番で起動して、`wrangler dev -c wrangler.test.toml --local --port 8787` を動かす。`FEREKO_RELAY_URL=http://127.0.0.1:8787` と `FEREKO_RELAY_TOKEN=local-test-token` を付けて会話プログラムを動かす。

## PC でふぇれ子と話す

```powershell
.venv\Scripts\python.exe -X utf8 tools\fereko_chat.py
```

- 中継経由で Claude を呼ぶ（`tools/relay_client.py`）。返事は文ごとに分けて、ONNX 版 Irodori で読み上げる
- 会話は `chat/history.json` に追記される。返事の中身（考えた部分の署名を含む）は届いたまま残す。`/reset` でやり直す（前の履歴は別名で残る）
- `/cost` で今日の使用額を見る

## 作り直し

```powershell
# スマホ用一式
.venv\Scripts\python.exe -I -X utf8 tools\pc_synth.py --pack models\irodori-mf-a7542a4\v4.1-small-mf --text "こんにちは。今日はいい天気ですね。" --out out\pc_trim_seed0.wav --pad trim --seed 0 --dump out\golden_trim_seed0
.venv\Scripts\python.exe -I -X utf8 tools\prep_phone_pack.py models\irodori-mf-a7542a4\v4.1-small-mf out\golden_trim_seed0 fereko-pack
```

```bash
# APK（Unity 6000.3.24f1 に入っている JDK 17・Android SDK 36・Gradle 9.1 を使う）
tools/gradle.sh assembleRelease
```

## ライセンス

モデル一式は `models/irodori-mf-a7542a4/LICENSES/` と `THIRD_PARTY_NOTICES.md` を参照。いちばん上のリポジトリは「other」。上流の README の書き方と食い違いがあるので、**人に配る前に整理が必要**（引き継ぎ資料のとおり）。この ONNX 版の音声には透かしが入らない。

## 自前の音声認識（0.9.0〜）

- モデル：ReazonSpeech k2 v2（`models/reazonspeech-k2-v2/`、rev 291488c、Apache-2.0）。int8 の encoder/decoder/joiner と `tokens.txt` を、スマホのモデル置き場（`files/pack`）に `tools/phone.sh push` で入れる
- アプリ側：`core/Fbank.java`（Kaldi 互換 80 次元 fbank、音は [-1, 1] のまま）＋ `core/ReazonAsr.java`（transducer の greedy search）。`LocalAsr` が共有する
- PC での確認：`tools/asr_ref.py <wav> <出力先> 1` で基準を作り、`tools/java/PcAsrCheck.java` で Java 版と比べる（トークン列が sherpa-onnx と完全に一致）
- マイク：`BargeInDetector` が、割り込み（BARGE_IN）と普通の聞き取り（LISTEN、🎤 / 会話モード）の両方で、話し終わり（0.9 秒の無音）まで録ってから認識する。割り込んだ言葉そのものが認識される
- チャット画面の **耳** ボタンで「自前 / Android」を切り替えられる
- スマホでの速さ（Nothing Phone (3)）：10〜12 秒の音声が 0.35〜0.4 秒、5 秒なら 0.18 秒。読み込み 2.6〜2.9 秒、メモリ（PSS）約 400〜480 MB

## 返事の速さ（0.9.1〜、中継 2026-10-09 版）

- 中継は `model`（`opus` / `sonnet` / `haiku`、省略で opus）を受け付ける。考える量はそれぞれいちばん少なくする：Opus 5.5 は止められないので effort low だけ、Sonnet 5.5 は `thinking: between_tools`、Haiku 5.5 は `thinking: disabled`。Opus 以外に送るときは、過去の返事の考えたブロックを中継で外す
- キャッシュは 1 時間（system と会話の末尾の両方）。`POST /v1/warm` は `max_tokens: 0` でキャッシュだけ作る。アプリはチャットを開いたときと「頭」を切り替えたときに呼ぶ
- アプリの **頭** ボタンで Opus → Sonnet → Haiku を切り替える
- `tools/latency_bench.py <history.json>` で、本物の中継を通して最初の文字までの時間を測る（お金がかかる）。2026-10-09、温めたあと：Opus 1.5〜2.0 秒、Sonnet 0.8 秒、Haiku 0.7〜1.0 秒。温める前の Opus は 2〜5 秒だった
- 0.9.4：文の終わりのあと、続きが 0.25 秒来なければその文をすぐ読み上げに回す（`SentenceSplitter.releaseEnded`）。Claude の文字はかたまりで届くので、次のかたまりを待つと 1 文目が 0.8 秒ほど遅れていた。遅れて届いた絵文字は捨てる。スマホで Opus：最初の文字 1.1 秒 → 声 2.0 秒（2026-10-09）

## 立ち絵（0.10〜、3D ができるまでの仮）

- `chat/fereko_stand.webp`（ユーザー作の全身絵、白い背景）をアプリに入れて、チャット画面の上に出す（`StandView`）。余白が広いので 1.45 倍にして、頭から胸あたりを見せる
- ゆっくり息をするように動き、しゃべっている間は声の大きさ（`StreamPlayer.level()`、いま聞こえている 20ms ごとの RMS）に合わせて上下する。口は動かない

## チャット画面のデザイン（0.11〜）

- 構成：ぼかした背景（部屋／外、`chat/bg_room.webp` `chat/bg_outside.webp`、Codex で作成）＋切り抜いた立ち絵（`chat/fereko_stand_cut.webp`、`tools/cut_stand.py` を ComfyUI の venv で実行）＋下の暗いガラスの帯に会話
- 参考にした他のアプリの見た目は真似しない（ユーザーの方針）。ふぇれ子の服に合わせた白黒と、目の赤（#E5484D）だけを差し色にする
- 会話（`ChatLog`）：吹き出しや名前札はなし。ふぇれ子の最新の返事は大きく太く、前の行は小さく薄く、キミの行は右寄せで小さく、「（割り込み）」などは中央に灰色。古い行は上に向かって消える
- 上の帯：名前、今の状態（待ってる／聞いてる／考え中／しゃべり中）、背景の切り替え、⋯（前のボタン類はここの設定に移した）
- 下：丸いマイク（`MicButton`、聞いている間は赤くなって脈打つ）と入力欄。話している間は「送る」が「止める」になる
- 準備の表示（0.11.2）：会話の下にあった状態の行はなくした。準備中は上の札が「支度中」（点が点滅）、声と耳が揃うと「待ってる」（点が明るい）、失敗したら「声が出ない」（札を押すと理由）。聞き取りのあとは「聞き取り中」。「聞き取れなかった」などは入力欄の上に 4 秒だけ出る（`hint`）。読み込み時間や前のターンの計測は ⋯ の設定のいちばん下
- テンポ（0.11.3）：`IrodoriEngine.speed` で予測の長さを割る（音の高さは変わらない）。設定の「テンポ」で 1.0 / 1.1 / 1.15 / 1.2 / 1.3、標準は 1.2（0.11.4、ユーザーが聞き比べて決めた）。スマホで 1.0 / 1.15 / 1.3 を作って自前の認識にかけ、3 つとも一字一句同じだった（`synth:<dec>:<dit>:<speed>` の自動実行）

## 記憶（0.12〜）

- 会話が 48 件を超えたら、古いほう（最後の約 24 件より前）を Sonnet で「覚えていること」のメモにまとめ（`core/ChatMemory`、約 7 秒・$0.02、裏で動く）、送る会話から外す。外した会話は `files/chat/history-archive.jsonl` に残る
- メモ（`files/chat/memory.md`、最大 2500 文字）と今日の日付は、キャラ設定のあとに付けて送る。見出しは「キミのこと／二人の話題・お約束／最近の出来事」
- ⋯ の設定の「記憶」で中身を見られる。「忘れさせる」で消せる。「最初から」ではメモは消えない
- PC での確認：`tools/java/PcMemoryCheck.java <history.json>`（本物の中継を通すのでお金がかかる）

## アイコンと起動（0.12.1〜）

- アプリ名は「ふぇれ子」。アイコンは切り抜いた立ち絵の顔（`tools/make_icon.py` を ComfyUI の venv で実行 → `res/mipmap-xxxhdpi/ic_launcher_fg.png` / `ic_launcher_bg.png`、`res/mipmap-anydpi-v26/ic_launcher.xml` のアダプティブアイコン）。背景は濃い灰色に目の色の赤をうっすら
- アイコンから開くのはチャット画面。前の最初の画面（モデル取り込み、自己テスト、速さの試験）は ⋯ の設定の「試験画面」から。試験画面の「会話へ」は下のチャットに戻る
- adb で開くとき：`am start -n local.fereko.probe/.ChatActivity`（もう外から開ける）

## 3D 表示（0.13〜）

- `web/avatar.html` と `web/avatar.mjs`：three.js r170 と @pixiv/three-vrm 3.5.5（と three-vrm-animation）で VRM を表示する。ライブラリは charakuru キットに入っていたもの（MIT、`web/vendor/*LICENSE.txt`）をコピーして APK に入れた
- `AvatarView`：透明な WebView。https://fereko.local/web/… は APK の assets から、/pack/… はモデル置き場から、アプリ自身が返す。それ以外の通信は全部拒否する
- VRM は APK にもリポジトリにも入れない（作者専用、再配布禁止の設定）。`tools/phone.sh push ferreko-*.vrm` でモデル置き場に入れる。一番新しい `ferreko*.vrm` を表示する。設定の「姿:3D/2D」で切り替える
- 画角は耳の先（モデルの一番上）から膝（leftLowerLeg の骨）まで。描画は 60fps が上限
- 記録：読み込み時に `avatar_load`（版、三角形の数、表情の名前、使える表情、足りない表情、揺れものの数、演技用の骨、GPU）、10 秒ごとに `avatar_perf`（fps、p50/p99 のフレーム時間、しゃべっていた割合）を runs.jsonl に書く
- 表情と VRMA ができるまでは動きを手で付けている：腕を下ろす、呼吸、揺れ、しゃべるとき頭が動く、聞くとき首をかしげる、考えるとき上を見る
- PC で試す：`.claude/launch.json` の fereko-avatar（web とモデルをまとめたフォルダーを http.server で配る）
- 0.14.0：VRMA の全身の反応を再生する（今は `turn_around` だけ）。`.vrma` をモデル置き場に入れると読み込む（`Fereko.anims()`）。three.js の AnimationMixer は使わず、各トラックを自分で補間してボーンに書く（Mixer だと turn_around の 180 度のところが正面に戻ってしまった）。前後 0.3 秒で手続きの姿勢から切り替え、二重に呼んでも最初からやり直さない。ふぇれ子を **タップすると振り向く**。後ろ向きの間は会話の帯を薄くして、しっぽを見せる。adb：`am start -n local.fereko.probe/.ChatAuto --es motion turn_around`
- PC で試すとき：`avatar.html?vrm=/pack/….vrm&anims=turn_around.vrma`。タブが隠れていると requestAnimationFrame が止まるので、`window.avatarStep(秒)` で時間を進め、`window.avatarDebug()` と `window.avatarSample(clip, track, t)` で確認する
- 0.15.0：声の仕上げ（デコーダー）の標準を `gpu16@36`（36 フレームずつ）にした。しゃべっている間の 3D の止まりが 0.7 秒から 0.35 秒になる（RTF は 0.6 くらい）
- 0.15.0：アプリを開いたときのあいさつ。前の会話から 30 分以上たっていたら、声の準備ができたところで「（アプリが開かれた。今は夜の…前に話してから…）」というお知らせを Claude に送り、ふぇれ子が一言話す。お知らせは画面には出さない（`ChatMemory.EVENT_OPEN`）。設定の「あいさつ」でオフにできる。1 回 $0.003 くらい

## 好感度（0.16〜）

- 0〜100、5 段階（他人行儀／知り合い／友だち／気になる／特別）。0 から始める（ユーザーの希望）
- 段階ごとの態度と駆け引きのルール（告白は低い段階で成立させない、繰り返すと冷める）は `chat/persona.md`
- Claude は返事の最後に「［好感度:+1］」を付ける。アプリはそれを読み上げにも表示にも出さず（`core/Affection.TagFilter`）、1 回 −5〜+3、1 日に上がるのは最大 +10 に抑えて保存する
- 今の値は「〔好感度 12/100・他人行儀〕」としてキミの発言の後ろに付けて送る（キャラ設定に入れるとキャッシュが毎回作り直しになるため）。画面と記憶のメモには出さない
- 変化したときだけ入力欄の上に「♡ +2」と出る。段階が変わったら段階名も。設定の「好感度」で数値を見られる、0 に戻せる
- PC 版（fereko_chat.py）はタグを取り除かないので、使うなら対応が要る

## キャラの切り替え（0.17〜）

- 設定の一番上「キャラ:…」で話す相手を選ぶ。ふぇれ子は組み込み（データは今まで通り files/chat/、声と VRM はモデル置き場）
- ほかのキャラは files/chars/<id>/ のフォルダ：`character.json`（name、caption＝声の説明、avatar＝vrm / image / live2d）、`persona.md`、あれば `speaker.f32` + `speaker.json`（登録した声）、`*.vrm` / `*.vrma`、`stand.png|webp`。見本は `chars/template`、スマホへは `tools/phone.sh pushchar C:/…/chars/<id>`
- キャラごと：会話の記録、記憶のメモ、記録の保管、好感度、最後に話した時刻、声の説明、登録した声を使うか。共通：頭、耳、デコーダー、テンポ、背景、会話モード、割り込み、あいさつ
- Live2D はまだ表示できない（avatar を live2d にすると立ち絵を出す）。あとで表示ページを足す

## 3D の動きと表情（0.18〜、motion-14）

- モデル：`ferreko-cloth-3.vrm`（cloth-2-secondary-2）、25 本の VRMA、`motion-manifest.json` をモデル置き場に入れる（GPT の `app-delivery/motion-14`）
- 土台：状態に合わせて idle / listen / think / talk（talk_2・talk_3 からランダム）を 0.3 秒でつなぐ。think は think_enter → think → think_exit
- 反応：上半身だけ入れ替え（入り 0.2 秒、抜け 0.3 秒、重みの合計は 1）。turn_around は全身。割り込まれたら反応を 0.2 秒で抜いて startle（振り向き中は終わってから）
- 顔：絵文字 → 感情（<感情> 本体、_lids × (1−まばたき)、_mouth はしゃべる間 0.3）。自然なまばたき＋giggle_hide・yawn の時刻つきの閉眼。口は声の大きさと文のかなから あいうえお。考え中は目線を上・横へ。😪🤭🙄 と びっくりは、それだけで sleepy_nod / giggle_hide / sigh_drop / surprised_recoil が付く
- 動きのタグ：Claude が文の頭に「［うなずく］」などを付ける（persona.md の「動き」）。アプリは読み上げにも表示にも出さず、その文が聞こえ始めたときに再生する（`TtsPipeline.currentSentence()`）。あいさつの最初の文には「手を振る」
- 試す：`am start -n local.fereko.probe/.ChatAuto --es say '［うなずく］まあね。…😏'` は Claude を使わずに同じ流れで話す
- 0.18.2：どの動きの上にも「呼吸と左右のわずかな揺れ」を重ねる（`applyLife`）。胸と背中の呼吸（3.6 秒）、肩の上下、上半身の左右の揺れ（7 秒）、頭のゆっくりした揺れ（9.3 秒と 5.7 秒）。idle は元から呼吸があるので 0.35 倍、振り向き中は 0.4 倍、ほかは 1 倍。脚と腰は動かさない

## SNS 用のデモ動画（0.18.3〜）

- `tools/demo_record.sh <秒> "セリフ|セリフ|…" [out.mp4]`：会話をリセットして（前の会話は別名で残る）、あいさつ→セリフを順に送る（本物の Claude）。画面を screenrecord で録り（音は入らない）、アプリが保存したふぇれ子の声（`--ez rec true` → files/rec/turn_<時刻>.f32）を PC で重ねる
- `tools/mux_demo.py`：切り出し（FROM/TO 秒）、852×1894・30fps の H.264＋AAC に変換（X の縦長の上限に収まる）。screenrecord は表示した時刻より約 0.45 秒遅れて録り始めるので、声をその分早める（LAG_MS）
- テスト用の入口から開いたときは、あいさつは自動では出ない。デモでは `--ez greet true` で出す。デモ中は会話モードの聞き取りを止める
