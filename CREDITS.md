# 参考にしたもの・使っているもの

このアプリは、次の公開されている成果の上に作っています。作者のみなさんに感謝します。
モデルの重みはこのリポジトリに入っていません。使うときは、それぞれの配布元のライセンスと利用条件に従ってください。

## 声の合成：Irodori-TTS

- **Irodori-TTS**（Chihiro Arata / Aratako さん）
  - 実装：https://github.com/Aratako/Irodori-TTS（MIT）。MeanFlow の推論手順は、コミット `89f9d8fbd4d51ea019867ee1197725ede1df13c5` を基準にしています（`tools/pc_synth.py`）
  - 重み：https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF（MIT）
- **Onsei 用の ONNX 変換版**：`raratu/Onsei-Irodori-v4.1-Small-MF`（Hugging Face、リリース `ios-mf-v2` = コミット `a7542a4`）
  - スマホでの実行（`IrodoriEngine`）は、この変換版の ONNX ファイル一式と実行の手順をそのまま手本にしています。具体的には、文字と声の説明の符号化、話者、長さの予測、文脈 KV、4 ステップの MeanFlow、DAC-VAE デコード、末尾の切り取りです
  - GPU 用のデコーダー（`tools/make_dec_tflite.sh`）は、この変換版の `dacvae_decode.onnx` を、固定の窓の大きさで LiteRT 用に変換したものです
- 中に含まれる部品のライセンス（変換版の `THIRD_PARTY_NOTICES.md` による）
  - Semantic-DACVAE-Japanese-32dim（Aratako さん、MIT）
  - DACVAE（Meta、Apache-2.0）
  - Descript Audio Codec（MIT）
  - ModernBERT-ja-310m（SB Intuitions、MIT）
  - llm-jp-3-150m のトークナイザー（Apache-2.0）
- **利用上の注意**（上流のモデルカードより）
  - 実在の人の声をまねるときは、本人のはっきりした同意が必要です
  - 人をだます、なりすます、ディープフェイクに使うことは禁止されています
  - 参照なしで作った声が、たまたま実在の人に似ることがあります
  - 法的な責任は使う人にあります
  - この ONNX 版の出力には電子透かしが入りません

## 音声認識：ReazonSpeech

- **ReazonSpeech k2 v2**（Reazon Holdings）：https://huggingface.co/reazon-research/reazonspeech-k2-v2（Apache-2.0）
- 特徴量の計算（`core/Fbank`）と transducer の greedy search（`core/ReazonAsr`）は、**sherpa-onnx** と **kaldi-native-fbank**（k2-fsa、Apache-2.0）の処理に合わせて書き、出力が一致することを確かめています（`tools/asr_ref.py`）

## 同梱しているもの

- `web/vendor/three.module.js`、`GLTFLoader.js`、`BufferGeometryUtils.js`：three.js（MIT、`THREE-LICENSE.txt`）
- `web/vendor/three-vrm.module.js`、`three-vrm-animation.module.js`：pixiv Inc.（MIT、同じフォルダーのライセンスファイル）
- それ以外のコードは、このリポジトリの `LICENSE`（MIT）

## 実行と変換に使っているもの

| 名前 | 用途 | ライセンス |
|---|---|---|
| ONNX Runtime（Microsoft） | Android と PC での推論 | MIT |
| LiteRT（Google、旧 TensorFlow Lite）と GPU delegate | スマホの GPU でのデコード | Apache-2.0 |
| onnx2tf（PINTO0309） | ONNX から LiteRT への変換 | MIT |
| three.js | 3D の表示（`web/vendor`、同梱） | MIT |
| three-vrm と three-vrm-animation（pixiv） | VRM と VRMA の読み込み（`web/vendor`、同梱） | MIT |
| Claude（Anthropic） | 会話。Cloudflare Workers の中継を通して Messages API を呼びます | API の利用規約による |

## 素材

- 3D モデル、モーション、キャラクターの絵、キャラ設定は、このリポジトリに入っていません（作者のものです）
