# セッション引継ぎ文書

## 警告: 必ず読んでから作業を開始すること

---

## プロジェクト最終目的

### 1. 最終ゴール
- **グルメサポートAI**アプリ（α版テスト中）のコンシェルジュモード
- 実写女性コンシェルジュがユーザーの問いかけに応答
- フロー: STT → LLM → TTS
- TTSの回答に**リップシンク**と**顔の表情**、**頭**、**上半身**をシンクロして動かす
- **GUAVA論文以上のクオリティ**を目指す
- **スマホ単体での動作**が必須

### 2. 現時点でのテスト目標
- 論文公式GitHub提供の`test.py`を動かして検証
- **元画像、参考動画**: コンシェルジュ用に既に論文に従って準備済み
- **学習済みデータ**: onnx, bin, PLYファイルは生成済み
- **テスト音声**: `C:\Users\hamad\GUAVA_gpro\test_audio\test.wav`

---

### 前回のClaudeが犯した失敗

1. **動作確認済みの `generate_ply_modal.py` があるのに、自分の知識でアレンジした**
   - ユーザーは「忠実にコピーして」と明確に指示した
   - Claudeは「pickleファイル破損を回避する賢い方法がある」と思い込んだ
   - OmegaConf/ConfigDictの動作を推測で修正し続けた
   - 結果: 10回以上の的外れな修正で状況を悪化させた

2. **ユーザーの警告を無視した**
   - 「Claudeの知識ベースだけでは絶対に解決できない！100回やっても無理！」
   - 「ちゃんと見てないでしょ？折角提示したpyを」
   - これらの警告を受けても、自分の推論を優先し続けた

3. **エラーの原因を誤診断し続けた**
   - 「Windowsが同期されていない」と何度も主張
   - 「Modalのキャッシュが原因」と主張
   - 実際は自分の修正コードが間違っていた

### 絶対にやってはいけないこと

1. **自分の知識でコードを「改良」しない**
   - OmegaConf, ConfigDict, pickle, CUDAなどの問題を推測で解決しようとしない
   - 動作する例があるなら、それをそのままコピーする

2. **エラーが出たら「環境のせい」にしない**
   - git sync, Modal cache などを疑う前に、自分のコードを疑う

3. **ユーザーの指示を自分の判断で変更しない**
   - 「忠実にコピー」と言われたら、本当に忠実にコピーする

### 現在の状態

- ブランチ: `claude/test-guava-implementation-iOiRv`
- 問題のファイル: `modal_audio_test.py`
- 動作確認済み参考ファイル: `generate_ply_modal.py`

### 正しい進め方

1. **まず `generate_ply_modal.py` を完全に読む**
   - Image定義、Volume設定、関数内のコード全て

2. **`modal_audio_test.py` を `generate_ply_modal.py` ベースで書き直す**
   - Image定義: 完全にコピー
   - Volume設定: 同じパターン
   - ConfigDict/OmegaConf操作: `generate_ply_modal.py` と同じ方法のみ使う

3. **音声処理の追加部分のみ、慎重に追加**
   - librosa, soundfile の依存関係追加
   - audio_to_flame_params 関数

4. **分からないことは推測せず、ユーザーに聞く**

### pickleファイル破損問題について

- `assets/SMPLX/SMPLX_to_J14.pkl` と `assets/FLAME/FLAME_masks/FLAME_masks.pkl` がASCII textになっている（本来はbinary）
- ユーザーが公式から再アップロードしたが、まだ破損している可能性あり
- **Claudeの知識で回避策を考えない** - ユーザーに確認する

### 参考: generate_ply_modal.py のVolume設定

```python
ehm_volume = modal.Volume.from_name("ehm-tracker-output", create_if_missing=True)
ply_output_volume = modal.Volume.from_name("guava-ply-output", create_if_missing=True)

@app.function(
    image=image,
    gpu="a10g",
    timeout=3600,
    volumes={
        "/root/EHM_results": ehm_volume,
        "/root/GUAVA/ply_outputs": ply_output_volume
    },
    env={"MEDIAPIPE_DISABLE_GPU": "1"}
)
```

### 参考: generate_ply_modal.py のConfig操作

```python
# この方法で動作している
OmegaConf.set_readonly(meta_cfg['DATASET'], False)
meta_cfg['DATASET']['data_path'] = data_path
```

---

**最重要**: Claudeの知識ベースで推論するな。動作する例に従え。
