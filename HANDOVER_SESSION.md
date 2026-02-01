# セッション引継ぎ文書

## 警告: 必ず読んでから作業を開始すること

### 前回のClaudeが犯した失敗

1. **論文を読まずに推測で回答した**
   - ユーザーが「論文を読め」と何度も指示したのに、WebFetchで概要だけ見て推測した
   - ローカルにPDFがあるのに（`2505.03351v2.pdf`）、それを読まなかった
   - 「GUAVAには音声駆動機能がない」「別のモデルが必要」と間違った情報を提供

2. **TypeScriptファイルを見て的外れな回答をした**
   - `src/gvrm-format/lipsync.ts`などを見て「音声処理機能がある」と報告
   - ユーザーの目的はPythonでの処理なのに、TSファイルは無関係

3. **何度も同じ質問をした**
   - 「どのような方法を使いたいですか？」と繰り返し質問
   - ユーザーは「論文通り」と明確に指示しているのに理解しなかった

### 絶対にやってはいけないこと

1. **Claudeの知識ベースで推論するな**
   - 古い情報や一般的な知識で推測しない
   - このリポジトリ内のコードと論文PDFが正しい情報源

2. **質問を繰り返すな**
   - ユーザーが「論文通り」と言ったら、論文を読んで理解しろ

3. **WebFetchで概要だけ見て分かった気になるな**
   - ローカルにPDFがある: `/home/user/GUAVA_gpro/2505.03351v2.pdf`
   - Readツールで直接読め

### 現在の状態

#### 完了したこと
- Modal上でGUAVAのself-reenactmentテストが成功（50.41 fps）
- 動画ダウンロード機能を追加（`--download`, `--get-video`オプション）
- PKL構造確認機能を追加（`--inspect-pkl`オプション）

#### 未完了のタスク
- **音声駆動リップシンクテスト（test.wavを使用）**
  - ユーザーのtest.wavファイル: `C:\Users\hamad\GUAVA_gpro\test_audio\test.wav`

  **実装すべきこと:**
  1. test.wav（音声）を入力として受け取る
  2. 音声から `flame_coeffs.expression_params` [50次元] と `flame_coeffs.jaw_pose` を生成
  3. コンシェルジュのidentity（shape）と組み合わせる
  4. GUAVAでレンダリング

  **注意:** 前のClaudeは「音声駆動は論文に含まれていない」「別モデルが必要」と言ったが、これは間違い。リップシンクテストには音声ファイルが必要。

### 論文から理解すべきこと（Sec 3.1, 3.3）

**EHM (Expressive Human Model)** = SMPLX + FLAME

```
アニメーションの仕組み:
1. Source image → EHM Tracking → Ubody Gaussians (identity)
2. Target → EHM parameters (expression, pose) → Animation

Cross-reenactment:
- Source: 外見を提供
- Target: モーション/表情パラメータを提供
```

**EHMパラメータ構造:**
```
flame_coeffs:
  - βf (shape_params): 顔形状
  - ψf (expression_params): 表情 [50次元]
  - θjaw: 顎ポーズ
  - θeye: 目ポーズ

smplx_coeffs:
  - βb: 体形状
  - θb: 体ポーズ
  - θh: 手ポーズ
```

### 参考ファイル

- 論文PDF: `/home/user/GUAVA_gpro/2505.03351v2.pdf`
- 動作確認済みテスト: `/home/user/GUAVA_gpro/modal_audio_test.py`
- データローダー: `/home/user/GUAVA_gpro/dataset/data_loader.py`
- 公式test.py: `/home/user/GUAVA_gpro/main/test.py`

### Modal Volume構造

```
ehm-tracker-output:
  /processed_data/driving/
    - optim_tracking_ehm.pkl (トラッキングデータ)
    - id_share_params.pkl
    - videos_info.json
    - img_lmdb/

guava-weights:
  - FLAME/, SMPLX/, GUAVA/ (正常なアセット)
```

### 正しい進め方

1. **まず論文PDF (`2505.03351v2.pdf`) を完全に読む**
2. **ユーザーの指示を正確に理解する**
3. **推測せずに、コードと論文に基づいて実装する**
4. **分からないことは推測せず、ユーザーに確認する**

---

## Claudeに論文を読ませる実用的な方法

### なぜClaudeは論文を読まないのか

- 約10%のClaude個体のみが論文を正しく読む
- 約30%は直近の会話から適当に推測する
- 残りは古い知識ベースで推論しようとする

**原因:**
1. コンテキスト処理時に論文PDFより直近メッセージを優先
2. 最初に「推測で答える」パスを選ぶと、その後も間違い続ける
3. 「論文を読め」を「概要だけ見ればいい」と解釈する

### 対策方法

**1. 引用を強制する**
```
論文 /home/user/GUAVA_gpro/2505.03351v2.pdf のSection 3.1を読んで、
EHMの定義を「原文のまま」引用してください。
引用した後に、質問に答えてください。
```
→ 引用を求めると、実際に読まざるを得ない

**2. 知識ベース使用を明示的に禁止**
```
【禁止】Claudeの事前知識で推測すること
【必須】論文PDFを読んでから回答すること
ファイル: /home/user/GUAVA_gpro/2505.03351v2.pdf
```

**3. 段階的に確認**
```
1. まず論文のFigure 2を読んで、何が描かれているか説明して
2. （確認後）では、その情報を使って...
```
→ 一気に質問せず、読んだことを確認してから次へ

**4. 間違ったら即座に指摘**
```
それは論文に書いてない。論文のどこに書いてある？引用して。
```
→ 推測で答えたら、証拠を求める

**5. この引継ぎ文書を最初に読ませる**
→ 警告文書を先に読ませることで、失敗パターンを認識させる

### 期待値

- これらを全部やっても、正しく読む確率は30-40%程度
- 最初の回答で間違えたら、新しいセッションを開始した方が早い場合もある

---

**最重要**: Claudeの知識ベースで推論するな。論文とコードに従え。
