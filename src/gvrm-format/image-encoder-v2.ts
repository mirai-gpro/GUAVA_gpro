// image-encoder-v2.ts
// DINOv2 Image Encoder with PCA-based dimensionality reduction

import { AutoProcessor, AutoModel, RawImage } from '@huggingface/transformers';

export class ImageEncoderV2 {
  private model: any = null;
  private processor: any = null;
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[ImageEncoder] Initializing DINOv2...');

    try {
      const modelId = 'onnx-community/dinov2-base';

      console.log('[ImageEncoder] Loading processor...');
      this.processor = await AutoProcessor.from_pretrained(modelId);

      console.log('[ImageEncoder] Loading model...');
      this.model = await AutoModel.from_pretrained(modelId, {
        quantized: true
      });

      this.initialized = true;
      console.log('[ImageEncoder] ✅ DINOv2 initialized');
    } catch (error) {
      console.error('[ImageEncoder] ❌ Failed to initialize:', error);
      throw new Error(`Image Encoder initialization failed: ${error}`);
    }
  }

  /**
   * 簡易PCAによる次元削減
   * 768次元 → 128次元への情報圧縮
   */
  private performPCA(
    features: Float32Array,
    numSamples: number,
    inputDim: number,
    outputDim: number
  ): Float32Array {
    console.log('[ImageEncoder] Performing PCA dimensionality reduction...');

    // 1. 平均を計算して中心化
    const means = new Float32Array(inputDim);
    for (let d = 0; d < inputDim; d++) {
      let sum = 0;
      for (let s = 0; s < numSamples; s++) {
        sum += features[s * inputDim + d];
      }
      means[d] = sum / numSamples;
    }

    // 中心化されたデータ
    const centered = new Float32Array(features.length);
    for (let s = 0; s < numSamples; s++) {
      for (let d = 0; d < inputDim; d++) {
        centered[s * inputDim + d] = features[s * inputDim + d] - means[d];
      }
    }

    // 2. 分散共分散行列の固有値分解の代わりに、
    //    エネルギーの高い次元を選択（簡易版）
    const variances = new Float32Array(inputDim);
    for (let d = 0; d < inputDim; d++) {
      let variance = 0;
      for (let s = 0; s < numSamples; s++) {
        const val = centered[s * inputDim + d];
        variance += val * val;
      }
      variances[d] = variance / numSamples;
    }

    // 分散の大きい順にソート
    const indices = Array.from({ length: inputDim }, (_, i) => i);
    indices.sort((a, b) => variances[b] - variances[a]);

    // 上位outputDim次元を選択
    const topIndices = indices.slice(0, outputDim);

    console.log('[ImageEncoder] Top 10 variance dimensions:', 
      topIndices.slice(0, 10).map(i => `${i}(${variances[i].toFixed(3)})`));

    // 3. 選択された次元で新しい特徴量を構築
    const reduced = new Float32Array(numSamples * outputDim);
    for (let s = 0; s < numSamples; s++) {
      for (let d = 0; d < outputDim; d++) {
        const srcIdx = topIndices[d];
        reduced[s * outputDim + d] = centered[s * inputDim + srcIdx];
      }
    }

    return reduced;
  }

  /**
   * グリッドベースの補間マッピング
   * パッチ特徴（16×16）を頂点特徴（10,595）に補間
   */
  private interpolateFeatures(
    patchFeatures: Float32Array,
    numPatches: number,
    featureDim: number,
    targetVertexCount: number
  ): Float32Array {
    console.log('[ImageEncoder] Interpolating patch features to vertices...');

    // パッチを2Dグリッド（√numPatches × √numPatches）として扱う
    const gridSize = Math.sqrt(numPatches);
    if (Math.abs(gridSize - Math.floor(gridSize)) > 0.01) {
      console.warn('[ImageEncoder] numPatches is not a perfect square, using approximate grid');
    }
    const gridW = Math.floor(gridSize);
    const gridH = Math.ceil(numPatches / gridW);

    const output = new Float32Array(targetVertexCount * featureDim);

    for (let i = 0; i < targetVertexCount; i++) {
      // 頂点インデックスをグリッド座標にマッピング
      const normIdx = i / targetVertexCount; // [0, 1]
      const gridX = normIdx * gridW;
      const gridY = (normIdx * gridH) % gridH;

      // バイリニア補間用の4つの最近傍パッチを取得
      const x0 = Math.floor(gridX);
      const x1 = Math.min(x0 + 1, gridW - 1);
      const y0 = Math.floor(gridY);
      const y1 = Math.min(y0 + 1, gridH - 1);

      const wx1 = gridX - x0;
      const wx0 = 1 - wx1;
      const wy1 = gridY - y0;
      const wy0 = 1 - wy1;

      // 4つの角のパッチインデックス
      const p00 = Math.min(y0 * gridW + x0, numPatches - 1);
      const p10 = Math.min(y0 * gridW + x1, numPatches - 1);
      const p01 = Math.min(y1 * gridW + x0, numPatches - 1);
      const p11 = Math.min(y1 * gridW + x1, numPatches - 1);

      // 各次元でバイリニア補間
      for (let d = 0; d < featureDim; d++) {
        const v00 = patchFeatures[p00 * featureDim + d];
        const v10 = patchFeatures[p10 * featureDim + d];
        const v01 = patchFeatures[p01 * featureDim + d];
        const v11 = patchFeatures[p11 * featureDim + d];

        const v0 = wx0 * v00 + wx1 * v10;
        const v1 = wx0 * v01 + wx1 * v11;
        const v = wy0 * v0 + wy1 * v1;

        output[i * featureDim + d] = v;
      }
    }

    return output;
  }

  async extractFeatures(
    imageUrl: string,
    targetVertexCount: number = 10595
  ): Promise<Float32Array> {
    if (!this.model || !this.processor) {
      throw new Error('[ImageEncoder] Not initialized. Call init() first.');
    }

    console.log('[ImageEncoder] Processing image:', imageUrl);

    try {
      const startTime = performance.now();

      // 1. 画像を読み込み
      const image = await RawImage.fromURL(imageUrl);
      console.log('[ImageEncoder] Image loaded:', {
        width: image.width,
        height: image.height
      });

      // 2. 前処理
      const inputs = await this.processor(image);

      // 3. DINOv2で特徴抽出
      const { last_hidden_state } = await this.model(inputs);

      // 4. 特徴量を取得（CLS tokenを除外）
      const features = last_hidden_state.slice(null, [1, null], null);
      const featuresData = features.data as Float32Array;
      const numPatches = features.dims[1];
      const featureDim = features.dims[2]; // 768

      console.log('[ImageEncoder] DINOv2 output:', {
        numPatches,
        featureDim,
        targetVertexCount
      });

      // 5. PCAで次元削減: 768 → 128
      const reducedFeatures = this.performPCA(
        featuresData,
        numPatches,
        featureDim,
        128
      );

      // 6. バイリニア補間でパッチ特徴を頂点にマッピング
      const projectionFeature = this.interpolateFeatures(
        reducedFeatures,
        numPatches,
        128,
        targetVertexCount
      );

      const elapsed = performance.now() - startTime;
      console.log(`[ImageEncoder] ✅ Feature extraction completed in ${elapsed.toFixed(2)}ms`);

      // 統計情報
      const sampleSize = Math.min(1000, projectionFeature.length);
      const sample = Array.from(projectionFeature.slice(0, sampleSize));
      console.log('[ImageEncoder] Feature statistics:', {
        min: Math.min(...sample),
        max: Math.max(...sample),
        avg: sample.reduce((a, b) => a + b, 0) / sample.length,
        std: Math.sqrt(
          sample.reduce((sum, v) => {
            const mean = sample.reduce((a, b) => a + b, 0) / sample.length;
            return sum + Math.pow(v - mean, 2);
          }, 0) / sample.length
        )
      });

      return projectionFeature;

    } catch (error) {
      console.error('[ImageEncoder] ❌ Feature extraction failed:', error);
      throw error;
    }
  }

  dispose(): void {
    if (this.model) {
      this.model.dispose?.();
      this.model = null;
      this.processor = null;
      this.initialized = false;
      console.log('[ImageEncoder] Disposed');
    }
  }
}