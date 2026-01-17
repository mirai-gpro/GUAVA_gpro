
// neural-refiner.ts
// Web / iOS / onnxruntime-web 安定版 Neural Refiner (修正版)

import * as ort from 'onnxruntime-web';

export class NeuralRefiner {
  private session: ort.InferenceSession | null = null;
  private initialized = false;

  private readonly MODEL_PATH = '/assets/refiner_512_websafe.onnx';
  private readonly FM_CHANNELS = 32;
  private readonly FM_SIZE = 512;

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[NeuralRefiner] Initializing...');

    try {
      // Template Decoderと同じ設定
      ort.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm'
      };
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;

      console.log('[NeuralRefiner] ONNX Runtime v1.17.3 configured');

      // セッション生成
      this.session = await ort.InferenceSession.create(
        this.MODEL_PATH,
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        }
      );

      this.initialized = true;
      console.log('[NeuralRefiner] ✅ Model loaded successfully');
    } catch (error) {
      console.error('[NeuralRefiner] ❌ Failed to load model:', error);
      console.error('[NeuralRefiner] Error details:', {
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        errorType: typeof error,
        errorCode: error
      });
      throw new Error(`Neural Refiner initialization failed: ${error}`);
    }
  }

  async run(
    coarseFM: Float32Array,
    idEmb: Float32Array
  ): Promise<Float32Array> {
    if (!this.session) {
      throw new Error('NeuralRefiner not initialized');
    }

    // 入力サイズ検証
    const expectedFMSize = this.FM_CHANNELS * this.FM_SIZE * this.FM_SIZE;

    if (coarseFM.length !== expectedFMSize) {
      throw new Error(
        `Invalid coarseFM size: ${coarseFM.length}, expected: ${expectedFMSize}`
      );
    }

    if (idEmb.length !== 256) {
      throw new Error(
        `Invalid idEmb size: ${idEmb.length}, expected: 256`
      );
    }

    // Tensor作成（512×512 モデル用）
    const fmTensor = new ort.Tensor('float32', coarseFM, [1, 32, this.FM_SIZE, this.FM_SIZE]);
    const idTensor = new ort.Tensor('float32', idEmb, [1, 256]);

    // 推論実行
    try {
      const outputs = await this.session.run({
        coarse_fm: fmTensor,
        id_emb: idTensor
      });

      const rawOutput = outputs.refined_rgb.data as Float32Array;
      const dims = outputs.refined_rgb.dims;

      console.log('[NeuralRefiner] Raw output info:', {
        length: rawOutput.length,
        dims: dims,
        type: outputs.refined_rgb.type
      });

      // 出力形状確認
      const expectedLength = 512 * 512 * 3;
      if (rawOutput.length !== expectedLength) {
        console.error('[NeuralRefiner] Unexpected output length!', {
          actual: rawOutput.length,
          expected: expectedLength,
          dims: dims
        });
      }

      // CHW → HWC 変換
      let out: Float32Array;
      const H = 512, W = 512, C = 3;

      if (dims.length === 4 && dims[1] === 3) {
        console.log('[NeuralRefiner] Converting CHW to HWC...');
        out = new Float32Array(H * W * C);
        
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            for (let c = 0; c < C; c++) {
              const srcIdx = c * H * W + h * W + w;
              const dstIdx = h * W * C + w * C + c;
              out[dstIdx] = rawOutput[srcIdx];
            }
          }
        }
      } else {
        out = rawOutput;
      }

      // 統計情報
      let min = Infinity, max = -Infinity, nonZero = 0, nanCount = 0, infCount = 0;

      for (let i = 0; i < out.length; i++) {
        const v = out[i];
        if (isNaN(v)) { nanCount++; continue; }
        if (!isFinite(v)) { infCount++; continue; }
        if (v !== 0) nonZero++;
        if (v < min) min = v;
        if (v > max) max = v;
      }

      console.log('[NeuralRefiner] Output stats:', {
        min, max, nonZeroCount: nonZero, nanCount, infCount,
        sampleValues: Array.from(out.slice(0, 30)).map(v => v.toFixed(4))
      });

      // NaN/Inf除去
      if (nanCount > 0 || infCount > 0) {
        console.warn('[NeuralRefiner] Cleaning invalid values...');
        for (let i = 0; i < out.length; i++) {
          if (!isFinite(out[i])) out[i] = 0;
        }
      }

      // 正規化: [-1, 1] → [0, 1] の固定変換（min-max ではなく）
      // Neural Refiner の出力は通常 [-1, 1] 付近を想定
      // 少し範囲外の値はクリップする
      console.log('[NeuralRefiner] Converting output to [0, 1] with fixed mapping...');
      for (let i = 0; i < out.length; i++) {
        // [-1, 1] → [0, 1]
        let normalized = (out[i] + 1) / 2;
        // クリップ
        out[i] = Math.max(0, Math.min(1, normalized));
      }

      // 変換後の統計
      let finalMin = Infinity, finalMax = -Infinity;
      for (let i = 0; i < 10000; i++) {
        if (out[i] < finalMin) finalMin = out[i];
        if (out[i] > finalMax) finalMax = out[i];
      }
      console.log('[NeuralRefiner] Final output range:', {
        min: finalMin.toFixed(4),
        max: finalMax.toFixed(4)
      });

      console.log('[NeuralRefiner] ✅ Normalization complete');
      return out;

    } catch (runError) {
      console.error('[NeuralRefiner] ❌ Inference failed:', runError);
      throw runError;
    }
  }

  async process(
    coarseFM: Float32Array,
    idEmb: Float32Array
  ): Promise<Float32Array> {
    return this.run(coarseFM, idEmb);
  }

  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
      this.initialized = false;
      console.log('[NeuralRefiner] Session disposed');
    }
  }
}
