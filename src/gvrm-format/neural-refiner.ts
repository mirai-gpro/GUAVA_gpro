// neural-refiner.ts
// Neural Refiner - Coarse Feature Map (32ch) → RGB (3ch)
// 論文準拠: StyleUNet/SimpleUNet based refiner

import * as ort from 'onnxruntime-web';

export class NeuralRefiner {
  private session: ort.InferenceSession | null = null;
  private initialized = false;
  private inputName: string = '';
  private outputName: string = '';

  private readonly MODEL_PATH = '/assets/refiner_512_websafe.onnx';
  private readonly FM_CHANNELS = 32;
  private readonly FM_SIZE = 512;

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[NeuralRefiner] Initializing...');

    try {
      // WASM設定
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

      // 動的に入出力名を取得
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];

      this.initialized = true;
      console.log('[NeuralRefiner] ✅ Model loaded successfully');
      console.log('[NeuralRefiner] 🔍 Input names:', this.session.inputNames);
      console.log('[NeuralRefiner] 🔍 Output names:', this.session.outputNames);
      console.log('[NeuralRefiner] Using input:', this.inputName, '→ output:', this.outputName);

    } catch (error) {
      console.error('[NeuralRefiner] ❌ Failed to load model:', error);
      throw new Error(`Neural Refiner initialization failed: ${error}`);
    }
  }

  /**
   * 32ch coarse feature map → 3ch RGB
   * @param coarseFM [32 * 512 * 512] CHW format
   * @param _idEmb 未使用（互換性のため残す）
   */
  async run(
    coarseFM: Float32Array,
    _idEmb?: Float32Array
  ): Promise<Float32Array> {
    if (!this.session) {
      throw new Error('NeuralRefiner not initialized');
    }

    console.log('[NeuralRefiner] run() called');

    // 入力サイズ検証
    const expectedFMSize = this.FM_CHANNELS * this.FM_SIZE * this.FM_SIZE;

    console.log('[NeuralRefiner] Input validation:', {
      coarseFM: coarseFM.length,
      expectedFM: expectedFMSize
    });

    if (coarseFM.length !== expectedFMSize) {
      throw new Error(
        `Invalid coarseFM size: ${coarseFM.length}, expected: ${expectedFMSize}`
      );
    }

    // 入力統計
    let fmMin = Infinity, fmMax = -Infinity, fmNonZero = 0;
    for (let i = 0; i < Math.min(10000, coarseFM.length); i++) {
      const v = coarseFM[i];
      if (v < fmMin) fmMin = v;
      if (v > fmMax) fmMax = v;
      if (v !== 0) fmNonZero++;
    }
    console.log('[NeuralRefiner] Input stats (first 10k):', {
      min: fmMin.toFixed(4),
      max: fmMax.toFixed(4),
      nonZeroRatio: (fmNonZero / 10000 * 100).toFixed(1) + '%'
    });

    console.log('[NeuralRefiner] Creating tensor...');

    // Tensor作成（512×512 モデル用）
    const fmTensor = new ort.Tensor('float32', coarseFM, [1, 32, this.FM_SIZE, this.FM_SIZE]);

    console.log('[NeuralRefiner] Running inference (WASM)...');
    const startTime = performance.now();

    // 推論実行 - 動的に取得した入力名を使用
    try {
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.inputName] = fmTensor;

      const outputs = await this.session.run(feeds);
      const elapsed = performance.now() - startTime;
      console.log('[NeuralRefiner] ✅ Inference complete:', elapsed.toFixed(1) + 'ms');

      const rawOutput = outputs[this.outputName].data as Float32Array;
      const dims = outputs[this.outputName].dims;

      console.log('[NeuralRefiner] Raw output info:', {
        length: rawOutput.length,
        dims: dims,
        type: outputs[this.outputName].type
      });

      // 出力形状確認
      const expectedLength = 512 * 512 * 3;
      if (rawOutput.length !== expectedLength) {
        console.warn('[NeuralRefiner] Output length mismatch:', {
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
        out = new Float32Array(rawOutput);
      }

      // 統計情報
      let min = Infinity, max = -Infinity, nonZero = 0, nanCount = 0;

      for (let i = 0; i < out.length; i++) {
        const v = out[i];
        if (isNaN(v)) { nanCount++; continue; }
        if (v !== 0) nonZero++;
        if (v < min) min = v;
        if (v > max) max = v;
      }

      console.log('[NeuralRefiner] Output stats:', {
        min: min.toFixed(4),
        max: max.toFixed(4),
        nonZeroRatio: (nonZero / out.length * 100).toFixed(1) + '%',
        nanCount
      });

      // NaN除去
      if (nanCount > 0) {
        console.warn('[NeuralRefiner] Cleaning NaN values...');
        for (let i = 0; i < out.length; i++) {
          if (!isFinite(out[i])) out[i] = 0;
        }
      }

      // 正規化
      // モデル出力が [-1, 1] の場合 → [0, 1] に変換
      // モデル出力が [0, 1] の場合 → そのまま
      if (min < 0) {
        console.log('[NeuralRefiner] Converting [-1,1] → [0,1]...');
        for (let i = 0; i < out.length; i++) {
          out[i] = Math.max(0, Math.min(1, (out[i] + 1) / 2));
        }
      } else {
        console.log('[NeuralRefiner] Output already in [0,1] range, clipping...');
        for (let i = 0; i < out.length; i++) {
          out[i] = Math.max(0, Math.min(1, out[i]));
        }
      }

      console.log('[NeuralRefiner] ✅ Processing complete');
      return out;

    } catch (runError) {
      console.error('[NeuralRefiner] ❌ Inference failed:', runError);
      throw runError;
    }
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
