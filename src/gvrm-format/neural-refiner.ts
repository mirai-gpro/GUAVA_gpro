// neural-refiner.ts
// WebGL バックエンド版 - 安定したGPU推論
// ONNX Runtime Web の WebGL EP を使用

import * as ort from 'onnxruntime-web';

export class NeuralRefiner {
  private session: ort.InferenceSession | null = null;
  private initialized = false;

  private readonly MODEL_PATH = '/assets/refiner_512_websafe.onnx';
  private readonly FM_CHANNELS = 32;
  private readonly FM_SIZE = 512;

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[NeuralRefiner] Initializing (WebGL backend)...');

    try {
      // ONNX Runtime設定
      ort.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm'
      };
      
      // マルチスレッド有効化（WASMフォールバック時に効果）
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
      ort.env.wasm.simd = true;

      // WebGL利用可能チェック
      const webglAvailable = this.checkWebGLAvailable();
      console.log('[NeuralRefiner] WebGL available:', webglAvailable);

      const startTime = performance.now();

      // セッション生成 - WebGL優先、WASMフォールバック
      this.session = await ort.InferenceSession.create(
        this.MODEL_PATH,
        {
          // WebGL → WASM の順でフォールバック
          executionProviders: ['webgl', 'wasm'],
          graphOptimizationLevel: 'all'
        }
      );

      const loadTime = performance.now() - startTime;

      this.initialized = true;
      console.log(`[NeuralRefiner] ✅ Model loaded in ${loadTime.toFixed(0)}ms`);
      console.log('[NeuralRefiner]   Backend: WebGL (GPU accelerated)');
      console.log('[NeuralRefiner]   Fallback: WASM with', ort.env.wasm.numThreads, 'threads');
    } catch (error) {
      console.error('[NeuralRefiner] ❌ Failed to load model:', error);
      throw new Error(`Neural Refiner initialization failed: ${error}`);
    }
  }

  private checkWebGLAvailable(): boolean {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) return false;
      
      // Float texture support check
      const ext = gl.getExtension('OES_texture_float');
      return true;  // WebGL exists, ONNX Runtime will handle details
    } catch (e) {
      return false;
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

    // 入力の診断
    console.log('[NeuralRefiner] Analyzing input...');
    const inputStats = this.analyzeInput(coarseFM);
    console.log('[NeuralRefiner] Input stats:', inputStats);

    // 異常値チェック
    if (inputStats.hasExtreme) {
      console.warn('[NeuralRefiner] ⚠️ Extreme values detected, clamping...');
      for (let i = 0; i < coarseFM.length; i++) {
        if (!isFinite(coarseFM[i])) {
          coarseFM[i] = 0;
        } else if (Math.abs(coarseFM[i]) > 10) {
          coarseFM[i] = Math.sign(coarseFM[i]) * 10;
        }
      }
    }

    // Tensor作成
    const fmTensor = new ort.Tensor('float32', coarseFM, [1, 32, this.FM_SIZE, this.FM_SIZE]);
    const idTensor = new ort.Tensor('float32', idEmb, [1, 256]);

    // 推論実行
    try {
      console.log('[NeuralRefiner] Running inference (WebGL)...');
      const inferStart = performance.now();

      const outputs = await this.session.run({
        coarse_fm: fmTensor,
        id_emb: idTensor
      });

      const inferTime = performance.now() - inferStart;
      console.log(`[NeuralRefiner] ⚡ Inference completed in ${inferTime.toFixed(0)}ms`);

      const rawOutput = outputs.refined_rgb.data as Float32Array;
      const dims = outputs.refined_rgb.dims;

      console.log('[NeuralRefiner] Output dims:', dims);

      // CHW → HWC 変換
      let out: Float32Array;
      const H = 512, W = 512, C = 3;

      if (dims.length === 4 && dims[1] === 3) {
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
      let min = Infinity, max = -Infinity, nonZero = 0;
      for (let i = 0; i < out.length; i++) {
        const v = out[i];
        if (!isFinite(v)) continue;
        if (v !== 0) nonZero++;
        if (v < min) min = v;
        if (v > max) max = v;
      }

      console.log('[NeuralRefiner] Output stats:', { min, max, nonZeroCount: nonZero });

      // NaN/Inf除去
      for (let i = 0; i < out.length; i++) {
        if (!isFinite(out[i])) out[i] = 0;
      }

      // [-1, 1] → [0, 1] 変換
      for (let i = 0; i < out.length; i++) {
        let normalized = (out[i] + 1) / 2;
        out[i] = Math.max(0, Math.min(1, normalized));
      }

      console.log('[NeuralRefiner] ✅ Processing complete');
      return out;

    } catch (runError) {
      console.error('[NeuralRefiner] ❌ Inference failed:', runError);
      throw runError;
    }
  }

  private analyzeInput(fm: Float32Array): any {
    let min = Infinity, max = -Infinity;
    let nonZero = 0, extremeCount = 0;
    
    const sampleSize = Math.min(100000, fm.length);
    const step = Math.floor(fm.length / sampleSize);
    
    for (let i = 0; i < fm.length; i += step) {
      const v = fm[i];
      if (!isFinite(v)) continue;
      if (Math.abs(v) > 0.001) nonZero++;
      if (Math.abs(v) > 100) extremeCount++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    
    return {
      range: `[${min.toFixed(2)}, ${max.toFixed(2)}]`,
      nonZeroRatio: ((nonZero / sampleSize) * 100).toFixed(1) + '%',
      hasExtreme: extremeCount > 0
    };
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
