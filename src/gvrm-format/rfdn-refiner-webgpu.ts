/**
 * neural-refiner-webgpu.ts
 *
 * SimpleUNet Neural Refiner (38MB)
 * StyleUNetのUNet部分を使用した軽量モデル
 *
 * 入力: 32ch Feature Map [1, 32, 512, 512] (値域 [0, 1] に正規化必須)
 * 出力: RGB画像 [1, 3, 512, 512] (クランプ [0, 1]、sigmoidなし)
 *
 * 使い方:
 *   const refiner = new RFDNRefiner();
 *   await refiner.init();
 *   const rgb = await refiner.process(featureMap32ch);
 */

import * as ort from 'onnxruntime-web';

export interface RefinerConfig {
  modelPath?: string;
  inputChannels?: number;
  inputSize?: number;
  useWebGPU?: boolean;
}

export class RFDNRefiner {
  private session: ort.InferenceSession | null = null;
  private initialized = false;

  // デフォルト設定
  private readonly config: Required<RefinerConfig>;

  constructor(config: RefinerConfig = {}) {
    this.config = {
      modelPath: config.modelPath ?? '/assets/simpleunet_trained.onnx',
      inputChannels: config.inputChannels ?? 32,
      inputSize: config.inputSize ?? 512,
      useWebGPU: config.useWebGPU ?? false,
    };
  }

  /**
   * 初期化
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[NeuralRefiner] Initializing...');
    console.log('[NeuralRefiner]   Model: simpleunet_trained.onnx (38MB)');
    console.log('[NeuralRefiner]   Input: 32ch × 512 × 512 (normalized to [0,1])');
    console.log('[NeuralRefiner]   Output: RGB × 512 × 512');

    try {
      // ONNX Runtime Web設定
      const executionProviders: string[] = [];

      if (this.config.useWebGPU) {
        // WebGPU使用 (対応ブラウザのみ)
        executionProviders.push('webgpu');
        console.log('[NeuralRefiner]   Backend: WebGPU');
      }
      
      // フォールバック: WASM
      executionProviders.push('wasm');

      // WASM設定
      ort.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm'
      };
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
      ort.env.wasm.simd = true;

      // セッション作成
      this.session = await ort.InferenceSession.create(
        this.config.modelPath,
        {
          executionProviders,
          graphOptimizationLevel: 'all'
        }
      );

      // 入出力情報を確認
      console.log('[NeuralRefiner]   Input names:', this.session.inputNames);
      console.log('[NeuralRefiner]   Output names:', this.session.outputNames);

      this.initialized = true;
      console.log('[NeuralRefiner] ✅ Initialized');

    } catch (error) {
      console.error('[NeuralRefiner] ❌ Initialization failed:', error);
      throw new Error(`Neural Refiner initialization failed: ${error}`);
    }
  }

  /**
   * 推論実行
   * @param featureMap 32チャンネル特徴マップ (CHW形式, Float32Array)
   * @returns RGB画像 (HWC形式, Float32Array, 値域 [0, 1])
   */
  async process(featureMap: Float32Array): Promise<Float32Array> {
    if (!this.session) {
      throw new Error('NeuralRefiner not initialized. Call init() first.');
    }

    const { inputChannels, inputSize } = this.config;
    const expectedSize = inputChannels * inputSize * inputSize;

    // 入力サイズ検証
    if (featureMap.length !== expectedSize) {
      throw new Error(
        `Invalid input size: ${featureMap.length}, expected: ${expectedSize} (${inputChannels}×${inputSize}×${inputSize})`
      );
    }

    // 入力統計
    const inputStats = this.computeStats(featureMap);
    console.log('[NeuralRefiner] Input stats:', inputStats);

    // NaN/Inf をクリーンアップ
    if (inputStats.hasInvalid) {
      console.warn('[NeuralRefiner] ⚠️ Cleaning invalid values in input...');
      for (let i = 0; i < featureMap.length; i++) {
        if (!isFinite(featureMap[i])) {
          featureMap[i] = 0;
        }
      }
    }
    
    // 極端な値をクリップ（GPUハングを防ぐ）- 閾値を緩和
    const absMax = Math.max(Math.abs(inputStats.min), Math.abs(inputStats.max));
    if (absMax > 1000) {
      console.warn(`[NeuralRefiner] ⚠️ Extreme values detected (max=${absMax.toFixed(1)}), clipping to [-100, 100]`);
      for (let i = 0; i < featureMap.length; i++) {
        if (featureMap[i] > 100) featureMap[i] = 100;
        if (featureMap[i] < -100) featureMap[i] = -100;
      }
    }

    // Tensor作成 [1, 32, 512, 512]
    const inputTensor = new ort.Tensor(
      'float32',
      featureMap,
      [1, inputChannels, inputSize, inputSize]
    );

    try {
      console.log('[NeuralRefiner] Running inference...');
      const startTime = performance.now();

      // 推論実行
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.session.inputNames[0]] = inputTensor;
      
      const outputs = await this.session.run(feeds);
      
      const endTime = performance.now();
      console.log(`[NeuralRefiner] Inference time: ${(endTime - startTime).toFixed(1)}ms`);

      // 出力取得
      const outputName = this.session.outputNames[0];
      const rawOutput = outputs[outputName].data as Float32Array;
      const dims = outputs[outputName].dims;

      console.log('[NeuralRefiner] Output dims:', dims);

      // CHW → HWC 変換 (sigmoid不要 - Python版と同様にクランプのみ)
      const H = inputSize;
      const W = inputSize;
      const C = 3;
      const output = new Float32Array(H * W * C);

      // 出力の統計を取得（デバッグ用）
      let rawMin = Infinity, rawMax = -Infinity;
      for (let i = 0; i < rawOutput.length; i++) {
        if (rawOutput[i] < rawMin) rawMin = rawOutput[i];
        if (rawOutput[i] > rawMax) rawMax = rawOutput[i];
      }
      console.log(`[NeuralRefiner] Raw output range: [${rawMin.toFixed(4)}, ${rawMax.toFixed(4)}]`);

      // v78: 出力処理の修正
      // モデル出力が pre-sigmoid 値の場合は sigmoid を適用
      // 線形マッピングは色差を消すので使用しない
      const useSigmoid = rawMin < -1 || rawMax > 2;  // pre-sigmoid値っぽい場合

      if (useSigmoid) {
        console.log('[NeuralRefiner] 🔧 Applying sigmoid (raw values suggest pre-sigmoid output)');
      }

      if (dims.length === 4 && dims[1] === 3) {
        // [1, 3, H, W] → [H, W, 3]
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            for (let c = 0; c < C; c++) {
              const srcIdx = c * H * W + h * W + w;
              const dstIdx = h * W * C + w * C + c;
              let val = rawOutput[srcIdx];

              if (useSigmoid) {
                // Sigmoid: 1 / (1 + exp(-x))
                val = 1 / (1 + Math.exp(-val));
              }

              output[dstIdx] = Math.max(0, Math.min(1, val));
            }
          }
        }
      } else {
        // そのままコピー
        for (let i = 0; i < H * W * C; i++) {
          let val = rawOutput[i];
          if (useSigmoid) {
            val = 1 / (1 + Math.exp(-val));
          }
          output[i] = Math.max(0, Math.min(1, val));
        }
      }

      // 出力統計（sigmoid後）
      let finalMin = Infinity, finalMax = -Infinity, finalSum = 0;
      for (let i = 0; i < output.length; i++) {
        if (output[i] < finalMin) finalMin = output[i];
        if (output[i] > finalMax) finalMax = output[i];
        finalSum += output[i];
      }
      console.log(`[NeuralRefiner] Final output: [${finalMin.toFixed(4)}, ${finalMax.toFixed(4)}], mean=${(finalSum/output.length).toFixed(4)}`);

      // 出力統計
      const outputStats = this.computeStats(output);
      console.log('[NeuralRefiner] Output stats:', outputStats);

      return output;

    } catch (error) {
      console.error('[NeuralRefiner] ❌ Inference failed:', error);
      throw error;
    }
  }

  /**
   * Canvas/ImageDataに直接描画
   */
  async processToCanvas(
    featureMap: Float32Array,
    canvas: HTMLCanvasElement
  ): Promise<void> {
    const rgb = await this.process(featureMap);
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Cannot get 2D context');

    const { inputSize } = this.config;
    canvas.width = inputSize;
    canvas.height = inputSize;

    const imageData = ctx.createImageData(inputSize, inputSize);
    const data = imageData.data;

    for (let i = 0; i < inputSize * inputSize; i++) {
      const srcIdx = i * 3;
      const dstIdx = i * 4;
      
      data[dstIdx + 0] = Math.round(rgb[srcIdx + 0] * 255); // R
      data[dstIdx + 1] = Math.round(rgb[srcIdx + 1] * 255); // G
      data[dstIdx + 2] = Math.round(rgb[srcIdx + 2] * 255); // B
      data[dstIdx + 3] = 255; // A
    }

    ctx.putImageData(imageData, 0, 0);
  }

  /**
   * 統計計算
   */
  private computeStats(data: Float32Array): {
    min: number;
    max: number;
    mean: number;
    hasInvalid: boolean;
  } {
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let validCount = 0;
    let hasInvalid = false;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!isFinite(v)) {
        hasInvalid = true;
        continue;
      }
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      validCount++;
    }

    return {
      min: min === Infinity ? 0 : min,
      max: max === -Infinity ? 0 : max,
      mean: validCount > 0 ? sum / validCount : 0,
      hasInvalid
    };
  }

  /**
   * リソース解放
   */
  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
      this.initialized = false;
      console.log('[NeuralRefiner] Disposed');
    }
  }

  /**
   * 初期化済みかどうか
   */
  get isInitialized(): boolean {
    return this.initialized;
  }
}


/**
 * 使用例
 */
export async function example() {
  // 1. 初期化
  const refiner = new RFDNRefiner({
    modelPath: '/assets/simpleunet_trained.onnx',
    useWebGPU: true  // WebGPU対応ブラウザで高速化
  });
  await refiner.init();

  // 2. 32ch特徴マップを用意 (Gaussian Splattingの出力)
  const featureMap = new Float32Array(32 * 512 * 512);
  // ... featureMapにデータを設定 ...

  // 3. 推論
  const rgb = await refiner.process(featureMap);
  // rgb: Float32Array (512 * 512 * 3), 値域 [0, 1]

  // 4. Canvasに描画
  const canvas = document.getElementById('output') as HTMLCanvasElement;
  await refiner.processToCanvas(featureMap, canvas);

  // 5. クリーンアップ
  refiner.dispose();
}
