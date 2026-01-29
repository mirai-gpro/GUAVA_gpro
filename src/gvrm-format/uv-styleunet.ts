// src/gvrm-format/uv-styleunet.ts
// UV StyleUNet: 35ch → 96ch (論文準拠)
// 入力: 32ch UV features + 3ch RGB = 35ch
// 出力: 96ch UV decoded features

import * as ort from 'onnxruntime-web/wasm';

/**
 * UV StyleUNet - UV branch の特徴変換
 *
 * 論文 Section 3.2:
 * "The UV feature map is processed by a StyleUNet to generate 96-channel features"
 */
export class UVStyleUNet {
  private session: ort.InferenceSession | null = null;
  private styleMappingSession: ort.InferenceSession | null = null;
  private baseFeature: Float32Array | null = null;  // [32, 512, 512]
  private initialized = false;

  /**
   * 初期化
   */
  async init(basePath: string = '/assets'): Promise<void> {
    if (this.initialized) return;

    console.log('[UVStyleUNet] Initializing...');

    try {
      // ONNX Runtime 設定
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      // 1. UV StyleUNet モデルをロード
      console.log('[UVStyleUNet] Loading uv_styleunet.onnx...');
      this.session = await ort.InferenceSession.create(
        `${basePath}/uv_styleunet.onnx`,
        { executionProviders: ['wasm'] }
      );
      console.log('[UVStyleUNet] ✅ StyleUNet loaded');
      console.log('[UVStyleUNet]   Input names:', this.session.inputNames);
      console.log('[UVStyleUNet]   Output names:', this.session.outputNames);

      // 2. Style Mapping モデルをロード
      console.log('[UVStyleUNet] Loading uv_style_mapping.onnx...');
      this.styleMappingSession = await ort.InferenceSession.create(
        `${basePath}/uv_style_mapping.onnx`,
        { executionProviders: ['wasm'] }
      );
      console.log('[UVStyleUNet] ✅ Style mapping loaded');

      // 3. Base Feature をロード
      console.log('[UVStyleUNet] Loading uv_base_feature.bin...');
      const response = await fetch(`${basePath}/uv_base_feature.bin`);
      if (!response.ok) {
        throw new Error(`Failed to load uv_base_feature.bin: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      this.baseFeature = new Float32Array(buffer);
      console.log('[UVStyleUNet] ✅ Base feature loaded:', {
        size: this.baseFeature.length,
        expectedSize: 32 * 512 * 512,
        shape: '[32, 512, 512]'
      });

      this.initialized = true;
      console.log('[UVStyleUNet] ✅ Initialization complete');

    } catch (error) {
      console.error('[UVStyleUNet] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Global feature から extra_style を生成
   * 768 → 512
   */
  async computeExtraStyle(globalFeature: Float32Array): Promise<Float32Array> {
    if (!this.styleMappingSession) {
      throw new Error('Style mapping not initialized');
    }

    const inputTensor = new ort.Tensor('float32', globalFeature, [1, 768]);
    const feeds = { 'global_feature': inputTensor };
    const outputs = await this.styleMappingSession.run(feeds);

    return new Float32Array(outputs['extra_style'].data as Float32Array);
  }

  /**
   * UV特徴を処理して96ch出力を生成
   *
   * @param uvFeatures32ch DINOv2 UV branch出力 [32, H, W] (HWC or CHW)
   * @param rgbImage ソース画像RGB [3, H, W] または [H, W, 3]
   * @param globalFeature DINOv2 global feature [768]
   * @param uvWidth UV map width (512)
   * @param uvHeight UV map height (512)
   * @returns 96ch feature map [96, uvHeight, uvWidth]
   */
  async forward(
    uvFeatures32ch: Float32Array,
    rgbImage: Float32Array,  // [3, H, W] normalized 0-1
    globalFeature: Float32Array,
    uvWidth: number = 512,
    uvHeight: number = 512
  ): Promise<Float32Array> {
    if (!this.session || !this.baseFeature) {
      throw new Error('UVStyleUNet not initialized');
    }

    console.log('[UVStyleUNet] Processing...');
    console.log('[UVStyleUNet]   UV features: ', uvFeatures32ch.length, '(expected:', 32 * uvWidth * uvHeight, ')');
    console.log('[UVStyleUNet]   RGB image: ', rgbImage.length, '(expected:', 3 * uvWidth * uvHeight, ')');

    // 1. Compute extra_style from global feature
    const extraStyle = await this.computeExtraStyle(globalFeature);
    console.log('[UVStyleUNet]   Extra style computed: [1, 512]');

    // 2. Concatenate UV features (32ch) + RGB (3ch) = 35ch
    const input35ch = new Float32Array(35 * uvHeight * uvWidth);

    // Copy 32ch UV features (assuming CHW format)
    for (let i = 0; i < 32 * uvHeight * uvWidth; i++) {
      input35ch[i] = uvFeatures32ch[i];
    }

    // Copy 3ch RGB (assuming CHW format)
    for (let i = 0; i < 3 * uvHeight * uvWidth; i++) {
      input35ch[32 * uvHeight * uvWidth + i] = rgbImage[i];
    }

    console.log('[UVStyleUNet]   Input 35ch prepared');

    // 3. Run StyleUNet: 35ch → 96ch
    const inputTensor = new ort.Tensor('float32', input35ch, [1, 35, uvHeight, uvWidth]);
    const styleTensor = new ort.Tensor('float32', extraStyle, [1, 512]);

    const feeds = {
      'uv_features': inputTensor,
      'extra_style': styleTensor
    };

    console.log('[UVStyleUNet]   Running StyleUNet inference...');
    const startTime = performance.now();
    const outputs = await this.session.run(feeds);
    const inferenceTime = performance.now() - startTime;
    console.log('[UVStyleUNet]   ✅ Inference complete:', inferenceTime.toFixed(1), 'ms');

    const output96ch = new Float32Array(outputs['output'].data as Float32Array);
    console.log('[UVStyleUNet]   Output 96ch:', output96ch.length, '(expected:', 96 * uvHeight * uvWidth, ')');

    return output96ch;
  }

  /**
   * 96ch StyleUNet出力 + 32ch base_feature = 128ch
   */
  addBaseFeature(styleunetOutput96ch: Float32Array, uvWidth: number = 512, uvHeight: number = 512): Float32Array {
    if (!this.baseFeature) {
      throw new Error('Base feature not loaded');
    }

    const output128ch = new Float32Array(128 * uvHeight * uvWidth);

    // Copy 96ch from StyleUNet output
    for (let i = 0; i < 96 * uvHeight * uvWidth; i++) {
      output128ch[i] = styleunetOutput96ch[i];
    }

    // Add 32ch base_feature
    for (let i = 0; i < 32 * uvHeight * uvWidth; i++) {
      output128ch[96 * uvHeight * uvWidth + i] = this.baseFeature[i];
    }

    console.log('[UVStyleUNet] Added base_feature: 96ch + 32ch = 128ch');
    return output128ch;
  }

  /**
   * リソース解放
   */
  dispose(): void {
    this.session = null;
    this.styleMappingSession = null;
    this.baseFeature = null;
    this.initialized = false;
    console.log('[UVStyleUNet] Disposed');
  }
}
