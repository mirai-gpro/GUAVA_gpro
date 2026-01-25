/**
 * Template Decoder - ONNX Runtime Wrapper
 * 
 * 高速なONNX Runtimeを使いつつ、モジュール化を実現
 * gvrm.tsの肥大化を防ぐためのラッパークラス
 */

import * as ort from 'onnxruntime-web';

export interface TemplateDecoderInput {
  projection_features: Float32Array;   // [N, 128]
  global_embedding: Float32Array;      // [768] ← CLS token直接（256ではない）
  num_vertices: number;
}

export interface TemplateDecoderOutput {
  positions: Float32Array;   // [N, 3]
  rotations: Float32Array;   // [N, 4]
  scales: Float32Array;      // [N, 3]
  opacities: Float32Array;   // [N, 1]
  colors: Float32Array;      // [N, 32]
}

/**
 * Template Decoder using ONNX Runtime
 * 
 * ONNXモデルを使用した高速推論
 * - GPU加速対応
 * - 公式実装と完全一致
 * - モジュール化されたAPI
 * 
 * 注意: 新しいバージョンではglobal_feature_mappingとbase_featuresが
 * ONNXモデルに統合されています
 */
export class TemplateDecoder {
  private session: ort.InferenceSession | null = null;
  private initialized: boolean = false;

  /**
   * Initialize Template Decoder
   * 
   * @param assetsPath - Path to assets directory
   */
  async init(assetsPath: string = '/assets'): Promise<void> {
    console.log('[TemplateDecoder] Initializing...');
    
    try {
      // Load ONNX model
      const modelPath = `${assetsPath}/template_decoder.onnx`;
      const dataPath = `${assetsPath}/template_decoder.onnx.data`;
      
      console.log(`[TemplateDecoder]   Loading ONNX model: ${modelPath}`);
      
      // モデルファイルをフェッチ
      const [modelResp, dataResp] = await Promise.all([
        fetch(modelPath),
        fetch(dataPath)
      ]);
      
      if (!modelResp.ok) {
        throw new Error(`Failed to load model: ${modelResp.status}`);
      }
      
      const modelBuffer = await modelResp.arrayBuffer();
      
      // 外部データファイルの処理
      const options: any = {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all'
      };
      
      if (dataResp.ok) {
        console.log(`[TemplateDecoder]   Loading external data: ${dataPath}`);
        const dataBuffer = await dataResp.arrayBuffer();
        options.externalData = [{
          path: 'template_decoder.onnx.data',
          data: dataBuffer
        }];
      } else {
        console.log(`[TemplateDecoder]   No external data file (embedded model)`);
      }
      
      this.session = await ort.InferenceSession.create(modelBuffer, options);
      
      console.log('[TemplateDecoder]   ✅ ONNX model loaded');
      console.log('[TemplateDecoder]   Model includes:');
      console.log('[TemplateDecoder]     - Global feature mapping (768→256)');
      console.log('[TemplateDecoder]     - Vertex base features (embedded)');
      console.log('[TemplateDecoder]     - GS decoder');
      
      this.initialized = true;
      console.log('[TemplateDecoder] ✅ Initialized');
      
    } catch (error) {
      console.error('[TemplateDecoder] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Run Template Decoder inference
   * 
   * @param input - Decoder input features
   * @returns Gaussian attributes
   */
  async forward(input: TemplateDecoderInput): Promise<TemplateDecoderOutput> {
    if (!this.initialized || !this.session) {
      throw new Error('Template Decoder not initialized');
    }

    const N = input.num_vertices;
    
    // Validate input dimensions
    if (input.projection_features.length !== N * 128) {
      throw new Error(`Invalid projection_features size: ${input.projection_features.length}, expected ${N * 128}`);
    }
    if (input.global_embedding.length !== 768) {
      throw new Error(`Invalid global_embedding size: ${input.global_embedding.length}, expected 768`);
    }

    // Create ONNX tensors
    // 新しいモデルはbase_featuresを内部に持っているため、外部から渡さない
    const feeds = {
      projection_features: new ort.Tensor('float32', input.projection_features, [1, N, 128]),
      global_embedding: new ort.Tensor('float32', input.global_embedding, [1, 768]),
      view_dirs: new ort.Tensor('float32', new Float32Array(27), [1, 27])
    };

    // Run inference
    const results = await this.session.run(feeds);

    // Extract outputs
    const rotations = results.rotation.data as Float32Array;
    const scales = results.scale.data as Float32Array;
    const opacities = results.opacity.data as Float32Array;
    const colors = results.rgb.data as Float32Array;

    // Positions are not modified by decoder (use input positions)
    const positions = new Float32Array(N * 3);

    return {
      positions,
      rotations,
      scales,
      opacities,
      colors
    };
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.session = null;
    this.initialized = false;
    console.log('[TemplateDecoder] Disposed');
  }
}

export default TemplateDecoder;