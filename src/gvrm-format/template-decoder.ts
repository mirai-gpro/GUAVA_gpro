// template-decoder.ts
// GUAVA Template Decoder - WASM直接指定版
// 新版 template_decoder.onnx 対応

import * as ort from 'onnxruntime-web/wasm';
import { encodeViewDirection } from './view-encoding';

interface GeometryData {
  vTemplate: Float32Array;
  uvCoord: Float32Array;
  baseFeature: Float32Array;
  numVertices: number;
}

export interface TemplateGaussianOutput {
  rgb: Float32Array;          // [N, 32] - 新版では 'rgb' (旧版: latent_32ch)
  opacity: Float32Array;      // [N, 1]
  scale: Float32Array;        // [N, 3]
  rotation: Float32Array;     // [N, 4]
  offset?: Float32Array;      // [N, 3] - 新版で追加
  idEmbedding256?: Float32Array; // [256] - 新版で追加
}

export class TemplateDecoder {
  private session: ort.InferenceSession | null = null;
  private geometryData: GeometryData | null = null;
  private initialized = false;

  async init(basePath: string = '/assets'): Promise<void> {
    if (this.initialized) return;

    console.log('[TemplateDecoder] Initializing (WASM direct paths)...');

    try {
      // ✅ WASM設定（iOS安定版）
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;
      
      // ✅ WASMファイルを直接指定（.mjsの読み込みを回避）
      ort.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm'
      };

      console.log('[TemplateDecoder] ONNX Runtime v1.17.3 configured (direct WASM paths)');

      // ONNXモデルをロード
      this.session = await ort.InferenceSession.create(
        `${basePath}/template_decoder.onnx`,
        { 
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: true,
          enableMemPattern: true
        }
      );

      console.log('[TemplateDecoder] ✅ Model loaded');

      // Debug: ONNXモデルの入力/出力名を確認
      console.log('[TemplateDecoder] 🔍 Input names:', this.session.inputNames);
      console.log('[TemplateDecoder] 🔍 Output names:', this.session.outputNames);

      await this.loadGeometryData(basePath);

      this.initialized = true;
      console.log('[TemplateDecoder] ✅ Initialization complete');

    } catch (error) {
      console.error('[TemplateDecoder] ❌ Failed:', error);
      throw new Error(`Template Decoder init failed: ${error}`);
    }
  }

  private async loadGeometryData(basePath: string): Promise<void> {
    const loadBinary = async (filename: string): Promise<Float32Array> => {
      const response = await fetch(`${basePath}/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load ${filename}: ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      return new Float32Array(buffer);
    };

    console.log('[TemplateDecoder] Loading geometry data...');

    const [vTemplate, uvCoord, baseFeature] = await Promise.all([
      loadBinary('v_template.bin'),
      loadBinary('uv_coord.bin'),
      loadBinary('vertex_base_feature.bin')
    ]);

    this.geometryData = {
      vTemplate,
      uvCoord,
      baseFeature,
      numVertices: 10595
    };

    console.log('[TemplateDecoder] ✅ Geometry loaded');
  }

  async generate(
    projectionFeature: Float32Array,
    globalEmbedding: Float32Array,
    viewDir: [number, number, number] = [0, 0, 1]  // デフォルト: 正面
  ): Promise<TemplateGaussianOutput> {
    if (!this.session || !this.geometryData) {
      throw new Error('[TemplateDecoder] Not initialized');
    }

    const { baseFeature, numVertices } = this.geometryData;

    const startTime = performance.now();

    // View direction を 27次元 SH encoding に変換
    const viewDirs = encodeViewDirection(viewDir);
    console.log('[TemplateDecoder] 📐 View direction:', viewDir, '→ 27ch SH encoding');

    // テンソル作成（新版 template_decoder.onnx のインターフェース）
    // バッチ次元を追加: [N, C] → [1, N, C]
    const projTensor = new ort.Tensor('float32', projectionFeature, [1, numVertices, 128]);
    const globalTensor = new ort.Tensor('float32', globalEmbedding, [1, 256]);
    const baseTensor = new ort.Tensor('float32', baseFeature, [1, numVertices, 128]);
    const viewTensor = new ort.Tensor('float32', viewDirs, [1, 27]);

    // 新版 ONNX 入力名:
    // ['projection_features', 'global_embedding', 'base_features', 'view_dirs']
    const outputs = await this.session.run({
      projection_features: projTensor,
      global_embedding: globalTensor,
      base_features: baseTensor,
      view_dirs: viewTensor
    });

    const elapsed = performance.now() - startTime;
    console.log(`[TemplateDecoder] ✅ Inference: ${elapsed.toFixed(2)}ms`);

    // 新版 ONNX 出力名:
    // ['rgb', 'opacity', 'scale', 'rotation', 'offset', 'id_embedding_256']
    return {
      rgb: outputs.rgb.data as Float32Array,
      opacity: outputs.opacity.data as Float32Array,
      scale: outputs.scale.data as Float32Array,
      rotation: outputs.rotation.data as Float32Array,
      offset: outputs.offset?.data as Float32Array,
      idEmbedding256: outputs.id_embedding_256?.data as Float32Array
    };
  }

  /**
   * ジオメトリデータを取得（gvrm.tsでマッピングに使用）
   */
  getGeometryData(): GeometryData | null {
    return this.geometryData;
  }

  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
      this.initialized = false;
    }
  }
}