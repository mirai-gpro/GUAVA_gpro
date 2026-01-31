// template-decoder.ts
// GUAVA Template Decoder - WASM直接指定版

import * as ort from 'onnxruntime-web/wasm';

interface GeometryData {
  vTemplate: Float32Array;
  uvCoord: Float32Array;
  baseFeature: Float32Array;
  numVertices: number;
}

export interface TemplateGaussianOutput {
  latent32ch: Float32Array;
  opacity: Float32Array;
  scale: Float32Array;
  rotation: Float32Array;
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
    idEmbedding: Float32Array
  ): Promise<TemplateGaussianOutput> {
    if (!this.session || !this.geometryData) {
      throw new Error('[TemplateDecoder] Not initialized');
    }

    const { baseFeature, numVertices } = this.geometryData;

    const startTime = performance.now();

    // Log expected vs actual input names for debugging
    const expectedInputs = this.session.inputNames;
    console.log('[TemplateDecoder] 🔍 ONNX expects inputs:', expectedInputs);
    console.log('[TemplateDecoder] 📐 Input shapes:', {
      projectionFeature: `[${numVertices}, 128]`,
      baseFeature: `[${numVertices}, 128]`,
      idEmbedding: '[768]'
    });

    // Create tensors
    const projTensor = new ort.Tensor('float32', projectionFeature, [numVertices, 128]);
    const baseTensor = new ort.Tensor('float32', baseFeature, [numVertices, 128]);
    const idTensor = new ort.Tensor('float32', idEmbedding, [768]);

    // Build feeds dynamically based on what the model expects
    const feeds: Record<string, ort.Tensor> = {};

    for (const inputName of expectedInputs) {
      const lowerName = inputName.toLowerCase();
      if (lowerName.includes('projection') || lowerName.includes('proj')) {
        feeds[inputName] = projTensor;
        console.log(`[TemplateDecoder]   ✓ Mapped ${inputName} ← projectionFeature`);
      } else if (lowerName.includes('base')) {
        feeds[inputName] = baseTensor;
        console.log(`[TemplateDecoder]   ✓ Mapped ${inputName} ← baseFeature`);
      } else if (lowerName.includes('id') || lowerName.includes('global') || lowerName.includes('embedding')) {
        feeds[inputName] = idTensor;
        console.log(`[TemplateDecoder]   ✓ Mapped ${inputName} ← idEmbedding`);
      } else {
        console.warn(`[TemplateDecoder]   ⚠ Unknown input: ${inputName}`);
      }
    }

    // Verify all inputs are mapped
    if (Object.keys(feeds).length !== expectedInputs.length) {
      console.error('[TemplateDecoder] ❌ Input mapping mismatch:', {
        expected: expectedInputs,
        mapped: Object.keys(feeds)
      });
      throw new Error(`Input mapping failed. Expected: ${expectedInputs.join(', ')}`);
    }

    const outputs = await this.session.run(feeds);

    const elapsed = performance.now() - startTime;
    console.log(`[TemplateDecoder] ✅ Inference: ${elapsed.toFixed(2)}ms`);

    // Log output names for verification
    console.log('[TemplateDecoder] 🔍 Output keys:', Object.keys(outputs));

    return {
      latent32ch: outputs.latent_32ch.data as Float32Array,
      opacity: outputs.opacity.data as Float32Array,
      scale: outputs.scale.data as Float32Array,
      rotation: outputs.rotation.data as Float32Array
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