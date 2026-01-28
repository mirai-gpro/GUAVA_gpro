// gvrm.ts
// 修正版 v74: SimpleUNet Neural Refiner (GUAVA pretrained weights)
// - SimpleUNet Refiner (38MB from GUAVA checkpoint)
// - 入力正規化: Gaussian出力を[0, 1]に変換
// - Sigmoid適用: 出力を[0, 1]に変換

import { ImageEncoder } from './image-encoder';
import { InverseTextureMapper, type EHMMeshData, type ImageFeatures } from './inverse-texture-mapping';
import TemplateDecoderWebGPU, {
  type TemplateDecoderInput,
  type TemplateDecoderOutput
} from './template-decoder-webgpu';
import { UVDecoder } from './uv-decoder';
import { RFDNRefiner } from './rfdn-refiner-webgpu';  // ← 新しい軽量Refiner
import { WebGLDisplay } from './webgl-display';
import { GuavaWebGPURendererPractical } from './guava-webgpu-renderer-practical';
import { GuavaWebGPURendererCompute } from './guava-webgpu-renderer-compute';
import { CameraUtils } from './camera-utils';
import { loadUVTriangleMapping, type UVTriangleMapping } from './webgl-uv-rasterizer';
import { UVFeatureMapper } from './uv-feature-mapper';

interface GVRMConfig {
  templatePath?: string;
  imagePath?: string;
  container?: HTMLElement;
  useWebGPURefiner?: boolean;  // WebGPU for refiner
}

interface GaussianData {
  positions: Float32Array;
  latents: Float32Array;
  opacity: Float32Array;
  scale: Float32Array;
  rotation: Float32Array;
  vertexCount: number;
}

interface PLYData {
  positions: Float32Array;
  vertexCount: number;
  faces: Uint32Array;
  faceCount: number;
  normals?: Float32Array;
  colors?: Float32Array;
}

/**
 * GVRM - GUAVA Virtual Reality Model
 *
 * v74: SimpleUNet Refiner対応版
 * - GUAVA公式チェックポイントからSimpleUNet抽出 (38MB)
 * - 入力正規化: Gaussian出力を[0, 1]に変換
 * - Sigmoid適用: 出力を[0, 1]に変換
 *
 * Pipeline:
 * 1. Image Encoder: Image → Features (projection + ID embedding)
 * 2. Template Decoder: 3入力 (projection, base, id) → Template Gaussians
 * 3. Inverse Texture Mapping: Image → UV features (optional)
 * 4. UV Decoder: UV features → UV Gaussians (optional)
 * 5. WebGPU Rendering: Gaussians → Coarse feature map (32ch)
 * 6. SimpleUNet Refiner: Coarse 32ch [0,1] → Refined RGB [0,1]
 */
export class GVRM {
  // Asset paths
  private templatePath: string = '/assets/avatar_web.ply';
  private imagePath: string = '/assets/source.png';
  private cameraConfigPath: string = '/assets/source_camera.json';
  private uvCoordsPath: string = '/assets/uv_coords.bin';
  private uvTriangleMappingPath: string = '/assets/uv_triangle_mapping.bin';
  private container: HTMLElement | null = null;
  private useWebGPURefiner: boolean = true;

  // UV Pipeline data
  private uvTriangleMapping: UVTriangleMapping | null = null;
  private uvFeatureMapper: UVFeatureMapper | null = null;
  private appearanceMap: Float32Array | null = null;  // [128, 518, 518] from Image Encoder
  private appearanceMapSize: number = 518;
  
  // Core modules
  private imageEncoder: ImageEncoder;
  private templateDecoder: TemplateDecoderWebGPU | null = null;
  private inverseTextureMapper: InverseTextureMapper;
  private uvDecoder: UVDecoder;
  private neuralRefiner: RFDNRefiner;  // SimpleUNet (GUAVA pretrained)
  private webglDisplay: WebGLDisplay | null = null;
  
  // Data
  private plyData: PLYData | null = null;
  private uvCoords: Float32Array | null = null;
  private uvMappingData: EHMMeshData | null = null;
  private templateGaussians: GaussianData | null = null;
  private uvGaussians: GaussianData | null = null;
  private visibilityMask: Uint8Array | null = null;  // Tracks which vertices are visible in source image
  // idEmbeddingは不要になった！
  
  // WebGPU
  private gpuDevice: GPUDevice | null = null;
  private gsCoarseRenderer: GuavaWebGPURendererPractical | null = null;
  private gsComputeRenderer: GuavaWebGPURendererCompute | null = null;
  private useComputeRenderer: boolean = true;  // ← 32チャンネル完全保持のためCompute Rendererを使用
  private debugBypassRFDN: boolean = false;  // v77: Refiner有効化（Gemini推奨）
  private debugInjectTestColors: boolean = false;  // テスト用: 虹色グラデーションを注入してレンダリング検証
  private readbackBuffers: GPUBuffer[] = [];
  private coarseFeatureArray: Float32Array | null = null;
  
  // State
  private initialized: boolean = false;
  private isRunning: boolean = false;
  private frameId: number | null = null;
  private frameCount: number = 0;
  private deviceLost: boolean = false;
  private recovering: boolean = false;
  private boundVisibilityHandler: (() => void) | null = null;
  
  constructor(config?: GVRMConfig) {
    if (config) {
      if (config.templatePath) this.templatePath = config.templatePath;
      if (config.imagePath) this.imagePath = config.imagePath;
      if (config.container) this.container = config.container;
      if (config.useWebGPURefiner !== undefined) this.useWebGPURefiner = config.useWebGPURefiner;
    }
    
    this.imageEncoder = new ImageEncoder();
    // templateDecoderは重みロード後に初期化
    this.inverseTextureMapper = new InverseTextureMapper(512);
    this.uvDecoder = new UVDecoder();
    
    // Neural Refiner: SimpleUNet (38MB from GUAVA pretrained weights)
    // StyleUNetのUNet部分を使用 (ModulatedConv2dがONNX非対応のため)
    this.neuralRefiner = new RFDNRefiner({
      modelPath: '/assets/simpleunet_trained.onnx',
      useWebGPU: false  // WASM使用（安定性優先）
    });
    
    console.log('[GVRM] Created (v86: GPU Compute Splatting 2026-01-27)');
    console.log('[GVRM] ════════════════════════════════════════════════════');
    console.log('[GVRM] 🔧 BUILD v86 - GPU compute splatting with unified buffer');
    console.log('[GVRM] ════════════════════════════════════════════════════');
  }
  
  async init(config?: GVRMConfig): Promise<void> {
    if (this.initialized) {
      console.warn('[GVRM] Already initialized');
      return;
    }
    
    try {
      console.log('[GVRM] 🚀 Initializing pipeline...');
      console.log('[GVRM] ═══════════════════════════════');
      console.log('[GVRM] 📦 Using SimpleUNet Refiner (38MB, GUAVA pretrained)');
      
      // 1. WebGPU setup
      console.log('[GVRM] Step 1/6: WebGPU initialization');
      if (!navigator.gpu) throw new Error('WebGPU not supported');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No GPU adapter');
      
      // 8つのRGBA16Floatテクスチャ用に制限を上げる
      const requiredLimits: GPUDeviceLimits = {};
      const adapterLimits = adapter.limits;

      // maxColorAttachmentBytesPerSample: 8 * RGBA16Float = 8 * 8 = 64 bytes必要
      if (adapterLimits.maxColorAttachmentBytesPerSample >= 128) {
        (requiredLimits as any).maxColorAttachmentBytesPerSample = 128;
        console.log('[GVRM]   Requesting maxColorAttachmentBytesPerSample: 128');
      } else if (adapterLimits.maxColorAttachmentBytesPerSample >= 64) {
        (requiredLimits as any).maxColorAttachmentBytesPerSample = 64;
        console.log('[GVRM]   Requesting maxColorAttachmentBytesPerSample: 64');
      }

      // v86: Gaussianバッファ用にストレージバッファサイズ上限を引き上げ
      // 1M+ Gaussians × 44 floats × 4 bytes ≈ 186MB > default 128MB
      const neededStorageSize = 512 * 1024 * 1024;  // 512MB
      if (adapterLimits.maxStorageBufferBindingSize >= neededStorageSize) {
        (requiredLimits as any).maxStorageBufferBindingSize = neededStorageSize;
        console.log(`[GVRM]   Requesting maxStorageBufferBindingSize: ${neededStorageSize / 1024 / 1024}MB`);
      } else {
        // アダプターの限界まで要求
        (requiredLimits as any).maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
        console.warn(`[GVRM]   ⚠️ Adapter maxStorageBufferBindingSize: ${adapterLimits.maxStorageBufferBindingSize / 1024 / 1024}MB (need ${neededStorageSize / 1024 / 1024}MB)`);
      }

      // maxBufferSize も引き上げ
      const neededBufferSize = 512 * 1024 * 1024;
      if (adapterLimits.maxBufferSize >= neededBufferSize) {
        (requiredLimits as any).maxBufferSize = neededBufferSize;
        console.log(`[GVRM]   Requesting maxBufferSize: ${neededBufferSize / 1024 / 1024}MB`);
      }
      
      this.gpuDevice = await adapter.requestDevice({
        requiredLimits: requiredLimits as any
      });
      this.setupDeviceLostHandler();
      this.setupVisibilityHandler();
      this.initReadbackBuffers(512, 512);
      console.log('[GVRM]   ✅ WebGPU ready');
      
      // 2. Display setup
      console.log('[GVRM] Step 2/6: Display setup');
      if (!this.container) {
        const autoContainer = document.getElementById('avatar3DContainer');
        if (autoContainer) this.container = autoContainer;
      }
      if (this.container) {
        this.webglDisplay = new WebGLDisplay(this.container, 512, 512);
        console.log('[GVRM]   ✅ Display ready');
      } else {
        console.warn('[GVRM]   ⚠️ No container (headless mode)');
      }
      
      // 3. Load assets
      console.log('[GVRM] Step 3/6: Loading assets');
      const plyData = await this.loadPLY(this.templatePath);
      this.plyData = plyData;
      console.log(`[GVRM]   ✅ PLY loaded: ${plyData.vertexCount} vertices, ${plyData.faceCount} faces`);
      
      try {
        this.uvCoords = await this.loadBinary(this.uvCoordsPath);
        console.log(`[GVRM]   ✅ UV coords loaded: ${this.uvCoords.length / 2} vertices`);
      } catch (e) {
        console.warn('[GVRM]   ⚠️ UV coords not found (template-only mode)');
      }

      // UV Triangle Mapping (for UV pipeline)
      try {
        this.uvTriangleMapping = await loadUVTriangleMapping(this.uvTriangleMappingPath);
        console.log(`[GVRM]   ✅ UV Triangle Mapping loaded: ${this.uvTriangleMapping.numValid.toLocaleString()} valid pixels`);

        // Initialize UV Feature Mapper
        this.uvFeatureMapper = new UVFeatureMapper({
          uvWidth: this.uvTriangleMapping.width,
          uvHeight: this.uvTriangleMapping.height,
          imageWidth: 518,
          imageHeight: 518
        });
        console.log('[GVRM]   ✅ UV Feature Mapper initialized');
      } catch (e) {
        console.warn('[GVRM]   ⚠️ UV Triangle Mapping not found (template-only mode)');
        this.uvTriangleMapping = null;
        this.uvFeatureMapper = null;
      }
      
      // 4. Initialize modules
      console.log('[GVRM] Step 4/6: Initializing modules');
      
      await Promise.all([
        this.imageEncoder.init(),
        this.uvDecoder.init('/assets'),
        this.neuralRefiner.init()  // SimpleUNet (38MB) - GUAVA pretrained weights
      ]);
      
      // Template Decoder initialization (WebGPU)
      this.templateDecoder = new TemplateDecoderWebGPU();
      await this.templateDecoder.init(this.gpuDevice!, '/assets');
      
      console.log('[GVRM]   ✅ All modules initialized');
      console.log('[GVRM]   📊 SimpleUNet Refiner: 38MB loaded (GUAVA pretrained)');
      
      // 5. Run inference pipeline
      console.log('[GVRM] Step 5/6: Running inference pipeline');
      await this.runInferencePipeline();
      console.log('[GVRM]   ✅ Inference complete');
      
      // 6. Setup renderer
      console.log('[GVRM] Step 6/6: Setting up renderer');
      await this.setupRenderer();
      console.log('[GVRM]   ✅ Renderer ready');
      
      this.initialized = true;
      this.isRunning = true;
      
      console.log('[GVRM] ═══════════════════════════════');
      console.log('[GVRM] ✅ Initialization complete!');
      console.log('[GVRM]   Template Gaussians: ', this.templateGaussians?.vertexCount || 0);
      console.log('[GVRM]   UV Gaussians: ', this.uvGaussians?.vertexCount || 0);
      console.log('[GVRM]   Total Gaussians: ', 
        (this.templateGaussians?.vertexCount || 0) + (this.uvGaussians?.vertexCount || 0));
      console.log('[GVRM]   🚀 SimpleUNet Refiner: Raw input (no normalization)');
      
      this.renderFrame();
      
    } catch (error) {
      console.error('[GVRM] ❌ Initialization failed:', error);
      throw error;
    }
  }
  
  /**
   * Run the inference pipeline (3入力Template Decoder対応版)
   * Note: idEmbeddingはNeural Refinerで使わないが、Template Decoderでは使用
   */
  private async runInferencePipeline(): Promise<void> {
    console.log('[GVRM] ─── Inference Pipeline ───');
    
    const vertexCount = this.uvCoords ? (this.uvCoords.length / 2) : 
                       (this.plyData!.vertexCount);
    
    console.log(`[GVRM] Using vertex count: ${vertexCount}`);
    
    // Prepare vertices
    const vertices = new Float32Array(vertexCount * 3);
    if (this.plyData!.positions.length >= vertexCount * 3) {
      vertices.set(this.plyData!.positions.subarray(0, vertexCount * 3));
    } else {
      vertices.set(this.plyData!.positions);
    }
    
    // ===== PHASE 1: Image Encoding =====
    console.log('[GVRM] Phase 1: Image encoding');
    console.log(`[GVRM]   Input image: ${this.imagePath}`);
    console.log(`[GVRM]   Vertices: ${vertexCount}`);
    
    const { projectionFeature, idEmbedding, visibilityMask, appearanceMap, appearanceMapSize } =
      await this.imageEncoder.extractFeaturesWithSourceCamera(
        this.imagePath,
        {},
        vertices,
        vertexCount,
        128
      );

    // Store visibility mask for opacity masking
    this.visibilityMask = visibilityMask;
    // Store appearance map for UV pipeline
    this.appearanceMap = appearanceMap;
    this.appearanceMapSize = appearanceMapSize;
    
    console.log('[GVRM]   ✅ Encoder output:');
    console.log(`[GVRM]      Projection features: [${vertexCount}, 128]`);
    const projStats = this.analyzeArray(projectionFeature);
    console.log(`[GVRM]        stats: min=${projStats.min.toFixed(4)}, max=${projStats.max.toFixed(4)}, nonZeros=${projStats.nonZeros}`);
    console.log(`[GVRM]      ID embedding (CLS token): [${idEmbedding.length}]`);
    const idStats = this.analyzeArray(idEmbedding);
    console.log(`[GVRM]        stats: min=${idStats.min.toFixed(4)}, max=${idStats.max.toFixed(4)}, nonZeros=${idStats.nonZeros}`);
    console.log(`[GVRM]      Appearance map: [128, ${appearanceMapSize}, ${appearanceMapSize}]`);
    const appStats = this.analyzeArray(appearanceMap);
    console.log(`[GVRM]        stats: min=${appStats.min.toFixed(4)}, max=${appStats.max.toFixed(4)}, nonZeros=${appStats.nonZeros}`);
    
    // ===== PHASE 2: Template Gaussian Decoding (完全版ONNX) =====
    console.log('[GVRM] Phase 2: Template Gaussian decoding');
    
    if (!this.templateDecoder) {
      throw new Error('Template Decoder not initialized');
    }
    
    // 入力準備（新しいバージョン：global_embeddingは768次元）
    const decoderInput: TemplateDecoderInput = {
      projection_features: projectionFeature,  // [N, 128]
      global_embedding: idEmbedding,           // [768] ← CLS token直接
      num_vertices: vertexCount
    };
    
    console.log('[GVRM]   Input validation:');
    console.log(`[GVRM]     projection_features: [${vertexCount}, 128]`);
    console.log(`[GVRM]     global_embedding: [${idEmbedding.length}] (CLS token)`);
    console.log(`[GVRM]     num_vertices: ${vertexCount}`);
    console.log(`[GVRM]   Note: base_features and global_mapping are embedded in ONNX`);
    
    const projInputStats = this.analyzeArray(projectionFeature);
    const idInputStats = this.analyzeArray(idEmbedding);
    console.log(`[GVRM]     Projection stats: min=${projInputStats.min.toFixed(4)}, max=${projInputStats.max.toFixed(4)}, nonZeros=${projInputStats.nonZeros}`);
    console.log(`[GVRM]     Global embedding stats: min=${idInputStats.min.toFixed(4)}, max=${idInputStats.max.toFixed(4)}, nonZeros=${idInputStats.nonZeros}`);
    
    // ONNX Template Decoderで推論実行
    console.log('[GVRM]   Running Complete Template Decoder (ONNX)...');
    const templateOutput: TemplateDecoderOutput = await this.templateDecoder.forward(decoderInput);
    
    // Set positions from vertices (decoder doesn't modify positions)
    templateOutput.positions = vertices;

    // Note: idEmbeddingはRFDN Refinerでは不要になった！
    // this.idEmbedding = templateOutput.id_embedding;  // ← 削除

    // ================================================================
    // 重要: 画像範囲外の頂点のopacityをゼロにする
    // Template Decoderは base_features と global_embedding から
    // 範囲外頂点にも非ゼロのopacityを出力するため、明示的にマスク
    // ================================================================
    let invisibleCount = 0;
    if (this.visibilityMask) {
      for (let i = 0; i < vertexCount; i++) {
        if (this.visibilityMask[i] === 0) {
          templateOutput.opacities[i] = 0;
          invisibleCount++;
        }
      }
      console.log(`[GVRM] ⚠️ Opacity masked: ${invisibleCount}/${vertexCount} out-of-bounds vertices set to opacity=0`);
    }

    this.templateGaussians = {
      positions: templateOutput.positions,  // [N, 3]
      latents: templateOutput.colors,       // [N, 32]
      opacity: templateOutput.opacities,    // [N, 1]
      scale: templateOutput.scales,         // [N, 3]
      rotation: templateOutput.rotations,   // [N, 4]
      vertexCount: vertexCount
    };

    // ======== DEBUG: テストカラー注入 ========
    // レンダリングパイプラインの検証用: 各頂点にY座標に基づく虹色を設定
    // 色が正しく表示されれば、レンダリングパイプラインは正常
    if (this.debugInjectTestColors) {
      console.log('[GVRM] 🧪🧪🧪 DEBUG: Injecting TEST COLORS (rainbow gradient)');
      const colors = this.templateGaussians.latents;
      const positions = this.templateGaussians.positions;

      // Y座標の範囲を取得
      let minY = Infinity, maxY = -Infinity;
      for (let i = 0; i < vertexCount; i++) {
        const y = positions[i * 3 + 1];
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const rangeY = maxY - minY || 1;

      // 各頂点に虹色を設定（HSV色相をY座標で変化）
      for (let i = 0; i < vertexCount; i++) {
        const y = positions[i * 3 + 1];
        const t = (y - minY) / rangeY;  // [0, 1]

        // HSVからRGBへ変換（H = t, S = 1, V = 1）
        const [r, g, b] = this.hsvToRgb(t, 1.0, 1.0);

        const offset = i * 32;
        colors[offset + 0] = r;  // R (values in [0,1])
        colors[offset + 1] = g;  // G
        colors[offset + 2] = b;  // B
      }

      console.log(`[GVRM]   Y range: [${minY.toFixed(2)}, ${maxY.toFixed(2)}]`);
      console.log('[GVRM]   Applied rainbow gradient based on Y coordinate');
      console.log('[GVRM]   If colors appear correctly → rendering pipeline is OK');
      console.log('[GVRM]   If still gray → problem in rendering, not in colors');
    }
    // ======== END DEBUG ========

    console.log('[GVRM]   ✅ Template Gaussians generated (ONNX)');
    console.log(`[GVRM]      Count: ${vertexCount}`);
    console.log(`[GVRM]      Positions: [${vertexCount}, 3]`);
    console.log(`[GVRM]      Colors (latent): [${vertexCount}, 32]`);
    console.log(`[GVRM]      Opacities: [${vertexCount}, 1]`);
    console.log(`[GVRM]      Scales: [${vertexCount}, 3]`);
    console.log(`[GVRM]      Rotations: [${vertexCount}, 4]`);
    
    // Output統計
    const opacityStats = this.analyzeArray(templateOutput.opacities);
    const scaleStats = this.analyzeArray(templateOutput.scales);
    const colorStats = this.analyzeArray(templateOutput.colors);
    const rotationStats = this.analyzeArray(templateOutput.rotations);
    console.log(`[GVRM]      Opacity stats: min=${opacityStats.min.toFixed(4)}, max=${opacityStats.max.toFixed(4)}`);
    console.log(`[GVRM]      Scale stats: min=${scaleStats.min.toFixed(4)}, max=${scaleStats.max.toFixed(4)}`);
    console.log(`[GVRM]      Color stats: min=${colorStats.min.toFixed(4)}, max=${colorStats.max.toFixed(4)}`);
    console.log(`[GVRM]      Rotation stats: min=${rotationStats.min.toFixed(4)}, max=${rotationStats.max.toFixed(4)}`);
    
    // NaN/Infinity チェック
    let nanCount = 0, infCount = 0;
    for (let i = 0; i < templateOutput.opacities.length; i++) {
      if (isNaN(templateOutput.opacities[i])) nanCount++;
      if (!isFinite(templateOutput.opacities[i])) infCount++;
    }
    if (nanCount > 0 || infCount > 0) {
      console.error(`[GVRM] ❌ Template Decoder output contains invalid values!`);
      console.error(`[GVRM]    NaN count: ${nanCount}, Inf count: ${infCount}`);
    }
    
    // ===== PHASE 3: UV Pipeline (Optional) =====
    if (this.uvTriangleMapping && this.uvFeatureMapper && this.appearanceMap && this.plyData) {
      console.log('[GVRM] Phase 3: UV pipeline');
      console.log(`[GVRM]   ✅ UV Triangle Mapping: ${this.uvTriangleMapping.numValid.toLocaleString()} valid pixels`);
      console.log(`[GVRM]   Resolution: ${this.uvTriangleMapping.width}x${this.uvTriangleMapping.height}`);

      try {
        // Step 1: Map appearance features to UV space
        // Note: Use FULL PLY positions, not truncated vertices (faces reference full mesh)
        console.log('[GVRM]   Step 1: Mapping appearance features to UV space...');
        console.log(`[GVRM]   Using full PLY positions: ${this.plyData.positions.length / 3} vertices`);
        const uvFeatures128 = this.uvFeatureMapper.mapToUV(
          this.appearanceMap,
          this.plyData.positions,  // Full mesh positions, not truncated vertices!
          this.plyData.faces,
          this.uvTriangleMapping
        );

        // Step 2: Add view direction embedding (128ch → 155ch)
        console.log('[GVRM]   Step 2: Adding view direction embedding...');
        const uvFeatures155 = this.uvFeatureMapper.addViewEmbedding(
          uvFeatures128,
          [0, 0, 1]  // Canonical view direction (looking at +Z)
        );

        console.log(`[GVRM]   UV Features shape: [155, ${this.uvTriangleMapping.height}, ${this.uvTriangleMapping.width}]`);
        const uvFeatStats = this.analyzeArray(uvFeatures155);
        console.log(`[GVRM]   UV Features stats: min=${uvFeatStats.min.toFixed(4)}, max=${uvFeatStats.max.toFixed(4)}, nonZeros=${uvFeatStats.nonZeros}`);

        // Step 3: Run UV Decoder
        console.log('[GVRM]   Step 3: Running UV Point Decoder...');
        const uvGaussianOutput = await this.uvDecoder.generate(
          uvFeatures155,
          this.uvTriangleMapping.width,
          this.uvTriangleMapping.height,
          this.uvTriangleMapping
        );

        console.log(`[GVRM]   ✅ UV Decoder output: ${uvGaussianOutput.uvCount.toLocaleString()} UV Gaussians`);

        // v86: Apply activation functions to UV decoder raw outputs
        // The ONNX model outputs raw values; activations must be applied here
        console.log('[GVRM]   Applying UV Gaussian activations...');

        // Opacity: sigmoid activation → [0, 1]
        const uvOpacity = uvGaussianOutput.opacity;
        for (let i = 0; i < uvOpacity.length; i++) {
          uvOpacity[i] = 1.0 / (1.0 + Math.exp(-uvOpacity[i]));
        }
        const opStats = this.analyzeArray(uvOpacity);
        console.log(`[GVRM]     Opacity (sigmoid): [${opStats.min.toFixed(4)}, ${opStats.max.toFixed(4)}], mean=${opStats.mean.toFixed(4)}`);

        // Scale: exp activation only (no extra multiplier)
        // PLY stores log(raw_scale * face_scaling), so exp() recovers the original value.
        // face_scaling (average edge length) is already baked into the PLY data
        // (see ubody_gaussian.py:243,283 — _uv_scaling_cano = _uv_scaling * face_scaling_nn)
        // The previous * 0.05 was an incorrect extra multiplier not present in Python.
        const uvScale = uvGaussianOutput.scale;
        for (let i = 0; i < uvScale.length; i++) {
          uvScale[i] = Math.exp(uvScale[i]);
        }
        const scStats = this.analyzeArray(uvScale);
        console.log(`[GVRM]     Scale (exp): [${scStats.min.toFixed(6)}, ${scStats.max.toFixed(6)}], mean=${scStats.mean.toFixed(6)}`);

        // Rotation: normalize quaternion
        const uvRot = uvGaussianOutput.rotation;
        for (let i = 0; i < uvRot.length; i += 4) {
          const q0 = uvRot[i], q1 = uvRot[i+1], q2 = uvRot[i+2], q3 = uvRot[i+3];
          const norm = Math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3) || 1;
          uvRot[i] = q0 / norm;
          uvRot[i+1] = q1 / norm;
          uvRot[i+2] = q2 / norm;
          uvRot[i+3] = q3 / norm;
        }

        // Step 4: Transform UV Gaussians to world space
        console.log('[GVRM]   Step 4: Transforming UV Gaussians to world space...');
        const worldGaussians = this.transformUVGaussiansToWorld(
          uvGaussianOutput,
          vertices,
          this.plyData.faces
        );

        // Store UV Gaussians
        this.uvGaussians = {
          positions: worldGaussians.positions,
          latents: uvGaussianOutput.latent32ch,
          opacity: uvGaussianOutput.opacity,
          scale: uvGaussianOutput.scale,
          rotation: uvGaussianOutput.rotation,
          vertexCount: uvGaussianOutput.uvCount
        };

        console.log('[GVRM]   ✅ UV Pipeline complete');
        console.log(`[GVRM]      UV Gaussians: ${this.uvGaussians.vertexCount.toLocaleString()}`);
        const uvPosStats = this.analyzeArray(worldGaussians.positions);
        console.log(`[GVRM]      Position stats: min=${uvPosStats.min.toFixed(4)}, max=${uvPosStats.max.toFixed(4)}`);

      } catch (error) {
        console.error('[GVRM]   ❌ UV Pipeline failed:', error);
        console.log('[GVRM]   Continuing with Template Gaussians only');
        this.uvGaussians = null;
      }
    } else {
      console.log('[GVRM] Phase 3: UV pipeline skipped');
      if (!this.uvTriangleMapping) console.log('[GVRM]   Reason: No UV Triangle Mapping');
      if (!this.uvFeatureMapper) console.log('[GVRM]   Reason: No UV Feature Mapper');
      if (!this.appearanceMap) console.log('[GVRM]   Reason: No Appearance Map');
    }
    
    console.log('[GVRM] ─────────────────────────────');
  }
  
  /**
   * Setup WebGPU renderer with canonical camera
   */
  private async setupRenderer(): Promise<void> {
    if (!this.templateGaussians || !this.gpuDevice) {
      throw new Error('Missing data for renderer setup');
    }
    
    let finalGaussians: GaussianData;
    
    if (this.uvGaussians) {
      const totalCount = this.templateGaussians.vertexCount + this.uvGaussians.vertexCount;
      
      finalGaussians = {
        positions: new Float32Array(totalCount * 3),
        latents: new Float32Array(totalCount * 32),
        opacity: new Float32Array(totalCount),
        scale: new Float32Array(totalCount * 3),
        rotation: new Float32Array(totalCount * 4),
        vertexCount: totalCount
      };
      
      finalGaussians.positions.set(this.templateGaussians.positions);
      finalGaussians.latents.set(this.templateGaussians.latents);
      finalGaussians.opacity.set(this.templateGaussians.opacity);
      finalGaussians.scale.set(this.templateGaussians.scale);
      finalGaussians.rotation.set(this.templateGaussians.rotation);
      
      const tCount = this.templateGaussians.vertexCount;
      finalGaussians.positions.set(this.uvGaussians.positions, tCount * 3);
      finalGaussians.latents.set(this.uvGaussians.latents, tCount * 32);
      finalGaussians.opacity.set(this.uvGaussians.opacity, tCount);
      finalGaussians.scale.set(this.uvGaussians.scale, tCount * 3);
      finalGaussians.rotation.set(this.uvGaussians.rotation, tCount * 4);
      
      console.log('[GVRM] Merged Gaussians:', {
        template: this.templateGaussians.vertexCount,
        uv: this.uvGaussians.vertexCount,
        total: totalCount
      });
    } else {
      finalGaussians = this.templateGaussians;
      console.log('[GVRM] Using template Gaussians only:', finalGaussians.vertexCount);
    }
    
    const viewMatrix = CameraUtils.getCanonicalViewMatrix();
    const projMatrix = CameraUtils.getProjMatrix(1.0);

    const cameraConfig = {
      viewMatrix: viewMatrix,
      projMatrix: projMatrix,
      imageWidth: 512,
      imageHeight: 512
    };

    if (this.useComputeRenderer) {
      // Compute Renderer: 32チャンネル完全保持 (フラグメントシェーダーのアルファブレンディングによるチャンネル欠落を回避)
      this.gsComputeRenderer = new GuavaWebGPURendererCompute(
        this.gpuDevice,
        finalGaussians,
        cameraConfig
      );
      await this.gsComputeRenderer.waitForInit();
      console.log('[GVRM] ✅ Compute Renderer configured (32 channels preserved)');
    } else {
      // Practical Renderer: 従来のフラグメントシェーダー方式 (8チャンネル欠落)
      this.gsCoarseRenderer = new GuavaWebGPURendererPractical(
        this.gpuDevice,
        finalGaussians,
        cameraConfig
      );
      console.log('[GVRM] Practical Renderer configured (warning: 8 channels lost to alpha blending)');
    }
  }
  
  // ===== GPU Device Recovery =====

  private setupDeviceLostHandler(): void {
    if (!this.gpuDevice) return;
    this.gpuDevice.lost.then((info) => {
      console.warn(`[GVRM] ⚠️ GPU device lost: ${info.message} (reason: ${info.reason})`);
      this.deviceLost = true;
      this.isRunning = false;
      if (this.frameId !== null) {
        cancelAnimationFrame(this.frameId);
        this.frameId = null;
      }
    });
  }

  private setupVisibilityHandler(): void {
    this.boundVisibilityHandler = () => this.onVisibilityChange();
    document.addEventListener('visibilitychange', this.boundVisibilityHandler);
  }

  private onVisibilityChange(): void {
    if (document.visibilityState === 'hidden') {
      // Tab going to background: pause render loop to avoid TDR
      console.log('[GVRM] Tab hidden - pausing render loop');
      this.isRunning = false;
      if (this.frameId !== null) {
        cancelAnimationFrame(this.frameId);
        this.frameId = null;
      }
    } else if (document.visibilityState === 'visible') {
      console.log('[GVRM] Tab visible - resuming');
      if (this.deviceLost) {
        console.log('[GVRM] Device was lost - starting recovery...');
        this.recoverFromDeviceLost();
      } else if (this.initialized && !this.isRunning) {
        this.isRunning = true;
        this.renderFrame();
      }
    }
  }

  private async recoverFromDeviceLost(): Promise<void> {
    if (this.recovering) return;
    this.recovering = true;

    try {
      console.log('[GVRM] 🔄 Recovering GPU device...');

      // Cleanup old GPU resources
      this.readbackBuffers.forEach(b => { try { b.destroy(); } catch (_) {} });
      this.readbackBuffers = [];
      if (this.gsComputeRenderer) {
        try { this.gsComputeRenderer.destroy(); } catch (_) {}
        this.gsComputeRenderer = null;
      }
      if (this.gsCoarseRenderer) {
        try { this.gsCoarseRenderer.destroy(); } catch (_) {}
        this.gsCoarseRenderer = null;
      }

      // Request new device
      if (!navigator.gpu) throw new Error('WebGPU not supported');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No GPU adapter');

      const requiredLimits: any = {};
      const adapterLimits = adapter.limits;

      if (adapterLimits.maxColorAttachmentBytesPerSample >= 128) {
        requiredLimits.maxColorAttachmentBytesPerSample = 128;
      }
      const neededSize = 512 * 1024 * 1024;
      if (adapterLimits.maxStorageBufferBindingSize >= neededSize) {
        requiredLimits.maxStorageBufferBindingSize = neededSize;
      }
      if (adapterLimits.maxBufferSize >= neededSize) {
        requiredLimits.maxBufferSize = neededSize;
      }

      this.gpuDevice = await adapter.requestDevice({ requiredLimits });
      this.deviceLost = false;
      this.setupDeviceLostHandler();

      console.log('[GVRM] ✅ New GPU device acquired');

      // Rebuild renderer with existing Gaussian data
      this.initReadbackBuffers(512, 512);
      await this.setupRenderer();
      this.frameCount = 0;

      console.log('[GVRM] ✅ GPU recovery complete - resuming render');
      this.isRunning = true;
      this.renderFrame();
    } catch (error) {
      console.error('[GVRM] ❌ GPU recovery failed:', error);
    } finally {
      this.recovering = false;
    }
  }

  /**
   * Render loop (SimpleUNet Refiner版)
   * - 入力: Gaussian出力を[0, 1]に正規化
   * - 出力: sigmoid適用後の[0, 1] RGB
   */
  private lastRenderTime: number = 0;
  private readonly MIN_FRAME_INTERVAL = 500; // 最低500ms間隔 (2fps)
  
  private renderFrame = async (): Promise<void> => {
    if (!this.isRunning || !this.initialized) return;
    
    // フレームレート制限
    const now = performance.now();
    if (now - this.lastRenderTime < this.MIN_FRAME_INTERVAL) {
      this.frameId = requestAnimationFrame(this.renderFrame);
      return;
    }
    this.lastRenderTime = now;
    
    try {
      this.frameCount++;

      let coarseFeatures: Float32Array;

      if (this.useComputeRenderer && this.gsComputeRenderer) {
        // Compute Renderer: 32チャンネル完全保持
        this.gsComputeRenderer.sort();
        this.gsComputeRenderer.render();

        // v86: Try unified buffer first (GPU splatting), fall back to legacy buffers (CPU splatting)
        const unifiedBuffer = this.gsComputeRenderer.getUnifiedOutputBuffer();
        if (unifiedBuffer) {
          coarseFeatures = await this.convertUnifiedBufferToFloat32Array(unifiedBuffer);
        } else {
          const outputBuffers = this.gsComputeRenderer.getOutputBuffers();
          coarseFeatures = await this.convertBuffersToFloat32Array(outputBuffers);
        }

        if (this.frameCount === 1) {
          console.log('[GVRM] 🚀 Using Compute Renderer (all 32 channels preserved)');
        }
      } else if (this.gsCoarseRenderer) {
        // Practical Renderer: 従来方式 (8チャンネル欠落あり)
        this.gsCoarseRenderer.sort();
        this.gsCoarseRenderer.render();

        const outputTextures = this.gsCoarseRenderer.getOutputTextures();
        coarseFeatures = await this.convertTexturesToFloat32Array(outputTextures);

        if (this.frameCount === 1) {
          console.log('[GVRM] ⚠️ Using Practical Renderer (8 channels interpolated)');
        }
      } else {
        throw new Error('No renderer available');
      }

      let displayRGB: Float32Array;

      if (this.debugBypassRFDN) {
        // DEBUG: RFDNをバイパスして最初3チャンネルを直接RGBとして表示
        // Template Decoder で ch 0-2 に sigmoid を適用済み
        // 強化されたコントラスト補正を適用
        const width = 512, height = 512;
        displayRGB = new Float32Array(width * height * 3);
        const pixelCount = width * height;

        // 🔧 FIX v76: GLOBAL min/max across ALL 3 channels (preserves color differences)
        // Per-channel stretching was destroying color info by normalizing each channel independently
        let globalMin = Infinity, globalMax = -Infinity;
        const chStats = [];
        for (let ch = 0; ch < 3; ch++) {
          let chMin = Infinity, chMax = -Infinity, chSum = 0, count = 0;
          for (let p = 0; p < pixelCount; p++) {
            const val = coarseFeatures[ch * pixelCount + p];
            if (isFinite(val) && val > 0.001) {  // 背景(0)を除外
              if (val < chMin) chMin = val;
              if (val > chMax) chMax = val;
              chSum += val;
              count++;
              // Track global min/max
              if (val < globalMin) globalMin = val;
              if (val > globalMax) globalMax = val;
            }
          }
          chStats.push({ min: chMin, max: chMax, mean: count > 0 ? chSum / count : 0.5, count });
        }

        const globalRange = globalMax - globalMin;

        // CHW → HWC変換 + GLOBAL コントラスト強調（色差を保持）
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const p = y * width + x;
            for (let c = 0; c < 3; c++) {
              const srcIdx = c * pixelCount + p;
              const dstIdx = p * 3 + c;
              let val = coarseFeatures[srcIdx];

              // 背景（0付近）はそのまま
              if (val < 0.001) {
                displayRGB[dstIdx] = 0;
                continue;
              }

              // 🔧 GLOBAL contrast stretch (same scale for all channels → preserves color!)
              if (globalRange > 0.01) {
                val = (val - globalMin) / globalRange;
              }

              // ガンマ補正（明るさ調整）- γ=0.6 で明るく（0.5は強すぎた）
              val = Math.pow(val, 0.6);

              displayRGB[dstIdx] = Math.max(0, Math.min(1, val));
            }
          }
        }

        if (this.frameCount === 1) {
          console.log('[GVRM] 🔧 DEBUG: Bypassing RFDN, using ch 0-2 with GLOBAL contrast');
          console.log(`[GVRM]   🔧 FIX v76: Using GLOBAL min/max to preserve color differences`);
          console.log(`[GVRM]   Global range: [${globalMin.toFixed(4)}, ${globalMax.toFixed(4)}]`);
          console.log(`[GVRM]   Per-channel stats (for reference):`);
          for (let ch = 0; ch < 3; ch++) {
            const chName = ['R', 'G', 'B'][ch];
            console.log(`[GVRM]   Ch ${ch} (${chName}): [${chStats[ch].min.toFixed(4)}, ${chStats[ch].max.toFixed(4)}], mean=${chStats[ch].mean.toFixed(4)}, pixels=${chStats[ch].count}`);
          }
          console.log('[GVRM]   Applied: GLOBAL contrast stretch [globalMin,globalMax]→[0,1] + gamma=0.6');

          // ======== 🔍🔍🔍 RGB CROSS-CHANNEL ANALYSIS ========
          // Check if R≈G≈B (causes gray output)
          console.log('[GVRM] 🔍🔍🔍 RGB CROSS-CHANNEL ANALYSIS:');

          // Sample 20 non-background pixels
          const samplePixels: {x: number, y: number, r: number, g: number, b: number}[] = [];
          for (let p = 0; p < pixelCount && samplePixels.length < 20; p++) {
            const r = coarseFeatures[0 * pixelCount + p];
            const g = coarseFeatures[1 * pixelCount + p];
            const b = coarseFeatures[2 * pixelCount + p];
            if (r > 0.01 || g > 0.01 || b > 0.01) {  // Non-background
              samplePixels.push({
                x: p % width,
                y: Math.floor(p / width),
                r, g, b
              });
            }
          }

          console.log('[GVRM]   Sample rendered pixels (RAW before contrast):');
          for (const sp of samplePixels.slice(0, 10)) {
            const diff_rg = Math.abs(sp.r - sp.g);
            const diff_gb = Math.abs(sp.g - sp.b);
            const diff_rb = Math.abs(sp.r - sp.b);
            console.log(`[GVRM]     (${sp.x},${sp.y}): R=${sp.r.toFixed(4)}, G=${sp.g.toFixed(4)}, B=${sp.b.toFixed(4)} | diff: R-G=${diff_rg.toFixed(4)}, G-B=${diff_gb.toFixed(4)}, R-B=${diff_rb.toFixed(4)}`);
          }

          // Compute variance of (R-G), (G-B), (R-B) for ALL non-background pixels
          let sumDiffRG = 0, sumDiffGB = 0, sumDiffRB = 0;
          let sumDiffRG2 = 0, sumDiffGB2 = 0, sumDiffRB2 = 0;
          let countNonBg = 0;

          for (let p = 0; p < pixelCount; p++) {
            const r = coarseFeatures[0 * pixelCount + p];
            const g = coarseFeatures[1 * pixelCount + p];
            const b = coarseFeatures[2 * pixelCount + p];
            if (r > 0.001 || g > 0.001 || b > 0.001) {
              const dRG = r - g;
              const dGB = g - b;
              const dRB = r - b;
              sumDiffRG += dRG;
              sumDiffGB += dGB;
              sumDiffRB += dRB;
              sumDiffRG2 += dRG * dRG;
              sumDiffGB2 += dGB * dGB;
              sumDiffRB2 += dRB * dRB;
              countNonBg++;
            }
          }

          const meanDiffRG = countNonBg > 0 ? sumDiffRG / countNonBg : 0;
          const meanDiffGB = countNonBg > 0 ? sumDiffGB / countNonBg : 0;
          const meanDiffRB = countNonBg > 0 ? sumDiffRB / countNonBg : 0;
          const varDiffRG = countNonBg > 0 ? sumDiffRG2 / countNonBg - meanDiffRG * meanDiffRG : 0;
          const varDiffGB = countNonBg > 0 ? sumDiffGB2 / countNonBg - meanDiffGB * meanDiffGB : 0;
          const varDiffRB = countNonBg > 0 ? sumDiffRB2 / countNonBg - meanDiffRB * meanDiffRB : 0;

          console.log(`[GVRM]   Cross-channel differences (${countNonBg} non-bg pixels):`);
          console.log(`[GVRM]     R-G: mean=${meanDiffRG.toFixed(6)}, σ=${Math.sqrt(varDiffRG).toFixed(6)}`);
          console.log(`[GVRM]     G-B: mean=${meanDiffGB.toFixed(6)}, σ=${Math.sqrt(varDiffGB).toFixed(6)}`);
          console.log(`[GVRM]     R-B: mean=${meanDiffRB.toFixed(6)}, σ=${Math.sqrt(varDiffRB).toFixed(6)}`);

          if (Math.sqrt(varDiffRG) < 0.01 && Math.sqrt(varDiffGB) < 0.01) {
            console.log('[GVRM]   ⚠️⚠️⚠️ PROBLEM DETECTED: R≈G≈B (colors are nearly identical!)');
            console.log('[GVRM]   This explains the gray output. The rendering pipeline is losing color diversity.');
          }
          // ======== END RGB ANALYSIS ========
        }
      } else {
        // Neural Refiner (SimpleUNet): 32ch特徴マップをそのまま入力（正規化なし）
        const stats = this.analyzeArray(coarseFeatures);

        // Guard: skip refiner if GPU readback returned degenerate data
        if (stats.min === stats.max || !isFinite(stats.min) || !isFinite(stats.max)) {
          console.warn(`[GVRM] ⚠️ Degenerate coarse features (min=max=${stats.min.toFixed(4)}), skipping frame`);
          this.frameId = requestAnimationFrame(this.renderFrame);
          return;
        }
        if (this.frameCount === 1) {
          console.log(`[GVRM] Coarse features (raw, no normalization): [${stats.min.toFixed(4)}, ${stats.max.toFixed(4)}]`);
        }

        // Pass raw coarse features directly to Refiner — no normalization.
        // Python pipeline (gaussian_render.py:73) passes rasterizer output directly:
        //   refine_images = self.nerual_refiner(rendered_images)
        // The previous normalizeToZeroOne was not present in the Python pipeline
        // and destroyed the learned feature distribution.
        displayRGB = await this.neuralRefiner.process(coarseFeatures);

        // 🔍 v77: Refiner output debug
        if (this.frameCount === 1) {
          console.log('[GVRM] 🚀 SimpleUNet Refiner OUTPUT:');
          const refinerStats = this.analyzeArray(displayRGB);
          console.log(`[GVRM]   Range: [${refinerStats.min.toFixed(4)}, ${refinerStats.max.toFixed(4)}]`);

          // RGB channel analysis
          const pixelCount = 512 * 512;
          let rSum = 0, gSum = 0, bSum = 0, count = 0;
          for (let i = 0; i < pixelCount; i++) {
            const r = displayRGB[i * 3 + 0];
            const g = displayRGB[i * 3 + 1];
            const b = displayRGB[i * 3 + 2];
            if (r > 0.01 || g > 0.01 || b > 0.01) {
              rSum += r; gSum += g; bSum += b; count++;
            }
          }
          if (count > 0) {
            console.log(`[GVRM]   RGB means (non-bg): R=${(rSum/count).toFixed(4)}, G=${(gSum/count).toFixed(4)}, B=${(bSum/count).toFixed(4)}`);
          }

          // Sample pixels
          console.log('[GVRM]   Sample pixels:');
          for (let y = 200; y < 210; y++) {
            const x = 256;
            const idx = (y * 512 + x) * 3;
            const r = displayRGB[idx], g = displayRGB[idx+1], b = displayRGB[idx+2];
            if (r > 0.01 || g > 0.01 || b > 0.01) {
              console.log(`[GVRM]     (${x},${y}): R=${r.toFixed(4)}, G=${g.toFixed(4)}, B=${b.toFixed(4)}`);
            }
          }
        }
      }

      // =============== v81: ガンマ補正とトーンマッピング ===============
      // AIの出力(Linear空間)を、モニタ表示用(sRGB空間)に明るく補正
      // これがないと、正しい色データでも「暗いグレー」に見えてしまう
      const exposureBoost = 1.3;  // 露出補正（明るさブースト）
      const gamma = 2.2;          // sRGB標準ガンマ

      for (let i = 0; i < displayRGB.length; i++) {
        let val = displayRGB[i];

        // 1. 露出補正 (Exposure Boost): 全体を明るくする
        val = val * exposureBoost;

        // 2. ガンマ補正 (Linear → sRGB): 暗部を持ち上げ、本来の色味を引き出す
        // 公式: val = val ^ (1 / gamma)
        if (val > 0) {
          val = Math.pow(val, 1.0 / gamma);
        }

        // クランプ [0, 1]
        displayRGB[i] = Math.max(0, Math.min(1, val));
      }

      if (this.frameCount === 1) {
        console.log('[GVRM] 🔧 v81: Applied gamma correction (Linear → sRGB)');
        console.log(`[GVRM]   Exposure boost: ${exposureBoost}x, Gamma: ${gamma}`);
        const finalStats = this.analyzeArray(displayRGB);
        console.log(`[GVRM]   After gamma: [${finalStats.min.toFixed(4)}, ${finalStats.max.toFixed(4)}]`);
      }
      // =============================================================

      if (this.webglDisplay) {
        this.webglDisplay.display(displayRGB, this.frameCount);
      }

      if (this.frameCount === 1) {
        const coarseStats = this.analyzeArray(coarseFeatures.slice(0, 10000));
        const displayStats = this.analyzeArray(displayRGB.slice(0, 10000));
        console.log('[GVRM] First frame stats:');
        console.log(`  Coarse features (32ch): min=${coarseStats.min.toFixed(4)}, max=${coarseStats.max.toFixed(4)}`);
        console.log(`  Display RGB: min=${displayStats.min.toFixed(4)}, max=${displayStats.max.toFixed(4)}`);
        if (!this.debugBypassRFDN) {
          console.log(`  🚀 SimpleUNet Refiner: Raw input (no normalization), output is final RGB`);
        }
      }

    } catch (error) {
      const msg = (error as Error)?.message || '';
      if (msg.includes('lost') || msg.includes('destroyed') || msg.includes('Device')) {
        console.warn('[GVRM] Render error (device lost):', msg);
        this.deviceLost = true;
        this.isRunning = false;
        // Trigger recovery immediately if tab is visible
        if (document.visibilityState === 'visible') {
          this.recoverFromDeviceLost();
        }
        return;
      }
      console.error('[GVRM] Render error:', error);
      this.isRunning = false;
      return;
    }
    
    this.frameId = requestAnimationFrame(this.renderFrame);
  };
  
  // ===== Helper Methods =====
  
  private initReadbackBuffers(width: number, height: number): void {
    if (!this.gpuDevice) return;
    
    this.readbackBuffers.forEach(b => b.destroy());
    this.readbackBuffers = [];
    
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    
    for (let i = 0; i < 8; i++) {
      this.readbackBuffers.push(this.gpuDevice.createBuffer({
        size: bytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      }));
    }
    
    this.coarseFeatureArray = new Float32Array(width * height * 32);
  }
  
  private async convertTexturesToFloat32Array(textures: GPUTexture[]): Promise<Float32Array> {
    if (!this.gpuDevice || !this.coarseFeatureArray) {
      throw new Error('Buffers not initialized');
    }
    
    const width = 512, height = 512;
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    
    const commandEncoder = this.gpuDevice.createCommandEncoder();
    for (let i = 0; i < 8; i++) {
      commandEncoder.copyTextureToBuffer(
        { texture: textures[i] },
        { buffer: this.readbackBuffers[i], bytesPerRow: bytesPerRow },
        [width, height]
      );
    }
    this.gpuDevice.queue.submit([commandEncoder.finish()]);
    
    await Promise.all(this.readbackBuffers.map(buf => buf.mapAsync(GPUMapMode.READ)));
    
    // Debug: 各MRTの統計
    const mrtStats: string[] = [];
    
    for (let i = 0; i < 8; i++) {
      const mappedRange = this.readbackBuffers[i].getMappedRange();
      const source = new Uint16Array(mappedRange);
      
      // Debug: 最初のMRTの生データサンプル
      if (i === 0 && this.frameCount === 1) {
        console.log('[GVRM] MRT0 raw Uint16 samples:', Array.from(source.slice(0, 20)));
      }
      
      let mrtMin = Infinity, mrtMax = -Infinity, mrtNaN = 0, mrtInf = 0;
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          for (let c = 0; c < 4; c++) {
            const srcIdx = y * (bytesPerRow / 2) + x * 4 + c;
            const dstIdx = (i * 4 + c) * width * height + y * width + x;
            const val = this.float16ToFloat32(source[srcIdx]);
            this.coarseFeatureArray[dstIdx] = val;
            
            if (isNaN(val)) mrtNaN++;
            else if (!isFinite(val)) mrtInf++;
            else {
              if (val < mrtMin) mrtMin = val;
              if (val > mrtMax) mrtMax = val;
            }
          }
        }
      }
      
      if (this.frameCount === 1) {
        mrtStats.push(`MRT${i}: [${mrtMin.toFixed(2)}, ${mrtMax.toFixed(2)}] NaN=${mrtNaN} Inf=${mrtInf}`);
      }

      this.readbackBuffers[i].unmap();
    }

    // ================================================================
    // 応急処置: アルファチャンネルに上書きされた欠落チャンネルを補間
    // MRTの4番目のチャンネル(A)はblending用alphaなので、
    // 実際のlatentチャンネル3,7,11,15,19,23,27,31が欠落している
    // 隣接チャンネルからコピーして補間
    // ================================================================
    const pixelCount = width * height;
    const missingChannels = [3, 7, 11, 15, 19, 23, 27, 31];
    for (const ch of missingChannels) {
      const srcCh = ch - 1;  // 隣接チャンネルからコピー
      const srcOffset = srcCh * pixelCount;
      const dstOffset = ch * pixelCount;
      for (let p = 0; p < pixelCount; p++) {
        this.coarseFeatureArray[dstOffset + p] = this.coarseFeatureArray[srcOffset + p];
      }
    }

    if (this.frameCount === 1) {
      console.log('[GVRM] ⚠️ Missing channel fix applied: ch 3,7,11,15,19,23,27,31 interpolated from adjacent');
    }
    
    if (this.frameCount === 1) {
      console.log('[GVRM] MRT readback stats:');
      mrtStats.forEach(s => console.log('  ' + s));
    }
    
    return this.coarseFeatureArray;
  }
  
  /**
   * Compute Renderer用: ストレージバッファから32チャンネルを読み出し
   * Practical Rendererと異なり、全32チャンネルが正しく保持されている
   */
  /**
   * v86: Read unified output buffer (32ch interleaved per pixel) into CHW Float32Array
   * Unified layout: [pixel0_ch0..ch31, pixel1_ch0..ch31, ...]
   * Output layout: CHW [ch0_pixel0..pixelN, ch1_pixel0..pixelN, ...]
   */
  private async convertUnifiedBufferToFloat32Array(buffer: GPUBuffer): Promise<Float32Array> {
    if (!this.gpuDevice) throw new Error('GPU device not initialized');

    const width = 512, height = 512;
    const pixelCount = width * height;
    const bufferSize = pixelCount * 32 * 4;

    if (!this.coarseFeatureArray || this.coarseFeatureArray.length !== pixelCount * 32) {
      this.coarseFeatureArray = new Float32Array(pixelCount * 32);
    }

    // Create readback buffer
    const readback = this.gpuDevice.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const commandEncoder = this.gpuDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, readback, 0, bufferSize);
    this.gpuDevice.queue.submit([commandEncoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(readback.getMappedRange());

    // Convert from interleaved (pixel-major) to CHW (channel-major)
    let globalMin = Infinity, globalMax = -Infinity, nanCount = 0;

    for (let p = 0; p < pixelCount; p++) {
      const srcBase = p * 32;
      for (let ch = 0; ch < 32; ch++) {
        const val = source[srcBase + ch];
        const dstIdx = ch * pixelCount + p;
        this.coarseFeatureArray[dstIdx] = val;

        if (isNaN(val)) nanCount++;
        else {
          if (val < globalMin) globalMin = val;
          if (val > globalMax) globalMax = val;
        }
      }
    }

    if (this.frameCount === 1) {
      console.log(`[GVRM] Unified buffer stats (32 channels): [${globalMin.toFixed(4)}, ${globalMax.toFixed(4)}] NaN=${nanCount}`);
    }

    readback.unmap();
    readback.destroy();

    return this.coarseFeatureArray;
  }

  private async convertBuffersToFloat32Array(buffers: GPUBuffer[]): Promise<Float32Array> {
    if (!this.gpuDevice) {
      throw new Error('GPU device not initialized');
    }

    const width = 512, height = 512;
    const pixelCount = width * height;

    // 出力配列を初期化
    if (!this.coarseFeatureArray || this.coarseFeatureArray.length !== pixelCount * 32) {
      this.coarseFeatureArray = new Float32Array(pixelCount * 32);
    }

    // 読み出し用バッファを作成
    const readbackSize = pixelCount * 4 * 4;  // 4 floats x 4 bytes
    const readbackBuffers: GPUBuffer[] = [];

    const commandEncoder = this.gpuDevice.createCommandEncoder();
    for (let i = 0; i < 8; i++) {
      const readback = this.gpuDevice.createBuffer({
        size: readbackSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      commandEncoder.copyBufferToBuffer(buffers[i], 0, readback, 0, readbackSize);
      readbackBuffers.push(readback);
    }
    this.gpuDevice.queue.submit([commandEncoder.finish()]);

    // 非同期で読み出し
    await Promise.all(readbackBuffers.map(buf => buf.mapAsync(GPUMapMode.READ)));

    // Debug: 統計情報
    const bufStats: string[] = [];

    for (let i = 0; i < 8; i++) {
      const mappedRange = readbackBuffers[i].getMappedRange();
      const source = new Float32Array(mappedRange);

      let bufMin = Infinity, bufMax = -Infinity, bufNaN = 0;

      for (let p = 0; p < pixelCount; p++) {
        for (let c = 0; c < 4; c++) {
          const srcIdx = p * 4 + c;
          const channel = i * 4 + c;
          const dstIdx = channel * pixelCount + p;
          const val = source[srcIdx];
          this.coarseFeatureArray[dstIdx] = val;

          if (isNaN(val)) bufNaN++;
          else {
            if (val < bufMin) bufMin = val;
            if (val > bufMax) bufMax = val;
          }
        }
      }

      if (this.frameCount === 1) {
        bufStats.push(`Buf${i}: [${bufMin.toFixed(2)}, ${bufMax.toFixed(2)}] NaN=${bufNaN}`);
      }

      readbackBuffers[i].unmap();
      readbackBuffers[i].destroy();
    }

    if (this.frameCount === 1) {
      console.log('[GVRM] Compute Renderer buffer stats (32 channels, no loss):');
      bufStats.forEach(s => console.log('  ' + s));
    }

    return this.coarseFeatureArray;
  }

  private float16ToFloat32(f16: number): number {
    const sign = (f16 & 0x8000) >> 15;
    const exponent = (f16 & 0x7C00) >> 10;
    const mantissa = f16 & 0x03FF;
    
    if (exponent === 0) {
      return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
    }
    if (exponent === 31) {
      return mantissa ? NaN : (sign ? -Infinity : Infinity);
    }
    
    return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  }
  
  private async loadPLY(url: string): Promise<PLYData> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load PLY: ${url}`);
    const buffer = await response.arrayBuffer();
    return this.parsePLY(buffer);
  }
  
  private parsePLY(buffer: ArrayBuffer): PLYData {
    const headerText = new TextDecoder('utf-8').decode(buffer.slice(0, 4000));
    const headerLines = headerText.split('\n');
    let vertexCount = 0, faceCount = 0, headerEnd = 0;
    let vertexProps: { name: string, type: string }[] = [];
    
    for (const line of headerLines) {
      const trimmed = line.trim();
      if (trimmed === 'end_header') {
        headerEnd += trimmed.length + 1;
        while (headerEnd < buffer.byteLength && new Uint8Array(buffer)[headerEnd] < 32) {
          headerEnd++;
        }
        break;
      }
      headerEnd += line.length + 1;
      
      if (trimmed.startsWith('element vertex')) {
        vertexCount = parseInt(trimmed.split(/\s+/)[2]);
      } else if (trimmed.startsWith('element face')) {
        faceCount = parseInt(trimmed.split(/\s+/)[2]);
      } else if (trimmed.startsWith('property')) {
        if (vertexCount > 0 && faceCount === 0) {
          const parts = trimmed.split(/\s+/);
          vertexProps.push({ type: parts[1], name: parts[2] });
        }
      }
    }
    
    const dataView = new DataView(buffer);
    let offset = headerEnd;
    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);
    
    for (let i = 0; i < vertexCount; i++) {
      for (const prop of vertexProps) {
        let val = 0;
        if (prop.type === 'float' || prop.type === 'float32') {
          val = dataView.getFloat32(offset, true);
          offset += 4;
        } else if (prop.type === 'uchar' || prop.type === 'uint8') {
          val = dataView.getUint8(offset);
          offset += 1;
        } else {
          offset += 4;
        }
        
        if (prop.name === 'x') positions[i * 3 + 0] = val;
        if (prop.name === 'y') positions[i * 3 + 1] = val;
        if (prop.name === 'z') positions[i * 3 + 2] = val;
        if (prop.name === 'nx') normals[i * 3 + 0] = val;
        if (prop.name === 'ny') normals[i * 3 + 1] = val;
        if (prop.name === 'nz') normals[i * 3 + 2] = val;
        if (prop.name === 'red') colors[i * 3 + 0] = val / 255.0;
        if (prop.name === 'green') colors[i * 3 + 1] = val / 255.0;
        if (prop.name === 'blue') colors[i * 3 + 2] = val / 255.0;
      }
    }
    
    const faces = new Uint32Array(faceCount * 3);
    for (let i = 0; i < faceCount; i++) {
      const numVerts = dataView.getUint8(offset);
      offset += 1;
      if (numVerts === 3) {
        faces[i * 3 + 0] = dataView.getUint32(offset, true); offset += 4;
        faces[i * 3 + 1] = dataView.getUint32(offset, true); offset += 4;
        faces[i * 3 + 2] = dataView.getUint32(offset, true); offset += 4;
      } else {
        offset += numVerts * 4;
      }
    }
    
    return { positions, vertexCount, faces, faceCount, normals, colors };
  }
  
  private async loadBinary(url: string): Promise<Float32Array> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load binary: ${url}`);
    return new Float32Array(await response.arrayBuffer());
  }
  
  /**
   * Normalize 32-channel features to [-1, 1] range for SimpleUNet
   * v79: Changed from [0,1] to [-1,1] - most image generation networks expect zero-centered input
   */
  private normalizeToZeroOne(features: Float32Array, logStats: boolean = false): Float32Array {
    const normalized = new Float32Array(features.length);

    // Compute min/max across all channels
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < features.length; i++) {
      const v = features[i];
      if (isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    const range = max - min || 1;

    // Normalize to [-1, 1] (NOT [0, 1])
    // Formula: normalized = ((v - min) / range) * 2.0 - 1.0
    for (let i = 0; i < features.length; i++) {
      const v = features[i];
      if (isFinite(v)) {
        normalized[i] = ((v - min) / range) * 2.0 - 1.0;
      } else {
        normalized[i] = 0;  // center value for [-1, 1]
      }
    }

    if (logStats) {
      console.log(`[GVRM] 🔧 v79: Normalizing features: [${min.toFixed(4)}, ${max.toFixed(4)}] → [-1, 1]`);
    }

    return normalized;
  }

  private analyzeArray(arr: Float32Array): { min: number; max: number; mean: number; nonZeros: number } {
    let min = Infinity, max = -Infinity, sum = 0, nonZeros = 0;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (Math.abs(v) > 0.001) nonZeros++;
    }
    return { min, max, mean: sum / arr.length, nonZeros };
  }

  /**
   * HSV to RGB 変換（デバッグ用）
   * @param h Hue [0, 1]
   * @param s Saturation [0, 1]
   * @param v Value [0, 1]
   * @returns [r, g, b] each in [0, 1]
   */
  private hsvToRgb(h: number, s: number, v: number): [number, number, number] {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);

    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      case 5: return [v, p, q];
      default: return [v, t, p];
    }
  }
  
  /**
   * Transform UV Gaussians from UV space to world space
   * Uses barycentric coordinates to interpolate world positions from mesh vertices
   */
  private transformUVGaussiansToWorld(
    uvOutput: import('./uv-decoder').UVGaussianOutput,
    vertices: Float32Array,
    faces: Uint32Array
  ): { positions: Float32Array } {
    const numGaussians = uvOutput.uvCount;
    const worldPositions = new Float32Array(numGaussians * 3);

    console.log(`[GVRM] Transforming ${numGaussians.toLocaleString()} UV Gaussians to world space...`);

    let validCount = 0;
    let invalidTriangleCount = 0;

    for (let i = 0; i < numGaussians; i++) {
      const triIdx = uvOutput.triangleIndices[i];
      const bary0 = uvOutput.barycentricCoords[i * 3 + 0];
      const bary1 = uvOutput.barycentricCoords[i * 3 + 1];
      const bary2 = uvOutput.barycentricCoords[i * 3 + 2];

      // Local position delta from UV decoder
      const localDx = uvOutput.localPositions[i * 3 + 0];
      const localDy = uvOutput.localPositions[i * 3 + 1];
      const localDz = uvOutput.localPositions[i * 3 + 2];

      // Get triangle vertex indices
      const v0Idx = faces[triIdx * 3 + 0];
      const v1Idx = faces[triIdx * 3 + 1];
      const v2Idx = faces[triIdx * 3 + 2];

      // Validate indices
      if (v0Idx * 3 + 2 >= vertices.length ||
          v1Idx * 3 + 2 >= vertices.length ||
          v2Idx * 3 + 2 >= vertices.length) {
        invalidTriangleCount++;
        // Set to origin
        worldPositions[i * 3 + 0] = 0;
        worldPositions[i * 3 + 1] = 0;
        worldPositions[i * 3 + 2] = 0;
        continue;
      }

      // Get vertex world positions
      const v0x = vertices[v0Idx * 3 + 0];
      const v0y = vertices[v0Idx * 3 + 1];
      const v0z = vertices[v0Idx * 3 + 2];

      const v1x = vertices[v1Idx * 3 + 0];
      const v1y = vertices[v1Idx * 3 + 1];
      const v1z = vertices[v1Idx * 3 + 2];

      const v2x = vertices[v2Idx * 3 + 0];
      const v2y = vertices[v2Idx * 3 + 1];
      const v2z = vertices[v2Idx * 3 + 2];

      // Barycentric interpolation: world position = Σ(bary_i * v_i)
      const baseX = bary0 * v0x + bary1 * v1x + bary2 * v2x;
      const baseY = bary0 * v0y + bary1 * v1y + bary2 * v2y;
      const baseZ = bary0 * v0z + bary1 * v1z + bary2 * v2z;

      // Add local position delta
      // Note: local delta is typically small and in tangent space
      // For now, add directly (may need rotation to world space in future)
      worldPositions[i * 3 + 0] = baseX + localDx;
      worldPositions[i * 3 + 1] = baseY + localDy;
      worldPositions[i * 3 + 2] = baseZ + localDz;

      validCount++;
    }

    console.log(`[GVRM]   Transformed: ${validCount.toLocaleString()} valid, ${invalidTriangleCount} invalid triangles`);

    return { positions: worldPositions };
  }

  /**
   * Update lip sync animation based on audio level
   * TODO: Implement actual lip sync deformation based on SMPL-X jaw parameters
   * @param audioLevel Audio level (0-1) for lip sync animation
   */
  updateLipSync(audioLevel: number): void {
    // Stub implementation - lip sync requires modifying SMPL-X jaw parameters
    // and re-running the Gaussian splatting pipeline
    // For now, this prevents crashes when called from external code
  }

  dispose(): void {
    this.isRunning = false;
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }
    if (this.boundVisibilityHandler) {
      document.removeEventListener('visibilitychange', this.boundVisibilityHandler);
      this.boundVisibilityHandler = null;
    }

    this.readbackBuffers.forEach(b => b.destroy());
    this.gsCoarseRenderer?.destroy();
    this.webglDisplay?.dispose();
    this.imageEncoder.dispose();
    if (this.templateDecoder) {
      // Template Decoderにdisposeメソッドがあれば呼び出す
    }
    this.inverseTextureMapper.dispose();
    this.uvDecoder.dispose();
    this.neuralRefiner.dispose();
    
    console.log('[GVRM] Disposed');
  }
}
