// gvrm.ts
// 修正版 v73: TemplateDecoderWebGPU対応 + scaleクランプ修正
// - RFDN Refiner (178KB軽量モデル)
// - TemplateDecoderWebGPU (scaleオーバーフロー修正済み)

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
 * v72: RFDN Refiner対応版
 * - 蒸留済み軽量Neural Refiner (178KB, 元の630倍圧縮)
 * - idEmbedding不要 (32ch入力のみ)
 * 
 * Pipeline:
 * 1. Image Encoder: Image → Features (projection + ID embedding)
 * 2. Template Decoder: 3入力 (projection, base, id) → Template Gaussians
 * 3. Inverse Texture Mapping: Image → UV features (optional)
 * 4. UV Decoder: UV features → UV Gaussians (optional)
 * 5. WebGPU Rendering: Gaussians → Coarse feature map (32ch)
 * 6. RFDN Refiner: Coarse 32ch → Refined RGB (idEmb不要！)
 */
export class GVRM {
  // Asset paths
  private templatePath: string = '/assets/avatar_web.ply';
  private imagePath: string = '/assets/source.png';
  private cameraConfigPath: string = '/assets/source_camera.json';
  private uvCoordsPath: string = '/assets/uv_coords.bin';
  private container: HTMLElement | null = null;
  private useWebGPURefiner: boolean = true;
  
  // Core modules
  private imageEncoder: ImageEncoder;
  private templateDecoder: TemplateDecoderWebGPU | null = null;
  private inverseTextureMapper: InverseTextureMapper;
  private uvDecoder: UVDecoder;
  private neuralRefiner: RFDNRefiner;  // ← RFDN (軽量版)
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
  private debugBypassRFDN: boolean = false;  // DEBUG OFF: Use RFDN to convert 32ch → RGB
  private readbackBuffers: GPUBuffer[] = [];
  private coarseFeatureArray: Float32Array | null = null;
  
  // State
  private initialized: boolean = false;
  private isRunning: boolean = false;
  private frameId: number | null = null;
  private frameCount: number = 0;
  
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
    
    // Neural Refiner: RFDN (178KB distilled model)
    // StyleUNet export is problematic, using RFDN for now
    const useStyleUNet = false;  // ← RFDNを使用（StyleUNetエクスポート問題）
    this.neuralRefiner = new RFDNRefiner({
      modelPath: useStyleUNet ? '/assets/styleunet_refiner.onnx' : '/assets/rfdn_refiner.onnx',
      useWebGPU: false  // WASM使用（安定性優先）
    });
    
    console.log('[GVRM] Created (v72: RFDN Refiner)');
  }
  
  async init(config?: GVRMConfig): Promise<void> {
    if (this.initialized) {
      console.warn('[GVRM] Already initialized');
      return;
    }
    
    try {
      console.log('[GVRM] 🚀 Initializing pipeline...');
      console.log('[GVRM] ═══════════════════════════════');
      console.log('[GVRM] 📦 Using RFDN Refiner (178KB, 630x smaller)');
      
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
      
      this.gpuDevice = await adapter.requestDevice({
        requiredLimits: requiredLimits as any
      });
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
      
      // 4. Initialize modules
      console.log('[GVRM] Step 4/6: Initializing modules');
      
      await Promise.all([
        this.imageEncoder.init(),
        this.uvDecoder.init('/assets'),
        this.neuralRefiner.init()  // RFDN (178KB) - 超高速ロード
      ]);
      
      // Template Decoder initialization (WebGPU)
      this.templateDecoder = new TemplateDecoderWebGPU();
      await this.templateDecoder.init(this.gpuDevice!, '/assets');
      
      console.log('[GVRM]   ✅ All modules initialized');
      console.log('[GVRM]   📊 RFDN Refiner: 178KB loaded (vs 107MB original)');
      
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
      console.log('[GVRM]   🚀 RFDN Refiner: No idEmbedding needed!');
      
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
    
    const { projectionFeature, idEmbedding, visibilityMask } =
      await this.imageEncoder.extractFeaturesWithSourceCamera(
        this.imagePath,
        {},
        vertices,
        vertexCount,
        128
      );

    // Store visibility mask for opacity masking
    this.visibilityMask = visibilityMask;
    
    console.log('[GVRM]   ✅ Encoder output:');
    console.log(`[GVRM]      Projection features: [${vertexCount}, 128]`);
    const projStats = this.analyzeArray(projectionFeature);
    console.log(`[GVRM]        stats: min=${projStats.min.toFixed(4)}, max=${projStats.max.toFixed(4)}, nonZeros=${projStats.nonZeros}`);
    console.log(`[GVRM]      ID embedding (CLS token): [${idEmbedding.length}]`);
    const idStats = this.analyzeArray(idEmbedding);
    console.log(`[GVRM]        stats: min=${idStats.min.toFixed(4)}, max=${idStats.max.toFixed(4)}, nonZeros=${idStats.nonZeros}`);
    
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
    if (this.uvMappingData) {
      console.log('[GVRM] Phase 3: UV pipeline');
      console.log('[GVRM]   ⚠️ UV pipeline currently disabled (requires EHM mesh data)');
    } else {
      console.log('[GVRM] Phase 3: UV pipeline skipped (no UV mapping data)');
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
  
  /**
   * Render loop (RFDN Refiner版 - idEmbedding不要)
   * フレームレート制限: GPUハング対策
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

        const outputBuffers = this.gsComputeRenderer.getOutputBuffers();
        coarseFeatures = await this.convertBuffersToFloat32Array(outputBuffers);

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
        // DEBUG: RFDNをバイパスして最初3チャンネルをRGBとして直接表示
        // これにより、Gaussian splatting自体が正しく動作しているかを確認
        const width = 512, height = 512;
        displayRGB = new Float32Array(width * height * 3);

        // 最初3チャンネルの統計を取得（正規化用）
        const pixelCount = width * height;
        let minVal = Infinity, maxVal = -Infinity;
        for (let ch = 0; ch < 3; ch++) {
          for (let p = 0; p < pixelCount; p++) {
            const val = coarseFeatures[ch * pixelCount + p];
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
          }
        }
        const range = maxVal - minVal || 1;

        // CHW → HWC変換 + 正規化 [0, 1]
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const p = y * width + x;
            for (let c = 0; c < 3; c++) {
              const srcIdx = c * pixelCount + p;
              const dstIdx = p * 3 + c;
              // 正規化: [minVal, maxVal] → [0, 1]
              displayRGB[dstIdx] = (coarseFeatures[srcIdx] - minVal) / range;
            }
          }
        }

        if (this.frameCount === 1) {
          console.log('[GVRM] 🔧 DEBUG: Bypassing RFDN, showing first 3 channels as RGB');
          console.log(`[GVRM]   Normalization: [${minVal.toFixed(4)}, ${maxVal.toFixed(4)}] → [0, 1]`);
        }
      } else {
        // RFDN Refiner: idEmbedding不要！32ch特徴マップのみ
        displayRGB = await this.neuralRefiner.process(coarseFeatures);
      }

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
          console.log(`  🚀 RFDN Refiner: No idEmbedding used (178KB model)`);
        }
      }

    } catch (error) {
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
  
  dispose(): void {
    this.isRunning = false;
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
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
