// gvrm.ts
// GUAVA pipeline implementation (WebGL GPU mode)
// 論文準拠: Real-time UV rasterization with GPU

import * as THREE from 'three';
import { ImageEncoder } from './image-encoder';
import { TemplateDecoder } from './template-decoder';
import { UVDecoder } from './uv-decoder';
import { WebGLUVRasterizer } from './webgl-uv-rasterizer';
import { InverseTextureMapper } from './inverse-texture-mapping';
import { RFDNRefiner } from './rfdn-refiner-webgpu';
import { WebGLDisplay } from './webgl-display';
import { GSViewer } from './gs';
import { UVStyleUNet } from './uv-styleunet';
import { computeViewDirection, concatenateWithViewEncoding } from './view-encoding';
import { GaussianRasterizerWebGPU, GaussianRasterInput, CameraParams as RasterizerCameraParams } from './gaussian-rasterizer-webgpu';

interface PLYData {
  vertices: Float32Array;
  triangles: Uint32Array;
  normals?: Float32Array;
  colors?: Float32Array;
  uvCoords?: Float32Array;  // UV coordinates per vertex [N * 2]
}

interface EHMMesh {
  vertices: Float32Array;
  triangles: Uint32Array;
  normals?: Float32Array;
}

interface GaussianData {
  positions: Float32Array;
  opacities: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  latents: Float32Array;
}

interface UVGaussianData extends GaussianData {
  triangleIndices: Uint32Array;
  barycentricCoords: Float32Array;
  worldPositions: Float32Array;
}

export interface CameraParams {
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
  aspect: number;
  near: number;
  far: number;
  width: number;
  height: number;
  viewMatrix: Float32Array;
  projMatrix: Float32Array;
  screenWidth: number;
  screenHeight: number;
}

/**
 * GVRM初期化設定
 * concierge-controller.ts互換
 */
export interface GVRMConfig {
  /** PLYテンプレートファイルのパス (例: '/assets/avatar_24p.ply') */
  templatePath?: string;
  /** ソース画像のパス (例: '/assets/source.png') */
  imagePath?: string;
  /** 表示コンテナ要素 */
  container?: HTMLElement;
}

export class GVRM {
  private imageEncoder: ImageEncoder;
  private templateDecoder: TemplateDecoder;
  private uvDecoder: UVDecoder;
  private uvStyleUNet: UVStyleUNet | null = null;  // 論文準拠: 35ch → 96ch (null in mobile mode)
  private webglRasterizer: WebGLUVRasterizer;
  private inverseMapper: InverseTextureMapper | null = null;
  private neuralRefiner: RFDNRefiner;
  private display: WebGLDisplay | null = null;
  private gsViewer: GSViewer | null = null;
  private gaussianRasterizer: GaussianRasterizerWebGPU | null = null;  // EWA Splatting rasterizer

  private plyData: PLYData | null = null;
  private templateMesh: EHMMesh | null = null;
  private templateGaussians: GaussianData | null = null;
  private uvGaussians: UVGaussianData | null = null;

  private initialized = false;
  private displayContainer: HTMLElement | null = null;

  // Configurable asset paths (concierge-controller.ts互換)
  private templatePath: string = '/assets/avatar_web.ply';
  private imagePath: string = '/assets/source.png';

  // Lip-sync state
  private currentLipSyncLevel: number = 0;

  // Neural Refiner用のidEmbedding (256ch)
  private idEmbedding256: Float32Array | null = null;

  // ソースカメラ設定（render時に使用）
  private sourceCameraConfig: {
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
  } | null = null;

  // UV Mapping（ワールド座標を含む）
  private uvMappingWorldPositions: Float32Array | null = null;

  // Ubody Gaussians data for rendering
  private ubodyGaussiansData: {
    positions: Float32Array;
    scales: Float32Array;
    rotations: Float32Array;
    opacities: Float32Array;
    latents: Float32Array;
    count: number;
  } | null = null;

  /**
   * コンストラクタ
   * @param displayContainer 表示コンテナ（オプション、init()のconfigでも指定可能）
   */
  constructor(displayContainer?: HTMLElement) {
    console.log('[GVRM] Constructor called (WebGL GPU mode)');

    // Store container reference but don't initialize display yet
    this.displayContainer = displayContainer || null;

    this.imageEncoder = new ImageEncoder();
    this.templateDecoder = new TemplateDecoder();
    this.uvDecoder = new UVDecoder();
    // uvStyleUNet is created conditionally in init() based on mobileMode
    this.webglRasterizer = new WebGLUVRasterizer();
    this.neuralRefiner = new RFDNRefiner({
      modelPath: '/assets/simpleunet_trained.onnx',
      useWebGPU: true  // WebGPU対応ブラウザで高速化
    });
  }

  /**
   * 初期化
   * @param config 設定オブジェクト（concierge-controller.ts互換）
   */
  async init(config?: GVRMConfig): Promise<void> {
    if (this.initialized) return;

    console.log('[GVRM] init() called');

    // Apply config if provided (concierge-controller.ts互換)
    if (config) {
      console.log('[GVRM] Config provided:', {
        templatePath: config.templatePath,
        imagePath: config.imagePath,
        hasContainer: !!config.container
      });

      if (config.templatePath) {
        this.templatePath = config.templatePath;
      }
      if (config.imagePath) {
        this.imagePath = config.imagePath;
      }
      if (config.container) {
        this.displayContainer = config.container;
      }
    }

    // StyleUNet (軽量モデル 3.69MB)
    this.uvStyleUNet = new UVStyleUNet();
    console.log('[GVRM] Using lightweight StyleUNet (3.69MB, 32x compressed)');

    console.log('[GVRM] Using paths:', {
      template: this.templatePath,
      image: this.imagePath
    });

    try {
      // Auto-detect display container if not provided
      if (!this.displayContainer) {
        const autoContainer = document.getElementById('avatar3DContainer');
        if (autoContainer) {
          this.displayContainer = autoContainer;
          console.log('[GVRM] Auto-detected display container: #avatar3DContainer');
        }
      }

      // Initialize WebGL display
      if (this.displayContainer) {
        console.log('[GVRM] Initializing WebGL display...');
        this.display = new WebGLDisplay(this.displayContainer, 512, 512);
        console.log('[GVRM] ✅ WebGL display initialized');
      } else {
        console.warn('[GVRM] No display container found, skipping display initialization');
      }

      console.log('[GVRM] 🚀 Starting GUAVA Pipeline (WebGL GPU mode)...');
      console.log('[GVRM] 📖 Paper-compliant: Real-time UV rasterization with GPU');

      await this.loadAssets();

      this.initialized = true;
      console.log('[GVRM] ✅ Initialization successful');

      // Initial render to display avatar using EWA Splatting
      if (this.display) {
        console.log('[GVRM] Performing initial render with EWA Splatting...');
        await this.render('ewa_splatting');
      }

    } catch (error) {
      console.error('[GVRM] ❌ Initialization failed:', error);
      throw error;
    }
  }

  private async loadAssets(): Promise<void> {
    console.log('[GVRM] Loading assets...');

    // ========== Step 0: Load source camera config first (needed for coordinate alignment) ==========
    console.log('[GVRM] Loading source camera config for coordinate alignment...');
    const sourceCameraConfig = await this.loadSourceCameraConfig();
    console.log('[GVRM] Source camera target:', sourceCameraConfig.target);

    // Store for render() use
    this.sourceCameraConfig = {
      position: sourceCameraConfig.position,
      target: sourceCameraConfig.target,
      fov: sourceCameraConfig.fov
    };

    // ========== Step 0.5: Load PLY file ==========
    // Use configurable templatePath (concierge-controller.ts互換)
    this.plyData = await this.loadPLY(this.templatePath, sourceCameraConfig.target);
    console.log('[GVRM] PLY loaded:', this.plyData.vertices.length / 3, 'vertices');

    // ========== Step 1: Load UV coordinates ==========
    const uvCoordsUrl = '/assets/uv_coords.bin';
    this.plyData.uvCoords = await this.loadUVCoords(uvCoordsUrl);
    console.log('[GVRM] UV coords loaded:', this.plyData.uvCoords.length / 2, 'vertices');

    // ========== Step 1.5: Load SMPLX faces (triangles) ==========
    // PLYファイルにfacesが含まれていない場合は別途ロード
    if (this.plyData.triangles.length === 0) {
      console.log('[GVRM] PLY has no faces, loading from smplx_faces.bin...');
      const facesUrl = '/assets/smplx_faces.bin';
      this.plyData.triangles = await this.loadSMPLXFaces(facesUrl);
      console.log('[GVRM] SMPLX faces loaded:', this.plyData.triangles.length / 3, 'triangles');
    }

    // ========== Step 2: Initialize modules ==========
    console.log('[GVRM] Step 2: Initializing modules...');
    
    console.log('[GVRM]   - Image Encoder (DINOv2)...');
    await this.imageEncoder.init();
    
    console.log('[GVRM]   - Template Decoder...');
    await this.templateDecoder.init('/assets');
    
    // Get template geometry data
    const geometryData = this.templateDecoder.getGeometryData();
    if (!geometryData) {
      throw new Error('[GVRM] Template geometry data not loaded');
    }
    
    const templateVertexCount = geometryData.numVertices;
    const templateVertices = this.plyData.vertices.slice(0, templateVertexCount * 3);
    
    console.log('[GVRM]   📊 Vertex configuration:', {
      totalPLY: (this.plyData.vertices.length / 3).toLocaleString(),
      template: templateVertexCount.toLocaleString(),
      ratio: ((templateVertexCount / (this.plyData.vertices.length / 3)) * 100).toFixed(1) + '%'
    });
    
    console.log('[GVRM]   - UV Decoder...');
    await this.uvDecoder.init('/assets');

    // UV StyleUNet (論文準拠: 35ch→96ch, 軽量モデル 3.69MB)
    if (this.uvStyleUNet) {
      console.log('[GVRM]   - UV StyleUNet (論文準拠: 35ch→96ch, 3.69MB)...');
      await this.uvStyleUNet.init({ basePath: '/assets' });
    }

    console.log('[GVRM]   - WebGL GPU Rasterizer...');
    await this.webglRasterizer.init();

    console.log('[GVRM]   - Neural Refiner...');
    await this.neuralRefiner.init();

    console.log('[GVRM] ✅ All modules initialized');

    // ========== Step 3: Extract appearance features ==========
    console.log('[GVRM] Step 3: Extracting appearance features...');

    // Use configurable imagePath (concierge-controller.ts互換)
    // Note: sourceCameraConfig already loaded at Step 0 for coordinate alignment

    const { projectionFeature, idEmbedding } = await this.imageEncoder.extractFeaturesWithSourceCamera(
      this.imagePath,
      sourceCameraConfig,
      templateVertices,
      templateVertexCount,
      128
    );
    
    console.log('[GVRM] ✅ Appearance features extracted');

    // ========== Step 4: Generate Template Gaussians ==========
    console.log('[GVRM] Step 4: Generating Template Gaussians...');

    // View direction を計算（カメラ → ターゲット）
    const viewDir = computeViewDirection(
      sourceCameraConfig.position as [number, number, number],
      sourceCameraConfig.target as [number, number, number]
    );
    console.log('[GVRM]   View direction:', viewDir);

    const templateOutput = await this.templateDecoder.generate(
      projectionFeature,
      idEmbedding,
      viewDir
    );

    // ========== CRITICAL FIX: Apply offset to template Gaussian positions ==========
    // Python版準拠: positions = v_template + offset
    let templatePositions: Float32Array;
    if (templateOutput.offset && templateOutput.offset.length === templateVertexCount * 3) {
      console.log('[GVRM]   Applying Template Decoder offset to positions...');
      templatePositions = new Float32Array(templateVertexCount * 3);
      for (let i = 0; i < templateVertexCount * 3; i++) {
        templatePositions[i] = templateVertices[i] + templateOutput.offset[i];
      }

      // Debug: offset statistics
      let offsetMin = Infinity, offsetMax = -Infinity, offsetSum = 0;
      for (let i = 0; i < templateOutput.offset.length; i++) {
        const v = templateOutput.offset[i];
        if (v < offsetMin) offsetMin = v;
        if (v > offsetMax) offsetMax = v;
        offsetSum += Math.abs(v);
      }
      console.log('[GVRM]   Offset stats:', {
        min: offsetMin.toFixed(4),
        max: offsetMax.toFixed(4),
        avgMagnitude: (offsetSum / templateOutput.offset.length).toFixed(4)
      });
    } else {
      console.warn('[GVRM]   ⚠️ No offset from Template Decoder, using raw vertices');
      templatePositions = templateVertices;
    }

    this.templateGaussians = {
      positions: templatePositions,  // Now with offset applied
      opacities: templateOutput.opacity,
      scales: templateOutput.scale,
      rotations: templateOutput.rotation,
      latents: templateOutput.rgb  // 新版: rgb (旧版: latent32ch)
    };

    // Python版準拠: 最初の3チャンネルにsigmoidを適用してRGBに変換
    this.applySigmoidToRGB(this.templateGaussians.latents, templateVertexCount);

    // Store idEmbedding256 for Neural Refiner
    if (templateOutput.idEmbedding256) {
      this.idEmbedding256 = templateOutput.idEmbedding256;
      console.log('[GVRM]   idEmbedding256 stored:', this.idEmbedding256.length, 'elements');
    }
    
    console.log('[GVRM] ✅ Template Gaussians generated:', {
      vertices: templateVertexCount.toLocaleString(),
      features: '32ch latent'
    });

    // ========== Step 5: Prepare EHM mesh ==========
    console.log('[GVRM] Step 5: Preparing EHM mesh...');
    console.log('[GVRM]   📖 Paper: "Given the tracked mesh..." = EHM mesh');
    
    this.templateMesh = {
      vertices: this.plyData.vertices,
      triangles: this.plyData.triangles,
      normals: this.plyData.normals
    };
    
    console.log('[GVRM] ✅ EHM mesh prepared:', {
      vertices: this.templateMesh.vertices.length / 3,
      triangles: this.templateMesh.triangles.length / 3
    });

    // ========== デバッグコード開始 ==========
    console.log('[Debug] === EHM Mesh Analysis ===');
    
    const vertices = this.templateMesh.vertices;
    const vertexCount = vertices.length / 3;
    
    // First 10 vertices
    console.log('[Debug] First 10 vertices:');
    for (let i = 0; i < Math.min(10, vertexCount); i++) {
      const x = vertices[i * 3];
      const y = vertices[i * 3 + 1];
      const z = vertices[i * 3 + 2];
      console.log(`  Vertex ${i}:`, [x.toFixed(4), y.toFixed(4), z.toFixed(4)]);
    }
    
    // Bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    for (let i = 0; i < vertexCount; i++) {
      const x = vertices[i * 3];
      const y = vertices[i * 3 + 1];
      const z = vertices[i * 3 + 2];
      
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
    
    console.log('[Debug] Mesh bounding box:', {
      x: [minX.toFixed(4), maxX.toFixed(4)],
      y: [minY.toFixed(4), maxY.toFixed(4)],
      z: [minZ.toFixed(4), maxZ.toFixed(4)]
    });
    
    // Center of mass
    let sumX = 0, sumY = 0, sumZ = 0;
    for (let i = 0; i < vertexCount; i++) {
      sumX += vertices[i * 3];
      sumY += vertices[i * 3 + 1];
      sumZ += vertices[i * 3 + 2];
    }
    
    const centerX = sumX / vertexCount;
    const centerY = sumY / vertexCount;
    const centerZ = sumZ / vertexCount;
    
    console.log('[Debug] Mesh center:', {
      x: centerX.toFixed(4),
      y: centerY.toFixed(4),
      z: centerZ.toFixed(4)
    });
    
    // Camera analysis
    console.log('[Debug] Camera configuration:', {
      position: sourceCameraConfig.position,
      target: sourceCameraConfig.target,
      fov: sourceCameraConfig.fov
    });
    
    // Distance from camera to mesh center
    const dx = centerX - sourceCameraConfig.position[0];
    const dy = centerY - sourceCameraConfig.position[1];
    const dz = centerZ - sourceCameraConfig.position[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    
    console.log('[Debug] Distance from camera to mesh center:', distance.toFixed(4));
    
    console.log('[Debug] === End of Analysis ===');
    // ========== デバッグコード終了 ==========

    // ========== Step 6: Map Template Gaussians to PLY ==========
    console.log('[GVRM] Step 6: Mapping Template Gaussians to PLY...');
    
    // Map template Gaussians to PLY vertices
    // (This step combines the coarse Gaussian attributes with PLY positions)
    
    console.log('[GVRM] ✅ Template Gaussians mapped');

    // ========== Step 7: Create Gaussian Splatting Viewer ==========
    console.log('[GVRM] Step 7: Creating Gaussian Splatting Viewer...');

    const plyVertexCount = this.plyData.vertices.length / 3;

    // GSViewerが期待するGaussianDataオブジェクトを構築
    const gaussianData = {
      positions: this.plyData.vertices,
      latents: this.templateGaussians.latents,
      opacity: this.templateGaussians.opacities,
      scale: this.templateGaussians.scales,
      rotation: this.templateGaussians.rotations,
      boneIndices: new Float32Array(plyVertexCount * 4),  // ダミー（スキニングなし）
      boneWeights: new Float32Array(plyVertexCount * 4),  // ダミー（スキニングなし）
      vertexCount: plyVertexCount
    };

    this.gsViewer = new GSViewer(gaussianData);

    console.log('[GVRM] ✅ GSViewer created');

    // ========== Step 8: Generate Coarse Feature Map ==========
    console.log('[GVRM] Step 8: Generating Coarse Feature Map...');
    // (GSViewer will generate this during rendering)
    console.log('[GVRM] ✅ Coarse Feature Map generated');

    // ========== Step 9: GPU UV Rasterization ==========
    console.log('[GVRM] Step 9: GPU UV Rasterization...');
    console.log('[GVRM]   ⚡ Using WebGL GPU for real-time rasterization');

    // MeshDataオブジェクトを構築
    if (!this.plyData.uvCoords) {
      throw new Error('[GVRM] UV coordinates not loaded. Please ensure uv_coords.bin is available.');
    }

    const meshData = {
      vertices: this.templateMesh.vertices,
      triangles: this.templateMesh.triangles,
      uvCoords: this.plyData.uvCoords,
      numVertices: this.templateMesh.vertices.length / 3,
      numTriangles: this.templateMesh.triangles.length / 3
    };

    const uvMapping = await this.webglRasterizer.rasterize(
      meshData,
      1024
    );
    
    console.log('[GVRM] ✅ GPU rasterization complete:', {
      resolution: '1024×1024',
      validPixels: uvMapping.validMask.reduce((sum, v) => sum + v, 0).toLocaleString(),
      coverage: (uvMapping.validMask.reduce((sum, v) => sum + v, 0) / (1024 * 1024) * 100).toFixed(1) + '%'
    });

    // Store world positions for UV Gaussians (Step 12で使用)
    this.uvMappingWorldPositions = uvMapping.worldPositions;

    // ========== Step 9.5: Initialize InverseTextureMapper ==========
    console.log('[GVRM] Step 9.5: Initializing InverseTextureMapper...');

    this.inverseMapper = new InverseTextureMapper();
    this.inverseMapper.initialize(1024, {
      position: sourceCameraConfig.position,
      target: sourceCameraConfig.target,
      fov: sourceCameraConfig.fov,
      viewport: { width: 518, height: 518 }
    });

    console.log('[GVRM] ✅ InverseTextureMapper initialized');

    // ========== Step 10: Inverse Texture Mapping (実際の投影) ==========
    console.log('[GVRM] Step 10: Inverse Texture Mapping (論文準拠)...');
    console.log('[GVRM]   📖 Paper: Project image-space features to UV-space');

    // Get UV branch features (32ch) from image encoder - these are in IMAGE space
    const imageBranchFeatures = this.imageEncoder.getUVFeatures();

    // Verify feature dimensions
    const imageFeatureSize = 518;
    const expectedSize = imageFeatureSize * imageFeatureSize * 32;
    console.log('[GVRM] Debug - Image branch features:', {
      length: imageBranchFeatures.length,
      expected: expectedSize,
      channels: 32,
      format: 'HWC [518, 518, 32]',
      match: imageBranchFeatures.length === expectedSize ? '✅' : '❌'
    });

    // 🔍 Debug: Image features statistics
    let imgMin = Infinity, imgMax = -Infinity, imgSum = 0, imgNonZero = 0;
    for (let i = 0; i < imageBranchFeatures.length; i++) {
      const v = imageBranchFeatures[i];
      if (v < imgMin) imgMin = v;
      if (v > imgMax) imgMax = v;
      imgSum += v;
      if (v !== 0) imgNonZero++;
    }
    console.log('[GVRM] 🔍 Image branch features stats:', {
      min: imgMin.toFixed(4),
      max: imgMax.toFixed(4),
      mean: (imgSum / imageBranchFeatures.length).toFixed(4),
      nonZeroRatio: (imgNonZero / imageBranchFeatures.length * 100).toFixed(1) + '%'
    });

    // ========== CRITICAL: Actually perform Inverse Texture Mapping ==========
    // Project image-space features AND RGB to UV-space using mesh world positions
    console.log('[GVRM]   🚀 Performing Inverse Texture Mapping...');
    console.log('[GVRM]   - Input 1: Image-space features (518×518×32)');
    console.log('[GVRM]   - Input 2: Image-space RGB (518×518×3)');
    console.log('[GVRM]   - Output: UV-space features (1024×1024×32) + RGB (1024×1024×3)');

    // Project 32ch DINO features to UV space
    const uvSpaceFeatures1024 = this.inverseMapper.map(
      uvMapping,
      imageBranchFeatures,
      { width: imageFeatureSize, height: imageFeatureSize },
      32  // 32 channels
    );

    // 🔍 Debug: UV-space features statistics
    let uvMin = Infinity, uvMax = -Infinity, uvSum = 0, uvNonZero = 0;
    for (let i = 0; i < uvSpaceFeatures1024.length; i++) {
      const v = uvSpaceFeatures1024[i];
      if (v < uvMin) uvMin = v;
      if (v > uvMax) uvMax = v;
      uvSum += v;
      if (v !== 0) uvNonZero++;
    }
    console.log('[GVRM] 🔍 UV-space features (1024×1024×32) stats:', {
      min: uvMin.toFixed(4),
      max: uvMax.toFixed(4),
      mean: (uvSum / uvSpaceFeatures1024.length).toFixed(4),
      nonZeroRatio: (uvNonZero / uvSpaceFeatures1024.length * 100).toFixed(1) + '%'
    });

    // ========== CRITICAL: Also project RGB to UV space (Python版準拠) ==========
    console.log('[GVRM]   🎨 Projecting RGB to UV space...');
    const sourceRGB = this.imageEncoder.getSourceImageRGBForMapping();
    const uvSpaceRGB1024 = this.inverseMapper.map(
      uvMapping,
      sourceRGB.data,
      { width: sourceRGB.width, height: sourceRGB.height },
      3  // 3 RGB channels
    );

    // 🔍 Debug: UV-space RGB statistics
    let rgbMin = Infinity, rgbMax = -Infinity, rgbSum = 0, rgbNonZero = 0;
    for (let i = 0; i < uvSpaceRGB1024.length; i++) {
      const v = uvSpaceRGB1024[i];
      if (v < rgbMin) rgbMin = v;
      if (v > rgbMax) rgbMax = v;
      rgbSum += v;
      if (v !== 0) rgbNonZero++;
    }
    console.log('[GVRM] 🔍 UV-space RGB (1024×1024×3) stats:', {
      min: rgbMin.toFixed(4),
      max: rgbMax.toFixed(4),
      mean: (rgbSum / uvSpaceRGB1024.length).toFixed(4),
      nonZeroRatio: (rgbNonZero / uvSpaceRGB1024.length * 100).toFixed(1) + '%'
    });

    console.log('[GVRM] ✅ Inverse Texture Mapping complete (features + RGB)');

    // ========== Step 10.5: Build 155ch UV features (論文準拠) ==========
    console.log('[GVRM] Step 10.5: Building 155ch UV features (論文準拠)...');
    console.log('[GVRM] 📖 Paper pipeline:');
    console.log('[GVRM]   1. 35ch (32 UV + 3 RGB) → StyleUNet → 96ch');
    console.log('[GVRM]   2. 96ch + 32ch base_feature = 128ch');
    console.log('[GVRM]   3. 128ch + 27ch view_dirs = 155ch');

    const uvResolution = 512;  // StyleUNet expects 512x512
    const uvPixels = uvResolution * uvResolution;
    const uvMappingRes = 1024;  // UV mapping resolution

    // Step 10.5.1: Resample UV-space features from 1024 to 512 (HWC→CHW format for StyleUNet)
    console.log('[GVRM]   Resampling UV-space features (1024→512, HWC→CHW)...');
    const uvFeatures32ch = new Float32Array(32 * uvResolution * uvResolution);

    // Resample with bilinear interpolation (HWC to CHW)
    // Source: uvSpaceFeatures1024 is [1024, 1024, 32] in HWC format (from InverseTextureMapper)
    // Target: uvFeatures32ch is [32, 512, 512] in CHW format (for StyleUNet)
    const scale = uvMappingRes / uvResolution;  // 1024 / 512 = 2
    for (let c = 0; c < 32; c++) {
      for (let ty = 0; ty < uvResolution; ty++) {
        for (let tx = 0; tx < uvResolution; tx++) {
          const sx = tx * scale;
          const sy = ty * scale;
          const sx0 = Math.floor(sx);
          const sy0 = Math.floor(sy);
          const sx1 = Math.min(sx0 + 1, uvMappingRes - 1);
          const sy1 = Math.min(sy0 + 1, uvMappingRes - 1);
          const wx = sx - sx0;
          const wy = sy - sy0;

          // Source is HWC format: [H, W, C] from InverseTextureMapper
          const v00 = uvSpaceFeatures1024[(sy0 * uvMappingRes + sx0) * 32 + c];
          const v10 = uvSpaceFeatures1024[(sy0 * uvMappingRes + sx1) * 32 + c];
          const v01 = uvSpaceFeatures1024[(sy1 * uvMappingRes + sx0) * 32 + c];
          const v11 = uvSpaceFeatures1024[(sy1 * uvMappingRes + sx1) * 32 + c];

          const top = v00 * (1 - wx) + v10 * wx;
          const bottom = v01 * (1 - wx) + v11 * wx;
          const value = top * (1 - wy) + bottom * wy;

          // Output is CHW format: [C, H, W] for StyleUNet
          uvFeatures32ch[c * uvResolution * uvResolution + ty * uvResolution + tx] = value;
        }
      }
    }
    console.log('[GVRM]   ✅ UV-space features prepared (1024→512, CHW format)');

    // 🔍 Debug: Resampled UV features statistics
    let resampledMin = Infinity, resampledMax = -Infinity, resampledSum = 0, resampledNonZero = 0;
    for (let i = 0; i < uvFeatures32ch.length; i++) {
      const v = uvFeatures32ch[i];
      if (v < resampledMin) resampledMin = v;
      if (v > resampledMax) resampledMax = v;
      resampledSum += v;
      if (v !== 0) resampledNonZero++;
    }
    console.log('[GVRM]   🔍 UV features 32ch (512×512) stats:', {
      min: resampledMin.toFixed(4),
      max: resampledMax.toFixed(4),
      mean: (resampledSum / uvFeatures32ch.length).toFixed(4),
      nonZeroRatio: (resampledNonZero / uvFeatures32ch.length * 100).toFixed(1) + '%'
    });

    // Step 10.5.1b: Resample UV-space RGB from 1024 to 512 (HWC→CHW format)
    // CRITICAL: Python版では RGB も UV空間に投影してから StyleUNet に渡す
    console.log('[GVRM]   Resampling UV-space RGB (1024→512, HWC→CHW)...');
    const uvRGB3ch = new Float32Array(3 * uvResolution * uvResolution);

    for (let c = 0; c < 3; c++) {
      for (let ty = 0; ty < uvResolution; ty++) {
        for (let tx = 0; tx < uvResolution; tx++) {
          const sx = tx * scale;
          const sy = ty * scale;
          const sx0 = Math.floor(sx);
          const sy0 = Math.floor(sy);
          const sx1 = Math.min(sx0 + 1, uvMappingRes - 1);
          const sy1 = Math.min(sy0 + 1, uvMappingRes - 1);
          const wx = sx - sx0;
          const wy = sy - sy0;

          // Source is HWC format: [H, W, 3] from InverseTextureMapper
          const v00 = uvSpaceRGB1024[(sy0 * uvMappingRes + sx0) * 3 + c];
          const v10 = uvSpaceRGB1024[(sy0 * uvMappingRes + sx1) * 3 + c];
          const v01 = uvSpaceRGB1024[(sy1 * uvMappingRes + sx0) * 3 + c];
          const v11 = uvSpaceRGB1024[(sy1 * uvMappingRes + sx1) * 3 + c];

          const top = v00 * (1 - wx) + v10 * wx;
          const bottom = v01 * (1 - wx) + v11 * wx;
          const value = top * (1 - wy) + bottom * wy;

          // Output is CHW format: [3, H, W] for StyleUNet
          uvRGB3ch[c * uvResolution * uvResolution + ty * uvResolution + tx] = value;
        }
      }
    }

    // 🔍 Debug: Resampled UV RGB statistics
    let rgbResMin = Infinity, rgbResMax = -Infinity, rgbResSum = 0, rgbResNonZero = 0;
    for (let i = 0; i < uvRGB3ch.length; i++) {
      const v = uvRGB3ch[i];
      if (v < rgbResMin) rgbResMin = v;
      if (v > rgbResMax) rgbResMax = v;
      rgbResSum += v;
      if (v !== 0) rgbResNonZero++;
    }
    console.log('[GVRM]   🔍 UV RGB 3ch (512×512) stats:', {
      min: rgbResMin.toFixed(4),
      max: rgbResMax.toFixed(4),
      mean: (rgbResSum / uvRGB3ch.length).toFixed(4),
      nonZeroRatio: (rgbResNonZero / uvRGB3ch.length * 100).toFixed(1) + '%'
    });
    console.log('[GVRM]   ✅ UV-space RGB prepared (1024→512, CHW format)');

    // View direction (already computed at Step 4)
    console.log('[GVRM]   View direction (reusing from Step 4):', viewDir);

    let uvFeatureMap: Float32Array;

    // ========== StyleUNet pipeline (論文準拠, 軽量モデル 3.69MB) ==========
    if (!this.uvStyleUNet) {
      throw new Error('[GVRM] UVStyleUNet not initialized');
    }

    console.log('[GVRM]   Using StyleUNet pipeline (3.69MB lightweight)');

    // Step 10.5.2: Use UV-space RGB (NOT image-space RGB!)
    // CRITICAL FIX: Python版では RGB も UV空間に投影してから使用
    console.log('[GVRM]   Using UV-space RGB (projected from image)...');
    const rgbImage = uvRGB3ch;  // Use UV-space RGB instead of image-space RGB
    console.log('[GVRM]   ✅ 3ch RGB prepared');

    // Step 10.5.3: Get global feature for style embedding
    console.log('[GVRM]   Getting global feature (768ch)...');
    const globalFeature = this.imageEncoder.getGlobalFeature();
    console.log('[GVRM]   ✅ Global feature ready');

    // Step 10.5.4: StyleUNet: 35ch → 96ch
    console.log('[GVRM]   Running StyleUNet (35ch → 96ch)...');
    const styleunetOutput = await this.uvStyleUNet.forward(
      uvFeatures32ch,
      rgbImage,
      globalFeature,
      uvResolution,
      uvResolution
    );
    console.log('[GVRM]   ✅ StyleUNet output: 96ch');

    // Step 10.5.5: Add base_feature: 96ch + 32ch = 128ch
    console.log('[GVRM]   Adding base_feature (96ch + 32ch = 128ch)...');
    const features128ch = this.uvStyleUNet.addBaseFeature(styleunetOutput, uvResolution, uvResolution);
    console.log('[GVRM]   ✅ 128ch features ready');

    // Step 10.5.6: Add view direction encoding: 128ch + 27ch = 155ch
    console.log('[GVRM]   Adding view direction encoding (128ch + 27ch = 155ch)...');
    uvFeatureMap = concatenateWithViewEncoding(features128ch, viewDir, uvResolution, uvResolution);
    console.log('[GVRM]   ✅ 155ch UV features ready (論文準拠)');

    // ========== Step 11: Generate UV Gaussians ==========
    console.log('[GVRM] Step 11: Generating UV Gaussians...');

    this.uvGaussians = await this.uvDecoder.generate(
      uvFeatureMap,
      uvResolution,
      uvResolution,
      uvMapping
    );
    
    console.log('[GVRM] ✅ UV Gaussians generated:', {
      count: this.uvGaussians.uvCount
    });

    // Python版準拠: UV Gaussians の最初の3チャンネルにsigmoidを適用
    this.applySigmoidToRGB(this.uvGaussians.latent32ch, this.uvGaussians.uvCount);

    // ========== Step 12: Create Ubody Gaussians (Template ⊕ UV) ==========
    console.log('[GVRM] Step 12: Creating Ubody Gaussians (Template ⊕ UV)...');

    const templateCount = this.templateGaussians.positions.length / 3;
    const uvCount = this.uvGaussians.uvCount;
    const totalCount = templateCount + uvCount;

    // UV Gaussians のワールド座標を計算
    // Python版準拠: face orientation回転 + face scaling + face center
    const uvWorldPositions = this.computeUVWorldPositions(
      uvMapping,
      this.uvGaussians.localPositions,
      uvCount,
      this.uvGaussians.triangleIndices,
      this.uvGaussians.barycentricCoords
    );

    // Concatenate all Gaussian properties
    const ubodyGaussians = {
      positions: this.concatenateArrays(this.templateGaussians.positions, uvWorldPositions),
      opacities: this.concatenateArrays(this.templateGaussians.opacities, this.uvGaussians.opacity),
      scales: this.concatenateArrays(this.templateGaussians.scales, this.uvGaussians.scale),
      rotations: this.concatenateArrays(this.templateGaussians.rotations, this.uvGaussians.rotation),
      latents: this.concatenateArrays(this.templateGaussians.latents, this.uvGaussians.latent32ch)
    };

    console.log('[GVRM] ✅ Ubody Gaussians created:', {
      total: totalCount.toLocaleString(),
      template: templateCount.toLocaleString(),
      uv: uvCount.toLocaleString()
    });

    // ========== Step 12.5: Update GSViewer with Ubody Gaussians ==========
    console.log('[GVRM] Step 12.5: Updating GSViewer with Ubody Gaussians...');
    console.log('[GVRM]   📖 Paper: Template + UV Gaussians combined for full avatar');

    // Recreate GSViewer with complete ubodyGaussians (Template ⊕ UV)
    if (this.gsViewer) {
      this.gsViewer.dispose();
    }

    const ubodyGaussianData = {
      positions: ubodyGaussians.positions,
      latents: ubodyGaussians.latents,
      opacity: ubodyGaussians.opacities,
      scale: ubodyGaussians.scales,
      rotation: ubodyGaussians.rotations,
      boneIndices: new Float32Array(totalCount * 4),  // ダミー（スキニングなし）
      boneWeights: new Float32Array(totalCount * 4),  // ダミー（スキニングなし）
      vertexCount: totalCount
    };

    this.gsViewer = new GSViewer(ubodyGaussianData);

    console.log('[GVRM] ✅ GSViewer updated with Ubody Gaussians:', {
      totalGaussians: totalCount.toLocaleString(),
      latentChannels: 32
    });

    // ========== Step 12.6: Store Ubody Gaussians for EWA Splatting ==========
    console.log('[GVRM] Step 12.6: Storing Ubody Gaussians for EWA Splatting...');
    this.ubodyGaussiansData = {
      positions: ubodyGaussians.positions,
      scales: ubodyGaussians.scales,
      rotations: ubodyGaussians.rotations,
      opacities: ubodyGaussians.opacities,
      latents: ubodyGaussians.latents,
      count: totalCount
    };

    // Initialize EWA Splatting rasterizer
    console.log('[GVRM]   Initializing EWA Splatting rasterizer...');
    this.gaussianRasterizer = new GaussianRasterizerWebGPU();
    await this.gaussianRasterizer.init();
    console.log('[GVRM] ✅ EWA Splatting rasterizer initialized');

    // ========== Final step: Pipeline complete ==========
    console.log('[GVRM] ✅ GUAVA Pipeline Complete! 🎉');
    console.log('[GVRM] 📊 Summary:', {
      mode: 'WebGL GPU (Real-time)',
      totalGaussians: totalCount.toLocaleString(),
      plyVertices: (this.plyData.vertices.length / 3).toLocaleString()
    });
  }

  private concatenateArrays(a: Float32Array, b: Float32Array): Float32Array {
    const result = new Float32Array(a.length + b.length);
    result.set(a, 0);
    result.set(b, a.length);
    return result;
  }

  /**
   * UV Gaussians のワールド座標を計算
   * Python版準拠: face orientation matrix による回転 + face scaling + face center
   *
   * Python実装 (ubody_gaussian.py line 238-239):
   *   face_orien_mat, face_scaling = compute_face_orientation(vertices, faces)
   *   xyz = einsum('bnij,bnj->bni', face_orien_mat, local_xyz)  # 回転
   *   world_pos = xyz * face_scaling + face_center  # スケール + 平行移動
   */
  private computeUVWorldPositions(
    uvMapping: { worldPositions: Float32Array; uvCoords: Uint16Array; triangleIndices: Uint32Array; barycentricCoords: Float32Array; width: number; height: number },
    localPositions: Float32Array,
    uvCount: number,
    triangleIndicesPerGaussian: Uint32Array,
    barycentricCoordsPerGaussian: Float32Array
  ): Float32Array {
    console.log('[GVRM]   Computing UV world positions with face orientation...');

    if (!this.templateMesh) {
      throw new Error('[GVRM] Template mesh not loaded');
    }

    const vertices = this.templateMesh.vertices;
    const triangles = this.templateMesh.triangles;
    const numFaces = triangles.length / 3;

    // Step 1: Compute face orientation matrices and scaling for all faces
    console.log('[GVRM]   Step 1: Computing face orientations for', numFaces, 'faces...');
    const faceOrientations = new Float32Array(numFaces * 9);  // 3x3 matrix per face
    const faceScaling = new Float32Array(numFaces);
    const faceCenters = new Float32Array(numFaces * 3);

    for (let f = 0; f < numFaces; f++) {
      const i0 = triangles[f * 3 + 0];
      const i1 = triangles[f * 3 + 1];
      const i2 = triangles[f * 3 + 2];

      // Get vertices
      const v0x = vertices[i0 * 3 + 0], v0y = vertices[i0 * 3 + 1], v0z = vertices[i0 * 3 + 2];
      const v1x = vertices[i1 * 3 + 0], v1y = vertices[i1 * 3 + 1], v1z = vertices[i1 * 3 + 2];
      const v2x = vertices[i2 * 3 + 0], v2y = vertices[i2 * 3 + 1], v2z = vertices[i2 * 3 + 2];

      // Face center
      faceCenters[f * 3 + 0] = (v0x + v1x + v2x) / 3;
      faceCenters[f * 3 + 1] = (v0y + v1y + v2y) / 3;
      faceCenters[f * 3 + 2] = (v0z + v1z + v2z) / 3;

      // a0 = normalize(v1 - v0) - tangent
      let a0x = v1x - v0x, a0y = v1y - v0y, a0z = v1z - v0z;
      const len0 = Math.sqrt(a0x * a0x + a0y * a0y + a0z * a0z) + 1e-8;
      a0x /= len0; a0y /= len0; a0z /= len0;

      // edge v2 - v0
      const e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;

      // a1 = normalize(cross(a0, v2-v0)) - normal
      let a1x = a0y * e2z - a0z * e2y;
      let a1y = a0z * e2x - a0x * e2z;
      let a1z = a0x * e2y - a0y * e2x;
      const len1 = Math.sqrt(a1x * a1x + a1y * a1y + a1z * a1z) + 1e-8;
      a1x /= len1; a1y /= len1; a1z /= len1;

      // a2 = -normalize(cross(a1, a0)) - bitangent
      let a2x = -(a1y * a0z - a1z * a0y);
      let a2y = -(a1z * a0x - a1x * a0z);
      let a2z = -(a1x * a0y - a1y * a0x);
      const len2 = Math.sqrt(a2x * a2x + a2y * a2y + a2z * a2z) + 1e-8;
      a2x /= len2; a2y /= len2; a2z /= len2;

      // Store orientation matrix [a0, a1, a2] as columns (row-major storage)
      // orientation[i, j] = column j, row i
      faceOrientations[f * 9 + 0] = a0x; faceOrientations[f * 9 + 1] = a1x; faceOrientations[f * 9 + 2] = a2x;
      faceOrientations[f * 9 + 3] = a0y; faceOrientations[f * 9 + 4] = a1y; faceOrientations[f * 9 + 5] = a2y;
      faceOrientations[f * 9 + 6] = a0z; faceOrientations[f * 9 + 7] = a1z; faceOrientations[f * 9 + 8] = a2z;

      // Face scaling = (edge0_length + height) / 2
      // s0 = length(v1 - v0)
      const s0 = len0;
      // s1 = abs(dot(a2, v2-v0))
      const s1 = Math.abs(a2x * e2x + a2y * e2y + a2z * e2z);
      faceScaling[f] = (s0 + s1) / 2;
    }

    console.log('[GVRM]   Step 2: Transforming', uvCount, 'UV Gaussian positions...');

    const worldPositions = new Float32Array(uvCount * 3);
    let validCount = 0;
    let invalidFaceCount = 0;

    for (let i = 0; i < uvCount; i++) {
      // Get the face index for this UV Gaussian
      const faceIdx = triangleIndicesPerGaussian[i];

      if (faceIdx >= numFaces) {
        invalidFaceCount++;
        continue;
      }

      // Get face orientation matrix
      const m00 = faceOrientations[faceIdx * 9 + 0];
      const m01 = faceOrientations[faceIdx * 9 + 1];
      const m02 = faceOrientations[faceIdx * 9 + 2];
      const m10 = faceOrientations[faceIdx * 9 + 3];
      const m11 = faceOrientations[faceIdx * 9 + 4];
      const m12 = faceOrientations[faceIdx * 9 + 5];
      const m20 = faceOrientations[faceIdx * 9 + 6];
      const m21 = faceOrientations[faceIdx * 9 + 7];
      const m22 = faceOrientations[faceIdx * 9 + 8];

      // Get face scaling
      const scale = faceScaling[faceIdx];

      // Get local position
      const lx = localPositions[i * 3 + 0];
      const ly = localPositions[i * 3 + 1];
      const lz = localPositions[i * 3 + 2];

      // Rotate by face orientation: xyz = M @ local
      const rx = m00 * lx + m01 * ly + m02 * lz;
      const ry = m10 * lx + m11 * ly + m12 * lz;
      const rz = m20 * lx + m21 * ly + m22 * lz;

      // Get face center using barycentric coords
      const bary0 = barycentricCoordsPerGaussian[i * 3 + 0];
      const bary1 = barycentricCoordsPerGaussian[i * 3 + 1];
      const bary2 = barycentricCoordsPerGaussian[i * 3 + 2];

      const i0 = triangles[faceIdx * 3 + 0];
      const i1 = triangles[faceIdx * 3 + 1];
      const i2 = triangles[faceIdx * 3 + 2];

      const cx = vertices[i0 * 3 + 0] * bary0 + vertices[i1 * 3 + 0] * bary1 + vertices[i2 * 3 + 0] * bary2;
      const cy = vertices[i0 * 3 + 1] * bary0 + vertices[i1 * 3 + 1] * bary1 + vertices[i2 * 3 + 1] * bary2;
      const cz = vertices[i0 * 3 + 2] * bary0 + vertices[i1 * 3 + 2] * bary1 + vertices[i2 * 3 + 2] * bary2;

      // World position = rotated_local * scale + face_center
      worldPositions[i * 3 + 0] = rx * scale + cx;
      worldPositions[i * 3 + 1] = ry * scale + cy;
      worldPositions[i * 3 + 2] = rz * scale + cz;

      validCount++;
    }

    // Debug output
    if (uvCount > 0) {
      const sampleIdx = Math.min(100, uvCount - 1);
      console.log('[GVRM]   Sample UV Gaussian transformation:', {
        index: sampleIdx,
        faceIdx: triangleIndicesPerGaussian[sampleIdx],
        localPos: [
          localPositions[sampleIdx * 3 + 0].toFixed(4),
          localPositions[sampleIdx * 3 + 1].toFixed(4),
          localPositions[sampleIdx * 3 + 2].toFixed(4)
        ],
        worldPos: [
          worldPositions[sampleIdx * 3 + 0].toFixed(4),
          worldPositions[sampleIdx * 3 + 1].toFixed(4),
          worldPositions[sampleIdx * 3 + 2].toFixed(4)
        ]
      });
    }

    console.log('[GVRM]   ✅ UV world positions computed with face orientation:', {
      count: uvCount,
      validTransforms: validCount,
      invalidFaces: invalidFaceCount
    });

    return worldPositions;
  }

  /**
   * 32ch latent features の最初の3チャンネル(RGB)にsigmoidを適用
   * Python版準拠: ubody_gaussian.py lines 186-187
   * @param latents [N, 32] 形式のlatent features
   * @param numVertices 頂点数
   */
  private applySigmoidToRGB(latents: Float32Array, numVertices: number): void {
    console.log('[GVRM] Applying sigmoid to first 3 channels (RGB)...');

    // latents is [N, 32] flattened, so latents[i * 32 + c] is channel c of vertex i
    for (let i = 0; i < numVertices; i++) {
      for (let c = 0; c < 3; c++) {  // Only first 3 channels (RGB)
        const idx = i * 32 + c;
        const raw = latents[idx];
        latents[idx] = 1.0 / (1.0 + Math.exp(-raw));  // sigmoid
      }
    }

    // Debug: check RGB range after sigmoid
    let minRGB = Infinity, maxRGB = -Infinity;
    for (let i = 0; i < Math.min(1000, numVertices); i++) {
      for (let c = 0; c < 3; c++) {
        const v = latents[i * 32 + c];
        if (v < minRGB) minRGB = v;
        if (v > maxRGB) maxRGB = v;
      }
    }
    console.log('[GVRM]   RGB range after sigmoid:', { min: minRGB.toFixed(4), max: maxRGB.toFixed(4) });
  }

  async render(debugMode: 'normal' | 'coarse_rgb' | 'ewa_splatting' = 'normal'): Promise<void> {
    if (!this.initialized || !this.gsViewer) {
      throw new Error('[GVRM] Not initialized');
    }

    if (!this.display) {
      console.warn('[GVRM] No display available, skipping render');
      return;
    }

    console.log('[GVRM] Rendering avatar... (mode:', debugMode, ')');

    const renderWidth = 512;
    const renderHeight = 512;

    // Step 1: Create camera from source camera config
    let camera: THREE.PerspectiveCamera | undefined;
    let cameraParams: RasterizerCameraParams | undefined;

    if (this.sourceCameraConfig) {
      const { position, target, fov } = this.sourceCameraConfig;
      camera = new THREE.PerspectiveCamera(fov, 1.0, 0.1, 100);
      camera.position.set(position[0], position[1], position[2]);
      camera.lookAt(target[0], target[1], target[2]);
      camera.updateMatrixWorld();

      // Create camera params for EWA rasterizer
      const viewMatrix = new Float32Array(16);
      const projMatrix = new Float32Array(16);
      camera.matrixWorldInverse.toArray(viewMatrix);
      camera.projectionMatrix.toArray(projMatrix);

      // Calculate focal lengths from FOV
      const fovRad = (fov * Math.PI) / 180;
      const tanFovY = Math.tan(fovRad / 2);
      const tanFovX = tanFovY * (renderWidth / renderHeight);
      const focalY = renderHeight / (2 * tanFovY);
      const focalX = renderWidth / (2 * tanFovX);

      cameraParams = {
        viewMatrix,
        projMatrix,
        focalX,
        focalY,
        tanFovX,
        tanFovY,
        width: renderWidth,
        height: renderHeight
      };

      console.log('[GVRM]   Using source camera:', {
        position,
        target,
        fov,
        focalX: focalX.toFixed(2),
        focalY: focalY.toFixed(2)
      });
    } else {
      console.warn('[GVRM]   No source camera config, using default');
    }

    let coarseFeatureMap: Float32Array;

    // Step 2: Render coarse feature map (32ch) with proper camera
    if (debugMode === 'ewa_splatting' && this.gaussianRasterizer && this.ubodyGaussiansData && cameraParams) {
      // Use EWA Splatting algorithm (proper 3D Gaussian projection)
      console.log('[GVRM]   🌟 Using EWA Splatting (anisotropic Gaussian projection)...');

      const rasterInput: GaussianRasterInput = {
        positions: this.ubodyGaussiansData.positions,
        scales: this.ubodyGaussiansData.scales,
        rotations: this.ubodyGaussiansData.rotations,
        opacities: this.ubodyGaussiansData.opacities,
        features: this.ubodyGaussiansData.latents,
        count: this.ubodyGaussiansData.count
      };

      coarseFeatureMap = await this.gaussianRasterizer.render(rasterInput, cameraParams, 0);
      console.log('[GVRM]   ✅ EWA Splatting complete:', coarseFeatureMap.length);
    } else {
      // Fallback to old GSViewer (simple point splatting)
      console.log('[GVRM]   Using GSViewer (simple point splatting)...');
      coarseFeatureMap = this.gsViewer.render(renderWidth, renderHeight, camera);
      console.log('[GVRM]   Coarse feature map rendered:', coarseFeatureMap.length);
    }

    let outputImage: Float32Array;

    if (debugMode === 'coarse_rgb') {
      // Debug mode: Display RGB from coarse features directly (bypass Neural Refiner)
      console.log('[GVRM]   🔍 Debug: Extracting RGB from coarse features (bypassing Neural Refiner)...');
      outputImage = this.gsViewer.extractRGBFromCoarseFeatures(coarseFeatureMap, renderWidth, renderHeight);
      console.log('[GVRM]   🔍 Debug: Direct RGB extraction complete');
    } else {
      // Normal mode: Neural refinement (32ch → 3ch RGB)
      outputImage = await this.neuralRefiner.process(coarseFeatureMap);
      console.log('[GVRM]   Neural refinement complete:', outputImage.length);
    }

    // Step 4: Display
    this.display.display(outputImage);
    console.log('[GVRM] ✅ Avatar rendered');
  }

  /**
   * リップシンク更新（concierge-controller.ts互換）
   * オーディオレベルに基づいて口の動きを更新
   * @param level 正規化されたオーディオレベル (0.0 - 1.0)
   */
  updateLipSync(level: number): void {
    // Clamp level to valid range
    this.currentLipSyncLevel = Math.max(0, Math.min(1, level));

    // TODO: 将来的にGSViewerにリップシンクパラメータを渡す
    // 現在はレベルを保存するのみ
    if (this.gsViewer) {
      // gsViewer.setLipSyncLevel(this.currentLipSyncLevel);
    }
  }

  /**
   * 現在のリップシンクレベルを取得
   */
  getLipSyncLevel(): number {
    return this.currentLipSyncLevel;
  }

  private async loadPLY(url: string, cameraTarget?: [number, number, number]): Promise<PLYData> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    
    // Parse PLY header
    const decoder = new TextDecoder('utf-8');
    const headerText = decoder.decode(arrayBuffer.slice(0, 10000));
    const headerEnd = headerText.indexOf('end_header');
    
    if (headerEnd === -1) {
      throw new Error('[GVRM] Invalid PLY file: no end_header');
    }
    
    const headerLines = headerText.substring(0, headerEnd).split('\n');
    
    let vertexCount = 0;
    let faceCount = 0;
    const vertexProperties: string[] = [];
    let inVertexSection = false;
    
    for (const line of headerLines) {
      const trimmed = line.trim();
      
      if (trimmed.startsWith('element vertex')) {
        vertexCount = parseInt(trimmed.split(/\s+/)[2]);
        inVertexSection = true;
      } else if (trimmed.startsWith('element face')) {
        faceCount = parseInt(trimmed.split(/\s+/)[2]);
        inVertexSection = false;
      } else if (trimmed.startsWith('property') && inVertexSection) {
        const parts = trimmed.split(' ');
        if (parts.length >= 3) {
          vertexProperties.push(parts[parts.length - 1]); // プロパティ名
        }
      }
    }
    
    console.log('[GVRM] PLYLoader: Header parsed:', {
      vertexCount,
      faceCount,
      vertexPropertyCount: vertexProperties.length,
      properties: vertexProperties
    });
    
    console.log('[GVRM] PLYLoader: Start Fetching', url);
    
    // Calculate header byte length
    const headerByteLength = headerText.indexOf('end_header') + 'end_header\n'.length;
    
    // Parse binary data
    const dataView = new DataView(arrayBuffer, headerByteLength);
    let offset = 0;
    
    const vertices = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);
    
    for (let i = 0; i < vertexCount; i++) {
      vertices[i * 3] = dataView.getFloat32(offset, true); offset += 4;
      vertices[i * 3 + 1] = dataView.getFloat32(offset, true); offset += 4;
      vertices[i * 3 + 2] = dataView.getFloat32(offset, true); offset += 4;
      
      normals[i * 3] = dataView.getFloat32(offset, true); offset += 4;
      normals[i * 3 + 1] = dataView.getFloat32(offset, true); offset += 4;
      normals[i * 3 + 2] = dataView.getFloat32(offset, true); offset += 4;

      // 色はfloat32形式 (SH係数: f_dc_0, f_dc_1, f_dc_2)
      // SH係数からRGBに変換: RGB = SH * 0.28209479177387814
      const SH_C0 = 0.28209479177387814;
      colors[i * 3] = dataView.getFloat32(offset, true) * SH_C0; offset += 4;
      colors[i * 3 + 1] = dataView.getFloat32(offset, true) * SH_C0; offset += 4;
      colors[i * 3 + 2] = dataView.getFloat32(offset, true) * SH_C0; offset += 4;

      // Skip remaining properties (scale_0, scale_1, scale_2)
      for (let j = 9; j < vertexProperties.length; j++) {
        offset += 4; // Assume float for simplicity
      }
    }
    
    // ========== 修正箇所: Auto-scaling & Coordinate Alignment ==========
    // スタックオーバーフローを回避するため、配列とスプレッド構文を使わない

    // Step 1: Calculate bounding box for scaling
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < vertexCount; i++) {
      const x = vertices[i * 3];
      const y = vertices[i * 3 + 1];
      const z = vertices[i * 3 + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    // Step 2: Auto-scale to target height (1.7m)
    const rawHeight = maxY - minY;
    const targetHeight = 1.7;
    const scaleFactor = targetHeight / rawHeight;

    console.log('[GVRM] Auto-scaling... Raw height:', rawHeight.toFixed(3) + 'm', '-> Normalized:', targetHeight.toFixed(3) + 'm', '(factor:', scaleFactor.toFixed(3) + ')');

    for (let i = 0; i < vertexCount * 3; i++) {
      vertices[i] *= scaleFactor;
    }

    // Step 3: Calculate mesh center after scaling (for coordinate alignment)
    let sumX = 0, sumY = 0, sumZ = 0;
    for (let i = 0; i < vertexCount; i++) {
      sumX += vertices[i * 3];
      sumY += vertices[i * 3 + 1];
      sumZ += vertices[i * 3 + 2];
    }

    const meshCenterX = sumX / vertexCount;
    const meshCenterY = sumY / vertexCount;
    const meshCenterZ = sumZ / vertexCount;

    console.log('[GVRM] Mesh center (after scaling):', {
      x: meshCenterX.toFixed(4),
      y: meshCenterY.toFixed(4),
      z: meshCenterZ.toFixed(4)
    });

    // Step 4: Automatic coordinate alignment with camera target
    if (cameraTarget) {
      // Align mesh center with camera target
      // X/Z: align to camera target
      // Y: align mesh center to camera target Y (typically looking at torso)
      const offsetX = cameraTarget[0] - meshCenterX;
      const offsetY = cameraTarget[1] - meshCenterY;
      const offsetZ = cameraTarget[2] - meshCenterZ;

      console.log('[GVRM] Auto-alignment with camera target:', {
        target: cameraTarget,
        offset: [offsetX.toFixed(4), offsetY.toFixed(4), offsetZ.toFixed(4)]
      });

      for (let i = 0; i < vertexCount; i++) {
        vertices[i * 3] += offsetX;
        vertices[i * 3 + 1] += offsetY;
        vertices[i * 3 + 2] += offsetZ;
      }

      console.log('[GVRM] ✅ Mesh automatically aligned to camera target (source image dependent)');
    }
    // ========== 修正終了 ==========
    
    // Parse faces
    const triangles = new Uint32Array(faceCount * 3);
    
    for (let i = 0; i < faceCount; i++) {
      const numVertices = dataView.getUint8(offset); offset += 1;
      
      if (numVertices !== 3) {
        throw new Error('[GVRM] PLYLoader: Only triangular faces are supported');
      }
      
      triangles[i * 3] = dataView.getUint32(offset, true); offset += 4;
      triangles[i * 3 + 1] = dataView.getUint32(offset, true); offset += 4;
      triangles[i * 3 + 2] = dataView.getUint32(offset, true); offset += 4;
    }
    
    return {
      vertices,
      triangles,
      normals,
      colors
    };
  }

  private async loadUVCoords(url: string): Promise<Float32Array> {
    console.log('[GVRM] Loading UV coordinates from:', url);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`[GVRM] Failed to load UV coords: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const uvCoords = new Float32Array(arrayBuffer);

    // Validate UV range
    let minU = Infinity, maxU = -Infinity;
    let minV = Infinity, maxV = -Infinity;

    const numVertices = uvCoords.length / 2;
    for (let i = 0; i < numVertices; i++) {
      const u = uvCoords[i * 2];
      const v = uvCoords[i * 2 + 1];
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }

    console.log('[GVRM] UV coords stats:', {
      vertices: numVertices,
      uRange: `[${minU.toFixed(4)}, ${maxU.toFixed(4)}]`,
      vRange: `[${minV.toFixed(4)}, ${maxV.toFixed(4)}]`
    });

    return uvCoords;
  }

  private async loadSMPLXFaces(url: string): Promise<Uint32Array> {
    console.log('[GVRM] Loading SMPLX faces from:', url);

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`[GVRM] Failed to load SMPLX faces: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const faces = new Uint32Array(arrayBuffer);

    const numFaces = faces.length / 3;

    // Validate face indices
    let minIdx = Infinity, maxIdx = 0;
    for (let i = 0; i < faces.length; i++) {
      if (faces[i] < minIdx) minIdx = faces[i];
      if (faces[i] > maxIdx) maxIdx = faces[i];
    }

    console.log('[GVRM] SMPLX faces stats:', {
      triangles: numFaces,
      indexRange: `[${minIdx}, ${maxIdx}]`
    });

    return faces;
  }

  private async loadSourceCameraConfig(): Promise<{
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
    imageWidth: number;
    imageHeight: number;
    debug?: {
      R_matrix?: number[][];
      T_vector?: number[];
    };
  }> {
    const response = await fetch('/assets/source_camera.json');
    const config = await response.json();
    
    console.log('[GVRM] Source camera config loaded:', {
      hasDebug: !!config.debug,
      hasRMatrix: !!config.debug?.R_matrix,
      hasTVector: !!config.debug?.T_vector
    });
    
    return {
      position: config.position,
      target: config.target,
      fov: config.fov,
      imageWidth: config.imageWidth,
      imageHeight: config.imageHeight,
      debug: config.debug
    };
  }

  dispose(): void {
    if (this.imageEncoder) this.imageEncoder.dispose();
    if (this.templateDecoder) this.templateDecoder.dispose();
    if (this.uvDecoder) this.uvDecoder.dispose();
    if (this.webglRasterizer) this.webglRasterizer.dispose();
    if (this.inverseMapper) this.inverseMapper.dispose();
    if (this.neuralRefiner) this.neuralRefiner.dispose();
    if (this.display) this.display.dispose();
    if (this.gsViewer) this.gsViewer.dispose();
    if (this.gaussianRasterizer) this.gaussianRasterizer.dispose();

    this.initialized = false;
    console.log('[GVRM] Disposed');
  }
}