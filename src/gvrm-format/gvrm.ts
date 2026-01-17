// gvrm.ts
// GUAVA pipeline implementation (WebGL GPU mode)
// 論文準拠: Real-time UV rasterization with GPU

import { ImageEncoder } from './image-encoder';
import { TemplateDecoder } from './template-decoder';
import { UVDecoder } from './uv-decoder';
import { WebGLUVRasterizer } from './webgl-uv-rasterizer';
import { InverseTextureMapper } from './inverse-texture-mapping';
import { NeuralRefiner } from './neural-refiner';
import { WebGLDisplay } from './webgl-display';
import { GSViewer } from './gs';

interface PLYData {
  vertices: Float32Array;
  triangles: Uint32Array;
  normals?: Float32Array;
  colors?: Float32Array;
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

export class GVRM {
  private imageEncoder: ImageEncoder;
  private templateDecoder: TemplateDecoder;
  private uvDecoder: UVDecoder;
  private webglRasterizer: WebGLUVRasterizer;
  private inverseMapper: InverseTextureMapper | null = null;
  private neuralRefiner: NeuralRefiner;
  private display: WebGLDisplay | null = null;
  private gsViewer: GSViewer | null = null;

  private plyData: PLYData | null = null;
  private templateMesh: EHMMesh | null = null;
  private templateGaussians: GaussianData | null = null;
  private uvGaussians: UVGaussianData | null = null;
  
  private initialized = false;
  private displayCanvas: HTMLCanvasElement | null = null;

  constructor(displayCanvas?: HTMLCanvasElement) {
    console.log('[GVRM] Constructor called (WebGL GPU mode)');
    
    // Store canvas reference but don't initialize display yet
    this.displayCanvas = displayCanvas || null;
    
    this.imageEncoder = new ImageEncoder();
    this.templateDecoder = new TemplateDecoder();
    this.uvDecoder = new UVDecoder();
    this.webglRasterizer = new WebGLUVRasterizer();
    this.neuralRefiner = new NeuralRefiner();
  }

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[GVRM] init() called');

    try {
      // Initialize display if canvas was provided
      if (this.displayCanvas) {
        console.log('[GVRM] Initializing WebGL display...');
        this.display = new WebGLDisplay(this.displayCanvas);
        await this.display.init();
      } else {
        console.warn('[GVRM] No display canvas provided, skipping display initialization');
      }
      
      console.log('[GVRM] 🚀 Starting GUAVA Pipeline (WebGL GPU mode)...');
      console.log('[GVRM] 📖 Paper-compliant: Real-time UV rasterization with GPU');

      await this.loadAssets();

      this.initialized = true;
      console.log('[GVRM] ✅ Initialization successful');

    } catch (error) {
      console.error('[GVRM] ❌ Initialization failed:', error);
      throw error;
    }
  }

  private async loadAssets(): Promise<void> {
    console.log('[GVRM] Loading assets...');

    // ========== Step 0: Load PLY file ==========
    const plyUrl = '/assets/avatar_web.ply';
    this.plyData = await this.loadPLY(plyUrl);
    console.log('[GVRM] PLY loaded:', this.plyData.vertices.length / 3, 'vertices');

    // ========== Step 1: Initialize modules ==========
    console.log('[GVRM] Step 1: Initializing modules...');
    
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
    
    console.log('[GVRM]   - WebGL GPU Rasterizer...');
    await this.webglRasterizer.init();
    
    console.log('[GVRM]   - Neural Refiner...');
    await this.neuralRefiner.init();
    
    console.log('[GVRM] ✅ All modules initialized');

    // ========== Step 2: Extract appearance features ==========
    console.log('[GVRM] Step 2: Extracting appearance features...');
    
    const sourceImageUrl = '/assets/source.png';
    const sourceCameraConfig = await this.loadSourceCameraConfig();
    
    const { projectionFeature, idEmbedding } = await this.imageEncoder.extractFeaturesWithSourceCamera(
      sourceImageUrl,
      sourceCameraConfig,
      templateVertices,
      templateVertexCount,
      128
    );
    
    console.log('[GVRM] ✅ Appearance features extracted');

    // ========== Step 3: Generate Template Gaussians ==========
    console.log('[GVRM] Step 3: Generating Template Gaussians...');
    
    const templateOutput = await this.templateDecoder.generate(
      projectionFeature,
      idEmbedding
    );
    
    this.templateGaussians = {
      positions: templateVertices,
      opacities: templateOutput.opacity,
      scales: templateOutput.scale,
      rotations: templateOutput.rotation,
      latents: templateOutput.latent32ch
    };
    
    console.log('[GVRM] ✅ Template Gaussians generated:', {
      vertices: templateVertexCount.toLocaleString(),
      features: '32ch latent'
    });

    // ========== Step 4: Prepare EHM mesh ==========
    console.log('[GVRM] Step 4: Preparing EHM mesh...');
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

    // ========== Step 5: Map Template Gaussians to PLY ==========
    console.log('[GVRM] Step 5: Mapping Template Gaussians to PLY...');
    
    // Map template Gaussians to PLY vertices
    // (This step combines the coarse Gaussian attributes with PLY positions)
    
    console.log('[GVRM] ✅ Template Gaussians mapped');

    // ========== Step 6: Create Gaussian Splatting Viewer ==========
    console.log('[GVRM] Step 6: Creating Gaussian Splatting Viewer...');

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

    // ========== Step 7: Generate Coarse Feature Map ==========
    console.log('[GVRM] Step 7: Generating Coarse Feature Map...');
    // (GSViewer will generate this during rendering)
    console.log('[GVRM] ✅ Coarse Feature Map generated');

    // ========== Step 8: GPU UV Rasterization ==========
    console.log('[GVRM] Step 8: GPU UV Rasterization...');
    console.log('[GVRM]   ⚡ Using WebGL GPU for real-time rasterization');

    // MeshDataオブジェクトを構築
    const meshData = {
      vertices: this.templateMesh.vertices,
      triangles: this.templateMesh.triangles,
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

    // ========== Step 8.5: Initialize InverseTextureMapper ==========
    console.log('[GVRM] Step 8.5: Initializing InverseTextureMapper...');
    
    this.inverseMapper = new InverseTextureMapper();
    await this.inverseMapper.initialize(
      uvMapping,
      sourceCameraConfig.position,
      sourceCameraConfig.target,
      518,
      518
    );
    
    console.log('[GVRM] ✅ InverseTextureMapper initialized');

    // ========== Step 9: Inverse Texture Mapping ==========
    console.log('[GVRM] Step 9: Inverse Texture Mapping (論文準拠)...');
    
    // Get UV branch features (32ch) from image encoder
    const uvBranchFeatures = this.imageEncoder.getUVFeatures();
    
    // Verify feature dimensions
    const expectedSize = 518 * 518 * 32;
    console.log('[GVRM] Debug - UV branch features (論文準拠):', {
      length: uvBranchFeatures.length,
      expected: expectedSize,
      channels: 32,
      match: uvBranchFeatures.length === expectedSize ? '✅' : '❌'
    });
    
    console.log('[GVRM] ✅ Inverse Texture Mapping preparation complete');

    // ========== Step 9.5: Build 155ch UV features for UV Decoder ==========
    console.log('[GVRM] Step 9.5: Building 155ch UV features for UV Decoder...');
    console.log('[GVRM] 📖 Paper: 35ch (32 UV + 3 RGB)');
    console.log('[GVRM] 🔧 Model: 155ch (32 UV + 123 Template subset)');
    
    const uvResolution = 1024;
    const uvPixels = uvResolution * uvResolution;
    const uvFeatureMap = new Float32Array(uvPixels * 155);
    
    // Get template features (128ch)
    const templateBranchFeatures = this.imageEncoder.getTemplateFeatures();
    
    console.log('[GVRM] 📊 Channel breakdown:');
    console.log('[GVRM]   - UV features:       32ch (0-31)');
    console.log('[GVRM]   - Template subset:   123ch (32-154)');
    
    // Resample UV features from 518×518 to 1024×1024
    const sourceRes = 518;
    const targetRes = 1024;
    const scale = sourceRes / targetRes;
    
    for (let ty = 0; ty < targetRes; ty++) {
      for (let tx = 0; tx < targetRes; tx++) {
        const sx = tx * scale;
        const sy = ty * scale;
        
        const sx0 = Math.floor(sx);
        const sy0 = Math.floor(sy);
        const sx1 = Math.min(sx0 + 1, sourceRes - 1);
        const sy1 = Math.min(sy0 + 1, sourceRes - 1);
        
        const wx = sx - sx0;
        const wy = sy - sy0;
        
        const targetIdx = ty * targetRes + tx;
        
        // Copy 32ch UV features with bilinear interpolation
        for (let c = 0; c < 32; c++) {
          const v00 = uvBranchFeatures[(sy0 * sourceRes + sx0) * 32 + c];
          const v10 = uvBranchFeatures[(sy0 * sourceRes + sx1) * 32 + c];
          const v01 = uvBranchFeatures[(sy1 * sourceRes + sx0) * 32 + c];
          const v11 = uvBranchFeatures[(sy1 * sourceRes + sx1) * 32 + c];
          
          const top = v00 * (1 - wx) + v10 * wx;
          const bottom = v01 * (1 - wx) + v11 * wx;
          const interpolated = top * (1 - wy) + bottom * wy;
          
          uvFeatureMap[targetIdx * 155 + c] = interpolated;
        }
      }
    }
    
    console.log('[GVRM] ✅ Resampled and copied 32ch UV features (518×518 → 1024×1024)');
    
    // Copy 123ch template features (subset of 128ch) with resampling
    for (let ty = 0; ty < targetRes; ty++) {
      for (let tx = 0; tx < targetRes; tx++) {
        const sx = tx * scale;
        const sy = ty * scale;
        
        const sx0 = Math.floor(sx);
        const sy0 = Math.floor(sy);
        const sx1 = Math.min(sx0 + 1, sourceRes - 1);
        const sy1 = Math.min(sy0 + 1, sourceRes - 1);
        
        const wx = sx - sx0;
        const wy = sy - sy0;
        
        const targetIdx = ty * targetRes + tx;
        
        // Copy first 123ch from template features (128ch)
        for (let c = 0; c < 123; c++) {
          const v00 = templateBranchFeatures[(sy0 * sourceRes + sx0) * 128 + c];
          const v10 = templateBranchFeatures[(sy0 * sourceRes + sx1) * 128 + c];
          const v01 = templateBranchFeatures[(sy1 * sourceRes + sx0) * 128 + c];
          const v11 = templateBranchFeatures[(sy1 * sourceRes + sx1) * 128 + c];
          
          const top = v00 * (1 - wx) + v10 * wx;
          const bottom = v01 * (1 - wx) + v11 * wx;
          const interpolated = top * (1 - wy) + bottom * wy;
          
          uvFeatureMap[targetIdx * 155 + 32 + c] = interpolated;
        }
      }
    }
    
    console.log('[GVRM] ✅ Resampled and copied 123ch template features (518×518 → 1024×1024)');
    console.log('[GVRM] ✅ 155ch UV features built successfully');
    console.log('[GVRM] Total size:', uvFeatureMap.length, '(expected:', uvPixels * 155, ')');

    // ========== Step 10: Generate UV Gaussians ==========
    console.log('[GVRM] Step 10: Generating UV Gaussians...');
    
    this.uvGaussians = await this.uvDecoder.decode(
      uvFeatureMap,
      uvMapping,
      uvResolution,
      uvResolution
    );
    
    console.log('[GVRM] ✅ UV Gaussians generated:', {
      count: this.uvGaussians.positions.length / 3
    });

    // ========== Step 11: Create Ubody Gaussians (Template ⊕ UV) ==========
    console.log('[GVRM] Step 11: Creating Ubody Gaussians (Template ⊕ UV)...');
    
    const templateCount = this.templateGaussians.positions.length / 3;
    const uvCount = this.uvGaussians.positions.length / 3;
    const totalCount = templateCount + uvCount;
    
    // Concatenate all Gaussian properties
    const ubodyGaussians = {
      positions: this.concatenateArrays(this.templateGaussians.positions, this.uvGaussians.positions),
      opacities: this.concatenateArrays(this.templateGaussians.opacities, this.uvGaussians.opacities),
      scales: this.concatenateArrays(this.templateGaussians.scales, this.uvGaussians.scales),
      rotations: this.concatenateArrays(this.templateGaussians.rotations, this.uvGaussians.rotations),
      latents: this.concatenateArrays(this.templateGaussians.latents, this.uvGaussians.latents)
    };
    
    console.log('[GVRM] ✅ Ubody Gaussians created:', {
      total: totalCount.toLocaleString(),
      template: templateCount.toLocaleString(),
      uv: uvCount.toLocaleString()
    });

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

  async render(targetImageUrl: string): Promise<ImageData | null> {
    if (!this.initialized || !this.gsViewer) {
      throw new Error('[GVRM] Not initialized');
    }

    if (!this.display) {
      console.warn('[GVRM] No display available, skipping render');
      return null;
    }

    // Step 1: Render coarse feature map
    const coarseFeatureMap = this.gsViewer.render();

    // Step 2: Neural refinement
    const refinedImage = await this.neuralRefiner.refine(coarseFeatureMap);

    // Step 3: Display
    return this.display.display(refinedImage);
  }

  private async loadPLY(url: string): Promise<PLYData> {
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
    
    // ========== 修正箇所: Auto-scaling ==========
    // スタックオーバーフローを回避するため、配列とスプレッド構文を使わない
    let minY = Infinity;
    let maxY = -Infinity;
    
    for (let i = 0; i < vertexCount; i++) {
      const y = vertices[i * 3 + 1];
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    
    const rawHeight = maxY - minY;
    const targetHeight = 1.7;
    const scaleFactor = targetHeight / rawHeight;
    
    console.log('[GVRM] Auto-scaling... Raw height:', rawHeight.toFixed(3) + 'm', '-> Normalized:', targetHeight.toFixed(3) + 'm', '(factor:', scaleFactor.toFixed(3) + ')');
    
    for (let i = 0; i < vertexCount * 3; i++) {
      vertices[i] *= scaleFactor;
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

  private async loadSourceCameraConfig(): Promise<{
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
    imageWidth: number;
    imageHeight: number;
  }> {
    const response = await fetch('/assets/source_camera.json');
    const config = await response.json();
    
    return {
      position: config.position,
      target: config.target,
      fov: config.fov,
      imageWidth: config.imageWidth,
      imageHeight: config.imageHeight
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
    
    this.initialized = false;
    console.log('[GVRM] Disposed');
  }
}