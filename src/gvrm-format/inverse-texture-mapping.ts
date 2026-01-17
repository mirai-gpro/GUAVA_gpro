/**
 * GUAVA Paper Compliant: Inverse Texture Mapping with Pose Space Mesh
 * 
 * 修正版 - 初期化とエラーハンドリングを改善
 */

interface CameraParams {
  position: number[];
  target: number[];
  fov: number;
  viewport: { width: number; height: number };
}

interface EHMMesh {
  vertices: Float32Array;   // [N, 3]
  triangles: Uint32Array;   // [M, 3]
  uvCoords: Float32Array;   // [N, 2]
}

interface UVMapping {
  triangleIndices: Uint32Array;    // [P] which triangle each UV pixel belongs to
  barycentricCoords: Float32Array; // [P, 3] barycentric coordinates
  worldPositions: Float32Array;    // [P, 3] 3D positions in world space
  validMask: Uint8Array;           // [P] 1 if valid, 0 if invalid
}

interface PoseParams {
  bodyPose: Float32Array;      // SMPLX pose parameters
  globalRotation: Float32Array; // 3x3 rotation matrix
  translation: Float32Array;    // [3] translation vector
}

/**
 * MeshData interface for WebGL UV Rasterizer
 * Used by webgl-uv-rasterizer.ts
 */
export interface MeshData {
  vertices: Float32Array;     // [N * 3] vertex positions (x, y, z)
  triangles: Uint32Array;     // [M * 3] triangle indices
  uvCoords: Float32Array;     // [N * 2] UV coordinates per vertex (u, v)
  numVertices: number;        // Number of vertices
  numTriangles: number;       // Number of triangles
}

/**
 * 🔧 修正版: Inverse Texture Mapping
 */
export class InverseTextureMapper {
  private uvResolution: number = 1024;
  private camera: CameraParams | null = null;
  private viewMatrix: Float32Array | null = null;
  private projMatrix: Float32Array | null = null;
  private initialized: boolean = false;
  
  constructor() {
    console.log('[InverseTextureMapper] Created');
    console.log('[InverseTextureMapper] ⚠️  Call initialize() before use');
  }
  
  /**
   * 🔧 修正: 初期化状態を追跡
   */
  initialize(uvResolution: number, camera: CameraParams): void {
    try {
      console.log('[InverseTextureMapper] Initializing...');
      
      // 入力検証
      if (!camera || !camera.position || !camera.target || !camera.viewport) {
        throw new Error('Invalid camera parameters');
      }
      
      if (uvResolution <= 0 || uvResolution > 4096) {
        throw new Error(`Invalid UV resolution: ${uvResolution}`);
      }
      
      this.uvResolution = uvResolution;
      this.camera = camera;
      
      // Build view and projection matrices
      this.viewMatrix = this.buildViewMatrix();
      this.projMatrix = this.buildProjectionMatrix();
      
      // 行列の検証
      if (!this.viewMatrix || !this.projMatrix) {
        throw new Error('Failed to build camera matrices');
      }
      
      this.initialized = true;
      
      console.log('[InverseTextureMapper] ✅ Initialized successfully');
      console.log(`  UV Resolution: ${uvResolution}×${uvResolution}`);
      console.log(`  Viewport: ${camera.viewport.width}×${camera.viewport.height}`);
      console.log(`  Camera position: [${camera.position.map(v => v.toFixed(2)).join(', ')}]`);
      console.log(`  Camera target: [${camera.target.map(v => v.toFixed(2)).join(', ')}]`);
      
    } catch (error) {
      this.initialized = false;
      console.error('[InverseTextureMapper] ❌ Initialization failed:', error);
      throw error;
    }
  }
  
  /**
   * 🔧 修正: 初期化チェックを改善、asyncを削除（不要なため）
   */
  map(
    uvMapping: UVMapping,
    appearanceFeatureMap: Float32Array,
    featureMapSize: { width: number; height: number },
    featureChannels: number = 128
  ): Float32Array {
    
    // 詳細な初期化チェック
    if (!this.initialized) {
      throw new Error(
        '[InverseTextureMapper] Not initialized! Call initialize(uvResolution, camera) first.'
      );
    }
    
    if (!this.camera) {
      throw new Error('[InverseTextureMapper] Camera not set');
    }
    
    if (!this.viewMatrix) {
      throw new Error('[InverseTextureMapper] View matrix not built');
    }
    
    if (!this.projMatrix) {
      throw new Error('[InverseTextureMapper] Projection matrix not built');
    }
    
    // 入力検証 - より詳細なエラーメッセージ
    if (!uvMapping) {
      throw new Error('[InverseTextureMapper] UV mapping is null or undefined');
    }
    
    const missingFields: string[] = [];
    if (!uvMapping.triangleIndices) missingFields.push('triangleIndices');
    if (!uvMapping.barycentricCoords) missingFields.push('barycentricCoords');
    if (!uvMapping.worldPositions) missingFields.push('worldPositions');
    if (!uvMapping.validMask) missingFields.push('validMask');
    
    if (missingFields.length > 0) {
      console.error('[InverseTextureMapper] UV mapping structure:', {
        hasTriangleIndices: !!uvMapping.triangleIndices,
        hasBarycentricCoords: !!uvMapping.barycentricCoords,
        hasWorldPositions: !!uvMapping.worldPositions,
        hasValidMask: !!uvMapping.validMask,
        allKeys: Object.keys(uvMapping)
      });
      throw new Error(
        `[InverseTextureMapper] Invalid UV mapping: missing fields [${missingFields.join(', ')}]`
      );
    }
    
    // 配列の長さチェック
    if (uvMapping.validMask.length === 0) {
      throw new Error('[InverseTextureMapper] UV mapping validMask is empty');
    }
    
    if (uvMapping.worldPositions.length === 0) {
      throw new Error('[InverseTextureMapper] UV mapping worldPositions is empty');
    }
    
    if (!appearanceFeatureMap || appearanceFeatureMap.length === 0) {
      throw new Error('[InverseTextureMapper] Invalid appearance feature map');
    }
    
    const expectedFeatureMapSize = featureMapSize.width * featureMapSize.height * featureChannels;
    if (appearanceFeatureMap.length !== expectedFeatureMapSize) {
      console.warn(
        `[InverseTextureMapper] ⚠️ Feature map size mismatch: ` +
        `expected ${expectedFeatureMapSize}, got ${appearanceFeatureMap.length}`
      );
    }
    
    console.log('[InverseTextureMapper] 🚀 Starting inverse texture mapping...');
    console.log(`  UV Resolution: ${this.uvResolution}×${this.uvResolution}`);
    console.log(`  Feature Map: ${featureMapSize.width}×${featureMapSize.height}×${featureChannels}`);
    console.log(`  Valid UV pixels: ${uvMapping.validMask.filter(v => v > 0).length.toLocaleString()}`);
    
    try {
      // Sample features from screen space to UV space
      const uvFeatures = this.sampleFeaturesSimplified(
        uvMapping,
        appearanceFeatureMap,
        featureMapSize,
        featureChannels
      );
      
      // Validate results
      const validCount = this.countValidSamples(uvFeatures, featureChannels);
      const totalPixels = this.uvResolution * this.uvResolution;
      
      console.log('[InverseTextureMapper] ✅ Mapping complete');
      console.log(`  Valid samples: ${validCount.toLocaleString()} / ${totalPixels.toLocaleString()} (${(validCount/totalPixels*100).toFixed(1)}%)`);
      
      if (validCount === 0) {
        console.error('[InverseTextureMapper] ❌ CRITICAL: No valid samples!');
        console.error('  Possible causes:');
        console.error('  - All vertices are behind the camera');
        console.error('  - UV mapping is invalid');
        console.error('  - Camera parameters are incorrect');
      } else if (validCount < totalPixels * 0.05) {
        console.warn(`[InverseTextureMapper] ⚠️ Low coverage: only ${(validCount/totalPixels*100).toFixed(1)}% of UV space is filled`);
      }
      
      return uvFeatures;
      
    } catch (error) {
      console.error('[InverseTextureMapper] ❌ Mapping failed:', error);
      throw error;
    }
  }
  
  /**
   * 🔧 修正: より堅牢なエラーハンドリング
   */
  private sampleFeaturesSimplified(
    uvMapping: UVMapping,
    featureMap: Float32Array,
    featureMapSize: { width: number; height: number },
    featureChannels: number
  ): Float32Array {
    
    const numPixels = this.uvResolution * this.uvResolution;
    const uvFeatures = new Float32Array(numPixels * featureChannels);
    
    let sampledCount = 0;
    let outOfBoundsCount = 0;
    let behindCameraCount = 0;
    
    for (let i = 0; i < numPixels; i++) {
      // UV mapping が無効な場合はスキップ
      if (uvMapping.validMask[i] === 0) continue;
      
      // Get 3D world position
      const wx = uvMapping.worldPositions[i * 3 + 0];
      const wy = uvMapping.worldPositions[i * 3 + 1];
      const wz = uvMapping.worldPositions[i * 3 + 2];
      
      // 位置の検証
      if (!isFinite(wx) || !isFinite(wy) || !isFinite(wz)) {
        continue;
      }
      
      // Project to screen space
      const [sx, sy, depth, visible] = this.projectToScreen(wx, wy, wz);
      
      if (!visible || depth < 0) {
        behindCameraCount++;
        continue;
      }
      
      // Check if within screen bounds (少し余裕を持たせる)
      if (sx < 0 || sx >= featureMapSize.width || sy < 0 || sy >= featureMapSize.height) {
        outOfBoundsCount++;
        continue;
      }
      
      // Sample feature map using bilinear interpolation
      const sampledFeature = this.bilinearSample(
        featureMap,
        featureMapSize.width,
        featureMapSize.height,
        featureChannels,
        sx,
        sy
      );
      
      // Copy to UV feature map
      for (let c = 0; c < featureChannels; c++) {
        uvFeatures[i * featureChannels + c] = sampledFeature[c];
      }
      
      sampledCount++;
    }
    
    console.log('[InverseTextureMapper] Sampling statistics:');
    console.log(`  Sampled: ${sampledCount.toLocaleString()}`);
    console.log(`  Behind camera: ${behindCameraCount.toLocaleString()}`);
    console.log(`  Out of bounds: ${outOfBoundsCount.toLocaleString()}`);
    
    return uvFeatures;
  }
  
  /**
   * Project 3D world point to screen space
   * 🔧 修正: より詳細なデバッグ情報
   */
  private projectToScreen(wx: number, wy: number, wz: number): [number, number, number, boolean] {
    if (!this.viewMatrix || !this.projMatrix || !this.camera) {
      return [0, 0, 0, false];
    }
    
    // Transform to view space
    const vx = this.viewMatrix[0] * wx + this.viewMatrix[4] * wy + this.viewMatrix[8] * wz + this.viewMatrix[12];
    const vy = this.viewMatrix[1] * wx + this.viewMatrix[5] * wy + this.viewMatrix[9] * wz + this.viewMatrix[13];
    const vz = this.viewMatrix[2] * wx + this.viewMatrix[6] * wy + this.viewMatrix[10] * wz + this.viewMatrix[14];
    const vw = this.viewMatrix[3] * wx + this.viewMatrix[7] * wy + this.viewMatrix[11] * wz + this.viewMatrix[15];
    
    // Check if behind camera (OpenGL: -Z is forward)
    if (vz >= 0) {
      return [0, 0, 0, false];
    }
    
    // Transform to clip space
    const cx = this.projMatrix[0] * vx + this.projMatrix[4] * vy + this.projMatrix[8] * vz + this.projMatrix[12] * vw;
    const cy = this.projMatrix[1] * vx + this.projMatrix[5] * vy + this.projMatrix[9] * vz + this.projMatrix[13] * vw;
    const cz = this.projMatrix[2] * vx + this.projMatrix[6] * vy + this.projMatrix[10] * vz + this.projMatrix[14] * vw;
    const cw = this.projMatrix[3] * vx + this.projMatrix[7] * vy + this.projMatrix[11] * vz + this.projMatrix[15] * vw;
    
    if (Math.abs(cw) < 1e-8) {
      return [0, 0, 0, false];
    }
    
    // Perspective divide
    const ndcX = cx / cw;
    const ndcY = cy / cw;
    const ndcZ = cz / cw;
    
    // Convert to screen coordinates
    const screenX = (ndcX + 1) * 0.5 * this.camera.viewport.width;
    const screenY = (1 - ndcY) * 0.5 * this.camera.viewport.height; // Flip Y
    
    return [screenX, screenY, ndcZ, true];
  }
  
  private buildViewMatrix(): Float32Array {
    if (!this.camera) throw new Error('Camera not initialized');
    
    const eye = this.camera.position;
    const target = this.camera.target;
    const up = [0, 1, 0]; // Y-up
    
    // Camera coordinate system
    const zAxis = [
      eye[0] - target[0],
      eye[1] - target[1],
      eye[2] - target[2]
    ];
    const zLen = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
    
    if (zLen < 1e-8) {
      throw new Error('Camera position and target are too close');
    }
    
    zAxis[0] /= zLen; zAxis[1] /= zLen; zAxis[2] /= zLen;
    
    const xAxis = [
      up[1] * zAxis[2] - up[2] * zAxis[1],
      up[2] * zAxis[0] - up[0] * zAxis[2],
      up[0] * zAxis[1] - up[1] * zAxis[0]
    ];
    const xLen = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
    
    if (xLen < 1e-8) {
      throw new Error('Camera up vector is parallel to view direction');
    }
    
    xAxis[0] /= xLen; xAxis[1] /= xLen; xAxis[2] /= xLen;
    
    const yAxis = [
      zAxis[1] * xAxis[2] - zAxis[2] * xAxis[1],
      zAxis[2] * xAxis[0] - zAxis[0] * xAxis[2],
      zAxis[0] * xAxis[1] - zAxis[1] * xAxis[0]
    ];
    
    // View matrix (column-major)
    return new Float32Array([
      xAxis[0], yAxis[0], zAxis[0], 0,
      xAxis[1], yAxis[1], zAxis[1], 0,
      xAxis[2], yAxis[2], zAxis[2], 0,
      -(xAxis[0]*eye[0] + xAxis[1]*eye[1] + xAxis[2]*eye[2]),
      -(yAxis[0]*eye[0] + yAxis[1]*eye[1] + yAxis[2]*eye[2]),
      -(zAxis[0]*eye[0] + zAxis[1]*eye[1] + zAxis[2]*eye[2]),
      1
    ]);
  }
  
  private buildProjectionMatrix(): Float32Array {
    if (!this.camera) throw new Error('Camera not initialized');
    
    const fov = this.camera.fov * Math.PI / 180; // Convert to radians
    const aspect = this.camera.viewport.width / this.camera.viewport.height;
    const near = 0.1;
    const far = 100.0;
    
    if (fov <= 0 || fov >= Math.PI) {
      throw new Error(`Invalid FOV: ${this.camera.fov}°`);
    }
    
    const f = 1.0 / Math.tan(fov / 2);
    
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / (near - far), -1,
      0, 0, (2 * far * near) / (near - far), 0
    ]);
  }
  
  private bilinearSample(
    data: Float32Array,
    width: number,
    height: number,
    channels: number,
    x: number,
    y: number
  ): Float32Array {
    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, width - 1);
    const y0 = Math.floor(y);
    const y1 = Math.min(y0 + 1, height - 1);
    
    const fx = x - x0;
    const fy = y - y0;
    
    const result = new Float32Array(channels);
    
    for (let c = 0; c < channels; c++) {
      const v00 = data[(y0 * width + x0) * channels + c] || 0;
      const v10 = data[(y0 * width + x1) * channels + c] || 0;
      const v01 = data[(y1 * width + x0) * channels + c] || 0;
      const v11 = data[(y1 * width + x1) * channels + c] || 0;
      
      result[c] = 
        v00 * (1 - fx) * (1 - fy) +
        v10 * fx * (1 - fy) +
        v01 * (1 - fx) * fy +
        v11 * fx * fy;
    }
    
    return result;
  }
  
  private countValidSamples(features: Float32Array, featureChannels: number): number {
    let count = 0;
    const numPixels = features.length / featureChannels;
    
    for (let i = 0; i < numPixels; i++) {
      let hasNonZero = false;
      for (let c = 0; c < featureChannels; c++) {
        if (Math.abs(features[i * featureChannels + c]) > 1e-6) {
          hasNonZero = true;
          break;
        }
      }
      if (hasNonZero) count++;
    }
    
    return count;
  }
  
  /**
   * 🆕 デバッグ用: 初期化状態を確認
   */
  isInitialized(): boolean {
    return this.initialized;
  }
  
  /**
   * 🆕 デバッグ用: カメラ情報を取得
   */
  getCameraInfo(): any {
    if (!this.camera) return null;
    return {
      position: this.camera.position,
      target: this.camera.target,
      fov: this.camera.fov,
      viewport: this.camera.viewport,
      initialized: this.initialized
    };
  }
}

/**
 * Integration with GUAVA pipeline
 */
export async function createInverseTextureMapperForGUAVA(
  sourceCameraPath: string,
  uvResolution: number = 1024
): Promise<InverseTextureMapper> {
  
  console.log('[GUAVA] Loading source camera parameters...');
  
  const response = await fetch(sourceCameraPath);
  const cameraConfig = await response.json();
  
  const camera: CameraParams = {
    position: cameraConfig.position,
    target: cameraConfig.target,
    fov: cameraConfig.fov,
    viewport: {
      width: cameraConfig.imageWidth,
      height: cameraConfig.imageHeight
    }
  };
  
  const mapper = new InverseTextureMapper();
  mapper.initialize(uvResolution, camera);
  
  return mapper;
}

/**
 * Simple wrapper for lazy initialization in GVRM class
 */
export function createUninitializedMapper(): InverseTextureMapper {
  return new InverseTextureMapper();
}
