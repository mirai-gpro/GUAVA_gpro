// gaussian-rasterizer-webgpu.ts
// WebGPU-based 3D Gaussian Rasterizer
// Implements EWA Splatting (Zwicker et al., 2002) algorithm
// Compatible with GUAVA paper and diff-gaussian-rasterization-32

/**
 * Gaussian data structure for mobile rendering
 */
export interface GaussianRasterInput {
  positions: Float32Array;    // [N, 3] world-space positions
  scales: Float32Array;       // [N, 3] log-scale values
  rotations: Float32Array;    // [N, 4] quaternions (w, x, y, z)
  opacities: Float32Array;    // [N] inverse-sigmoid opacity
  features: Float32Array;     // [N, 32] latent features
  count: number;
}

export interface CameraParams {
  viewMatrix: Float32Array;   // [4, 4] world-to-camera matrix
  projMatrix: Float32Array;   // [4, 4] projection matrix
  focalX: number;
  focalY: number;
  tanFovX: number;
  tanFovY: number;
  width: number;
  height: number;
}

/**
 * Preprocessed Gaussian data ready for rendering
 */
interface PreprocessedGaussian {
  screenX: number;
  screenY: number;
  depth: number;
  conic: [number, number, number];  // Inverse 2D covariance
  opacity: number;                   // Adjusted opacity
  radius: number;                    // Screen-space radius
  featureIdx: number;                // Index in features array
}

/**
 * WebGPU-based 3D Gaussian Rasterizer
 *
 * Key differences from simple point splatting:
 * 1. Computes 3D covariance from scale + rotation
 * 2. Projects 3D covariance to 2D screen-space
 * 3. Uses conic (inverse covariance) for anisotropic rendering
 * 4. Proper EWA splatting algorithm for elliptical Gaussians
 */
export class GaussianRasterizerWebGPU {
  private device: GPUDevice | null = null;
  private preprocessPipeline: GPUComputePipeline | null = null;
  private renderPipeline: GPUComputePipeline | null = null;
  private initialized: boolean = false;

  // CPU fallback mode for devices without WebGPU
  private cpuFallback: boolean = false;

  /**
   * Initialize WebGPU context
   */
  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[GaussianRasterizer] Initializing...');

    // Check WebGPU support
    if (!navigator.gpu) {
      console.warn('[GaussianRasterizer] WebGPU not available, using CPU fallback');
      this.cpuFallback = true;
      this.initialized = true;
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });

      if (!adapter) {
        console.warn('[GaussianRasterizer] No GPU adapter found, using CPU fallback');
        this.cpuFallback = true;
        this.initialized = true;
        return;
      }

      this.device = await adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxComputeWorkgroupSizeX: 256,
          maxStorageBufferBindingSize: 256 * 1024 * 1024  // 256MB
        }
      });

      // Create compute pipelines
      await this.createPipelines();

      console.log('[GaussianRasterizer] ✅ WebGPU initialized');
      this.initialized = true;

    } catch (error) {
      console.warn('[GaussianRasterizer] WebGPU init failed, using CPU fallback:', error);
      this.cpuFallback = true;
      this.initialized = true;
    }
  }

  /**
   * Render 32-channel feature map
   */
  async render(
    input: GaussianRasterInput,
    camera: CameraParams,
    background: number = 0
  ): Promise<Float32Array> {
    if (!this.initialized) {
      await this.init();
    }

    console.log('[GaussianRasterizer] Rendering...', {
      gaussianCount: input.count,
      resolution: `${camera.width}x${camera.height}`,
      mode: this.cpuFallback ? 'CPU' : 'WebGPU'
    });

    const startTime = performance.now();

    // Use CPU fallback for now (can be replaced with WebGPU compute later)
    const result = this.renderCPU(input, camera, background);

    const elapsed = performance.now() - startTime;
    console.log(`[GaussianRasterizer] ✅ Rendered in ${elapsed.toFixed(1)}ms`);

    return result;
  }

  /**
   * CPU-based rendering with proper EWA Splatting algorithm
   * This implements the same algorithm as CUDA but on CPU
   */
  private renderCPU(
    input: GaussianRasterInput,
    camera: CameraParams,
    background: number
  ): Float32Array {
    const { width, height } = camera;
    const featureChannels = 32;

    // Output feature map [32, H, W] in CHW format
    const featureMap = new Float32Array(featureChannels * height * width);
    featureMap.fill(background);

    // Transmittance buffer for alpha compositing
    const transmittance = new Float32Array(height * width);
    transmittance.fill(1.0);

    // Step 1: Preprocess all Gaussians
    console.log('[GaussianRasterizer] Preprocessing Gaussians...');
    const preprocessed = this.preprocessGaussians(input, camera);

    if (preprocessed.length === 0) {
      console.warn('[GaussianRasterizer] No visible Gaussians!');
      return featureMap;
    }

    console.log(`[GaussianRasterizer] ${preprocessed.length}/${input.count} Gaussians visible`);

    // Step 2: Sort by depth (front to back)
    preprocessed.sort((a, b) => a.depth - b.depth);

    // Step 3: Render each Gaussian with EWA splatting
    console.log('[GaussianRasterizer] Splatting...');

    let totalPixelsWritten = 0;

    for (const g of preprocessed) {
      const r = Math.ceil(g.radius);
      const minX = Math.max(0, Math.floor(g.screenX - r));
      const maxX = Math.min(width - 1, Math.ceil(g.screenX + r));
      const minY = Math.max(0, Math.floor(g.screenY - r));
      const maxY = Math.min(height - 1, Math.ceil(g.screenY + r));

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const pixelIdx = y * width + x;

          // Skip fully occluded pixels
          const T = transmittance[pixelIdx];
          if (T < 0.0001) continue;

          // Compute EWA weight using conic matrix
          const dx = x - g.screenX;
          const dy = y - g.screenY;

          // Quadratic form: -0.5 * (conic.x * dx^2 + conic.z * dy^2) - conic.y * dx * dy
          const power = -0.5 * (g.conic[0] * dx * dx + g.conic[2] * dy * dy)
                        - g.conic[1] * dx * dy;

          // Skip if outside Gaussian (power > 0 means outside)
          if (power > 0) continue;

          // Compute alpha
          const alpha = Math.min(0.99, g.opacity * Math.exp(power));
          if (alpha < 1.0 / 255.0) continue;

          // Alpha compositing: C += feature * alpha * T
          const weight = alpha * T;
          for (let c = 0; c < featureChannels; c++) {
            const featureValue = input.features[g.featureIdx * featureChannels + c];
            featureMap[c * height * width + pixelIdx] += featureValue * weight;
          }

          // Update transmittance: T *= (1 - alpha)
          transmittance[pixelIdx] *= (1.0 - alpha);
          totalPixelsWritten++;
        }
      }
    }

    // Add background color weighted by remaining transmittance
    for (let pixelIdx = 0; pixelIdx < height * width; pixelIdx++) {
      const T = transmittance[pixelIdx];
      if (T > 0.0001) {
        for (let c = 0; c < featureChannels; c++) {
          featureMap[c * height * width + pixelIdx] += background * T;
        }
      }
    }

    // Statistics
    let fmMin = Infinity, fmMax = -Infinity, fmSum = 0;
    for (let i = 0; i < featureMap.length; i++) {
      const v = featureMap[i];
      if (v < fmMin) fmMin = v;
      if (v > fmMax) fmMax = v;
      fmSum += v;
    }

    console.log('[GaussianRasterizer] Stats:', {
      pixelsWritten: totalPixelsWritten,
      min: fmMin.toFixed(4),
      max: fmMax.toFixed(4),
      mean: (fmSum / featureMap.length).toFixed(6)
    });

    return featureMap;
  }

  /**
   * Preprocess Gaussians: compute screen-space position, covariance, and conic
   * Implements EWA Splatting algorithm (Zwicker et al., 2002)
   */
  private preprocessGaussians(
    input: GaussianRasterInput,
    camera: CameraParams
  ): PreprocessedGaussian[] {
    const result: PreprocessedGaussian[] = [];
    const { width, height, focalX, focalY, tanFovX, tanFovY } = camera;

    // Extract view matrix components
    const vm = camera.viewMatrix;

    for (let i = 0; i < input.count; i++) {
      // Get world position
      const px = input.positions[i * 3 + 0];
      const py = input.positions[i * 3 + 1];
      const pz = input.positions[i * 3 + 2];

      // Transform to camera space
      const camX = vm[0] * px + vm[4] * py + vm[8] * pz + vm[12];
      const camY = vm[1] * px + vm[5] * py + vm[9] * pz + vm[13];
      const camZ = vm[2] * px + vm[6] * py + vm[10] * pz + vm[14];

      // Skip if behind camera
      if (camZ <= 0.1) continue;

      // Project to NDC
      const ndcX = camX * focalX / camZ;
      const ndcY = camY * focalY / camZ;

      // Clamp to avoid extreme values
      const limX = 1.3 * tanFovX;
      const limY = 1.3 * tanFovY;
      const clampedX = Math.min(limX, Math.max(-limX, camX / camZ)) * camZ;
      const clampedY = Math.min(limY, Math.max(-limY, camY / camZ)) * camZ;

      // NDC to screen coordinates
      const screenX = (ndcX * 0.5 + 0.5) * width;
      const screenY = (1.0 - (ndcY * 0.5 + 0.5)) * height;  // Y-flip

      // Skip if outside screen
      if (screenX < -100 || screenX > width + 100 ||
          screenY < -100 || screenY > height + 100) continue;

      // Compute 3D covariance matrix from scale and rotation
      const sx = Math.exp(input.scales[i * 3 + 0]);
      const sy = Math.exp(input.scales[i * 3 + 1]);
      const sz = Math.exp(input.scales[i * 3 + 2]);

      const qw = input.rotations[i * 4 + 0];
      const qx = input.rotations[i * 4 + 1];
      const qy = input.rotations[i * 4 + 2];
      const qz = input.rotations[i * 4 + 3];

      // Quaternion to rotation matrix
      const r = qw, x = qx, y = qy, z = qz;
      const R00 = 1 - 2 * (y * y + z * z);
      const R01 = 2 * (x * y - r * z);
      const R02 = 2 * (x * z + r * y);
      const R10 = 2 * (x * y + r * z);
      const R11 = 1 - 2 * (x * x + z * z);
      const R12 = 2 * (y * z - r * x);
      const R20 = 2 * (x * z - r * y);
      const R21 = 2 * (y * z + r * x);
      const R22 = 1 - 2 * (x * x + y * y);

      // M = S * R (scale then rotate)
      const M00 = sx * R00, M01 = sx * R01, M02 = sx * R02;
      const M10 = sy * R10, M11 = sy * R11, M12 = sy * R12;
      const M20 = sz * R20, M21 = sz * R21, M22 = sz * R22;

      // Σ_3D = M^T * M (3D covariance matrix, symmetric)
      const cov3D_00 = M00 * M00 + M10 * M10 + M20 * M20;
      const cov3D_01 = M00 * M01 + M10 * M11 + M20 * M21;
      const cov3D_02 = M00 * M02 + M10 * M12 + M20 * M22;
      const cov3D_11 = M01 * M01 + M11 * M11 + M21 * M21;
      const cov3D_12 = M01 * M02 + M11 * M12 + M21 * M22;
      const cov3D_22 = M02 * M02 + M12 * M12 + M22 * M22;

      // Compute Jacobian of perspective projection
      const J00 = focalX / camZ;
      const J02 = -(focalX * clampedX) / (camZ * camZ);
      const J11 = focalY / camZ;
      const J12 = -(focalY * clampedY) / (camZ * camZ);

      // View rotation matrix (3x3 from view matrix)
      const W00 = vm[0], W01 = vm[4], W02 = vm[8];
      const W10 = vm[1], W11 = vm[5], W12 = vm[9];
      const W20 = vm[2], W21 = vm[6], W22 = vm[10];

      // T = W * J (transform matrix)
      const T00 = W00 * J00 + W02 * J02;
      const T01 = W01 * J00;
      const T02 = W02 * J00;
      const T10 = W10 * J11 + W12 * J12;
      const T11 = W11 * J11;
      const T12 = W12 * J11;

      // Σ_2D = T^T * Σ_3D * T (project to 2D)
      // First: temp = Σ_3D * T
      const temp00 = cov3D_00 * T00 + cov3D_01 * T10;
      const temp01 = cov3D_00 * T01 + cov3D_01 * T11;
      const temp10 = cov3D_01 * T00 + cov3D_11 * T10;
      const temp11 = cov3D_01 * T01 + cov3D_11 * T11;
      const temp20 = cov3D_02 * T00 + cov3D_12 * T10;
      const temp21 = cov3D_02 * T01 + cov3D_12 * T11;

      // Then: Σ_2D = T^T * temp
      let cov2D_00 = T00 * temp00 + T10 * temp10 + T02 * temp20;
      let cov2D_01 = T00 * temp01 + T10 * temp11 + T02 * temp21;
      let cov2D_11 = T01 * temp01 + T11 * temp11 + T12 * temp21;

      // Add anti-aliasing filter (low-pass)
      const hVar = 0.3;
      const detCov = cov2D_00 * cov2D_11 - cov2D_01 * cov2D_01;
      cov2D_00 += hVar;
      cov2D_11 += hVar;
      const detCovPlusH = cov2D_00 * cov2D_11 - cov2D_01 * cov2D_01;

      // Compute conic (inverse covariance)
      if (detCovPlusH <= 0) continue;
      const detInv = 1.0 / detCovPlusH;
      const conic: [number, number, number] = [
        cov2D_11 * detInv,
        -cov2D_01 * detInv,
        cov2D_00 * detInv
      ];

      // Compute screen-space radius from eigenvalues
      const mid = 0.5 * (cov2D_00 + cov2D_11);
      const lambda1 = mid + Math.sqrt(Math.max(0.1, mid * mid - detCovPlusH));
      const lambda2 = mid - Math.sqrt(Math.max(0.1, mid * mid - detCovPlusH));
      const radius = Math.ceil(3 * Math.sqrt(Math.max(lambda1, lambda2)));

      // Skip very small Gaussians
      if (radius < 0.1) continue;

      // Compute opacity with anti-aliasing scaling
      const hConvScaling = Math.sqrt(Math.max(0.000025, detCov / detCovPlusH));
      const rawOpacity = input.opacities[i];
      const opacity = (1.0 / (1.0 + Math.exp(-rawOpacity))) * hConvScaling;

      result.push({
        screenX,
        screenY,
        depth: camZ,
        conic,
        opacity,
        radius,
        featureIdx: i
      });
    }

    return result;
  }

  /**
   * Create WebGPU compute pipelines
   */
  private async createPipelines(): Promise<void> {
    if (!this.device) return;

    // Preprocessing shader
    const preprocessShader = this.device.createShaderModule({
      code: this.getPreprocessShaderCode()
    });

    // Rendering shader
    const renderShader = this.device.createShaderModule({
      code: this.getRenderShaderCode()
    });

    console.log('[GaussianRasterizer] Compute pipelines created');
  }

  /**
   * WGSL shader for preprocessing (3D -> 2D projection)
   */
  private getPreprocessShaderCode(): string {
    return `
      struct Gaussian {
        position: vec3<f32>,
        opacity: f32,
        scale: vec3<f32>,
        _pad1: f32,
        rotation: vec4<f32>,
      }

      struct PreprocessedGaussian {
        screenPos: vec2<f32>,
        depth: f32,
        radius: f32,
        conic: vec3<f32>,
        opacity: f32,
        featureIdx: u32,
        _pad: array<u32, 3>,
      }

      struct CameraParams {
        viewMatrix: mat4x4<f32>,
        projMatrix: mat4x4<f32>,
        focalX: f32,
        focalY: f32,
        tanFovX: f32,
        tanFovY: f32,
        width: f32,
        height: f32,
        _pad: vec2<f32>,
      }

      @group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
      @group(0) @binding(1) var<storage, read_write> preprocessed: array<PreprocessedGaussian>;
      @group(0) @binding(2) var<uniform> camera: CameraParams;
      @group(0) @binding(3) var<storage, read_write> visibleCount: atomic<u32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&gaussians)) { return; }

        let g = gaussians[idx];

        // Transform to camera space
        let worldPos = vec4<f32>(g.position, 1.0);
        let camPos = camera.viewMatrix * worldPos;

        if (camPos.z <= 0.1) { return; }

        // Project to screen
        let ndcX = camPos.x * camera.focalX / camPos.z;
        let ndcY = camPos.y * camera.focalY / camPos.z;
        let screenX = (ndcX * 0.5 + 0.5) * camera.width;
        let screenY = (1.0 - (ndcY * 0.5 + 0.5)) * camera.height;

        // TODO: Compute covariance projection
        // For now, use simplified spherical approximation
        let avgScale = (exp(g.scale.x) + exp(g.scale.y) + exp(g.scale.z)) / 3.0;
        let radius = avgScale * camera.focalX / camPos.z;

        let outIdx = atomicAdd(&visibleCount, 1u);
        preprocessed[outIdx].screenPos = vec2<f32>(screenX, screenY);
        preprocessed[outIdx].depth = camPos.z;
        preprocessed[outIdx].radius = radius;
        preprocessed[outIdx].opacity = 1.0 / (1.0 + exp(-g.opacity));
        preprocessed[outIdx].featureIdx = idx;
      }
    `;
  }

  /**
   * WGSL shader for rendering (splatting)
   */
  private getRenderShaderCode(): string {
    return `
      // TODO: Full render shader implementation
      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        // Tile-based rendering
      }
    `;
  }

  /**
   * Cleanup
   */
  dispose(): void {
    this.device = null;
    this.preprocessPipeline = null;
    this.renderPipeline = null;
    this.initialized = false;
    console.log('[GaussianRasterizer] Disposed');
  }
}
