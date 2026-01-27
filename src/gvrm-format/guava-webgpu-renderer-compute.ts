// guava-webgpu-renderer-compute.ts
// Compute shader-based Gaussian splatting renderer
// Preserves all 32 latent channels (no channel loss from alpha blending)
//
// Key difference from fragment shader approach:
// - Fragment shader with hardware blending loses every 4th channel (used for alpha)
// - Compute shader has full control over accumulation and preserves all channels
//
// v86: Unified output buffer to avoid WebGPU binding limit (8 storage buffers per stage)
// - Single buffer: width * height * 32 floats
// - GPU compute splatting with tile-based dispatch

import { CameraUtils } from './camera-utils';

export interface CameraConfig {
    viewMatrix: Float32Array;
    projMatrix: Float32Array;
    imageWidth?: number;
    imageHeight?: number;
    position?: any; target?: any; fov?: any; debug?: any;
}

export interface GaussianData {
    positions: Float32Array;
    latents: Float32Array;
    opacity: Float32Array;
    scale: Float32Array;
    rotation: Float32Array;
    vertexCount: number;
}

interface SortedGaussian {
    index: number;
    depth: number;
    screenX: number;
    screenY: number;
    screenRadius: number;
}

export class GuavaWebGPURendererCompute {
    private device: GPUDevice;
    private gaussianData: GaussianData;
    private cameraConfig: CameraConfig;

    private width: number;
    private height: number;

    // Output textures (8 x RGBA = 32 channels) - for compatibility
    private outputTextures: GPUTexture[] = [];

    // Compute pipeline and resources
    private clearPipeline: GPUComputePipeline | null = null;
    private splatPipeline: GPUComputePipeline | null = null;
    private gaussianBuffer: GPUBuffer | null = null;

    // v86: Single unified output buffer (replaces 8 separate buffers)
    private unifiedOutputBuffer: GPUBuffer | null = null;
    private outputBuffers: GPUBuffer[] = [];  // Legacy, kept for getOutputBuffers() compatibility

    private uniformBuffer: GPUBuffer | null = null;

    // GPU compute resources
    private sortedIndicesBuffer: GPUBuffer | null = null;
    private sortedDataBuffer: GPUBuffer | null = null;
    private atomicBuffer: GPUBuffer | null = null;  // Fixed-point atomic accumulation buffer

    // Bind group layouts (stored for dynamic bind group creation)
    private clearBindGroupLayout: GPUBindGroupLayout | null = null;
    private splatBindGroupLayout: GPUBindGroupLayout | null = null;
    private convertBindGroupLayout: GPUBindGroupLayout | null = null;
    private convertPipeline: GPUComputePipeline | null = null;

    private depthArray: Float32Array;
    private indexArray: Uint32Array;
    private sortedGaussians: SortedGaussian[] = [];

    private initPromise: Promise<void>;
    private isInitialized = false;
    private sortCalled = false;
    private renderCount = 0;
    private useGPUSplatting = true;  // v86: Enable GPU splatting

    constructor(device: GPUDevice, data: GaussianData, camera: CameraConfig) {
        this.device = device;
        this.gaussianData = data;
        this.cameraConfig = camera;
        this.width = camera.imageWidth || 512;
        this.height = camera.imageHeight || 512;

        const count = data.vertexCount;
        this.depthArray = new Float32Array(count);
        this.indexArray = new Uint32Array(count);
        for (let i = 0; i < count; i++) this.indexArray[i] = i;

        console.log('[ComputeRenderer] ═══════════════════════════════════════════');
        console.log('[ComputeRenderer] 🔧 BUILD v86 - GPU compute splatting');
        console.log('[ComputeRenderer] ═══════════════════════════════════════════');
        console.log('[ComputeRenderer] Constructor called with:');
        console.log(`  vertexCount: ${count.toLocaleString()}`);
        console.log(`  dimensions: ${this.width}x${this.height}`);
        console.log(`  positions: ${data.positions.length.toLocaleString()} floats`);
        console.log(`  latents: ${data.latents.length.toLocaleString()} floats`);

        this.initPromise = this.init();
    }

    public async waitForInit(): Promise<void> {
        await this.initPromise;
    }

    private async init() {
        try {
            this.createOutputTextures();
            this.createUnifiedOutputBuffer();
            this.createGaussianBuffer();
            this.createUniformBuffer();
            this.createSortedBuffers();
            await this.createPipelines();
            this.isInitialized = true;
            console.log('[ComputeRenderer] ✅ Initialization complete (GPU compute splatting)');
        } catch (error) {
            console.error('[ComputeRenderer] Initialization failed:', error);
            this.useGPUSplatting = false;
            console.warn('[ComputeRenderer] Falling back to CPU splatting');
            this.isInitialized = true;
        }
    }

    private createOutputTextures(): void {
        this.outputTextures.forEach(t => t.destroy());
        this.outputTextures = [];

        for (let i = 0; i < 8; i++) {
            const texture = this.device.createTexture({
                size: [this.width, this.height],
                format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING |
                       GPUTextureUsage.COPY_SRC |
                       GPUTextureUsage.TEXTURE_BINDING
            });
            this.outputTextures.push(texture);
        }
        console.log('[ComputeRenderer] Created 8 output textures (32 channels total)');
    }

    /**
     * v86: Create single unified output buffer (32 channels per pixel)
     * Replaces 8 separate buffers to avoid WebGPU binding limit
     */
    private createUnifiedOutputBuffer(): void {
        this.unifiedOutputBuffer?.destroy();
        this.atomicBuffer?.destroy();

        // Single buffer: width * height * 32 channels * 4 bytes (float output)
        const bufferSize = this.width * this.height * 32 * 4;
        this.unifiedOutputBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'unified-output-32ch'
        });

        // Atomic buffer for fixed-point accumulation: width * height * 32 * 4 bytes (i32)
        this.atomicBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'atomic-accumulator'
        });

        console.log(`[ComputeRenderer] Created unified output buffer: ${(bufferSize / 1024 / 1024).toFixed(2)} MB (32 channels)`);
        console.log(`[ComputeRenderer] Created atomic buffer: ${(bufferSize / 1024 / 1024).toFixed(2)} MB`);

        // Also create legacy output buffers for getOutputBuffers() compatibility
        this.outputBuffers.forEach(b => b.destroy());
        this.outputBuffers = [];
        const legacyBufferSize = this.width * this.height * 4 * 4;
        for (let i = 0; i < 8; i++) {
            const buffer = this.device.createBuffer({
                size: legacyBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                label: `legacy-output-${i}`
            });
            this.outputBuffers.push(buffer);
        }
    }

    /**
     * Create buffers for sorted Gaussian data
     */
    private createSortedBuffers(): void {
        this.sortedIndicesBuffer?.destroy();
        this.sortedDataBuffer?.destroy();

        const maxGaussians = this.gaussianData.vertexCount;

        // Sorted indices buffer
        this.sortedIndicesBuffer = this.device.createBuffer({
            size: maxGaussians * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'sorted-indices'
        });

        // Sorted screen-space data: [screenX, screenY, radius, opacity] per Gaussian
        this.sortedDataBuffer = this.device.createBuffer({
            size: maxGaussians * 16,  // 4 floats per Gaussian
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'sorted-screen-data'
        });

        console.log(`[ComputeRenderer] Created sorted buffers for ${maxGaussians.toLocaleString()} Gaussians`);
    }

    private createGaussianBuffer(): void {
        // Pack all Gaussian data into a single buffer
        // Format per Gaussian: pos(3) + opacity(1) + scale(3) + pad(1) + rot(4) + latent(32) = 44 floats
        const count = this.gaussianData.vertexCount;
        const floatsPerGaussian = 44;
        const data = new Float32Array(count * floatsPerGaussian);

        const gd = this.gaussianData;
        for (let i = 0; i < count; i++) {
            const offset = i * floatsPerGaussian;
            data[offset + 0] = gd.positions[i * 3 + 0];
            data[offset + 1] = gd.positions[i * 3 + 1];
            data[offset + 2] = gd.positions[i * 3 + 2];
            data[offset + 3] = gd.opacity[i];
            data[offset + 4] = gd.scale[i * 3 + 0];
            data[offset + 5] = gd.scale[i * 3 + 1];
            data[offset + 6] = gd.scale[i * 3 + 2];
            data[offset + 7] = 0; // padding
            data[offset + 8] = gd.rotation[i * 4 + 0];
            data[offset + 9] = gd.rotation[i * 4 + 1];
            data[offset + 10] = gd.rotation[i * 4 + 2];
            data[offset + 11] = gd.rotation[i * 4 + 3];
            for (let j = 0; j < 32; j++) {
                data[offset + 12 + j] = gd.latents[i * 32 + j];
            }
        }

        this.gaussianBuffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.gaussianBuffer.getMappedRange()).set(data);
        this.gaussianBuffer.unmap();

        console.log('[ComputeRenderer] Created Gaussian buffer:', count, 'Gaussians');
    }

    private createUniformBuffer(): void {
        // v86: Simplified uniforms: width(1) + height(1) + numGaussians(1) + pad(1) = 4 u32s = 16 bytes
        this.uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'uniforms'
        });
    }

    private async createPipelines(): Promise<void> {
        // v86: GPU compute pipelines with unified buffer
        // Using per-Gaussian approach with fixed-point atomic accumulation
        // This avoids O(pixels * Gaussians) complexity of per-pixel approach

        // Clear shader - initialize output buffer to 0
        const clearShaderCode = /* wgsl */`
            @group(0) @binding(0) var<storage, read_write> output: array<i32>;

            struct Uniforms {
                width: u32,
                height: u32,
                numGaussians: u32,
                pad: u32
            }
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let pixelIdx = gid.x;
                let totalPixels = uniforms.width * uniforms.height;

                if (pixelIdx >= totalPixels) {
                    return;
                }

                // Clear all 32 channels to 0 (using i32 for atomic operations)
                let baseIdx = pixelIdx * 32u;
                for (var ch = 0u; ch < 32u; ch++) {
                    output[baseIdx + ch] = 0;
                }
            }
        `;

        // Splat shader - per-Gaussian approach
        // Each thread processes one Gaussian and atomically accumulates to affected pixels
        // Uses fixed-point arithmetic (scale by 2^14) for atomic operations
        const splatShaderCode = /* wgsl */`
            struct Gaussian {
                pos: vec3<f32>,
                opacity: f32,
                scale: vec3<f32>,
                pad: f32,
                rot: vec4<f32>,
                latent: array<f32, 32>
            }

            struct SortedData {
                screenX: f32,
                screenY: f32,
                radius: f32,
                opacity: f32
            }

            struct Uniforms {
                width: u32,
                height: u32,
                numGaussians: u32,
                pad: u32
            }

            @group(0) @binding(0) var<storage, read_write> output: array<atomic<i32>>;
            @group(0) @binding(1) var<uniform> uniforms: Uniforms;
            @group(0) @binding(2) var<storage, read> gaussians: array<Gaussian>;
            @group(0) @binding(3) var<storage, read> sortedIndices: array<u32>;
            @group(0) @binding(4) var<storage, read> sortedData: array<SortedData>;

            const FIXED_POINT_SCALE: f32 = 16384.0;  // 2^14

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let gi = gid.x;
                if (gi >= uniforms.numGaussians) {
                    return;
                }

                let gaussianIdx = sortedIndices[gi];
                let sd = sortedData[gi];
                let g = gaussians[gaussianIdx];

                // Skip nearly transparent Gaussians
                if (sd.opacity < 0.001) {
                    return;
                }

                // Compute bounding box (3-sigma)
                let radius3 = sd.radius * 3.0;
                let minX = max(0, i32(sd.screenX - radius3));
                let maxX = min(i32(uniforms.width) - 1, i32(sd.screenX + radius3));
                let minY = max(0, i32(sd.screenY - radius3));
                let maxY = min(i32(uniforms.height) - 1, i32(sd.screenY + radius3));

                let radiusSq = sd.radius * sd.radius + 0.0001;

                // Splat to affected pixels
                for (var py = minY; py <= maxY; py++) {
                    for (var px = minX; px <= maxX; px++) {
                        let dx = f32(px) - sd.screenX;
                        let dy = f32(py) - sd.screenY;
                        let r2 = (dx * dx + dy * dy) / radiusSq;

                        if (r2 > 9.0) {
                            continue;
                        }

                        // Gaussian weight (simplified: no transmittance for atomic approach)
                        let gaussianWeight = exp(-0.5 * r2);
                        let weight = gaussianWeight * sd.opacity;

                        if (weight < 0.001) {
                            continue;
                        }

                        // Atomic accumulate all 32 channels using fixed-point
                        let pixelIdx = u32(py) * uniforms.width + u32(px);
                        let baseIdx = pixelIdx * 32u;

                        for (var ch = 0u; ch < 32u; ch++) {
                            let value = g.latent[ch] * weight;
                            let fixedValue = i32(value * FIXED_POINT_SCALE);
                            atomicAdd(&output[baseIdx + ch], fixedValue);
                        }
                    }
                }
            }
        `;

        // Convert shader - convert fixed-point back to float
        const convertShaderCode = /* wgsl */`
            @group(0) @binding(0) var<storage, read> input: array<i32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            struct Uniforms {
                width: u32,
                height: u32,
                numGaussians: u32,
                pad: u32
            }
            @group(0) @binding(2) var<uniform> uniforms: Uniforms;

            const FIXED_POINT_SCALE: f32 = 16384.0;  // 2^14

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let pixelIdx = gid.x;
                let totalPixels = uniforms.width * uniforms.height;

                if (pixelIdx >= totalPixels) {
                    return;
                }

                let baseIdx = pixelIdx * 32u;
                for (var ch = 0u; ch < 32u; ch++) {
                    let fixedValue = input[baseIdx + ch];
                    output[baseIdx + ch] = f32(fixedValue) / FIXED_POINT_SCALE;
                }
            }
        `;

        try {
            // Create clear pipeline
            const clearShaderModule = this.device.createShaderModule({
                code: clearShaderCode,
                label: 'clear-shader'
            });

            const clearBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
                ],
                label: 'clear-bind-group-layout'
            });

            this.clearPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [clearBindGroupLayout],
                    label: 'clear-pipeline-layout'
                }),
                compute: {
                    module: clearShaderModule,
                    entryPoint: 'main'
                },
                label: 'clear-pipeline'
            });

            // Create splat pipeline
            const splatShaderModule = this.device.createShaderModule({
                code: splatShaderCode,
                label: 'splat-shader'
            });

            const splatBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
                ],
                label: 'splat-bind-group-layout'
            });

            this.splatPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [splatBindGroupLayout],
                    label: 'splat-pipeline-layout'
                }),
                compute: {
                    module: splatShaderModule,
                    entryPoint: 'main'
                },
                label: 'splat-pipeline'
            });

            // Create convert pipeline
            const convertShaderModule = this.device.createShaderModule({
                code: convertShaderCode,
                label: 'convert-shader'
            });

            const convertBindGroupLayout = this.device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
                ],
                label: 'convert-bind-group-layout'
            });

            this.convertPipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [convertBindGroupLayout],
                    label: 'convert-pipeline-layout'
                }),
                compute: {
                    module: convertShaderModule,
                    entryPoint: 'main'
                },
                label: 'convert-pipeline'
            });

            // Store layouts for bind group creation in gpuSplat
            this.clearBindGroupLayout = clearBindGroupLayout;
            this.splatBindGroupLayout = splatBindGroupLayout;
            this.convertBindGroupLayout = convertBindGroupLayout;

            console.log('[ComputeRenderer] ✅ GPU compute pipelines created (5 bindings max, within limit)');
            this.useGPUSplatting = true;

        } catch (error) {
            console.error('[ComputeRenderer] Failed to create GPU pipelines:', error);
            this.useGPUSplatting = false;
            throw error;
        }
    }

    public sort(): void {
        if (!this.isInitialized) {
            console.warn('[ComputeRenderer] sort() called but not ready');
            return;
        }

        const positions = this.gaussianData.positions;
        const count = this.gaussianData.vertexCount;
        const viewMatrix = this.cameraConfig.viewMatrix;

        // Clear sorted list
        this.sortedGaussians = [];

        // 公式実装に合わせた投影 (image-encoder.ts と同じ方式)
        // tanfov = 1/24, canonical camera at (0, 0.6, 22)
        const invTanFov = 24.0;

        // Project Gaussians to screen space and compute depth
        for (let i = 0; i < count; i++) {
            const px = positions[i * 3 + 0];
            const py = positions[i * 3 + 1];
            const pz = positions[i * 3 + 2];

            // Transform to view space (same as image-encoder.ts)
            // View matrix is identity rotation + translation (0, 0.6, 22)
            const vx = px + viewMatrix[12];  // px + 0
            const vy = py + viewMatrix[13];  // py + 0.6
            const vz = pz + viewMatrix[14];  // pz + 22

            // Skip if behind camera (vz should be positive for visible objects)
            if (vz <= 0.1) continue;

            // Perspective projection (same formula as image-encoder.ts)
            const imgX = (vx * invTanFov) / vz;
            const imgY = (vy * invTanFov) / vz;

            // Convert from NDC [-1, 1] to screen [0, width/height]
            const screenX = (imgX + 1.0) * this.width * 0.5;
            const screenY = (1.0 - imgY) * this.height * 0.5;  // Y軸反転

            // Compute screen radius based on scale and distance
            const scale = this.gaussianData.scale;
            const maxScale = Math.max(scale[i*3+0], scale[i*3+1], scale[i*3+2]);
            const screenRadius = maxScale * invTanFov / vz * this.width * 0.5;

            // Only include Gaussians that might be visible (with generous margin)
            const margin = screenRadius * 3;  // 3σ for Gaussian
            if (screenX + margin < 0 || screenX - margin > this.width) continue;
            if (screenY + margin < 0 || screenY - margin > this.height) continue;

            this.sortedGaussians.push({
                index: i,
                depth: vz,
                screenX: screenX,
                screenY: screenY,
                screenRadius: screenRadius
            });
        }

        // Sort back to front (larger depth = further away, render first)
        this.sortedGaussians.sort((a, b) => b.depth - a.depth);

        if (!this.sortCalled) {
            this.sortCalled = true;
            console.log('[ComputeRenderer] First sort() complete:');
            console.log(`  Total Gaussians: ${count}`);
            console.log(`  Visible after culling: ${this.sortedGaussians.length}`);
            if (this.sortedGaussians.length > 0) {
                const first = this.sortedGaussians[0];
                const last = this.sortedGaussians[this.sortedGaussians.length - 1];
                console.log(`  First (back): idx=${first.index}, depth=${first.depth.toFixed(4)}, screen=(${first.screenX.toFixed(1)}, ${first.screenY.toFixed(1)}), radius=${first.screenRadius.toFixed(2)}`);
                console.log(`  Last (front): idx=${last.index}, depth=${last.depth.toFixed(4)}, screen=(${last.screenX.toFixed(1)}, ${last.screenY.toFixed(1)}), radius=${last.screenRadius.toFixed(2)}`);
            }

            // ======== 🔍 DEBUG: INPUT LATENT RGB DIVERSITY CHECK ========
            console.log('[ComputeRenderer] 🔍🔍🔍 INPUT LATENT RGB DIVERSITY CHECK:');
            const latents = this.gaussianData.latents;

            // Check first 10 visible Gaussians
            console.log('[ComputeRenderer]   First 10 visible Gaussians (ch 0,1,2 = RGB):');
            for (let j = 0; j < Math.min(10, this.sortedGaussians.length); j++) {
                const idx = this.sortedGaussians[j].index;
                const r = latents[idx * 32 + 0];
                const g = latents[idx * 32 + 1];
                const b = latents[idx * 32 + 2];
                const diffRG = Math.abs(r - g);
                const diffGB = Math.abs(g - b);
                console.log(`[ComputeRenderer]     Gaussian ${idx}: R=${r.toFixed(4)}, G=${g.toFixed(4)}, B=${b.toFixed(4)} | diff: R-G=${diffRG.toFixed(4)}, G-B=${diffGB.toFixed(4)}`);
            }

            // Compute RGB diversity for ALL visible Gaussians
            let sumR = 0, sumG = 0, sumB = 0;
            let sumRG_diff = 0, sumGB_diff = 0;
            let sumRG_diff2 = 0, sumGB_diff2 = 0;

            for (const sg of this.sortedGaussians) {
                const idx = sg.index;
                const r = latents[idx * 32 + 0];
                const g = latents[idx * 32 + 1];
                const b = latents[idx * 32 + 2];
                sumR += r;
                sumG += g;
                sumB += b;
                const dRG = r - g;
                const dGB = g - b;
                sumRG_diff += dRG;
                sumGB_diff += dGB;
                sumRG_diff2 += dRG * dRG;
                sumGB_diff2 += dGB * dGB;
            }

            const n = this.sortedGaussians.length;
            if (n > 0) {
                const meanR = sumR / n;
                const meanG = sumG / n;
                const meanB = sumB / n;
                const meanRG = sumRG_diff / n;
                const meanGB = sumGB_diff / n;
                const varRG = sumRG_diff2 / n - meanRG * meanRG;
                const varGB = sumGB_diff2 / n - meanGB * meanGB;

                console.log(`[ComputeRenderer]   Overall stats for ${n} visible Gaussians:`);
                console.log(`[ComputeRenderer]     Mean R=${meanR.toFixed(4)}, G=${meanG.toFixed(4)}, B=${meanB.toFixed(4)}`);
                console.log(`[ComputeRenderer]     R-G diff: mean=${meanRG.toFixed(6)}, σ=${Math.sqrt(varRG).toFixed(6)}`);
                console.log(`[ComputeRenderer]     G-B diff: mean=${meanGB.toFixed(6)}, σ=${Math.sqrt(varGB).toFixed(6)}`);

                if (Math.sqrt(varRG) < 0.05 && Math.sqrt(varGB) < 0.05) {
                    console.log(`[ComputeRenderer]   ⚠️ WARNING: Input latents have R≈G≈B (low diversity)`);
                } else {
                    console.log(`[ComputeRenderer]   ✅ Input latents have RGB color diversity`);
                }
            }
            // ======== END DEBUG ========
        }
    }

    public render(): void {
        if (!this.isInitialized) {
            console.warn('[ComputeRenderer] Not ready to render');
            return;
        }

        this.renderCount++;
        const startTime = performance.now();

        if (this.useGPUSplatting && this.splatPipeline && this.clearPipeline) {
            this.gpuSplat();
            if (this.renderCount === 1) {
                const elapsed = performance.now() - startTime;
                console.log(`[ComputeRenderer] ✅ First render() complete (GPU splat, ${elapsed.toFixed(1)}ms)`);
            }
        } else {
            // Fallback to CPU splatting
            this.cpuSplat();
            if (this.renderCount === 1) {
                const elapsed = performance.now() - startTime;
                console.log(`[ComputeRenderer] First render() complete (CPU fallback, ${elapsed.toFixed(1)}ms)`);
            }
        }
    }

    /**
     * GPU compute-based splatting
     * v86: Uses unified buffer with fixed-point atomic accumulation
     *
     * 3-pass approach:
     * 1. Clear atomic buffer to 0
     * 2. Splat Gaussians with atomic accumulation (per-Gaussian dispatch)
     * 3. Convert fixed-point to float
     */
    private gpuSplat(): void {
        const width = this.width;
        const height = this.height;
        const numGaussians = this.sortedGaussians.length;
        const pixelCount = width * height;

        // Update uniform buffer
        const uniformData = new Uint32Array([width, height, numGaussians, 0]);
        this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);

        // Upload sorted Gaussian data
        const sortedIndices = new Uint32Array(numGaussians);
        const sortedData = new Float32Array(numGaussians * 4);

        for (let i = 0; i < numGaussians; i++) {
            const sg = this.sortedGaussians[i];
            sortedIndices[i] = sg.index;
            sortedData[i * 4 + 0] = sg.screenX;
            sortedData[i * 4 + 1] = sg.screenY;
            sortedData[i * 4 + 2] = sg.screenRadius;
            sortedData[i * 4 + 3] = this.gaussianData.opacity[sg.index];
        }

        this.device.queue.writeBuffer(this.sortedIndicesBuffer!, 0, sortedIndices);
        this.device.queue.writeBuffer(this.sortedDataBuffer!, 0, sortedData);

        // Create bind groups dynamically (needed because buffers may change)
        const clearBindGroup = this.device.createBindGroup({
            layout: this.clearBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: this.atomicBuffer! } },
                { binding: 1, resource: { buffer: this.uniformBuffer! } }
            ],
            label: 'clear-bind-group'
        });

        const splatBindGroup = this.device.createBindGroup({
            layout: this.splatBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: this.atomicBuffer! } },
                { binding: 1, resource: { buffer: this.uniformBuffer! } },
                { binding: 2, resource: { buffer: this.gaussianBuffer! } },
                { binding: 3, resource: { buffer: this.sortedIndicesBuffer! } },
                { binding: 4, resource: { buffer: this.sortedDataBuffer! } }
            ],
            label: 'splat-bind-group'
        });

        const convertBindGroup = this.device.createBindGroup({
            layout: this.convertBindGroupLayout!,
            entries: [
                { binding: 0, resource: { buffer: this.atomicBuffer! } },
                { binding: 1, resource: { buffer: this.unifiedOutputBuffer! } },
                { binding: 2, resource: { buffer: this.uniformBuffer! } }
            ],
            label: 'convert-bind-group'
        });

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();

        // Pass 1: Clear atomic buffer
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.clearPipeline!);
            passEncoder.setBindGroup(0, clearBindGroup);
            const workgroupCount = Math.ceil(pixelCount / 256);
            passEncoder.dispatchWorkgroups(workgroupCount);
            passEncoder.end();
        }

        // Pass 2: Splat Gaussians (per-Gaussian dispatch)
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.splatPipeline!);
            passEncoder.setBindGroup(0, splatBindGroup);
            const workgroupCount = Math.ceil(numGaussians / 256);
            passEncoder.dispatchWorkgroups(workgroupCount);
            passEncoder.end();
        }

        // Pass 3: Convert fixed-point to float
        {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.convertPipeline!);
            passEncoder.setBindGroup(0, convertBindGroup);
            const workgroupCount = Math.ceil(pixelCount / 256);
            passEncoder.dispatchWorkgroups(workgroupCount);
            passEncoder.end();
        }

        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);

        // Copy unified buffer to legacy output buffers (for compatibility)
        this.copyUnifiedToLegacyBuffers();

        if (this.renderCount === 1) {
            console.log(`[ComputeRenderer] GPU splat: ${numGaussians.toLocaleString()} Gaussians, ${width}x${height}`);
            console.log(`[ComputeRenderer]   Workgroups: ${Math.ceil(numGaussians / 256)} (splat)`);
        }
    }

    /**
     * Copy from unified buffer to 8 legacy buffers for getOutputBuffers() compatibility
     */
    private copyUnifiedToLegacyBuffers(): void {
        const width = this.width;
        const height = this.height;
        const pixelCount = width * height;

        // Read unified buffer and distribute to legacy buffers
        // We need to do this on GPU to avoid CPU readback
        // For now, use staging buffer approach

        // Create staging buffer to read unified output
        const stagingBuffer = this.device.createBuffer({
            size: pixelCount * 32 * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            label: 'staging-unified'
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.unifiedOutputBuffer!, 0,
            stagingBuffer, 0,
            pixelCount * 32 * 4
        );
        this.device.queue.submit([commandEncoder.finish()]);

        // Map and copy to legacy buffers
        stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const data = new Float32Array(stagingBuffer.getMappedRange());

            // Distribute to 8 legacy buffers (4 channels each)
            const legacyData: Float32Array[] = [];
            for (let i = 0; i < 8; i++) {
                legacyData.push(new Float32Array(pixelCount * 4));
            }

            for (let p = 0; p < pixelCount; p++) {
                const srcBase = p * 32;
                for (let buf = 0; buf < 8; buf++) {
                    const dstBase = p * 4;
                    for (let ch = 0; ch < 4; ch++) {
                        legacyData[buf][dstBase + ch] = data[srcBase + buf * 4 + ch];
                    }
                }
            }

            stagingBuffer.unmap();
            stagingBuffer.destroy();

            // Upload to legacy buffers
            for (let i = 0; i < 8; i++) {
                this.device.queue.writeBuffer(this.outputBuffers[i], 0, legacyData[i]);
            }
        });
    }

    /**
     * CPU-based Gaussian splatting that preserves all 32 channels
     * This is slower than GPU but guarantees correctness
     *
     * 背景色 = 0.0 (Python diff-gaussian-rasterization と同じ)
     */
    private cpuSplat(): void {
        const width = this.width;
        const height = this.height;
        const pixelCount = width * height;

        // Initialize output arrays (8 textures x 4 channels = 32)
        // 背景 = 0.0 (Python実装に合わせる)
        const outputs: Float32Array[] = [];
        for (let i = 0; i < 8; i++) {
            outputs.push(new Float32Array(pixelCount * 4).fill(0.0)); // bg = 0.0
        }

        // Transmittance per pixel (starts at 1.0)
        const transmittance = new Float32Array(pixelCount).fill(1.0);

        // Splat each Gaussian (already sorted back to front)
        const data = this.gaussianData;

        for (const sg of this.sortedGaussians) {
            const i = sg.index;
            const opacity = data.opacity[i];

            // Skip nearly transparent Gaussians
            if (opacity < 0.001) continue;

            const screenX = sg.screenX;
            const screenY = sg.screenY;
            const radius = sg.screenRadius;

            // Determine bounding box
            const minX = Math.max(0, Math.floor(screenX - radius * 3));
            const maxX = Math.min(width - 1, Math.ceil(screenX + radius * 3));
            const minY = Math.max(0, Math.floor(screenY - radius * 3));
            const maxY = Math.min(height - 1, Math.ceil(screenY + radius * 3));

            // Get latent values for this Gaussian
            const latent = new Float32Array(32);
            for (let j = 0; j < 32; j++) {
                latent[j] = data.latents[i * 32 + j];
            }

            // Splat to each pixel in bounding box
            for (let py = minY; py <= maxY; py++) {
                for (let px = minX; px <= maxX; px++) {
                    const dx = px - screenX;
                    const dy = py - screenY;
                    const r2 = (dx * dx + dy * dy) / (radius * radius + 0.0001);

                    if (r2 > 9.0) continue; // 3 sigma cutoff

                    // Gaussian weight
                    const gaussian = Math.exp(-0.5 * r2);
                    const alpha = gaussian * opacity;

                    if (alpha < 0.001) continue;

                    const pixelIdx = py * width + px;
                    const T = transmittance[pixelIdx];

                    if (T < 0.001) continue; // Pixel is fully covered

                    const weight = alpha * T;

                    // Accumulate all 32 latent channels
                    // 正しい3DGS式: C = Σ(c_i * α_i * T_i)
                    for (let ch = 0; ch < 32; ch++) {
                        const texIdx = Math.floor(ch / 4);
                        const chInTex = ch % 4;
                        const bufIdx = pixelIdx * 4 + chInTex;

                        // accumulated += color * alpha * T
                        outputs[texIdx][bufIdx] += latent[ch] * weight;
                    }

                    // Update transmittance
                    transmittance[pixelIdx] = T * (1.0 - alpha);
                }
            }
        }

        // ======== 🔍 DEBUG: OUTPUT RGB CHECK AFTER SPLATTING ========
        if (this.renderCount === 1) {
            console.log('[ComputeRenderer] 🔍🔍🔍 OUTPUT RGB CHECK (after splatting):');

            // Find non-background pixels and check their RGB
            const sampleOutputs: {x: number, y: number, r: number, g: number, b: number, T: number}[] = [];
            for (let p = 0; p < pixelCount && sampleOutputs.length < 15; p++) {
                const r = outputs[0][p * 4 + 0];
                const g = outputs[0][p * 4 + 1];
                const b = outputs[0][p * 4 + 2];
                const T = transmittance[p];
                // Non-background = T < 1.0 (something was rendered here)
                if (T < 0.99) {
                    sampleOutputs.push({
                        x: p % width,
                        y: Math.floor(p / width),
                        r, g, b, T
                    });
                }
            }

            console.log('[ComputeRenderer]   Sample output pixels (accumulated RGB before background):');
            for (const sp of sampleOutputs.slice(0, 10)) {
                const diffRG = Math.abs(sp.r - sp.g);
                const diffGB = Math.abs(sp.g - sp.b);
                console.log(`[ComputeRenderer]     (${sp.x},${sp.y}): R=${sp.r.toFixed(4)}, G=${sp.g.toFixed(4)}, B=${sp.b.toFixed(4)}, T=${sp.T.toFixed(4)} | diff: R-G=${diffRG.toFixed(4)}, G-B=${diffGB.toFixed(4)}`);
            }

            // Check overall diversity in outputs
            let sumDiffRG = 0, sumDiffGB = 0;
            let sumDiffRG2 = 0, sumDiffGB2 = 0;
            let countRendered = 0;

            for (let p = 0; p < pixelCount; p++) {
                if (transmittance[p] < 0.99) {
                    const r = outputs[0][p * 4 + 0];
                    const g = outputs[0][p * 4 + 1];
                    const b = outputs[0][p * 4 + 2];
                    const dRG = r - g;
                    const dGB = g - b;
                    sumDiffRG += dRG;
                    sumDiffGB += dGB;
                    sumDiffRG2 += dRG * dRG;
                    sumDiffGB2 += dGB * dGB;
                    countRendered++;
                }
            }

            if (countRendered > 0) {
                const meanDiffRG = sumDiffRG / countRendered;
                const meanDiffGB = sumDiffGB / countRendered;
                const varDiffRG = sumDiffRG2 / countRendered - meanDiffRG * meanDiffRG;
                const varDiffGB = sumDiffGB2 / countRendered - meanDiffGB * meanDiffGB;

                console.log(`[ComputeRenderer]   Output RGB diversity (${countRendered} rendered pixels):`);
                console.log(`[ComputeRenderer]     R-G diff: mean=${meanDiffRG.toFixed(6)}, σ=${Math.sqrt(varDiffRG).toFixed(6)}`);
                console.log(`[ComputeRenderer]     G-B diff: mean=${meanDiffGB.toFixed(6)}, σ=${Math.sqrt(varDiffGB).toFixed(6)}`);

                if (Math.sqrt(varDiffRG) < 0.01 && Math.sqrt(varDiffGB) < 0.01) {
                    console.log(`[ComputeRenderer]   ⚠️⚠️⚠️ PROBLEM: Output colors have R≈G≈B (no diversity)`);
                    console.log(`[ComputeRenderer]   The splatting process is averaging out the color differences!`);
                } else {
                    console.log(`[ComputeRenderer]   ✅ Output maintains RGB diversity`);
                }
            }
        }
        // ======== END DEBUG ========

        // Upload results to GPU buffers
        for (let i = 0; i < 8; i++) {
            this.device.queue.writeBuffer(this.outputBuffers[i], 0, outputs[i]);
        }
    }

    public getOutputTextures(): GPUTexture[] {
        // Copy from storage buffers to textures
        const commandEncoder = this.device.createCommandEncoder();

        for (let i = 0; i < 8; i++) {
            commandEncoder.copyBufferToTexture(
                { buffer: this.outputBuffers[i], bytesPerRow: this.width * 4 * 4 },
                { texture: this.outputTextures[i] },
                [this.width, this.height]
            );
        }

        this.device.queue.submit([commandEncoder.finish()]);

        return this.outputTextures;
    }

    public getOutputBuffers(): GPUBuffer[] {
        return this.outputBuffers;
    }

    destroy(): void {
        this.gaussianBuffer?.destroy();
        this.uniformBuffer?.destroy();
        this.unifiedOutputBuffer?.destroy();
        this.atomicBuffer?.destroy();
        this.sortedIndicesBuffer?.destroy();
        this.sortedDataBuffer?.destroy();
        this.outputBuffers.forEach(b => b.destroy());
        this.outputTextures.forEach(t => t.destroy());
    }
}
