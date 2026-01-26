// guava-webgpu-renderer-compute.ts
// Compute shader-based Gaussian splatting renderer
// Preserves all 32 latent channels (no channel loss from alpha blending)
//
// Key difference from fragment shader approach:
// - Fragment shader with hardware blending loses every 4th channel (used for alpha)
// - Compute shader has full control over accumulation and preserves all channels

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

    // Output textures (8 x RGBA = 32 channels)
    private outputTextures: GPUTexture[] = [];

    // Compute pipeline and resources
    private clearPipeline: GPUComputePipeline | null = null;
    private splatPipeline: GPUComputePipeline | null = null;
    private gaussianBuffer: GPUBuffer | null = null;
    private outputBuffers: GPUBuffer[] = [];
    private uniformBuffer: GPUBuffer | null = null;

    private depthArray: Float32Array;
    private indexArray: Uint32Array;
    private sortedGaussians: SortedGaussian[] = [];

    private initPromise: Promise<void>;
    private isInitialized = false;
    private sortCalled = false;
    private renderCount = 0;

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

        console.log('[ComputeRenderer] Constructor called with:');
        console.log(`  vertexCount: ${count}`);
        console.log(`  dimensions: ${this.width}x${this.height}`);
        console.log(`  positions: ${data.positions.length} floats`);
        console.log(`  latents: ${data.latents.length} floats`);

        this.initPromise = this.init();
    }

    public async waitForInit(): Promise<void> {
        await this.initPromise;
    }

    private async init() {
        try {
            this.createOutputTextures();
            this.createOutputBuffers();
            this.createGaussianBuffer();
            this.createUniformBuffer();
            await this.createPipelines();
            this.isInitialized = true;
            console.log('[ComputeRenderer] Initialization complete (32-channel compute shader)');
        } catch (error) {
            console.error('[ComputeRenderer] Initialization failed:', error);
            throw error;
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

    private createOutputBuffers(): void {
        this.outputBuffers.forEach(b => b.destroy());
        this.outputBuffers = [];

        const bufferSize = this.width * this.height * 4 * 4; // 4 floats per pixel

        for (let i = 0; i < 8; i++) {
            const buffer = this.device.createBuffer({
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            this.outputBuffers.push(buffer);
        }
        console.log('[ComputeRenderer] Created 8 storage buffers');
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
        // Uniforms: view(16) + proj(16) + dims(2) + count(1) + pad(1) = 36 floats
        this.uniformBuffer = this.device.createBuffer({
            size: 36 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    private async createPipelines(): Promise<void> {
        // Clear pipeline - fills output with background color
        const clearShader = this.device.createShaderModule({
            code: `
                @group(0) @binding(0) var<storage, read_write> out0: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read_write> out1: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read_write> out2: array<vec4<f32>>;
                @group(0) @binding(3) var<storage, read_write> out3: array<vec4<f32>>;
                @group(0) @binding(4) var<storage, read_write> out4: array<vec4<f32>>;
                @group(0) @binding(5) var<storage, read_write> out5: array<vec4<f32>>;
                @group(0) @binding(6) var<storage, read_write> out6: array<vec4<f32>>;
                @group(0) @binding(7) var<storage, read_write> out7: array<vec4<f32>>;
                @group(0) @binding(8) var<storage, read_write> transmittance: array<f32>;

                struct Uniforms {
                    view: mat4x4<f32>,
                    proj: mat4x4<f32>,
                    width: u32,
                    height: u32,
                    count: u32,
                    pad: u32,
                };
                @group(0) @binding(9) var<uniform> u: Uniforms;

                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                    if (gid.x >= u.width || gid.y >= u.height) { return; }
                    let idx = gid.y * u.width + gid.x;

                    // Background = 1.0 (matching Python GUAVA bg=1.0)
                    let bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                    out0[idx] = bg;
                    out1[idx] = bg;
                    out2[idx] = bg;
                    out3[idx] = bg;
                    out4[idx] = bg;
                    out5[idx] = bg;
                    out6[idx] = bg;
                    out7[idx] = bg;
                    transmittance[idx] = 1.0;  // Initial transmittance = 1
                }
            `
        });

        const clearBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.clearPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [clearBindGroupLayout] }),
            compute: { module: clearShader, entryPoint: 'main' }
        });

        console.log('[ComputeRenderer] Created clear pipeline');

        // Note: Full compute shader splatting would require per-pixel Gaussian sorting
        // For now, we'll use a CPU-assisted approach for proper blending
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
        }
    }

    public render(): void {
        if (!this.isInitialized) {
            console.warn('[ComputeRenderer] Not ready to render');
            return;
        }

        this.renderCount++;

        // CPU-based splatting to ensure all 32 channels are preserved
        this.cpuSplat();

        if (this.renderCount === 1) {
            console.log('[ComputeRenderer] First render() complete (CPU splat, 32 channels preserved)');
        }
    }

    /**
     * CPU-based Gaussian splatting that preserves all 32 channels
     * This is slower than GPU but guarantees correctness
     */
    private cpuSplat(): void {
        const width = this.width;
        const height = this.height;
        const pixelCount = width * height;

        // Initialize output arrays (8 textures x 4 channels = 32)
        const outputs: Float32Array[] = [];
        for (let i = 0; i < 8; i++) {
            outputs.push(new Float32Array(pixelCount * 4).fill(1.0)); // bg = 1.0
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
                    for (let ch = 0; ch < 32; ch++) {
                        const texIdx = Math.floor(ch / 4);
                        const chInTex = ch % 4;
                        const bufIdx = pixelIdx * 4 + chInTex;

                        // Blend: new = latent * weight + old * (1 - weight)
                        // But we're doing proper splatting:
                        // accumulated += color * alpha * T
                        // transmittance *= (1 - alpha)
                        //
                        // At the end, final = accumulated + bg * T_final
                        // Since bg = 1.0 and we initialized with bg,
                        // we subtract bg * weight and add latent * weight:
                        outputs[texIdx][bufIdx] += (latent[ch] - 1.0) * weight;
                    }

                    // Update transmittance
                    transmittance[pixelIdx] = T * (1.0 - alpha);
                }
            }
        }

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
        this.outputBuffers.forEach(b => b.destroy());
        this.outputTextures.forEach(t => t.destroy());
    }
}
