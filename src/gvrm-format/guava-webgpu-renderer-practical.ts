// guava-webgpu-renderer-practical.ts
// v66: TemplateDecoderWebGPU対応版
// - opacity/scale: シェーダーでは変換なし（TypeScriptで適用済み）
// - ブレンディング: src*1 + dst*(1-src.a) (premultiplied alpha)

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

export class GuavaWebGPURendererPractical {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline | null = null;
    private vertexBuffer: GPUBuffer | null = null;
    private instanceBuffer: GPUBuffer | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private bindGroup: GPUBindGroup | null = null;
    private renderTargets: GPUTexture[] = [];
    
    private gaussianData: GaussianData;
    private cameraConfig: CameraConfig;
    
    private width: number;
    private height: number;
    private readonly numMRTs = 8;
    private readonly floatsPerInstance = 44; 

    private depthArray: Float32Array;
    private indexArray: Uint32Array;
    private sortedInstanceData: Float32Array;
    
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
        this.sortedInstanceData = new Float32Array(count * this.floatsPerInstance);
        for (let i = 0; i < count; i++) this.indexArray[i] = i;

        console.log('[Renderer] Constructor called with:');
        console.log(`  vertexCount: ${count}`);
        console.log(`  positions: ${data.positions.length} floats`);
        console.log(`  latents: ${data.latents.length} floats`);
        console.log(`  opacity: ${data.opacity.length} floats`);
        console.log(`  scale: ${data.scale.length} floats`);
        console.log(`  rotation: ${data.rotation.length} floats`);
        
        let posNaN = 0, posZero = 0;
        for (let i = 0; i < Math.min(100, data.positions.length); i++) {
            if (isNaN(data.positions[i])) posNaN++;
            if (data.positions[i] === 0) posZero++;
        }
        console.log(`  positions sample (first 100): NaN=${posNaN}, zeros=${posZero}`);
        console.log(`  positions[0..8]: ${Array.from(data.positions.slice(0, 9)).map(v => v.toFixed(4)).join(', ')}`);
        
        let opNaN = 0, opZero = 0;
        for (let i = 0; i < Math.min(100, data.opacity.length); i++) {
            if (isNaN(data.opacity[i])) opNaN++;
            if (data.opacity[i] === 0) opZero++;
        }
        console.log(`  opacity sample (first 100): NaN=${opNaN}, zeros=${opZero}`);
        console.log(`  opacity[0..9]: ${Array.from(data.opacity.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}`);

        this.initPromise = this.init();
    }
    
    public async waitForInit(): Promise<void> {
        await this.initPromise;
    }

    private async init() {
        try {
            this.setupBuffers();
            this.initRenderTargets();
            this.pipeline = await this.createRenderPipeline();
            this.createBindGroup();
            this.isInitialized = true;
            console.log('[Renderer] ✅ Initialized (v65: 二重変換修正)');
        } catch (error) {
            console.error('[Renderer] Init failed:', error);
        }
    }

    private initRenderTargets() {
        this.renderTargets.forEach(t => t.destroy());
        this.renderTargets = [];
        for (let i = 0; i < this.numMRTs; i++) {
            this.renderTargets.push(this.device.createTexture({
                size: [this.width, this.height, 1],
                format: 'rgba16float',
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            }));
        }
        console.log(`[Renderer] Created ${this.numMRTs} render targets (${this.width}x${this.height})`);
    }

    private setupBuffers() {
        const quadVertices = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        this.vertexBuffer = this.device.createBuffer({
            size: quadVertices.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(quadVertices);
        this.vertexBuffer.unmap();

        const instanceBufferSize = this.gaussianData.vertexCount * this.floatsPerInstance * 4;
        this.instanceBuffer = this.device.createBuffer({
            size: instanceBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        console.log(`[Renderer] Instance buffer created: ${instanceBufferSize} bytes`);
        
        const uniformSize = 32 * 4;
        this.uniformBuffer = this.device.createBuffer({
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        
        const uniformData = new Float32Array(32);
        uniformData.set(this.cameraConfig.viewMatrix, 0);
        uniformData.set(this.cameraConfig.projMatrix, 16);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
        
        console.log('[Renderer] View matrix:', Array.from(this.cameraConfig.viewMatrix.slice(0, 16)).map(v => v.toFixed(3)).join(', '));
        console.log('[Renderer] Proj matrix:', Array.from(this.cameraConfig.projMatrix.slice(0, 16)).map(v => v.toFixed(3)).join(', '));
    }

    public updateGaussianData(newData: GaussianData) {
        this.gaussianData = newData;
        this.sort();
    }

    public sort() {
        if (!this.instanceBuffer || !this.isInitialized) {
            console.warn('[Renderer] sort() called but not ready');
            return;
        }
        
        const positions = this.gaussianData.positions;
        const count = this.gaussianData.vertexCount;

        for (let i = 0; i < count; i++) {
            this.depthArray[i] = positions[i * 3 + 2]; 
            this.indexArray[i] = i;
        }
        
        // Sort back to front for proper alpha blending
        const depthRef = this.depthArray;
        const indices = Array.from(this.indexArray);
        indices.sort((a, b) => depthRef[a] - depthRef[b]);
        
        const data = this.gaussianData;
        const fpi = this.floatsPerInstance;
        
        for (let sortIdx = 0; sortIdx < count; sortIdx++) {
            const origIdx = indices[sortIdx];
            const outOffset = sortIdx * fpi;
            
            this.sortedInstanceData[outOffset + 0] = data.positions[origIdx * 3 + 0];
            this.sortedInstanceData[outOffset + 1] = data.positions[origIdx * 3 + 1];
            this.sortedInstanceData[outOffset + 2] = data.positions[origIdx * 3 + 2];
            this.sortedInstanceData[outOffset + 3] = data.opacity[origIdx];
            this.sortedInstanceData[outOffset + 4] = data.scale[origIdx * 3 + 0];
            this.sortedInstanceData[outOffset + 5] = data.scale[origIdx * 3 + 1];
            this.sortedInstanceData[outOffset + 6] = data.scale[origIdx * 3 + 2];
            this.sortedInstanceData[outOffset + 7] = 0;
            this.sortedInstanceData[outOffset + 8] = data.rotation[origIdx * 4 + 0];
            this.sortedInstanceData[outOffset + 9] = data.rotation[origIdx * 4 + 1];
            this.sortedInstanceData[outOffset + 10] = data.rotation[origIdx * 4 + 2];
            this.sortedInstanceData[outOffset + 11] = data.rotation[origIdx * 4 + 3];
            
            for (let j = 0; j < 32; j++) {
                this.sortedInstanceData[outOffset + 12 + j] = data.latents[origIdx * 32 + j];
            }
        }
        
        this.device.queue.writeBuffer(this.instanceBuffer, 0, this.sortedInstanceData);
        
        if (!this.sortCalled) {
            this.sortCalled = true;
            console.log('[Renderer] First sort() complete:');
            console.log(`  Sorted ${count} instances`);
            console.log(`  Instance data size: ${this.sortedInstanceData.length} floats`);
            
            const first = this.sortedInstanceData;
            console.log(`  First instance (${fpi} floats):`);
            console.log(`    pos: [${first[0].toFixed(4)}, ${first[1].toFixed(4)}, ${first[2].toFixed(4)}]`);
            console.log(`    opacity: ${first[3].toFixed(4)}`);
            console.log(`    scale: [${first[4].toFixed(4)}, ${first[5].toFixed(4)}, ${first[6].toFixed(4)}]`);
            console.log(`    rotation: [${first[8].toFixed(4)}, ${first[9].toFixed(4)}, ${first[10].toFixed(4)}, ${first[11].toFixed(4)}]`);
            console.log(`    latent[0..3]: [${first[12].toFixed(4)}, ${first[13].toFixed(4)}, ${first[14].toFixed(4)}, ${first[15].toFixed(4)}]`);
            
            let nanCount = 0;
            for (let i = 0; i < this.sortedInstanceData.length; i++) {
                if (isNaN(this.sortedInstanceData[i])) nanCount++;
            }
            console.log(`  NaN count in instance data: ${nanCount}`);
            
            let minDepth = Infinity, maxDepth = -Infinity;
            for (let i = 0; i < count; i++) {
                const d = this.depthArray[i];
                if (d < minDepth) minDepth = d;
                if (d > maxDepth) maxDepth = d;
            }
            console.log(`  Depth range: [${minDepth.toFixed(4)}, ${maxDepth.toFixed(4)}]`);
        }
    }

    private async createRenderPipeline(): Promise<GPURenderPipeline> {
        const shaderModule = this.device.createShaderModule({
            code: `
                struct Uniforms { view: mat4x4<f32>, proj: mat4x4<f32> };
                @group(0) @binding(0) var<uniform> u: Uniforms;
                
                struct VertexInput {
                    @location(0) quadPos: vec2<f32>,
                };
                struct InstanceInput {
                    @location(1) center: vec3<f32>,
                    @location(2) opacity: f32,
                    @location(3) scale: vec3<f32>,
                    @location(4) rotation: vec4<f32>,
                    @location(5) latent0: vec4<f32>, @location(6) latent1: vec4<f32>, @location(7) latent2: vec4<f32>, @location(8) latent3: vec4<f32>,
                    @location(9) latent4: vec4<f32>, @location(10) latent5: vec4<f32>, @location(11) latent6: vec4<f32>, @location(12) latent7: vec4<f32>,
                };
                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) quadOffset: vec2<f32>,
                    @location(1) opacity: f32,
                    @location(2) latent0: vec4<f32>, @location(3) latent1: vec4<f32>, @location(4) latent2: vec4<f32>, @location(5) latent3: vec4<f32>,
                    @location(6) latent4: vec4<f32>, @location(7) latent5: vec4<f32>, @location(8) latent6: vec4<f32>, @location(9) latent7: vec4<f32>,
                };

                @vertex
                fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
                    var out: VertexOutput;
                    
                    let worldPos = vec4<f32>(inst.center, 1.0);
                    let viewPos = u.view * worldPos;
                    let projPos = u.proj * viewPos;
                    
                    let ndcCenter = projPos.xy / projPos.w;
                    
                    let dist = abs(viewPos.z);
                    let focal = 24.0; 
                    
                    // Scale: TypeScriptで既にexp適用済み
                    let sX = max(inst.scale.x, 0.001) * focal / max(dist, 0.1);
                    let sY = max(inst.scale.y, 0.001) * focal / max(dist, 0.1);
                    
                    let clampedSX = clamp(sX, 0.0001, 2.0);
                    let clampedSY = clamp(sY, 0.0001, 2.0);
                    
                    let offset = in.quadPos * vec2<f32>(clampedSX, clampedSY);
                    
                    let depth = clamp(projPos.z / projPos.w, 0.0, 1.0);
                    
                    out.position = vec4<f32>(ndcCenter + offset, depth, 1.0);
                    out.quadOffset = in.quadPos;
                    out.opacity = inst.opacity;  // TypeScriptで既にsigmoid適用済み 
                    
                    out.latent0 = inst.latent0; out.latent1 = inst.latent1; out.latent2 = inst.latent2; out.latent3 = inst.latent3;
                    out.latent4 = inst.latent4; out.latent5 = inst.latent5; out.latent6 = inst.latent6; out.latent7 = inst.latent7;
                    return out;
                }

                struct FragmentOutput {
                    @location(0) out0: vec4<f32>, @location(1) out1: vec4<f32>, @location(2) out2: vec4<f32>, @location(3) out3: vec4<f32>,
                    @location(4) out4: vec4<f32>, @location(5) out5: vec4<f32>, @location(6) out6: vec4<f32>, @location(7) out7: vec4<f32>,
                };

                @fragment
                fn fs_main(in: VertexOutput) -> FragmentOutput {
                    let r2 = dot(in.quadOffset, in.quadOffset);
                    if (r2 > 1.0) { discard; }
                    
                    let gaussian = exp(-0.5 * r2);
                    let alpha = gaussian * in.opacity;
                    
                    // 3D Gaussian Splatting blending:
                    // C = Σ c_i * α_i * T_i
                    // where T_i = Π(1 - α_j) for j < i
                    //
                    // With blend: src * 1 + dst * (1 - src.a)
                    // Output premultiplied: color * alpha
                    
                    var out: FragmentOutput;
                    out.out0 = vec4<f32>(in.latent0.xyz * alpha, alpha);
                    out.out1 = vec4<f32>(in.latent1.xyz * alpha, alpha);
                    out.out2 = vec4<f32>(in.latent2.xyz * alpha, alpha);
                    out.out3 = vec4<f32>(in.latent3.xyz * alpha, alpha);
                    out.out4 = vec4<f32>(in.latent4.xyz * alpha, alpha);
                    out.out5 = vec4<f32>(in.latent5.xyz * alpha, alpha);
                    out.out6 = vec4<f32>(in.latent6.xyz * alpha, alpha);
                    out.out7 = vec4<f32>(in.latent7.xyz * alpha, alpha);
                    return out;
                }
            `
        });

        // Correct blending for Gaussian Splatting:
        // dst = src * 1 + dst * (1 - src.a)
        const colorTargets: GPUColorTargetState[] = [];
        for (let i = 0; i < this.numMRTs; i++) {
            colorTargets.push({
                format: 'rgba16float',
                blend: {
                    color: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                    alpha: {
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha',
                        operation: 'add',
                    },
                },
                writeMask: GPUColorWrite.ALL,
            });
        }

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }]
        });
        const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        const descriptor: GPURenderPipelineDescriptor = {
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
                buffers: [
                    { arrayStride: 2 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] },
                    { arrayStride: this.floatsPerInstance * 4, stepMode: 'instance',
                      attributes: [
                          { shaderLocation: 1, offset: 0, format: 'float32x3' },
                          { shaderLocation: 2, offset: 12, format: 'float32' },
                          { shaderLocation: 3, offset: 16, format: 'float32x3' },
                          { shaderLocation: 4, offset: 32, format: 'float32x4' },
                          { shaderLocation: 5, offset: 48, format: 'float32x4' },
                          { shaderLocation: 6, offset: 64, format: 'float32x4' },
                          { shaderLocation: 7, offset: 80, format: 'float32x4' },
                          { shaderLocation: 8, offset: 96, format: 'float32x4' },
                          { shaderLocation: 9, offset: 112, format: 'float32x4' },
                          { shaderLocation: 10, offset: 128, format: 'float32x4' },
                          { shaderLocation: 11, offset: 144, format: 'float32x4' },
                          { shaderLocation: 12, offset: 160, format: 'float32x4' },
                      ]
                    }
                ]
            },
            fragment: { module: shaderModule, entryPoint: 'fs_main', targets: colorTargets },
            primitive: { topology: 'triangle-strip' },
        };

        return this.device.createRenderPipeline(descriptor);
    }

    private createBindGroup() {
        if (!this.pipeline) return;
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer! } }]
        });
    }

    public render() {
        if (!this.pipeline || !this.bindGroup || !this.renderTargets.length || !this.isInitialized) {
            console.warn('[Renderer] Not ready to render');
            return;
        }
        
        this.renderCount++;
        
        const commandEncoder = this.device.createCommandEncoder();
        
        // Clear with bg=1.0 (matching Python GUAVA's bg=1.0)
        const colorAttachments: GPURenderPassColorAttachment[] = this.renderTargets.map(tex => ({
            view: tex.createView(),
            clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
            loadOp: 'clear' as GPULoadOp,
            storeOp: 'store' as GPUStoreOp,
        }));

        const passEncoder = commandEncoder.beginRenderPass({ colorAttachments });
        passEncoder.setPipeline(this.pipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setVertexBuffer(1, this.instanceBuffer);
        passEncoder.draw(4, this.gaussianData.vertexCount);
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);
        
        if (this.renderCount === 1) {
            console.log('[Renderer] First render() complete:');
            console.log(`  Drew ${this.gaussianData.vertexCount} instances with 4 vertices each`);
            console.log('  Background: 1.0 (matching Python GUAVA bg=1.0)');
            console.log('  Blending: src*1 + dst*(1-src.a)');
        }
    }

    public getOutputTextures(): GPUTexture[] { return this.renderTargets; }

    destroy(): void {
        this.vertexBuffer?.destroy();
        this.instanceBuffer?.destroy();
        this.uniformBuffer?.destroy();
        this.renderTargets.forEach(t => t.destroy());
    }
}
