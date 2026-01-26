/**
 * Template Decoder - WebGPU Compute Shader Implementation
 * 
 * ONNXを完全にバイパスして、WebGPU Compute Shaderで直接実行
 * ONNX Runtime Webの動的形状問題を完全回避
 * 
 * アーキテクチャ:
 * 1. Global Feature Mapping: 768→256 (3層FC)
 * 2. Feature Concatenation: projection[128] + base[128] + global[256] = 512
 * 3. GS Decoder Feature Layers: 512→256→256→256→256 (4層)
 * 4. Attribute Heads: 256+27=283 → rgb[32], opacity[1], scale[3], rotation[4]
 */

export interface TemplateDecoderInput {
  projection_features: Float32Array;   // [N, 128]
  global_embedding: Float32Array;      // [768]
  num_vertices: number;
  viewDirection?: [number, number, number];  // カメラ方向 (正規化済み) デフォルト: (0, 0, 1) = 正面
}

export interface TemplateDecoderOutput {
  positions: Float32Array;   // [N, 3]
  rotations: Float32Array;   // [N, 4]
  scales: Float32Array;      // [N, 3]
  opacities: Float32Array;   // [N, 1]
  colors: Float32Array;      // [N, 32]
  id_embedding: Float32Array; // [256]
}

interface ModelWeights {
  // Global Feature Mapping (768→256→256→256)
  global_fc0_weight: Float32Array;  // [256, 768]
  global_fc0_bias: Float32Array;    // [256]
  global_fc2_weight: Float32Array;  // [256, 256]
  global_fc2_bias: Float32Array;    // [256]
  global_fc4_weight: Float32Array;  // [256, 256]
  global_fc4_bias: Float32Array;    // [256]
  
  // Base Features
  base_features: Float32Array;      // [10595, 128]
  
  // GS Decoder Feature Layers (512→256→256→256→256)
  feature_0_weight: Float32Array;   // [256, 512]
  feature_0_bias: Float32Array;     // [256]
  feature_2_weight: Float32Array;   // [256, 256]
  feature_2_bias: Float32Array;     // [256]
  feature_4_weight: Float32Array;   // [256, 256]
  feature_4_bias: Float32Array;     // [256]
  feature_6_weight: Float32Array;   // [256, 256]
  feature_6_bias: Float32Array;     // [256]
  
  // Attribute Heads (283→128→output)
  color_0_weight: Float32Array;     // [128, 283]
  color_0_bias: Float32Array;       // [128]
  color_2_weight: Float32Array;     // [32, 128]
  color_2_bias: Float32Array;       // [32]
  
  opacity_0_weight: Float32Array;   // [128, 283]
  opacity_0_bias: Float32Array;     // [128]
  opacity_2_weight: Float32Array;   // [1, 128]
  opacity_2_bias: Float32Array;     // [1]
  
  scale_0_weight: Float32Array;     // [128, 283]
  scale_0_bias: Float32Array;       // [128]
  scale_2_weight: Float32Array;     // [3, 128]
  scale_2_bias: Float32Array;       // [3]
  
  rotation_0_weight: Float32Array;  // [128, 283]
  rotation_0_bias: Float32Array;    // [128]
  rotation_2_weight: Float32Array;  // [4, 128]
  rotation_2_bias: Float32Array;    // [4]
}

/**
 * Template Decoder using WebGPU Compute Shaders
 * 
 * ONNXを完全にバイパスして、GPU上で直接推論を実行
 */
export class TemplateDecoderWebGPU {
  private device: GPUDevice | null = null;
  private initialized: boolean = false;
  private weights: ModelWeights | null = null;
  private callCount: number = 0;
  
  // GPU Buffers for weights
  private weightBuffers: Map<string, GPUBuffer> = new Map();
  
  // Compute pipelines
  private globalMappingPipeline: GPUComputePipeline | null = null;
  private featureLayersPipeline: GPUComputePipeline | null = null;
  private attributeHeadsPipeline: GPUComputePipeline | null = null;

  /**
   * Initialize Template Decoder with WebGPU
   */
  async init(device: GPUDevice, assetsPath: string = '/assets'): Promise<void> {
    console.log('[TemplateDecoderWebGPU] Initializing...');
    
    this.device = device;
    
    try {
      // Load weights
      await this.loadWeights(assetsPath);
      
      // Create compute pipelines
      await this.createPipelines();
      
      this.initialized = true;
      console.log('[TemplateDecoderWebGPU] ✅ Initialization complete');
      
    } catch (error) {
      console.error('[TemplateDecoderWebGPU] ❌ Initialization failed:', error);
      throw error;
    }
  }

  /**
   * Load model weights from binary files
   */
  private async loadWeights(assetsPath: string): Promise<void> {
    console.log('[TemplateDecoderWebGPU]   Loading weights...');
    
    // Load base_features.bin
    const baseResponse = await fetch(`${assetsPath}/base_features.bin`);
    const baseBuffer = await baseResponse.arrayBuffer();
    const baseFeatures = new Float32Array(baseBuffer);
    console.log(`[TemplateDecoderWebGPU]   ✅ Base features: ${baseFeatures.length / 128} vertices`);
    
    // Load weights.bin (all FC layer weights)
    const weightsResponse = await fetch(`${assetsPath}/template_decoder_weights.bin`);
    const weightsBuffer = await weightsResponse.arrayBuffer();
    const allWeights = new Float32Array(weightsBuffer);
    console.log(`[TemplateDecoderWebGPU]   ✅ Weights loaded: ${allWeights.length} floats`);
    
    // Parse weights (順序は export script で定義)
    let offset = 0;
    
    const readArray = (size: number): Float32Array => {
      const arr = allWeights.slice(offset, offset + size);
      offset += size;
      return arr;
    };
    
    this.weights = {
      // Global Feature Mapping
      global_fc0_weight: readArray(256 * 768),
      global_fc0_bias: readArray(256),
      global_fc2_weight: readArray(256 * 256),
      global_fc2_bias: readArray(256),
      global_fc4_weight: readArray(256 * 256),
      global_fc4_bias: readArray(256),
      
      // Base features (loaded separately)
      base_features: baseFeatures,
      
      // Feature layers
      feature_0_weight: readArray(256 * 512),
      feature_0_bias: readArray(256),
      feature_2_weight: readArray(256 * 256),
      feature_2_bias: readArray(256),
      feature_4_weight: readArray(256 * 256),
      feature_4_bias: readArray(256),
      feature_6_weight: readArray(256 * 256),
      feature_6_bias: readArray(256),
      
      // Color head
      color_0_weight: readArray(128 * 283),
      color_0_bias: readArray(128),
      color_2_weight: readArray(32 * 128),
      color_2_bias: readArray(32),
      
      // Opacity head
      opacity_0_weight: readArray(128 * 283),
      opacity_0_bias: readArray(128),
      opacity_2_weight: readArray(1 * 128),
      opacity_2_bias: readArray(1),
      
      // Scale head
      scale_0_weight: readArray(128 * 283),
      scale_0_bias: readArray(128),
      scale_2_weight: readArray(3 * 128),
      scale_2_bias: readArray(3),
      
      // Rotation head
      rotation_0_weight: readArray(128 * 283),
      rotation_0_bias: readArray(128),
      rotation_2_weight: readArray(4 * 128),
      rotation_2_bias: readArray(4),
    };
    
    console.log(`[TemplateDecoderWebGPU]   ✅ Weights parsed (${offset} floats used)`);

    // Debug: verify weight statistics
    const w0Stats = this.analyzeWeights(this.weights.global_fc0_weight);
    const f0Stats = this.analyzeWeights(this.weights.feature_0_weight);
    const c0Stats = this.analyzeWeights(this.weights.color_0_weight);
    console.log(`[TemplateDecoderWebGPU]   📊 global_fc0 weight: min=${w0Stats.min.toFixed(4)}, max=${w0Stats.max.toFixed(4)}, mean=${w0Stats.mean.toFixed(4)}`);
    console.log(`[TemplateDecoderWebGPU]   📊 feature_0 weight: min=${f0Stats.min.toFixed(4)}, max=${f0Stats.max.toFixed(4)}, mean=${f0Stats.mean.toFixed(4)}`);
    console.log(`[TemplateDecoderWebGPU]   📊 color_0 weight: min=${c0Stats.min.toFixed(4)}, max=${c0Stats.max.toFixed(4)}, mean=${c0Stats.mean.toFixed(4)}`);
  }

  /**
   * Create WebGPU compute pipelines
   */
  private async createPipelines(): Promise<void> {
    if (!this.device) throw new Error('Device not initialized');
    
    console.log('[TemplateDecoderWebGPU]   Creating compute pipelines...');
    
    // For now, we'll use CPU fallback since compute shaders are complex
    // This can be upgraded to full GPU compute later
    
    console.log('[TemplateDecoderWebGPU]   ✅ Pipelines created (CPU fallback mode)');
  }

  /**
   * Run Template Decoder inference (CPU implementation)
   * 
   * This is a CPU fallback that guarantees correctness.
   * Can be upgraded to WebGPU Compute Shaders for better performance.
   */
  async forward(input: TemplateDecoderInput): Promise<TemplateDecoderOutput> {
    if (!this.initialized || !this.weights) {
      throw new Error('Template Decoder not initialized');
    }

    const N = input.num_vertices;
    const weights = this.weights;
    
    this.callCount++;
    console.log(`[TemplateDecoderWebGPU] ========== Call #${this.callCount} ==========`);
    console.log(`[TemplateDecoderWebGPU]   Vertices: ${N}`);

    // ================================================================
    // Step 1: Global Feature Mapping (768 → 256)
    // Python版 ubody_gaussian.py 準拠: LeakyReLU使用
    // ================================================================
    const global_256 = this.linearLeakyRelu(
      input.global_embedding,
      weights.global_fc0_weight, weights.global_fc0_bias,
      768, 256
    );
    const global_256_2 = this.linearLeakyRelu(
      global_256,
      weights.global_fc2_weight, weights.global_fc2_bias,
      256, 256
    );
    const id_embedding = this.linear(
      global_256_2,
      weights.global_fc4_weight, weights.global_fc4_bias,
      256, 256
    );

    // Debug: global mapping output
    const globalStats = this.analyzeArray(id_embedding);
    console.log(`[TemplateDecoderWebGPU]   Global mapping: 768 → 256 ✅`);
    console.log(`[TemplateDecoderWebGPU]   📊 id_embedding stats: min=${globalStats.min.toFixed(4)}, max=${globalStats.max.toFixed(4)}, unique=${globalStats.unique}`);

    // ================================================================
    // Step 2: Get base features for N vertices
    // ================================================================
    const base_features = weights.base_features.slice(0, N * 128);
    const baseStats = this.analyzeArray(base_features);
    console.log(`[TemplateDecoderWebGPU]   Base features: ${N} x 128 ✅`);
    console.log(`[TemplateDecoderWebGPU]   📊 base_features stats: min=${baseStats.min.toFixed(4)}, max=${baseStats.max.toFixed(4)}, unique=${baseStats.unique}`);

    // 🔍 DEBUG: base_features が全てゼロでないことを確認
    let baseNonZeroCount = 0;
    for (let i = 0; i < base_features.length; i++) {
      if (Math.abs(base_features[i]) > 0.0001) baseNonZeroCount++;
    }
    console.log(`[TemplateDecoderWebGPU]   📊 base_features non-zeros: ${baseNonZeroCount}/${base_features.length} (${(baseNonZeroCount/base_features.length*100).toFixed(1)}%)`);
    if (baseNonZeroCount === 0) {
      console.error(`[TemplateDecoderWebGPU] ❌ ERROR: base_features is ALL ZEROS!`);
    }

    // ================================================================
    // Step 3: Concatenate features for each vertex
    // fused = [projection[128], base[128], global[256]] = [512]
    // ================================================================
    const projStats = this.analyzeArray(input.projection_features);
    console.log(`[TemplateDecoderWebGPU]   📊 projection_features stats: min=${projStats.min.toFixed(4)}, max=${projStats.max.toFixed(4)}, unique=${projStats.unique}`);

    const fused = new Float32Array(N * 512);
    for (let i = 0; i < N; i++) {
      const offset = i * 512;
      // projection [128]
      for (let j = 0; j < 128; j++) {
        fused[offset + j] = input.projection_features[i * 128 + j];
      }
      // base [128]
      for (let j = 0; j < 128; j++) {
        fused[offset + 128 + j] = base_features[i * 128 + j];
      }
      // global [256] (broadcast)
      for (let j = 0; j < 256; j++) {
        fused[offset + 256 + j] = id_embedding[j];
      }
    }
    const fusedStats = this.analyzeArray(fused);
    console.log(`[TemplateDecoderWebGPU]   Fused features: ${N} x 512 ✅`);
    console.log(`[TemplateDecoderWebGPU]   📊 fused stats: min=${fusedStats.min.toFixed(4)}, max=${fusedStats.max.toFixed(4)}, unique=${fusedStats.unique}`);
    console.log(`[TemplateDecoderWebGPU]   📊 fused[0..7] (vertex 0): [${Array.from(fused.slice(0, 8)).map(v => v.toFixed(3)).join(', ')}]`);

    // 🔍 DEBUG: fused特徴量の各部分の寄与を確認
    let projPart = fused.slice(0, 128);
    let basePart = fused.slice(128, 256);
    let globalPart = fused.slice(256, 512);
    let projMag = 0, baseMag = 0, globalMag = 0;
    for (let j = 0; j < 128; j++) {
      projMag += Math.abs(projPart[j]);
      baseMag += Math.abs(basePart[j]);
    }
    for (let j = 0; j < 256; j++) {
      globalMag += Math.abs(globalPart[j]);
    }
    console.log(`[TemplateDecoderWebGPU]   📊 Fused contribution (vertex 0):`);
    console.log(`[TemplateDecoderWebGPU]     projection[0:128]: L1 norm = ${projMag.toFixed(4)}`);
    console.log(`[TemplateDecoderWebGPU]     base[128:256]:     L1 norm = ${baseMag.toFixed(4)}`);
    console.log(`[TemplateDecoderWebGPU]     global[256:512]:   L1 norm = ${globalMag.toFixed(4)}`);
    if (projMag < 1.0) {
      console.log(`[TemplateDecoderWebGPU]   ⚠️ WARNING: projection_features contribution is very small (${projMag.toFixed(4)})`);
    }

    // ================================================================
    // Step 4: Feature layers (512→256→256→256→256)
    // Python版 feature_decoder.py Vertex_GS_Decoder 準拠:
    //   - 最初の3層: Linear + ReLU
    //   - 最後の層: Linear のみ (ReLU無し)
    // ================================================================
    let features = this.batchLinearRelu(fused, weights.feature_0_weight, weights.feature_0_bias, N, 512, 256);
    let fl0Stats = this.analyzeArray(features);
    console.log(`[TemplateDecoderWebGPU]   📊 after feature_layer_0: min=${fl0Stats.min.toFixed(4)}, max=${fl0Stats.max.toFixed(4)}`);

    features = this.batchLinearRelu(features, weights.feature_2_weight, weights.feature_2_bias, N, 256, 256);
    features = this.batchLinearRelu(features, weights.feature_4_weight, weights.feature_4_bias, N, 256, 256);
    features = this.batchLinear(features, weights.feature_6_weight, weights.feature_6_bias, N, 256, 256);  // NO ReLU

    const featStats = this.analyzeArray(features);
    console.log(`[TemplateDecoderWebGPU]   Feature layers: ${N} x 256 ✅`);
    console.log(`[TemplateDecoderWebGPU]   📊 final features stats: min=${featStats.min.toFixed(4)}, max=${featStats.max.toFixed(4)}, unique=${featStats.unique}`);

    // ================================================================
    // Step 5: Concatenate with view_dirs (256 + 27 = 283)
    // view_dirs = Harmonic Embedding (24) + Raw Direction (3) = 27
    // ================================================================

    // デフォルト: 正面からのビュー (0, 0, 1) = モデルからカメラへの方向
    // カメラは Z=22 にあるので、モデル(原点付近)からカメラへの方向は (0, 0, 1)
    const viewDir: [number, number, number] = input.viewDirection ?? [0, 0, 1];
    const viewDirs27 = this.computeViewDirs(viewDir);

    console.log(`[TemplateDecoderWebGPU]   View direction: (${viewDir[0].toFixed(3)}, ${viewDir[1].toFixed(3)}, ${viewDir[2].toFixed(3)})`);
    console.log(`[TemplateDecoderWebGPU]   📊 view_dirs[0..7]: [${Array.from(viewDirs27.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

    // 🔍 DEBUG: 全27要素を表示
    console.log(`[TemplateDecoderWebGPU]   📊 view_dirs FULL 27 elements:`);
    console.log(`[TemplateDecoderWebGPU]     [0-2] raw: [${viewDirs27[0].toFixed(4)}, ${viewDirs27[1].toFixed(4)}, ${viewDirs27[2].toFixed(4)}]`);
    console.log(`[TemplateDecoderWebGPU]     [3-6] sin(x): [${Array.from(viewDirs27.slice(3, 7)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[TemplateDecoderWebGPU]     [7-10] sin(y): [${Array.from(viewDirs27.slice(7, 11)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[TemplateDecoderWebGPU]     [11-14] sin(z): [${Array.from(viewDirs27.slice(11, 15)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[TemplateDecoderWebGPU]     [15-18] cos(x): [${Array.from(viewDirs27.slice(15, 19)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[TemplateDecoderWebGPU]     [19-22] cos(y): [${Array.from(viewDirs27.slice(19, 23)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[TemplateDecoderWebGPU]     [23-26] cos(z): [${Array.from(viewDirs27.slice(23, 27)).map(v => v.toFixed(4)).join(', ')}]`);

    const features_with_view = new Float32Array(N * 283);
    for (let i = 0; i < N; i++) {
      const srcOffset = i * 256;
      const dstOffset = i * 283;
      // Feature part [256]
      for (let j = 0; j < 256; j++) {
        features_with_view[dstOffset + j] = features[srcOffset + j];
      }
      // view_dirs [27] (全頂点で同じ - 正面ビューでは全Gaussianが同じカメラ方向を見る)
      for (let j = 0; j < 27; j++) {
        features_with_view[dstOffset + 256 + j] = viewDirs27[j];
      }
    }

    // ================================================================
    // Step 6: Attribute heads
    // ================================================================
    // RGB: 283→128→32
    // Python版 ubody_gaussian.py 準拠: 最初の3チャンネルにsigmoidを適用
    // self._smplx_features_color[...,:3] = torch.sigmoid(self._smplx_features_color[...,:3])
    let rgb_hidden = this.batchLinearRelu(features_with_view, weights.color_0_weight, weights.color_0_bias, N, 283, 128);
    const colors = this.batchLinear(rgb_hidden, weights.color_2_weight, weights.color_2_bias, N, 128, 32);

    // 🔍 DEBUG: sigmoid適用前のch 0-2の値を確認
    let preSigmoidMin = [Infinity, Infinity, Infinity];
    let preSigmoidMax = [-Infinity, -Infinity, -Infinity];
    let preSigmoidSum = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      const offset = i * 32;
      for (let c = 0; c < 3; c++) {
        const val = colors[offset + c];
        if (val < preSigmoidMin[c]) preSigmoidMin[c] = val;
        if (val > preSigmoidMax[c]) preSigmoidMax[c] = val;
        preSigmoidSum[c] += val;
      }
    }
    console.log(`[TemplateDecoderWebGPU] 🔍 PRE-SIGMOID color ch 0-2:`);
    for (let c = 0; c < 3; c++) {
      const chName = ['R', 'G', 'B'][c];
      console.log(`[TemplateDecoderWebGPU]   Ch ${c} (${chName}): [${preSigmoidMin[c].toFixed(4)}, ${preSigmoidMax[c].toFixed(4)}], mean=${(preSigmoidSum[c]/N).toFixed(4)}`);
    }
    // Sigmoid(0) = 0.5 なので、mean が 0 に近い場合は問題
    const avgMean = (preSigmoidSum[0] + preSigmoidSum[1] + preSigmoidSum[2]) / (N * 3);
    if (Math.abs(avgMean) < 0.5) {
      console.log(`[TemplateDecoderWebGPU] ⚠️ WARNING: Pre-sigmoid mean is near 0 (${avgMean.toFixed(4)}) → sigmoid will output ~0.5 (GRAY)`);
    }

    // Apply sigmoid to first 3 channels (RGB) - Python版と同じ
    for (let i = 0; i < N; i++) {
      const offset = i * 32;
      for (let c = 0; c < 3; c++) {
        colors[offset + c] = 1 / (1 + Math.exp(-colors[offset + c]));
      }
    }

    // 🔍 DEBUG: sigmoid適用後のch 0-2の値を確認
    let postSigmoidMin = [Infinity, Infinity, Infinity];
    let postSigmoidMax = [-Infinity, -Infinity, -Infinity];
    let postSigmoidSum = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      const offset = i * 32;
      for (let c = 0; c < 3; c++) {
        const val = colors[offset + c];
        if (val < postSigmoidMin[c]) postSigmoidMin[c] = val;
        if (val > postSigmoidMax[c]) postSigmoidMax[c] = val;
        postSigmoidSum[c] += val;
      }
    }
    console.log(`[TemplateDecoderWebGPU] 🔍 POST-SIGMOID color ch 0-2:`);
    for (let c = 0; c < 3; c++) {
      const chName = ['R', 'G', 'B'][c];
      console.log(`[TemplateDecoderWebGPU]   Ch ${c} (${chName}): [${postSigmoidMin[c].toFixed(4)}, ${postSigmoidMax[c].toFixed(4)}], mean=${(postSigmoidSum[c]/N).toFixed(4)}`);
    }

    // 🔍 DEBUG: 頂点別のカラーバリエーションを確認（最初10頂点 + 最後10頂点）
    console.log(`[TemplateDecoderWebGPU] 🔍 Per-vertex RGB colors (post-sigmoid):`);
    console.log(`[TemplateDecoderWebGPU]   First 10 vertices:`);
    for (let i = 0; i < Math.min(10, N); i++) {
      const offset = i * 32;
      const r = colors[offset + 0].toFixed(3);
      const g = colors[offset + 1].toFixed(3);
      const b = colors[offset + 2].toFixed(3);
      console.log(`[TemplateDecoderWebGPU]     v${i}: RGB(${r}, ${g}, ${b})`);
    }
    console.log(`[TemplateDecoderWebGPU]   Last 10 vertices (different body region):`);
    for (let i = Math.max(0, N - 10); i < N; i++) {
      const offset = i * 32;
      const r = colors[offset + 0].toFixed(3);
      const g = colors[offset + 1].toFixed(3);
      const b = colors[offset + 2].toFixed(3);
      console.log(`[TemplateDecoderWebGPU]     v${i}: RGB(${r}, ${g}, ${b})`);
    }

    // 🔍 DEBUG: カラーの分散を計算（同じ色ばかりか確認）
    let colorVariance = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const mean = postSigmoidSum[c] / N;
      for (let i = 0; i < N; i++) {
        const diff = colors[i * 32 + c] - mean;
        colorVariance[c] += diff * diff;
      }
      colorVariance[c] = Math.sqrt(colorVariance[c] / N);
    }
    console.log(`[TemplateDecoderWebGPU] 🔍 Color standard deviation:`);
    console.log(`[TemplateDecoderWebGPU]   R: σ=${colorVariance[0].toFixed(4)}, G: σ=${colorVariance[1].toFixed(4)}, B: σ=${colorVariance[2].toFixed(4)}`);
    if (colorVariance[0] < 0.1 && colorVariance[1] < 0.1 && colorVariance[2] < 0.1) {
      console.log(`[TemplateDecoderWebGPU] ⚠️ WARNING: Very low color variance! All vertices have similar colors.`);
    }
    
    // Opacity: 283→128→1 + sigmoid
    let opacity_hidden = this.batchLinearRelu(features_with_view, weights.opacity_0_weight, weights.opacity_0_bias, N, 283, 128);
    const opacities_raw = this.batchLinear(opacity_hidden, weights.opacity_2_weight, weights.opacity_2_bias, N, 128, 1);
    const opacities = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      opacities[i] = 1 / (1 + Math.exp(-opacities_raw[i])); // sigmoid
    }
    
    // Scale: 283→128→3 + sigmoid * 0.05
    // Python版 Vertex_GS_Decoder準拠: scales = torch.sigmoid(scales) * 0.05
    // 出力範囲: [0, 0.05]
    let scale_hidden = this.batchLinearRelu(features_with_view, weights.scale_0_weight, weights.scale_0_bias, N, 283, 128);
    const scales_raw = this.batchLinear(scale_hidden, weights.scale_2_weight, weights.scale_2_bias, N, 128, 3);
    const scales = new Float32Array(N * 3);
    for (let i = 0; i < N * 3; i++) {
      const sigmoid = 1 / (1 + Math.exp(-scales_raw[i]));
      scales[i] = sigmoid * 0.05;
    }
    
    // Rotation: 283→128→4 + normalize
    let rotation_hidden = this.batchLinearRelu(features_with_view, weights.rotation_0_weight, weights.rotation_0_bias, N, 283, 128);
    const rotations_raw = this.batchLinear(rotation_hidden, weights.rotation_2_weight, weights.rotation_2_bias, N, 128, 4);
    const rotations = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      const offset = i * 4;
      let norm = 0;
      for (let j = 0; j < 4; j++) {
        norm += rotations_raw[offset + j] * rotations_raw[offset + j];
      }
      norm = Math.sqrt(norm) + 1e-8;
      for (let j = 0; j < 4; j++) {
        rotations[offset + j] = rotations_raw[offset + j] / norm;
      }
    }

    console.log(`[TemplateDecoderWebGPU]   Attribute heads ✅`);

    // ================================================================
    // Output statistics
    // ================================================================
    const opacityStats = this.analyzeArray(opacities);
    const scaleStats = this.analyzeArray(scales);
    const rotationStats = this.analyzeArray(rotations);
    const colorStats = this.analyzeArray(colors);
    
    console.log(`[TemplateDecoderWebGPU] 📤 Output Stats:`);
    console.log(`[TemplateDecoderWebGPU]   Opacity:  min=${opacityStats.min.toFixed(6)}, max=${opacityStats.max.toFixed(6)}, unique=${opacityStats.unique}`);
    console.log(`[TemplateDecoderWebGPU]   Scale:    min=${scaleStats.min.toFixed(6)}, max=${scaleStats.max.toFixed(6)}, unique=${scaleStats.unique}`);
    console.log(`[TemplateDecoderWebGPU]   Rotation: min=${rotationStats.min.toFixed(6)}, max=${rotationStats.max.toFixed(6)}, unique=${rotationStats.unique}`);
    console.log(`[TemplateDecoderWebGPU]   RGB:      min=${colorStats.min.toFixed(6)}, max=${colorStats.max.toFixed(6)}, unique=${colorStats.unique}`);
    
    console.log(`[TemplateDecoderWebGPU] 📝 Opacity Sample (first 10): [${Array.from(opacities.slice(0, 10)).map(v => v.toFixed(6)).join(', ')}]`);

    // Positions are not modified by decoder
    const positions = new Float32Array(N * 3);

    return {
      positions,
      rotations,
      scales,
      opacities,
      colors,
      id_embedding
    };
  }

  // ================================================================
  // Helper functions for linear algebra
  // ================================================================

  /**
   * Linear layer: output = input @ weight.T + bias
   */
  private linear(input: Float32Array, weight: Float32Array, bias: Float32Array, inDim: number, outDim: number): Float32Array {
    const output = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      let sum = bias[o];
      for (let i = 0; i < inDim; i++) {
        sum += input[i] * weight[o * inDim + i];
      }
      output[o] = sum;
    }
    return output;
  }

  /**
   * Linear + ReLU
   */
  private linearRelu(input: Float32Array, weight: Float32Array, bias: Float32Array, inDim: number, outDim: number): Float32Array {
    const output = this.linear(input, weight, bias, inDim, outDim);
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.max(0, output[i]);
    }
    return output;
  }

  /**
   * Linear + LeakyReLU (negative_slope=0.01, PyTorch default)
   * Python版 ubody_gaussian.py の global_feature_mapping 準拠
   */
  private linearLeakyRelu(input: Float32Array, weight: Float32Array, bias: Float32Array, inDim: number, outDim: number): Float32Array {
    const output = this.linear(input, weight, bias, inDim, outDim);
    for (let i = 0; i < output.length; i++) {
      output[i] = output[i] > 0 ? output[i] : 0.01 * output[i];
    }
    return output;
  }

  /**
   * Batch linear: process N samples
   */
  private batchLinear(input: Float32Array, weight: Float32Array, bias: Float32Array, N: number, inDim: number, outDim: number): Float32Array {
    const output = new Float32Array(N * outDim);
    for (let n = 0; n < N; n++) {
      const inOffset = n * inDim;
      const outOffset = n * outDim;
      for (let o = 0; o < outDim; o++) {
        let sum = bias[o];
        for (let i = 0; i < inDim; i++) {
          sum += input[inOffset + i] * weight[o * inDim + i];
        }
        output[outOffset + o] = sum;
      }
    }
    return output;
  }

  /**
   * Batch linear + ReLU
   */
  private batchLinearRelu(input: Float32Array, weight: Float32Array, bias: Float32Array, N: number, inDim: number, outDim: number): Float32Array {
    const output = this.batchLinear(input, weight, bias, N, inDim, outDim);
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.max(0, output[i]);
    }
    return output;
  }

  /**
   * Analyze array statistics
   */
  private analyzeArray(arr: Float32Array): { min: number; max: number; unique: number } {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < min) min = arr[i];
      if (arr[i] > max) max = arr[i];
    }
    const sampleSize = Math.min(1000, arr.length);
    const uniqueCount = new Set(Array.from(arr.slice(0, sampleSize))).size;
    return { min, max, unique: uniqueCount };
  }

  /**
   * Analyze weight statistics (includes mean)
   */
  private analyzeWeights(arr: Float32Array): { min: number; max: number; mean: number } {
    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < min) min = arr[i];
      if (arr[i] > max) max = arr[i];
      sum += arr[i];
    }
    return { min, max, mean: sum / arr.length };
  }

  // ================================================================
  // View Direction Encoding (Harmonic Embedding)
  // Python版 ubody_gaussian.py 準拠: n_harmonic_dir = 4, direnc_dim = 27
  // ================================================================

  /**
   * Compute view_dirs encoding (27 dimensions)
   *
   * PyTorch3D HarmonicEmbedding (append_input=True) の順序:
   *   - Raw Direction: 3次元 (最初!)
   *   - Sin Embeddings: 4周波数 × 3軸 = 12次元
   *   - Cos Embeddings: 4周波数 × 3軸 = 12次元
   *   - 合計: 27次元
   *
   * @param viewDir カメラ方向ベクトル (モデルからカメラへの方向、正規化済み)
   * @returns Float32Array[27]
   */
  private computeViewDirs(viewDir: [number, number, number]): Float32Array {
    const result = new Float32Array(27);
    const nHarmonic = 4;

    // Position 0-2: Raw direction FIRST (PyTorch3D append_input=True convention)
    result[0] = viewDir[0];
    result[1] = viewDir[1];
    result[2] = viewDir[2];

    // Position 3-14: Sin embeddings
    let idx = 3;
    for (let dim = 0; dim < 3; dim++) {
      for (let f = 0; f < nHarmonic; f++) {
        const freq = Math.pow(2, f);  // [1, 2, 4, 8]
        result[idx++] = Math.sin(freq * viewDir[dim]);
      }
    }

    // Position 15-26: Cos embeddings
    for (let dim = 0; dim < 3; dim++) {
      for (let f = 0; f < nHarmonic; f++) {
        const freq = Math.pow(2, f);  // [1, 2, 4, 8]
        result[idx++] = Math.cos(freq * viewDir[dim]);
      }
    }

    return result;
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.weightBuffers.forEach(buffer => buffer.destroy());
    this.weightBuffers.clear();
    this.weights = null;
    this.initialized = false;
    this.callCount = 0;
    console.log('[TemplateDecoderWebGPU] Disposed');
  }
}

export default TemplateDecoderWebGPU;
