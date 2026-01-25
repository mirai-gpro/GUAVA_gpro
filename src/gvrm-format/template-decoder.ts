/**
 * Template Decoder - Pure TypeScript Implementation (Debug Version)
 * NaN問題を特定するためのデバッグ版
 */

export interface TemplateDecoderInput {
  projection_features: Float32Array;
  global_embedding: Float32Array;
  num_vertices: number;
}

export interface TemplateDecoderOutput {
  positions: Float32Array;
  rotations: Float32Array;
  scales: Float32Array;
  opacities: Float32Array;
  colors: Float32Array;
  id_embedding: Float32Array;
}

interface ModelWeights {
  global_fc0_weight: Float32Array;
  global_fc0_bias: Float32Array;
  global_fc2_weight: Float32Array;
  global_fc2_bias: Float32Array;
  global_fc4_weight: Float32Array;
  global_fc4_bias: Float32Array;
  base_features: Float32Array;
  feature_0_weight: Float32Array;
  feature_0_bias: Float32Array;
  feature_2_weight: Float32Array;
  feature_2_bias: Float32Array;
  feature_4_weight: Float32Array;
  feature_4_bias: Float32Array;
  feature_6_weight: Float32Array;
  feature_6_bias: Float32Array;
  color_0_weight: Float32Array;
  color_0_bias: Float32Array;
  color_2_weight: Float32Array;
  color_2_bias: Float32Array;
  opacity_0_weight: Float32Array;
  opacity_0_bias: Float32Array;
  opacity_2_weight: Float32Array;
  opacity_2_bias: Float32Array;
  scale_0_weight: Float32Array;
  scale_0_bias: Float32Array;
  scale_2_weight: Float32Array;
  scale_2_bias: Float32Array;
  rotation_0_weight: Float32Array;
  rotation_0_bias: Float32Array;
  rotation_2_weight: Float32Array;
  rotation_2_bias: Float32Array;
}

export class TemplateDecoder {
  private initialized: boolean = false;
  private weights: ModelWeights | null = null;
  private callCount: number = 0;

  async init(assetsPath: string = '/assets'): Promise<void> {
    console.log('[TemplateDecoder] Initializing (Pure TypeScript - Debug Version)...');
    
    try {
      await this.loadWeights(assetsPath);
      this.initialized = true;
      console.log('[TemplateDecoder] ✅ Initialization complete');
    } catch (error) {
      console.error('[TemplateDecoder] ❌ Initialization failed:', error);
      throw error;
    }
  }

  private async loadWeights(assetsPath: string): Promise<void> {
    console.log('[TemplateDecoder]   Loading base_features...');
    
    const baseResponse = await fetch(`${assetsPath}/base_features.bin`);
    if (!baseResponse.ok) throw new Error(`Failed to load base_features.bin`);
    const baseBuffer = await baseResponse.arrayBuffer();
    const baseFeatures = new Float32Array(baseBuffer);
    console.log(`[TemplateDecoder]   ✅ Base features: ${baseFeatures.length / 128} vertices`);
    
    // Check for NaN in base features
    let nanCount = 0;
    for (let i = 0; i < Math.min(1000, baseFeatures.length); i++) {
      if (!isFinite(baseFeatures[i])) nanCount++;
    }
    console.log(`[TemplateDecoder]   Base features NaN check (first 1000): ${nanCount} invalid`);
    console.log(`[TemplateDecoder]   Base features sample: [${Array.from(baseFeatures.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
    
    console.log('[TemplateDecoder]   Loading weights...');
    const weightsResponse = await fetch(`${assetsPath}/template_decoder_weights.bin`);
    if (!weightsResponse.ok) throw new Error(`Failed to load weights`);
    const weightsBuffer = await weightsResponse.arrayBuffer();
    const allWeights = new Float32Array(weightsBuffer);
    console.log(`[TemplateDecoder]   ✅ Weights: ${allWeights.length} floats`);
    
    // Check for NaN in weights
    nanCount = 0;
    for (let i = 0; i < allWeights.length; i++) {
      if (!isFinite(allWeights[i])) nanCount++;
    }
    console.log(`[TemplateDecoder]   Weights NaN check: ${nanCount} invalid out of ${allWeights.length}`);
    console.log(`[TemplateDecoder]   Weights sample: [${Array.from(allWeights.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
    
    let offset = 0;
    const readArray = (size: number, name: string): Float32Array => {
      if (offset + size > allWeights.length) {
        console.error(`[TemplateDecoder] ❌ Buffer overflow: ${name} needs ${size} floats at offset ${offset}, but only ${allWeights.length} available`);
        throw new Error(`Buffer overflow reading ${name}`);
      }
      const arr = allWeights.slice(offset, offset + size);
      
      // Check this specific array
      let hasNaN = false;
      for (let i = 0; i < arr.length; i++) {
        if (!isFinite(arr[i])) { hasNaN = true; break; }
      }
      if (hasNaN) {
        console.warn(`[TemplateDecoder]   ⚠️ ${name} contains NaN/Inf`);
      }
      
      offset += size;
      return arr;
    };
    
    this.weights = {
      global_fc0_weight: readArray(256 * 768, 'global_fc0_weight'),
      global_fc0_bias: readArray(256, 'global_fc0_bias'),
      global_fc2_weight: readArray(256 * 256, 'global_fc2_weight'),
      global_fc2_bias: readArray(256, 'global_fc2_bias'),
      global_fc4_weight: readArray(256 * 256, 'global_fc4_weight'),
      global_fc4_bias: readArray(256, 'global_fc4_bias'),
      base_features: baseFeatures,
      feature_0_weight: readArray(256 * 512, 'feature_0_weight'),
      feature_0_bias: readArray(256, 'feature_0_bias'),
      feature_2_weight: readArray(256 * 256, 'feature_2_weight'),
      feature_2_bias: readArray(256, 'feature_2_bias'),
      feature_4_weight: readArray(256 * 256, 'feature_4_weight'),
      feature_4_bias: readArray(256, 'feature_4_bias'),
      feature_6_weight: readArray(256 * 256, 'feature_6_weight'),
      feature_6_bias: readArray(256, 'feature_6_bias'),
      color_0_weight: readArray(128 * 283, 'color_0_weight'),
      color_0_bias: readArray(128, 'color_0_bias'),
      color_2_weight: readArray(32 * 128, 'color_2_weight'),
      color_2_bias: readArray(32, 'color_2_bias'),
      opacity_0_weight: readArray(128 * 283, 'opacity_0_weight'),
      opacity_0_bias: readArray(128, 'opacity_0_bias'),
      opacity_2_weight: readArray(1 * 128, 'opacity_2_weight'),
      opacity_2_bias: readArray(1, 'opacity_2_bias'),
      scale_0_weight: readArray(128 * 283, 'scale_0_weight'),
      scale_0_bias: readArray(128, 'scale_0_bias'),
      scale_2_weight: readArray(3 * 128, 'scale_2_weight'),
      scale_2_bias: readArray(3, 'scale_2_bias'),
      rotation_0_weight: readArray(128 * 283, 'rotation_0_weight'),
      rotation_0_bias: readArray(128, 'rotation_0_bias'),
      rotation_2_weight: readArray(4 * 128, 'rotation_2_weight'),
      rotation_2_bias: readArray(4, 'rotation_2_bias'),
    };
    
    console.log(`[TemplateDecoder]   ✅ Parsed ${offset} floats`);
  }

  async forward(input: TemplateDecoderInput): Promise<TemplateDecoderOutput> {
    if (!this.initialized || !this.weights) {
      throw new Error('Not initialized');
    }

    const N = input.num_vertices;
    const W = this.weights;
    
    this.callCount++;
    console.log(`[TemplateDecoder] ========== Call #${this.callCount} ==========`);
    console.log(`[TemplateDecoder]   N = ${N}`);

    // Debug input
    console.log(`[TemplateDecoder] 🔍 Input check:`);
    console.log(`[TemplateDecoder]   projection length: ${input.projection_features.length}, expected: ${N * 128}`);
    console.log(`[TemplateDecoder]   global length: ${input.global_embedding.length}, expected: 768`);
    
    // Check input for NaN
    let projNaN = 0, globalNaN = 0;
    for (let i = 0; i < input.projection_features.length; i++) {
      if (!isFinite(input.projection_features[i])) projNaN++;
    }
    for (let i = 0; i < input.global_embedding.length; i++) {
      if (!isFinite(input.global_embedding[i])) globalNaN++;
    }
    console.log(`[TemplateDecoder]   projection NaN: ${projNaN}, global NaN: ${globalNaN}`);

    // ================================================================
    // Step 1: Global Feature Mapping (768 → 256)
    // ================================================================
    console.log(`[TemplateDecoder] Step 1: Global mapping`);
    
    const global_1 = this.linearRelu(input.global_embedding, W.global_fc0_weight, W.global_fc0_bias, 768, 256);
    console.log(`[TemplateDecoder]   After fc0: ${this.checkArray(global_1)}`);
    
    const global_2 = this.linearRelu(global_1, W.global_fc2_weight, W.global_fc2_bias, 256, 256);
    console.log(`[TemplateDecoder]   After fc2: ${this.checkArray(global_2)}`);
    
    const id_embedding = this.linear(global_2, W.global_fc4_weight, W.global_fc4_bias, 256, 256);
    console.log(`[TemplateDecoder]   After fc4 (id_embedding): ${this.checkArray(id_embedding)}`);

    // ================================================================
    // Step 2: Get base features
    // ================================================================
    console.log(`[TemplateDecoder] Step 2: Base features`);
    const base_features = W.base_features.slice(0, N * 128);
    console.log(`[TemplateDecoder]   Base features: ${this.checkArray(base_features)}`);

    // ================================================================
    // Step 3: Concatenate features
    // ================================================================
    console.log(`[TemplateDecoder] Step 3: Concatenate (N=${N})`);
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
      // global [256]
      for (let j = 0; j < 256; j++) {
        fused[offset + 256 + j] = id_embedding[j];
      }
    }
    console.log(`[TemplateDecoder]   Fused: ${this.checkArray(fused)}`);

    // ================================================================
    // Step 4: Feature layers
    // ================================================================
    console.log(`[TemplateDecoder] Step 4: Feature layers`);
    
    let features = this.batchLinearRelu(fused, W.feature_0_weight, W.feature_0_bias, N, 512, 256);
    console.log(`[TemplateDecoder]   After layer 0: ${this.checkArray(features)}`);
    
    features = this.batchLinearRelu(features, W.feature_2_weight, W.feature_2_bias, N, 256, 256);
    console.log(`[TemplateDecoder]   After layer 2: ${this.checkArray(features)}`);
    
    features = this.batchLinearRelu(features, W.feature_4_weight, W.feature_4_bias, N, 256, 256);
    console.log(`[TemplateDecoder]   After layer 4: ${this.checkArray(features)}`);
    
    features = this.batchLinearRelu(features, W.feature_6_weight, W.feature_6_bias, N, 256, 256);
    console.log(`[TemplateDecoder]   After layer 6: ${this.checkArray(features)}`);

    // ================================================================
    // Step 5: Add view_dirs
    // ================================================================
    console.log(`[TemplateDecoder] Step 5: Add view_dirs`);
    const features_with_view = new Float32Array(N * 283);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < 256; j++) {
        features_with_view[i * 283 + j] = features[i * 256 + j];
      }
      // view_dirs [27] = 0
    }
    console.log(`[TemplateDecoder]   Features+view: ${this.checkArray(features_with_view)}`);

    // ================================================================
    // Step 6: Attribute heads
    // ================================================================
    console.log(`[TemplateDecoder] Step 6: Attribute heads`);
    
    // Color
    let rgb_h = this.batchLinearRelu(features_with_view, W.color_0_weight, W.color_0_bias, N, 283, 128);
    const colors = this.batchLinear(rgb_h, W.color_2_weight, W.color_2_bias, N, 128, 32);
    console.log(`[TemplateDecoder]   Colors: ${this.checkArray(colors)}`);
    
    // Opacity
    let op_h = this.batchLinearRelu(features_with_view, W.opacity_0_weight, W.opacity_0_bias, N, 283, 128);
    const op_raw = this.batchLinear(op_h, W.opacity_2_weight, W.opacity_2_bias, N, 128, 1);
    const opacities = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      opacities[i] = 1 / (1 + Math.exp(-op_raw[i]));
    }
    console.log(`[TemplateDecoder]   Opacities: ${this.checkArray(opacities)}`);
    
    // Scale - ✅ FIXED: Output in LOG-SPACE (renderer will apply exp())
    // The renderer shader does: exp(inst.scale.x) * focal / dist
    // So we should NOT apply exp() here - just clamp to reasonable range
    let sc_h = this.batchLinearRelu(features_with_view, W.scale_0_weight, W.scale_0_bias, N, 283, 128);
    const sc_raw = this.batchLinear(sc_h, W.scale_2_weight, W.scale_2_bias, N, 128, 3);
    const scales = new Float32Array(N * 3);
    
    // Debug: Check raw scale distribution
    let rawMin = Infinity, rawMax = -Infinity, clampedCount = 0;
    for (let i = 0; i < N * 3; i++) {
      if (sc_raw[i] < rawMin) rawMin = sc_raw[i];
      if (sc_raw[i] > rawMax) rawMax = sc_raw[i];
    }
    
    // Clamp log-space scale to prevent exp() overflow in shader
    // exp(-10) ≈ 0.00005, exp(5) ≈ 148 - reasonable range for Gaussian scale
    const SCALE_CLAMP_MIN = -10.0;
    const SCALE_CLAMP_MAX = 5.0;
    
    for (let i = 0; i < N * 3; i++) {
      // Output in LOG-SPACE (clamped) - NO exp() here!
      const clamped = Math.max(SCALE_CLAMP_MIN, Math.min(SCALE_CLAMP_MAX, sc_raw[i]));
      if (clamped !== sc_raw[i]) clampedCount++;
      scales[i] = clamped;  // ✅ LOG-SPACE output
    }
    
    console.log(`[TemplateDecoder]   Scale raw range: [${rawMin.toFixed(4)}, ${rawMax.toFixed(4)}] (log-space)`);
    console.log(`[TemplateDecoder]   Scale clamped: ${clampedCount}/${N * 3} values (${(clampedCount / (N * 3) * 100).toFixed(1)}%)`);
    console.log(`[TemplateDecoder]   Scales (log-space): ${this.checkArray(scales)}`);
    
    // Rotation
    let rot_h = this.batchLinearRelu(features_with_view, W.rotation_0_weight, W.rotation_0_bias, N, 283, 128);
    const rot_raw = this.batchLinear(rot_h, W.rotation_2_weight, W.rotation_2_bias, N, 128, 4);
    const rotations = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      let norm = 0;
      for (let j = 0; j < 4; j++) {
        norm += rot_raw[i * 4 + j] ** 2;
      }
      norm = Math.sqrt(norm) + 1e-8;
      for (let j = 0; j < 4; j++) {
        rotations[i * 4 + j] = rot_raw[i * 4 + j] / norm;
      }
    }
    console.log(`[TemplateDecoder]   Rotations: ${this.checkArray(rotations)}`);

    console.log(`[TemplateDecoder] ✅ Forward complete`);

    return {
      positions: new Float32Array(N * 3),
      rotations,
      scales,
      opacities,
      colors,
      id_embedding
    };
  }

  // ================================================================
  // Helper functions
  // ================================================================

  private checkArray(arr: Float32Array): string {
    let min = Infinity, max = -Infinity, nanCount = 0;
    for (let i = 0; i < arr.length; i++) {
      if (!isFinite(arr[i])) { nanCount++; continue; }
      if (arr[i] < min) min = arr[i];
      if (arr[i] > max) max = arr[i];
    }
    return `len=${arr.length}, min=${min.toFixed(4)}, max=${max.toFixed(4)}, NaN=${nanCount}`;
  }

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

  private linearRelu(input: Float32Array, weight: Float32Array, bias: Float32Array, inDim: number, outDim: number): Float32Array {
    const output = this.linear(input, weight, bias, inDim, outDim);
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.max(0, output[i]);
    }
    return output;
  }

  private batchLinear(input: Float32Array, weight: Float32Array, bias: Float32Array, N: number, inDim: number, outDim: number): Float32Array {
    const output = new Float32Array(N * outDim);
    for (let n = 0; n < N; n++) {
      for (let o = 0; o < outDim; o++) {
        let sum = bias[o];
        for (let i = 0; i < inDim; i++) {
          sum += input[n * inDim + i] * weight[o * inDim + i];
        }
        output[n * outDim + o] = sum;
      }
    }
    return output;
  }

  private batchLinearRelu(input: Float32Array, weight: Float32Array, bias: Float32Array, N: number, inDim: number, outDim: number): Float32Array {
    const output = this.batchLinear(input, weight, bias, N, inDim, outDim);
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.max(0, output[i]);
    }
    return output;
  }

  dispose(): void {
    this.weights = null;
    this.initialized = false;
  }
}

export default TemplateDecoder;