// image-encoder.ts
// デバッグ版: 各ステップの出力統計を詳細にログ
// 参照: ubody_gaussian.py, data_loader.py, test_dinov2_518.py

import * as ort from 'onnxruntime-web';
import { RawImage } from '@huggingface/transformers';

/**
 * 公式実装の定数 (data_loader.py: load_canonical_render_prams)
 */
const CANONICAL_W2C = {
    R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T: [0.0, 0.6, 22.0]
};
const INV_TAN_FOV = 24.0;

export interface SourceCameraConfig {
    debug?: boolean;
}

export class ImageEncoder {
    private dinov2Session: ort.InferenceSession | null = null;
    private encoderSession: ort.InferenceSession | null = null;
    private initialized = false;

    async init(): Promise<void> {
        if (this.initialized) return;
        
        try {
            console.log('[ImageEncoder] Initializing (Official Python Port)...');
            
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
            ort.env.wasm.proxy = false;

            const [modelResp, dataResp, encResp] = await Promise.all([
                fetch('/assets/dinov2_518.onnx'),
                fetch('/assets/dinov2_518.onnx.data'),
                fetch('/assets/dino_encoder.onnx')
            ]);

            const modelBuffer = await modelResp.arrayBuffer();
            const options: any = { executionProviders: ['wasm'] };
            
            if (dataResp.ok) {
                const dataBuffer = await dataResp.arrayBuffer();
                options.externalData = [{ 
                    path: 'dinov2_518.onnx.data', 
                    data: dataBuffer 
                }];
            }
            
            this.dinov2Session = await ort.InferenceSession.create(modelBuffer, options);

            const encBuffer = await encResp.arrayBuffer();
            this.encoderSession = await ort.InferenceSession.create(encBuffer, { 
                executionProviders: ['wasm'] 
            });
            
            this.initialized = true;
            console.log('[ImageEncoder] ✅ Initialized');
            console.log('[ImageEncoder]   Input: 518x518 RGB');
            console.log('[ImageEncoder]   DINOv2-base: 768ch patches (37x37)');
            console.log('[ImageEncoder]   Encoder: 128ch appearance map (37x37 → 518x518)');
            console.log('[ImageEncoder]   Note: Global mapping is now in Template Decoder');
        } catch (error) {
            console.error('[ImageEncoder] ❌ Init failed:', error);
            throw error;
        }
    }

    /**
     * 配列の統計情報を取得
     */
    private getArrayStats(arr: Float32Array): { min: number; max: number; mean: number; nonZero: number; nanCount: number; uniqueApprox: number } {
        let min = Infinity;
        let max = -Infinity;
        let sum = 0;
        let nonZero = 0;
        let nanCount = 0;
        const sampleSet = new Set<string>();
        
        for (let i = 0; i < arr.length; i++) {
            const v = arr[i];
            if (isNaN(v)) {
                nanCount++;
                continue;
            }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            if (v !== 0) nonZero++;
            
            // サンプリングしてユニーク値を近似
            if (i % 100 === 0) {
                sampleSet.add(v.toFixed(4));
            }
        }
        
        return {
            min,
            max,
            mean: sum / arr.length,
            nonZero,
            nanCount,
            uniqueApprox: sampleSet.size
        };
    }

    private preprocessImage(image: RawImage): Float32Array {
        const width = 518;
        const height = 518;
        const pixels = new Float32Array(3 * width * height);
        const data = image.data;
        const channels = image.channels || 4;

        const stride = width * height;
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                const idx = h * width + w;
                const srcIdx = idx * channels;

                for (let c = 0; c < 3; c++) {
                    const value = data[srcIdx + c] / 255.0;
                    const normalized = (value - mean[c]) / std[c];
                    pixels[c * stride + idx] = normalized;
                }
            }
        }

        return pixels;
    }

    /**
     * DINOv2 patch tokensを2D feature mapに変換
     * Input: [numPatches * patchDim] flatten patches
     * Output: [patchDim, gridSize, gridSize] CHW format
     */
    private reshapePatches(
        patchData: Float32Array, 
        numPatches: number, 
        patchDim: number
    ): { data: Float32Array; size: number } {
        const gridSize = Math.sqrt(numPatches); // 37
        
        if (!Number.isInteger(gridSize)) {
            throw new Error(`Invalid patch count: ${numPatches} (expected 1369 = 37²)`);
        }

        const featureMap = new Float32Array(patchDim * gridSize * gridSize);
        
        // Reshape [N, D] → [D, H, W]
        for (let p = 0; p < numPatches; p++) {
            const py = Math.floor(p / gridSize);
            const px = p % gridSize;
            
            for (let d = 0; d < patchDim; d++) {
                const dstIdx = d * (gridSize * gridSize) + py * gridSize + px;
                const srcIdx = p * patchDim + d;
                featureMap[dstIdx] = patchData[srcIdx];
            }
        }

        return { data: featureMap, size: gridSize };
    }

    /**
     * Projection Sampling - 公式実装の正確な移植
     * appearanceMap (37x37 or 518x518) から頂点位置に基づいて特徴をサンプリング
     *
     * Python参照: padding_mode='border' (範囲外は端の値を使用)
     */
    private sampleProjectionFeatures(
        vertices: Float32Array,
        vertexCount: number,
        featureMap: Float32Array,
        mapSize: number,
        featureDim: number
    ): { features: Float32Array; visibilityMask: Uint8Array } {
        const output = new Float32Array(vertexCount * featureDim);
        const visibilityMask = new Uint8Array(vertexCount);
        const mapStride = mapSize * mapSize;

        const R = CANONICAL_W2C.R;
        const T = CANONICAL_W2C.T;
        const invtanfov = INV_TAN_FOV;

        let inBoundsCount = 0;
        let outOfBoundsCount = 0;

        for (let i = 0; i < vertexCount; i++) {
            const vx = vertices[i * 3 + 0];
            const vy = vertices[i * 3 + 1];
            const vz = vertices[i * 3 + 2];

            // World to Camera transform
            const cx = R[0][0] * vx + R[0][1] * vy + R[0][2] * vz + T[0];
            const cy = R[1][0] * vx + R[1][1] * vy + R[1][2] * vz + T[1];
            const cz = R[2][0] * vx + R[2][1] * vy + R[2][2] * vz + T[2];

            // Perspective projection
            const divZ = cz + 1e-7;
            const imgX = (cx * invtanfov) / divZ;
            const imgY = (cy * invtanfov) / divZ;

            // Grid sample (align_corners=False)
            let u = ((imgX + 1.0) * mapSize - 1.0) / 2.0;
            let v = ((imgY + 1.0) * mapSize - 1.0) / 2.0;

            // padding_mode='border': 範囲外は端にクランプ（Python参照と同じ）
            const wasOutOfBounds = u < 0 || u > mapSize - 1 || v < 0 || v > mapSize - 1;
            if (wasOutOfBounds) {
                outOfBoundsCount++;
                visibilityMask[i] = 0;  // 範囲外
            } else {
                inBoundsCount++;
                visibilityMask[i] = 1;  // 範囲内
            }

            // Clamp to border (padding_mode='border')
            u = Math.max(0, Math.min(mapSize - 1, u));
            v = Math.max(0, Math.min(mapSize - 1, v));

            const x0 = Math.floor(u);
            const y0 = Math.floor(v);
            const x1 = Math.min(x0 + 1, mapSize - 1);
            const y1 = Math.min(y0 + 1, mapSize - 1);

            const wx = u - x0;
            const wy = v - y0;

            // Bilinear interpolation
            for (let c = 0; c < featureDim; c++) {
                const plane = c * mapStride;

                const v00 = featureMap[plane + y0 * mapSize + x0];
                const v10 = featureMap[plane + y0 * mapSize + x1];
                const v01 = featureMap[plane + y1 * mapSize + x0];
                const v11 = featureMap[plane + y1 * mapSize + x1];

                const val =
                    v00 * (1 - wx) * (1 - wy) +
                    v10 * wx * (1 - wy) +
                    v01 * (1 - wx) * wy +
                    v11 * wx * wy;

                output[i * featureDim + c] = val;
            }
        }

        console.log(`[ImageEncoder] Projection sampling: ${inBoundsCount}/${vertexCount} vertices in bounds`);
        console.log(`[ImageEncoder] ⚠️ Out of bounds vertices (border padding): ${outOfBoundsCount}`);
        return { features: output, visibilityMask };
    }

    /**
     * メイン処理: 画像から特徴抽出
     * 
     * 公式実装フロー:
     * 1. DINOv2-baseで特徴抽出 (518x518 → 37x37 patches, 768ch)
     * 2. Encoderでアップサンプリング (768ch → 128ch, 37x37 → 518x518)
     * 3. Projection samplingで頂点特徴抽出 (128ch)
     * 4. Global feature (CLS token) を768次元で返す
     */
    async extractFeaturesWithSourceCamera(
        imageUrl: string,
        _config: SourceCameraConfig,
        vertices: Float32Array,
        vertexCount: number,
        featureDim: number = 128
    ): Promise<{ projectionFeature: Float32Array; idEmbedding: Float32Array; visibilityMask: Uint8Array }> {
        if (!this.dinov2Session || !this.encoderSession) {
            throw new Error('[ImageEncoder] Not initialized');
        }

        console.log('[ImageEncoder] Processing image...');

        // 1. Load and preprocess image
        const image = await RawImage.fromURL(imageUrl);
        const resized = await image.resize(518, 518);
        const preprocessed = this.preprocessImage(resized);
        
        // 🔍 DEBUG: 前処理後の画像データ
        const preprocStats = this.getArrayStats(preprocessed);
        console.log(`[ImageEncoder] 🔍 Preprocessed image stats:`);
        console.log(`[ImageEncoder]   range: [${preprocStats.min.toFixed(4)}, ${preprocStats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   mean: ${preprocStats.mean.toFixed(4)}, nonZero: ${preprocStats.nonZero}/${preprocessed.length}`);
        
        const pixelTensor = new ort.Tensor(
            'float32', 
            preprocessed, 
            [1, 3, 518, 518]
        );

        // 2. DINOv2-base inference
        console.log('[ImageEncoder] Running DINOv2-base...');
        const dinov2Out = await this.dinov2Session.run({ 'pixel_values': pixelTensor });
        const hiddenState = dinov2Out['last_hidden_state'].data as Float32Array;
        
        // 🔧 FIX: DINOv2-base outputs 768 dimensions, not 1024
        const patchDim = 768;  // DINOv2-base = 768, DINOv2-large = 1024
        const numPatches = 1369;  // 37 × 37
        
        console.log(`[ImageEncoder] DINOv2 output: ${hiddenState.length} floats`);
        console.log(`[ImageEncoder] Expected: ${(1 + numPatches) * patchDim} = ${(1 + numPatches) * patchDim}`);
        
        // 🔍 DEBUG: DINOv2出力の統計
        const dinov2Stats = this.getArrayStats(hiddenState);
        console.log(`[ImageEncoder] 🔍 DINOv2 output stats:`);
        console.log(`[ImageEncoder]   range: [${dinov2Stats.min.toFixed(4)}, ${dinov2Stats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   mean: ${dinov2Stats.mean.toFixed(4)}, nonZero: ${dinov2Stats.nonZero}/${hiddenState.length} (${(dinov2Stats.nonZero/hiddenState.length*100).toFixed(1)}%)`);
        console.log(`[ImageEncoder]   NaN count: ${dinov2Stats.nanCount}, unique approx: ${dinov2Stats.uniqueApprox}`);
        
        // Validate output shape
        const expectedLength = (1 + numPatches) * patchDim;  // 1370 × 768 = 1,052,160
        if (hiddenState.length !== expectedLength) {
            throw new Error(`DINOv2 output mismatch: got ${hiddenState.length}, expected ${expectedLength}`);
        }
        
        // CLS token is first token
        const clsToken = hiddenState.subarray(0, patchDim);
        // Patch tokens are remaining tokens
        const patchTokens = hiddenState.subarray(patchDim);
        
        // 🔍 DEBUG: CLS token統計
        const clsStats = this.getArrayStats(clsToken);
        console.log(`[ImageEncoder] 🔍 CLS token stats:`);
        console.log(`[ImageEncoder]   range: [${clsStats.min.toFixed(4)}, ${clsStats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   nonZero: ${clsStats.nonZero}/${clsToken.length}`);
        
        console.log(`[ImageEncoder] CLS token: ${clsToken.length} dims`);
        console.log(`[ImageEncoder] Patch tokens: ${patchTokens.length} floats (${patchTokens.length / patchDim} patches)`);

        // 3. Reshape patches to 2D feature map
        console.log('[ImageEncoder] Reshaping patches...');
        const { data: patchFeatureMap, size: patchMapSize } = 
            this.reshapePatches(patchTokens, numPatches, patchDim);
        
        // 🔍 DEBUG: Reshape後の統計
        const reshapeStats = this.getArrayStats(patchFeatureMap);
        console.log(`[ImageEncoder] 🔍 Reshaped feature map stats:`);
        console.log(`[ImageEncoder]   range: [${reshapeStats.min.toFixed(4)}, ${reshapeStats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   nonZero: ${reshapeStats.nonZero}/${patchFeatureMap.length} (${(reshapeStats.nonZero/patchFeatureMap.length*100).toFixed(1)}%)`);
        
        console.log(`[ImageEncoder] Feature map: [${patchDim}, ${patchMapSize}, ${patchMapSize}]`);

        // 4. Encoder (Conv upsampling: 768ch → 128ch, 37x37 → 518x518)
        console.log('[ImageEncoder] Running encoder...');
        
        const encoderIn = new ort.Tensor(
            'float32', 
            patchFeatureMap, 
            [1, patchDim, patchMapSize, patchMapSize]
        );
        const encoderOut = await this.encoderSession.run({ 'dinov2_features': encoderIn });
        
        const appearanceMap = encoderOut[this.encoderSession.outputNames[0]].data as Float32Array;
        
        // Get output shape from tensor
        const outputShape = encoderOut[this.encoderSession.outputNames[0]].dims;
        const appearanceDim = outputShape[1] as number;
        const outputMapSize = outputShape[2] as number;
        
        console.log(`[ImageEncoder] Appearance map shape: [1, ${appearanceDim}, ${outputMapSize}, ${outputMapSize}]`);
        
        // 🔍 DEBUG: Encoder出力（appearance map）の統計 - これが重要！
        const appearanceStats = this.getArrayStats(appearanceMap);
        console.log(`[ImageEncoder] 🔍🔍🔍 ENCODER OUTPUT (appearance map) stats:`);
        console.log(`[ImageEncoder]   range: [${appearanceStats.min.toFixed(4)}, ${appearanceStats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   mean: ${appearanceStats.mean.toFixed(4)}`);
        console.log(`[ImageEncoder]   nonZero: ${appearanceStats.nonZero}/${appearanceMap.length} (${(appearanceStats.nonZero/appearanceMap.length*100).toFixed(1)}%)`);
        console.log(`[ImageEncoder]   NaN count: ${appearanceStats.nanCount}`);
        console.log(`[ImageEncoder]   unique approx: ${appearanceStats.uniqueApprox}`);
        
        // 🔍 DEBUG: サンプル値
        console.log(`[ImageEncoder]   sample[0..9]: [${Array.from(appearanceMap.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}]`);

        // 5. Projection sampling
        console.log('[ImageEncoder] Projection sampling...');
        const { features: projectionFeature, visibilityMask } = this.sampleProjectionFeatures(
            vertices,
            vertexCount,
            appearanceMap,
            outputMapSize,
            featureDim  // Use first 128 channels
        );

        // 🔍 DEBUG: Projection sampling後の統計
        const projStats = this.getArrayStats(projectionFeature);
        console.log(`[ImageEncoder] 🔍 Projection features stats:`);
        console.log(`[ImageEncoder]   range: [${projStats.min.toFixed(4)}, ${projStats.max.toFixed(4)}]`);
        console.log(`[ImageEncoder]   nonZero: ${projStats.nonZero}/${projectionFeature.length} (${(projStats.nonZero/projectionFeature.length*100).toFixed(1)}%)`);

        // Count visible vertices
        let visibleCount = 0;
        for (let i = 0; i < vertexCount; i++) {
            if (visibilityMask[i]) visibleCount++;
        }
        console.log(`[ImageEncoder] 👁️ Visibility mask: ${visibleCount}/${vertexCount} vertices visible`);

        // 6. ID embedding from CLS token (768 dims)
        console.log('[ImageEncoder] Extracting CLS token (768ch)...');

        const idEmbedding = new Float32Array(768);
        for (let i = 0; i < 768; i++) {
            idEmbedding[i] = clsToken[i];
        }

        console.log('[ImageEncoder] ✅ Feature extraction complete');
        console.log(`[ImageEncoder]   Projection features: ${vertexCount} x ${featureDim}`);
        console.log(`[ImageEncoder]   ID embedding (CLS token): 768`);
        console.log(`[ImageEncoder]   Visibility mask: ${visibleCount} visible vertices`);

        return { projectionFeature, idEmbedding, visibilityMask };
    }

    dispose(): void {
        this.dinov2Session?.release();
        this.encoderSession?.release();
        this.initialized = false;
        console.log('[ImageEncoder] Disposed');
    }
}