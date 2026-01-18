// image-encoder.ts
// DINOv2 + DINO Encoder 完全ONNX版
// 技術仕様書 Section 3.1: 518×518入力 → 37×37パッチ(1369パッチ)
// 修正版: Camera JSON の R_matrix/T_vector を使用

import * as ort from 'onnxruntime-web';
import { RawImage } from '@huggingface/transformers';

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

export interface SourceCameraConfig {
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
  imageWidth: number;
  imageHeight: number;
  debug?: {
    R_matrix?: number[][];
    T_vector?: number[];
  };
}

export class ImageEncoder {
  private dinov2Session: ort.InferenceSession | null = null;
  private encoderSession: ort.InferenceSession | null = null;
  private initialized = false;
  
  // 📖 論文準拠: 2つのブランチを保存
  private templateFeatures: Float32Array | null = null;  // 518×518×128
  private uvFeatures: Float32Array | null = null;        // 518×518×32

  async init(): Promise<void> {
    if (this.initialized) return;

    console.log('[ImageEncoder] Initializing ONNX models (37×37 patch support)...');

    try {
      // ONNX Runtime設定
      ort.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort-wasm.wasm'
      };
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;

      console.log('[ImageEncoder] ONNX Runtime v1.17.3 configured');

      // 1. DINOv2 ONNXモデルをロード(518×518入力 → 37×37パッチ)
      console.log('[ImageEncoder] Loading DINOv2 ONNX (518×518 input)...');

      // 外部データファイルをロード(.onnx.dataが存在する場合)
      try {
        const dataResponse = await fetch('/assets/dinov2_518.onnx.data');
        if (dataResponse.ok) {
          console.log('[ImageEncoder] Loading external data file...');
          const externalData = await dataResponse.arrayBuffer();
          this.dinov2Session = await ort.InferenceSession.create('/assets/dinov2_518.onnx', {
            externalData: [{
              path: 'dinov2_518.onnx.data',
              data: externalData
            }]
          });
        } else {
          // 外部データファイルがない場合は通常ロード
          this.dinov2Session = await ort.InferenceSession.create('/assets/dinov2_518.onnx');
        }
      } catch {
        // 外部データファイルがない場合は通常ロード
        this.dinov2Session = await ort.InferenceSession.create('/assets/dinov2_518.onnx');
      }

      console.log('[ImageEncoder] 📋 DINOv2 input names:', this.dinov2Session.inputNames);
      console.log('[ImageEncoder] 📋 DINOv2 output names:', this.dinov2Session.outputNames);

      // 2. DINO Encoder ONNXモデルをロード(37×37 → 518×518)
      console.log('[ImageEncoder] Loading DINO Encoder ONNX...');
      this.encoderSession = await ort.InferenceSession.create('/assets/dino_encoder.onnx');
      console.log('[ImageEncoder] 📋 Encoder input names:', this.encoderSession.inputNames);
      console.log('[ImageEncoder] 📋 Encoder output names:', this.encoderSession.outputNames);

      this.initialized = true;
      console.log('[ImageEncoder] ✅ Initialized with 37×37 patch support');
    } catch (error) {
      console.error('[ImageEncoder] ❌ Failed to initialize:', error);
      throw error;
    }
  }

  /**
   * DINOv2用の前処理(正規化)
   * mean = [0.485, 0.456, 0.406]
   * std = [0.229, 0.224, 0.225]
   */
  private preprocessImage(image: RawImage): Float32Array {
    const width = 518;
    const height = 518;
    const pixels = new Float32Array(3 * width * height);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // RawImageのデータはRGBA形式
    const imageData = image.data;

    for (let c = 0; c < 3; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const srcIdx = (y * width + x) * 4 + c;
          const dstIdx = c * width * height + y * width + x;
          const normalized = (imageData[srcIdx] / 255.0 - mean[c]) / std[c];
          pixels[dstIdx] = normalized;
        }
      }
    }

    return pixels;
  }

  /**
   * DINOv2のパッチ特徴を2D特徴マップに変換
   * 技術仕様書 Section 3.1: 518×518入力 → 37×37パッチ(1369パッチ)
   */
  private reshapePatchesToFeatureMap(
    patchData: Float32Array,
    numPatches: number,
    patchDim: number
  ): { data: Float32Array; height: number; width: number } {
    const gridSize = Math.sqrt(numPatches);

    if (!Number.isInteger(gridSize)) {
      throw new Error(`Invalid number of patches: ${numPatches}`);
    }

    // [numPatches, patchDim] → [patchDim, gridSize, gridSize]
    const featureMap = new Float32Array(patchDim * gridSize * gridSize);

    for (let p = 0; p < numPatches; p++) {
      const py = Math.floor(p / gridSize);
      const px = p % gridSize;

      for (let d = 0; d < patchDim; d++) {
        const srcIdx = p * patchDim + d;
        const dstIdx = d * gridSize * gridSize + py * gridSize + px;
        featureMap[dstIdx] = patchData[srcIdx];
      }
    }

    return { data: featureMap, height: gridSize, width: gridSize };
  }

  /**
   * ソースカメラ設定を使用した特徴抽出(GUAVA論文準拠)
   * 完全ONNX版: DINOv2もONNXで実行し、37×37パッチを確実に取得
   */
  async extractFeaturesWithSourceCamera(
    imageUrl: string,
    cameraConfig: SourceCameraConfig,
    vertices: Float32Array,
    vertexCount: number,
    featureDim: number = 128
  ): Promise<{ projectionFeature: Float32Array; idEmbedding: Float32Array }> {
    if (!this.dinov2Session || !this.encoderSession) {
      throw new Error('[ImageEncoder] Not initialized. Call init() first.');
    }

    console.log('[ImageEncoder] Processing image (ONNX mode):', imageUrl);

    try {
      const startTime = performance.now();

      // 1. 画像読み込みと518×518リサイズ
      const image = await RawImage.fromURL(imageUrl);
      console.log('[ImageEncoder] Original image:', {
        width: image.width,
        height: image.height
      });

      const resized = await image.resize(518, 518);
      console.log('[ImageEncoder] Resized to 518×518');

      // 2. DINOv2前処理(正規化)
      const normalized = this.preprocessImage(resized);

      // 3. DINOv2 ONNX実行
      console.log('[ImageEncoder] Running DINOv2 ONNX...');
      const dinov2Tensor = new ort.Tensor('float32', normalized, [1, 3, 518, 518]);
      const dinov2Result = await this.dinov2Session.run({
        'pixel_values': dinov2Tensor
      });

      const hiddenState = dinov2Result['last_hidden_state'].data as Float32Array;
      const totalTokens = dinov2Result['last_hidden_state'].dims[1] as number;
      const patchDim = dinov2Result['last_hidden_state'].dims[2] as number;

      console.log('[ImageEncoder] DINOv2 output:', {
        totalTokens,
        patchDim,
        expectedTokens: 1370  // 1 CLS + 37×37
      });

      // 4. CLSトークンとパッチを分離
      const clsData = hiddenState.slice(0, patchDim);
      const patchData = hiddenState.slice(patchDim);
      const numPatches = totalTokens - 1;

      console.log('[ImageEncoder] Patches:', {
        numPatches,
        gridSize: `${Math.sqrt(numPatches)}×${Math.sqrt(numPatches)}`
      });

      // 検証: 37×37パッチであることを確認
      if (numPatches !== 37 * 37) {
        console.error(`[ImageEncoder] ❌ Expected 1369 patches, got ${numPatches}`);
        throw new Error(`Invalid patch count: ${numPatches}`);
      }

      console.log('[ImageEncoder] ✅ DINOv2 output: 37×37 patches confirmed');

      // 5. パッチを2D特徴マップに変換
      const { data: featureMapData, height: fmHeight, width: fmWidth } =
        this.reshapePatchesToFeatureMap(patchData, numPatches, patchDim);

      console.log('[ImageEncoder] Feature map reshaped:', {
        channels: patchDim,
        height: fmHeight,
        width: fmWidth
      });

      // 6. DINO Encoder ONNX実行(37×37 → 518×518)
      console.log('[ImageEncoder] Running DINO Encoder ONNX...');

      const encoderTensor = new ort.Tensor('float32', featureMapData, [1, patchDim, fmHeight, fmWidth]);
      const encoderResult = await this.encoderSession.run({
        'dinov2_features': encoderTensor
      });

      const outputKey = this.encoderSession.outputNames[0];
      const appearanceTensor = encoderResult[outputKey];

      if (!appearanceTensor) {
        throw new Error(`Output '${outputKey}' not found in results`);
      }

      console.log('[ImageEncoder] Appearance features:', {
        outputName: outputKey,
        shape: appearanceTensor.dims,
        type: appearanceTensor.type
      });

      // 7. Appearance特徴マップを取得
      const appearanceData = appearanceTensor.data as Float32Array;
      const appearanceHeight = appearanceTensor.dims[2] as number;
      const appearanceWidth = appearanceTensor.dims[3] as number;

      // 検証
      if (appearanceHeight !== 518 || appearanceWidth !== 518) {
        console.warn(`[ImageEncoder] ⚠️ Expected 518×518, got ${appearanceWidth}×${appearanceHeight}`);
      } else {
        console.log('[ImageEncoder] ✅ Appearance feature map: 518×518 confirmed');
      }

      // 8. 実際の特徴マップサイズでカメラパラメータを構築
      const camera = this.buildCameraParamsFromConfig(cameraConfig, appearanceWidth, appearanceHeight);

      // 9. Projection Sampling
      const projectionFeature = this.projectionSampling(
        appearanceData,
        appearanceWidth,
        appearanceHeight,
        featureDim,
        vertices,
        vertexCount,
        camera
      );

      // 10. ID Embedding生成
      const idEmbedding = this.createIdEmbedding(clsData, patchDim, 256);

      // 11. 特徴量の正規化
      this.normalizeFeatures(projectionFeature, vertexCount, featureDim);

      const elapsed = performance.now() - startTime;
      console.log(`[ImageEncoder] ✅ Feature extraction completed in ${elapsed.toFixed(2)}ms`);

      // 統計情報
      const sampleSize = Math.min(1000, projectionFeature.length);
      const sample = Array.from(projectionFeature.slice(0, sampleSize));
      const mean = sample.reduce((a, b) => a + b, 0) / sample.length;
      const std = Math.sqrt(sample.reduce((sum, v) => sum + (v - mean) ** 2, 0) / sample.length);

      console.log('[ImageEncoder] Projection feature statistics:', {
        min: Math.min(...sample).toFixed(4),
        max: Math.max(...sample).toFixed(4),
        mean: mean.toFixed(4),
        std: std.toFixed(4),
        nonZeroRatio: (sample.filter(v => Math.abs(v) > 0.001).length / sample.length).toFixed(3)
      });


      // 📖 論文準拠: 2つのブランチに分離
      // Appendix B.2: "transform its dimensions to 32 and 128"
      const appearanceChannels = appearanceTensor.dims[1] as number;
      
      console.log('[ImageEncoder] 📖 論文準拠: Feature branches separation');
      console.log('[ImageEncoder] Encoder output channels:', appearanceChannels);
      
      const numPixels = appearanceWidth * appearanceHeight;
      
      if (appearanceChannels >= 128 + 32) {
        this.templateFeatures = new Float32Array(numPixels * 128);
        this.uvFeatures = new Float32Array(numPixels * 32);
        
        for (let i = 0; i < numPixels; i++) {
          for (let c = 0; c < 128; c++) {
            this.templateFeatures[i * 128 + c] = appearanceData[c * numPixels + i];
          }
          for (let c = 0; c < 32; c++) {
            this.uvFeatures[i * 32 + c] = appearanceData[(128 + c) * numPixels + i];
          }
        }
        
        console.log('[ImageEncoder] ✅ Separated: 128ch (template) + 32ch (UV)');
        
      } else if (appearanceChannels === 128) {
        console.warn('[ImageEncoder] ⚠️ Encoder outputs 128ch only, using subset for UV (32ch)');
        
        this.templateFeatures = new Float32Array(numPixels * 128);
        this.uvFeatures = new Float32Array(numPixels * 32);
        
        for (let i = 0; i < numPixels; i++) {
          for (let c = 0; c < 128; c++) {
            this.templateFeatures[i * 128 + c] = appearanceData[c * numPixels + i];
          }
          for (let c = 0; c < 32; c++) {
            this.uvFeatures[i * 32 + c] = appearanceData[c * numPixels + i];
          }
        }
        
        console.log('[ImageEncoder] ✅ Separated: 128ch (template) + 32ch subset (UV)');
        
      } else {
        throw new Error(`Unexpected channel count: ${appearanceChannels}. Expected >= 128`);
      }
      
      console.log('[ImageEncoder] Feature branches saved:', {
        templateSize: this.templateFeatures.length,
        uvSize: this.uvFeatures.length,
        templateChannels: 128,
        uvChannels: 32
      });
      return { projectionFeature, idEmbedding };

    } catch (error) {
      console.error('[ImageEncoder] ❌ Feature extraction failed:', error);
      throw error;
    }
  }

  /**
   * 画像から特徴抽出(CameraParams直接指定版)
   * ✅ FIX: 渡されたカメラパラメータを使用 (ハードコード値を削除)
   */
  async extractFeatures(
    imageUrl: string,
    vertices: Float32Array,
    vertexCount: number,
    camera: CameraParams,
    featureDim: number = 128
  ): Promise<{ projectionFeature: Float32Array; idEmbedding: Float32Array }> {
    // ✅ 渡されたカメラパラメータを使用 (論文準拠)
    const cameraConfig: SourceCameraConfig = {
      position: camera.position,
      target: camera.target,
      fov: camera.fov,
      imageWidth: camera.width,
      imageHeight: camera.height
    };

    console.log('[ImageEncoder] ✅ Using actual camera parameters:', {
      position: cameraConfig.position,
      target: cameraConfig.target,
      fov: cameraConfig.fov,
      resolution: `${cameraConfig.imageWidth}×${cameraConfig.imageHeight}`
    });

    return this.extractFeaturesWithSourceCamera(
      imageUrl,
      cameraConfig,
      vertices,
      vertexCount,
      featureDim
    );
  }

  /**
   * 頂点をスクリーン座標に投影
   */
  private projectVertex(
    vx: number, vy: number, vz: number,
    viewMatrix: Float32Array,
    projMatrix: Float32Array,
    screenWidth: number,
    screenHeight: number
  ): [number, number, number, number] {
    // View変換
    const vx_view = viewMatrix[0] * vx + viewMatrix[4] * vy + viewMatrix[8] * vz + viewMatrix[12];
    const vy_view = viewMatrix[1] * vx + viewMatrix[5] * vy + viewMatrix[9] * vz + viewMatrix[13];
    const vz_view = viewMatrix[2] * vx + viewMatrix[6] * vy + viewMatrix[10] * vz + viewMatrix[14];
    const vw_view = viewMatrix[3] * vx + viewMatrix[7] * vy + viewMatrix[11] * vz + viewMatrix[15];

    // Projection変換
    const vx_clip = projMatrix[0] * vx_view + projMatrix[4] * vy_view + projMatrix[8] * vz_view + projMatrix[12] * vw_view;
    const vy_clip = projMatrix[1] * vx_view + projMatrix[5] * vy_view + projMatrix[9] * vz_view + projMatrix[13] * vw_view;
    const vz_clip = projMatrix[2] * vx_view + projMatrix[6] * vy_view + projMatrix[10] * vz_view + projMatrix[14] * vw_view;
    const vw_clip = projMatrix[3] * vx_view + projMatrix[7] * vy_view + projMatrix[11] * vz_view + projMatrix[15] * vw_view;

    // NDC変換とスクリーン座標変換
    if (Math.abs(vw_clip) < 1e-6) {
      return [0, 0, 0, 0];
    }

    const x_ndc = vx_clip / vw_clip;
    const y_ndc = vy_clip / vw_clip;
    const z_ndc = vz_clip / vw_clip;

    const screenX = (x_ndc * 0.5 + 0.5) * screenWidth;
    const screenY = (1.0 - (y_ndc * 0.5 + 0.5)) * screenHeight;

    return [screenX, screenY, z_ndc, vw_clip];
  }

  /**
   * 特徴マップから線形補間でサンプリング
   */
  private sampleFeatureMapAt(
    featureMap: Float32Array,
    mapWidth: number,
    mapHeight: number,
    featureDim: number,
    screenX: number,
    screenY: number,
    output: Float32Array,
    outputOffset: number
  ): void {
    const x = Math.max(0, Math.min(screenX, mapWidth - 1));
    const y = Math.max(0, Math.min(screenY, mapHeight - 1));

    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, mapWidth - 1);
    const y0 = Math.floor(y);
    const y1 = Math.min(y0 + 1, mapHeight - 1);

    const wx = x - x0;
    const wy = y - y0;

    // CHW形式: index = channel * H * W + y * W + x
    const spatialSize = mapHeight * mapWidth;

    for (let d = 0; d < featureDim; d++) {
      const channelOffset = d * spatialSize;
      const v00 = featureMap[channelOffset + y0 * mapWidth + x0] || 0;
      const v10 = featureMap[channelOffset + y0 * mapWidth + x1] || 0;
      const v01 = featureMap[channelOffset + y1 * mapWidth + x0] || 0;
      const v11 = featureMap[channelOffset + y1 * mapWidth + x1] || 0;

      const top = v00 * (1 - wx) + v10 * wx;
      const bottom = v01 * (1 - wx) + v11 * wx;
      output[outputOffset + d] = top * (1 - wy) + bottom * wy;
    }
  }

  /**
   * Projection Sampling
   */
  private projectionSampling(
    featureMap: Float32Array,
    mapWidth: number,
    mapHeight: number,
    featureDim: number,
    vertices: Float32Array,
    vertexCount: number,
    camera: CameraParams
  ): Float32Array {
    console.log('[ImageEncoder] Projection sampling:', {
      vertexCount,
      featureDim,
      mapSize: `${mapWidth}×${mapHeight}`
    });

    const projectionFeatures = new Float32Array(vertexCount * featureDim);
    let visibleCount = 0;

    // 🔍 デバッグ: 最初の10頂点の投影結果をログ
    const debugCount = Math.min(10, vertexCount);

    for (let i = 0; i < vertexCount; i++) {
      const vx = vertices[i * 3];
      const vy = vertices[i * 3 + 1];
      const vz = vertices[i * 3 + 2];

      const [screenX, screenY, depth, clipW] = this.projectVertex(
        vx, vy, vz,
        camera.viewMatrix,
        camera.projMatrix,
        mapWidth,
        mapHeight
      );

      const isVisible = clipW > 0 && depth >= -1 && depth <= 1 &&
        screenX >= 0 && screenX < mapWidth &&
        screenY >= 0 && screenY < mapHeight;

      if (isVisible) visibleCount++;

      // 🔍 最初の10頂点をログ
      if (i < debugCount) {
        console.log(`[DEBUG] Vertex ${i}:`, {
          world: [vx.toFixed(4), vy.toFixed(4), vz.toFixed(4)],
          screen: [screenX.toFixed(2), screenY.toFixed(2)],
          depth: depth.toFixed(4),
          clipW: clipW.toFixed(4),
          isVisible
        });
      }

      this.sampleFeatureMapAt(
        featureMap,
        mapWidth,
        mapHeight,
        featureDim,
        screenX,
        screenY,
        projectionFeatures,
        i * featureDim
      );
    }

    console.log('[ImageEncoder] Visible vertices:', visibleCount, '/', vertexCount);

    if (visibleCount === 0) {
      console.warn('[ImageEncoder] ⚠️ No visible vertices! Check camera parameters.');
    }

    return projectionFeatures;
  }

  /**
   * ID embeddingを生成
   */
  private createIdEmbedding(clsToken: Float32Array, patchDim: number, outputDim: number): Float32Array {
    const idEmbedding = new Float32Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      const srcIdx = Math.floor((i / outputDim) * patchDim);
      idEmbedding[i] = clsToken[srcIdx] || 0;
    }
    return idEmbedding;
  }

  /**
   * 特徴量を正規化
   */
  private normalizeFeatures(
    features: Float32Array,
    numVertices: number,
    featureDim: number
  ): void {
    for (let d = 0; d < featureDim; d++) {
      let sum = 0;
      for (let v = 0; v < numVertices; v++) {
        sum += features[v * featureDim + d];
      }
      const mean = sum / numVertices;

      let variance = 0;
      for (let v = 0; v < numVertices; v++) {
        const diff = features[v * featureDim + d] - mean;
        variance += diff * diff;
      }
      const std = Math.sqrt(variance / numVertices) + 1e-8;

      for (let v = 0; v < numVertices; v++) {
        features[v * featureDim + d] = (features[v * featureDim + d] - mean) / std;
      }
    }
  }

  /**
   * ✅ ソースカメラ設定からカメラパラメータを構築
   * 修正版: Camera JSON の R_matrix/T_vector を優先使用
   */
  buildCameraParamsFromConfig(
    config: SourceCameraConfig, 
    featureMapWidth: number, 
    featureMapHeight: number
  ): CameraParams {
    const { position, target, fov } = config;
    const R_matrix = config.debug?.R_matrix;
    const T_vector = config.debug?.T_vector;

    let viewMatrix: Float32Array;

    if (R_matrix && T_vector) {
      console.log('[ImageEncoder] ✅ Using verified R_matrix/T_vector from Camera JSON');
      console.log('[ImageEncoder] R_matrix:', R_matrix);
      console.log('[ImageEncoder] T_vector (original):', T_vector);
      
      // ⚠️ T_vector の符号を反転（Camera JSON の座標系に依存）
      // 標準的な View Matrix: [R^T | -R^T * eye]
      // しかし T_vector が既に R @ eye の場合、符号反転が必要
      const T_corrected = [
        -T_vector[0],
        -T_vector[1],
        -T_vector[2]
      ];
      
      console.log('[ImageEncoder] T_vector (corrected):', T_corrected);
      
      // Row-major (JSON) → Column-major (WebGL) 変換
      viewMatrix = new Float32Array([
        R_matrix[0][0], R_matrix[1][0], R_matrix[2][0], 0,
        R_matrix[0][1], R_matrix[1][1], R_matrix[2][1], 0,
        R_matrix[0][2], R_matrix[1][2], R_matrix[2][2], 0,
        T_corrected[0],  T_corrected[1],  T_corrected[2],  1
      ]);

      console.log('[ImageEncoder] View Matrix (column-major):', Array.from(viewMatrix));
    } else {
      // Fallback: position/target から計算
      console.warn('[ImageEncoder] ⚠️ R_matrix not found, computing from position/target');
      viewMatrix = this.computeLookAtMatrix(position, target);
    }

    // Projection Matrix (WebGL column-major形式)
    // Standard OpenGL perspective projection
    const fovRad = fov * Math.PI / 180;
    const aspect = featureMapWidth / featureMapHeight;
    const f = 1 / Math.tan(fovRad / 2);
    const near = 0.01;
    const far = 100;

    // Column-major storage:
    // Column 0: [f/aspect, 0, 0, 0]
    // Column 1: [0, f, 0, 0]
    // Column 2: [0, 0, (far+near)/(near-far), -1]
    // Column 3: [0, 0, (2*far*near)/(near-far), 0]
    const projMatrix = new Float32Array([
      f / aspect, 0,  0,  0,
      0,          f,  0,  0,
      0,          0,  (far + near) / (near - far),  -1,
      0,          0,  (2 * far * near) / (near - far),  0
    ]);

    console.log('[ImageEncoder] Camera parameters:', {
      position: Array.from(position),
      target: Array.from(target),
      fov,
      featureMapSize: `${featureMapWidth}×${featureMapHeight}`,
      near,
      far
    });

    return {
      position,
      target,
      fov,
      aspect,
      near,
      far,
      width: featureMapWidth,
      height: featureMapHeight,
      viewMatrix,
      projMatrix,
      screenWidth: featureMapWidth,
      screenHeight: featureMapHeight
    };
  }

  /**
   * 標準的な lookAt による View Matrix 計算 (Fallback)
   */
  private computeLookAtMatrix(
    position: [number, number, number],
    target: [number, number, number]
  ): Float32Array {
    // Z軸: eye から target への逆方向 (カメラは -Z を見る)
    let zx = position[0] - target[0];
    let zy = position[1] - target[1];
    let zz = position[2] - target[2];
    const zlen = Math.sqrt(zx * zx + zy * zy + zz * zz);
    zx /= zlen; 
    zy /= zlen; 
    zz /= zlen;

    // X軸: worldUp × Z (右方向)
    const upX = 0, upY = 1, upZ = 0;  // Y-up
    let xx = upY * zz - upZ * zy;
    let xy = upZ * zx - upX * zz;
    let xz = upX * zy - upY * zx;
    const xlen = Math.sqrt(xx * xx + xy * xy + xz * xz);
    
    if (xlen < 1e-6) {
      // カメラが真上または真下を向いている場合の処理
      console.warn('[ImageEncoder] ⚠️ Camera is looking straight up/down, using fallback X-axis');
      xx = 1; xy = 0; xz = 0;
    } else {
      xx /= xlen; 
      xy /= xlen; 
      xz /= xlen;
    }

    // Y軸: Z × X (上方向)
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    // Translation: -R^T * eye
    const tx = -(xx * position[0] + xy * position[1] + xz * position[2]);
    const ty = -(yx * position[0] + yy * position[1] + yz * position[2]);
    const tz = -(zx * position[0] + zy * position[1] + zz * position[2]);

    // View Matrix (OpenGL/WebGL column-major)
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      tx, ty, tz, 1
    ]);
  }

  /**
   * 📖 論文準拠: UV branch の features を取得 (32ch)
   */
  getUVFeatures(): Float32Array {
    if (!this.uvFeatures) {
      throw new Error('[ImageEncoder] UV features not available. Call extractFeaturesWithSourceCamera() first.');
    }
    return this.uvFeatures;
  }

  /**
   * 📖 論文準拠: Template branch の features を取得 (128ch)
   */
  getTemplateFeatures(): Float32Array {
    if (!this.templateFeatures) {
      throw new Error('[ImageEncoder] Template features not available. Call extractFeaturesWithSourceCamera() first.');
    }
    return this.templateFeatures;
  }

  /**
   * 互換性のため: appearance feature map を取得 (128ch)
   * @deprecated Use getTemplateFeatures() instead
   */
  getAppearanceFeatureMap(): Float32Array {
    return this.getTemplateFeatures();
  }

  /**
   * リソースを解放
   */
  dispose(): void {
    if (this.dinov2Session) {
      this.dinov2Session.release();
      this.dinov2Session = null;
    }
    if (this.encoderSession) {
      this.encoderSession.release();
      this.encoderSession = null;
    }
    this.initialized = false;
    console.log('[ImageEncoder] Disposed');
  }
}
