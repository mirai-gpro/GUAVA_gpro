// image-encoder.ts
// DINOv2 + DINO Encoder 完全ONNX版
// 技術仕様書 Section 3.1: 518×518入力 → 37×37パッチ(1369パッチ)
// 修正版: 配列形式のposition/target対応

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
}

export class ImageEncoder {
  private dinov2Session: ort.InferenceSession | null = null;
  private encoderSession: ort.InferenceSession | null = null;
  private initialized = false;
  
  // ? 論文準拠: 2つのブランチを保存
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

      console.log('[ImageEncoder] ?? DINOv2 input names:', this.dinov2Session.inputNames);
      console.log('[ImageEncoder] ?? DINOv2 output names:', this.dinov2Session.outputNames);

      // 2. DINO Encoder ONNXモデルをロード(37×37 → 518×518)
      console.log('[ImageEncoder] Loading DINO Encoder ONNX...');
      this.encoderSession = await ort.InferenceSession.create('/assets/dino_encoder.onnx');
      console.log('[ImageEncoder] ?? Encoder input names:', this.encoderSession.inputNames);
      console.log('[ImageEncoder] ?? Encoder output names:', this.encoderSession.outputNames);

      this.initialized = true;
      console.log('[ImageEncoder] ? Initialized with 37×37 patch support');
    } catch (error) {
      console.error('[ImageEncoder] ? Failed to initialize:', error);
      throw error;
    }
  } // ← この閉じ括弧が抜けていました

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
        console.error(`[ImageEncoder] ? Expected 1369 patches, got ${numPatches}`);
        throw new Error(`Invalid patch count: ${numPatches}`);
      }

      console.log('[ImageEncoder] ? DINOv2 output: 37×37 patches confirmed');

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
        console.warn(`[ImageEncoder] ?? Expected 518×518, got ${appearanceWidth}×${appearanceHeight}`);
      } else {
        console.log('[ImageEncoder] ? Appearance feature map: 518×518 confirmed');
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
      console.log(`[ImageEncoder] ? Feature extraction completed in ${elapsed.toFixed(2)}ms`);

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


      // ? 論文準拠: 2つのブランチに分離
      // Appendix B.2: "transform its dimensions to 32 and 128"
      const appearanceChannels = appearanceTensor.dims[1] as number;
      
      console.log('[ImageEncoder] ?? 論文準拠: Feature branches separation');
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
        
        console.log('[ImageEncoder] ? Separated: 128ch (template) + 32ch (UV)');
        
      } else if (appearanceChannels === 128) {
        console.warn('[ImageEncoder] ?? Encoder outputs 128ch only, using subset for UV (32ch)');
        
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
        
        console.log('[ImageEncoder] ? Separated: 128ch (template) + 32ch subset (UV)');
        
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
      console.error('[ImageEncoder] ? Feature extraction failed:', error);
      throw error;
    }
  }

  /**
   * 画像から特徴抽出(CameraParams直接指定版)
   * ?? FIX: 渡されたカメラパラメータを使用 (ハードコード値を削除)
   */
  async extractFeatures(
    imageUrl: string,
    vertices: Float32Array,
    vertexCount: number,
    camera: CameraParams,
    featureDim: number = 128
  ): Promise<{ projectionFeature: Float32Array; idEmbedding: Float32Array }> {
    // ? 渡されたカメラパラメータを使用 (論文準拠)
    const cameraConfig: SourceCameraConfig = {
      position: camera.position,
      target: camera.target,
      fov: camera.fov,
      imageWidth: camera.width,
      imageHeight: camera.height
    };

    console.log('[ImageEncoder] ?? Using actual camera parameters:', {
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
   * 3D頂点を2Dスクリーン座標に投影
   */
  private projectVertex(
    vx: number, vy: number, vz: number,
    viewMatrix: Float32Array,
    projMatrix: Float32Array,
    screenWidth: number,
    screenHeight: number
  ): [number, number, number, number] {
    // View transform (column-major)
    const viewX = viewMatrix[0] * vx + viewMatrix[4] * vy + viewMatrix[8] * vz + viewMatrix[12];
    const viewY = viewMatrix[1] * vx + viewMatrix[5] * vy + viewMatrix[9] * vz + viewMatrix[13];
    const viewZ = viewMatrix[2] * vx + viewMatrix[6] * vy + viewMatrix[10] * vz + viewMatrix[14];
    const viewW = viewMatrix[3] * vx + viewMatrix[7] * vy + viewMatrix[11] * vz + viewMatrix[15];

    // Projection transform (column-major)
    const clipX = projMatrix[0] * viewX + projMatrix[4] * viewY + projMatrix[8] * viewZ + projMatrix[12] * viewW;
    const clipY = projMatrix[1] * viewX + projMatrix[5] * viewY + projMatrix[9] * viewZ + projMatrix[13] * viewW;
    const clipZ = projMatrix[2] * viewX + projMatrix[6] * viewY + projMatrix[10] * viewZ + projMatrix[14] * viewW;
    const clipW = projMatrix[3] * viewX + projMatrix[7] * viewY + projMatrix[11] * viewZ + projMatrix[15] * viewW;

    // Perspective division
    const safeW = Math.abs(clipW) > 1e-6 ? clipW : 1e-6;
    const ndcX = clipX / safeW;
    const ndcY = clipY / safeW;
    const depth = clipZ / safeW;

    // NDC → Screen
    const screenX = (ndcX * 0.5 + 0.5) * screenWidth;
    const screenY = (1.0 - (ndcY * 0.5 + 0.5)) * screenHeight;

    return [screenX, screenY, depth, clipW];
  }

  /**
   * Feature mapから2D位置でバイリニアサンプリング
   * ONNX出力はCHW形式: [1, featureDim, height, width]
   * インデックス計算: d * H * W + y * W + x
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
      console.warn('[ImageEncoder] ?? No visible vertices! Check camera parameters.');
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
   * ? ソースカメラ設定からカメラパラメータを構築(配列形式対応)
   * GUAVA論文: ソース画像撮影時のカメラパラメータを使用
   * 修正版: WebGL column-major形式に準拠
   */
  buildCameraParamsFromConfig(
    config: SourceCameraConfig, 
    featureMapWidth: number, 
    featureMapHeight: number
  ): CameraParams {
    const { position, target, fov } = config;

    // ? 配列形式で方向ベクトルを計算
    const dx = target[0] - position[0];
    const dy = target[1] - position[1];
    const dz = target[2] - position[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // 前方向ベクトル (カメラが見る方向)
    const fx = dx / len;
    const fy = dy / len;
    const fz = dz / len;

    // 上方向ベクトル (仮定: Y軸が上)
    let ux = 0, uy = 1, uz = 0;

    // 右方向ベクトル = forward × up
    let rx = fy * uz - fz * uy;
    let ry = fz * ux - fx * uz;
    let rz = fx * uy - fy * ux;
    const rlen = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    // 真の上方向ベクトル = right × forward
    ux = ry * fz - rz * fy;
    uy = rz * fx - rx * fz;
    uz = rx * fy - ry * fx;

    // 平行移動成分を計算(配列形式対応)
    const tx = -(rx * position[0] + ry * position[1] + rz * position[2]);
    const ty = -(ux * position[0] + uy * position[1] + uz * position[2]);
    const tz = (fx * position[0] + fy * position[1] + fz * position[2]);

    // View Matrix (WebGL column-major形式)
    // 列ベクトルの順: right, up, -forward, translation
    const viewMatrix = new Float32Array([
      rx,  ry,  rz,  0,
      ux,  uy,  uz,  0,
      -fx, -fy, -fz, 0,
      tx,  ty,  tz,  1
    ]);

    // Projection Matrix (WebGL column-major形式)
    const fovRad = fov * Math.PI / 180;
    const aspect = featureMapWidth / featureMapHeight;
    const f = 1 / Math.tan(fovRad / 2);
    const near = 0.01;
    const far = 100;

    const projMatrix = new Float32Array([
      f / aspect, 0,  0,   0,
      0,          f,  0,   0,
      0,          0,  (far + near) / (near - far),  (2 * far * near) / (near - far),
      0,          0,  -1,  0
    ]);

    console.log('[ImageEncoder] Built camera params from config:', {
      position: [position[0], position[1], position[2]],
      target: [target[0], target[1], target[2]],
      fov,
      featureMapSize: `${featureMapWidth}×${featureMapHeight}`
    });

    return {
      position,
      target,
      fov,
      aspect,
      near: 0.01,
      far: 100,
      width: featureMapWidth,
      height: featureMapHeight,
      viewMatrix,
      projMatrix,
      screenWidth: featureMapWidth,
      screenHeight: featureMapHeight
    };
  }

  /**
   * ? 論文準拠: UV branch の features を取得 (32ch)
   */
  getUVFeatures(): Float32Array {
    if (!this.uvFeatures) {
      throw new Error('[ImageEncoder] UV features not available. Call extractFeaturesWithSourceCamera() first.');
    }
    return this.uvFeatures;
  }

  /**
   * ? 論文準拠: Template branch の features を取得 (128ch)
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