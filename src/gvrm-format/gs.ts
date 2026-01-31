// src/gvrm-format/gs.ts
// Gaussian Splatting Renderer - GUAVA論文準拠
// Template Decoderの出力 (latent32ch, opacity, scale, rotation) を使用

import * as THREE from 'three';

interface GaussianData {
  positions: Float32Array;      // [N, 3]
  latents: Float32Array;        // [N, 32]
  opacity: Float32Array;        // [N, 1]
  scale: Float32Array;          // [N, 3]
  rotation: Float32Array;       // [N, 4] quaternion
  boneIndices: Float32Array;
  boneWeights: Float32Array;
  vertexCount: number;
}

const vertexShader = `
  attribute vec4 latentTile;    // 4ch単位の特徴量
  attribute float opacity;       // Gaussian不透明度
  attribute vec3 gaussianScale;  // Gaussianスケール
  attribute vec4 boneIndices, boneWeights;

  uniform mat4 boneMatrices[64];
  uniform float basePointSize;

  varying vec4 vFeature;
  varying float vOpacity;

  void main() {
    // スキニング
    mat4 skinMatrix = boneWeights.x * boneMatrices[int(boneIndices.x)] +
                     boneWeights.y * boneMatrices[int(boneIndices.y)] +
                     boneWeights.z * boneMatrices[int(boneIndices.z)] +
                     boneWeights.w * boneMatrices[int(boneIndices.w)];

    vec4 posedPos = skinMatrix * vec4(position, 1.0);
    vec4 mvPosition = modelViewMatrix * posedPos;
    gl_Position = projectionMatrix * mvPosition;

    // ポイントサイズ: Gaussianスケールと距離に基づく
    // スケールの平均を使用（異方性スケールの近似）
    float avgScale = (gaussianScale.x + gaussianScale.y + gaussianScale.z) / 3.0;
    // スケール値を適切な範囲にマップ（学習済みスケールは通常-5〜2程度）
    float scaleFactor = exp(clamp(avgScale, -5.0, 2.0));

    // 距離に応じたサイズ調整
    float depth = -mvPosition.z;
    gl_PointSize = basePointSize * scaleFactor * (300.0 / max(depth, 0.1));

    // 最小/最大サイズ制限
    gl_PointSize = clamp(gl_PointSize, 1.0, 64.0);

    vFeature = latentTile;
    vOpacity = opacity;
  }
`;

const fragmentShader = `
  varying vec4 vFeature;
  varying float vOpacity;

  void main() {
    // 円形ガウシアンプロファイル
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);

    // Gaussian falloff: exp(-dist^2 / sigma^2)
    // sigma = 0.25 で中心から端まで滑らかに減衰
    float sigma = 0.25;
    float gaussian = exp(-dist * dist / (2.0 * sigma * sigma));

    // 端で完全に透明
    if (dist > 0.5) discard;

    // 不透明度を適用（sigmoid活性化されたopacity）
    // opacity値は学習済みなので、sigmoidで[0,1]に変換
    float alpha = 1.0 / (1.0 + exp(-vOpacity));
    alpha *= gaussian;

    // α < 0.01 はスキップ
    if (alpha < 0.01) discard;

    // 特徴量にαを掛けて出力（alpha blending準備）
    gl_FragColor = vec4(vFeature.rgb * alpha, alpha);
  }
`;

// Alpha blending用シェーダー（accumulation）
const accumulateFragShader = `
  varying vec4 vFeature;
  varying float vOpacity;

  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);

    float sigma = 0.25;
    float gaussian = exp(-dist * dist / (2.0 * sigma * sigma));

    if (dist > 0.5) discard;

    float alpha = 1.0 / (1.0 + exp(-vOpacity));
    alpha *= gaussian;

    if (alpha < 0.01) discard;

    // RGBA: 特徴量 * alpha, alpha
    gl_FragColor = vec4(vFeature.rgb * alpha, vFeature.a * alpha);
  }
`;

export class GSViewer {
  public mesh: THREE.Points;
  private geometry: THREE.BufferGeometry;
  private latentData: Float32Array;
  private opacityData: Float32Array;
  private scaleData: Float32Array;
  private rotationData: Float32Array;
  private vertexCount: number;

  constructor(data: GaussianData) {
    this.vertexCount = data.vertexCount;
    this.latentData = data.latents;
    this.opacityData = data.opacity || new Float32Array(data.vertexCount).fill(0); // sigmoid(0) = 0.5
    this.scaleData = data.scale || new Float32Array(data.vertexCount * 3).fill(0);
    this.rotationData = data.rotation || new Float32Array(data.vertexCount * 4);

    console.log('[GSViewer] Initializing Gaussian Splatting...', {
      vertexCount: this.vertexCount,
      latentsLength: this.latentData.length,
      hasOpacity: !!data.opacity,
      hasScale: !!data.scale,
      hasRotation: !!data.rotation
    });

    // 統計情報
    this.logAttributeStats('opacity', this.opacityData, 1);
    this.logAttributeStats('scale', this.scaleData, 3);

    // Latent features statistics (32 channels)
    this.logLatentStats();

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    this.geometry.setAttribute('boneIndices', new THREE.BufferAttribute(data.boneIndices, 4));
    this.geometry.setAttribute('boneWeights', new THREE.BufferAttribute(data.boneWeights, 4));

    // Gaussian属性を設定
    this.geometry.setAttribute('opacity', new THREE.BufferAttribute(this.opacityData, 1));
    this.geometry.setAttribute('gaussianScale', new THREE.BufferAttribute(this.scaleData, 3));

    // 初期状態として最初の4chをセット
    this.updateLatentTile(0);

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        boneMatrices: { value: new Float32Array(16 * 64) },
        basePointSize: { value: 15.0 }
      },
      depthTest: true,
      depthWrite: false,  // Alpha blending用
      transparent: true,
      blending: THREE.NormalBlending  // ✅ AdditiveBlending → NormalBlending に変更
    });

    this.mesh = new THREE.Points(this.geometry, material);
    this.mesh.frustumCulled = false;

    console.log('[GSViewer] ✅ Gaussian Splatting initialized');
  }

  private logAttributeStats(name: string, data: Float32Array, stride: number): void {
    if (!data || data.length === 0) {
      console.log(`[GSViewer] ${name}: empty`);
      return;
    }

    let min = Infinity, max = -Infinity, sum = 0, nanCount = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isNaN(v)) { nanCount++; continue; }
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const mean = sum / (data.length - nanCount);

    console.log(`[GSViewer] ${name} stats:`, {
      count: data.length / stride,
      min: min.toFixed(4),
      max: max.toFixed(4),
      mean: mean.toFixed(4),
      nanCount
    });
  }

  private logLatentStats(): void {
    if (!this.latentData || this.latentData.length === 0) {
      console.log('[GSViewer] latents: empty');
      return;
    }

    // Per-channel statistics
    const channelStats: { ch: number; min: number; max: number; mean: number }[] = [];

    for (let c = 0; c < 32; c++) {
      let min = Infinity, max = -Infinity, sum = 0;
      for (let i = 0; i < this.vertexCount; i++) {
        const v = this.latentData[i * 32 + c];
        if (isNaN(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
      }
      const mean = sum / this.vertexCount;
      channelStats.push({ ch: c, min, max, mean });
    }

    // Log first 3 channels (RGB) separately
    console.log('[GSViewer] Latent RGB channels (ch0-2, should be sigmoid activated):');
    for (let c = 0; c < 3; c++) {
      const s = channelStats[c];
      console.log(`  Ch${c} (${['R', 'G', 'B'][c]}): [${s.min.toFixed(4)}, ${s.max.toFixed(4)}], mean=${s.mean.toFixed(4)}`);
    }

    // Overall latent stats
    let overallMin = Infinity, overallMax = -Infinity, overallSum = 0;
    for (const s of channelStats) {
      if (s.min < overallMin) overallMin = s.min;
      if (s.max > overallMax) overallMax = s.max;
      overallSum += s.mean;
    }

    console.log('[GSViewer] Latent overall:', {
      channels: 32,
      vertices: this.vertexCount,
      minAcrossAll: overallMin.toFixed(4),
      maxAcrossAll: overallMax.toFixed(4),
      avgChannelMean: (overallSum / 32).toFixed(4)
    });
  }

  public updateLatentTile(tileIndex: number) {
    // 32chの中からi番目の4chセット(RGBA)を抽出
    // Tile 0: ch0-3, Tile 1: ch4-7, ..., Tile 7: ch28-31

    if (tileIndex < 0 || tileIndex >= 8) {
      console.error(`[GSViewer] Invalid tileIndex: ${tileIndex}, must be 0-7`);
      return;
    }

    const tile = new Float32Array(this.vertexCount * 4);
    const startCh = tileIndex * 4;

    let minVal = Infinity, maxVal = -Infinity;
    let nanCount = 0, zeroCount = 0;

    for (let i = 0; i < this.vertexCount; i++) {
      const baseIdx = i * 32;

      for (let c = 0; c < 4; c++) {
        const srcIdx = baseIdx + startCh + c;
        const dstIdx = i * 4 + c;

        if (srcIdx >= this.latentData.length) {
          tile[dstIdx] = 0;
          continue;
        }

        let value = this.latentData[srcIdx];

        if (isNaN(value)) {
          nanCount++;
          value = 0;
        } else if (!isFinite(value)) {
          value = 0;
        }

        if (value === 0) zeroCount++;

        tile[dstIdx] = value;

        if (isFinite(value)) {
          minVal = Math.min(minVal, value);
          maxVal = Math.max(maxVal, value);
        }
      }
    }

    const totalValues = this.vertexCount * 4;

    // 最初のタイルのみ詳細ログ
    if (tileIndex === 0) {
      console.log(`[GSViewer] Tile ${tileIndex}:`, {
        nonZeros: totalValues - zeroCount,
        min: minVal === Infinity ? 0 : minVal.toFixed(4),
        max: maxVal === -Infinity ? 0 : maxVal.toFixed(4),
        nanCount
      });
    }

    this.geometry.setAttribute('latentTile', new THREE.BufferAttribute(tile, 4));
  }

  public updateBones(matrices: Float32Array) {
    (this.mesh.material as THREE.ShaderMaterial).uniforms.boneMatrices.value.set(matrices);
  }

  public setPointSize(size: number) {
    (this.mesh.material as THREE.ShaderMaterial).uniforms.basePointSize.value = size;
  }

  /**
   * Render 32ch coarse feature map
   * 論文 Section 3.3: "splatting → coarse feature map (32ch) → refiner → RGB"
   *
   * @param width Output width (default 512)
   * @param height Output height (default 512)
   * @param camera Camera for projection (optional, uses default if not provided)
   * @returns Float32Array [32, height, width] in CHW format
   */
  public render(width: number = 512, height: number = 512, camera?: THREE.Camera): Float32Array {
    console.log('[GSViewer] Rendering 32ch coarse feature map...');
    console.log('[GSViewer]   vertexCount:', this.vertexCount);
    console.log('[GSViewer]   latentData length:', this.latentData.length);

    const featureMap = new Float32Array(32 * height * width);
    const weightMap = new Float32Array(height * width);  // For normalization

    // Default camera if not provided
    const projMatrix = camera
      ? new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      : this.createDefaultProjection(width, height);

    console.log('[GSViewer]   Camera provided:', !!camera);
    if (camera) {
      console.log('[GSViewer]   Camera position:', camera.position.toArray());
    }

    // Get position attribute
    const posAttr = this.geometry.getAttribute('position') as THREE.BufferAttribute;
    const positions = posAttr.array as Float32Array;

    // Compute depths for sorting
    const gaussianDepths: { index: number; depth: number }[] = [];
    const tempVec = new THREE.Vector4();

    for (let i = 0; i < this.vertexCount; i++) {
      tempVec.set(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2],
        1.0
      );
      tempVec.applyMatrix4(projMatrix);
      const depth = tempVec.z / tempVec.w;
      gaussianDepths.push({ index: i, depth });
    }

    // Sort by depth (front to back for transmittance)
    gaussianDepths.sort((a, b) => a.depth - b.depth);

    // Debug: depth distribution
    let depthSkipped = 0, wSkipped = 0, outOfBounds = 0;
    const depthStats = { min: Infinity, max: -Infinity };
    for (const { depth } of gaussianDepths) {
      if (depth < depthStats.min) depthStats.min = depth;
      if (depth > depthStats.max) depthStats.max = depth;
    }
    console.log('[GSViewer]   Depth range:', depthStats.min.toFixed(3), 'to', depthStats.max.toFixed(3));

    // Screen coordinate tracking
    const screenStats = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };
    const ndcStats = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity };

    // Transmittance buffer (per pixel)
    const transmittance = new Float32Array(height * width).fill(1.0);

    let splatCount = 0;

    // Splat each Gaussian
    for (const { index: i, depth } of gaussianDepths) {
      // Skip if behind camera
      if (depth < -1 || depth > 1) {
        depthSkipped++;
        continue;
      }

      // Project to screen space
      tempVec.set(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2],
        1.0
      );
      tempVec.applyMatrix4(projMatrix);

      if (tempVec.w <= 0) {
        wSkipped++;
        continue;
      }

      const ndcX = tempVec.x / tempVec.w;
      const ndcY = tempVec.y / tempVec.w;

      // Track NDC stats
      if (ndcX < ndcStats.minX) ndcStats.minX = ndcX;
      if (ndcX > ndcStats.maxX) ndcStats.maxX = ndcX;
      if (ndcY < ndcStats.minY) ndcStats.minY = ndcY;
      if (ndcY > ndcStats.maxY) ndcStats.maxY = ndcY;

      // NDC to pixel coordinates
      const px = Math.floor((ndcX * 0.5 + 0.5) * width);
      const py = Math.floor((1.0 - (ndcY * 0.5 + 0.5)) * height);  // Y-flip

      // Track screen stats
      if (px < screenStats.minX) screenStats.minX = px;
      if (px > screenStats.maxX) screenStats.maxX = px;
      if (py < screenStats.minY) screenStats.minY = py;
      if (py > screenStats.maxY) screenStats.maxY = py;

      // Track out of bounds
      if (px < 0 || px >= width || py < 0 || py >= height) {
        outOfBounds++;
      }

      // Get Gaussian attributes
      const opacity = this.opacityData[i];
      const alpha = 1.0 / (1.0 + Math.exp(-opacity));  // sigmoid

      // Get scale for splat radius
      const scaleX = this.scaleData[i * 3];
      const scaleY = this.scaleData[i * 3 + 1];
      const scaleZ = this.scaleData[i * 3 + 2];
      const avgScale = (Math.exp(scaleX) + Math.exp(scaleY) + Math.exp(scaleZ)) / 3;

      // Compute splat radius in pixels (simplified)
      const depthFactor = Math.max(0.1, 1.0 - depth * 0.5);
      const radius = Math.max(1, Math.min(16, avgScale * 50 * depthFactor));

      // Splat to neighboring pixels
      const r = Math.ceil(radius);
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const x = px + dx;
          const y = py + dy;

          if (x < 0 || x >= width || y < 0 || y >= height) continue;

          // Gaussian falloff
          const dist2 = (dx * dx + dy * dy) / (radius * radius);
          if (dist2 > 1.0) continue;

          const gaussian = Math.exp(-dist2 * 2.0);
          const pixelAlpha = alpha * gaussian;

          if (pixelAlpha < 0.001) continue;

          const pixelIdx = y * width + x;
          const T = transmittance[pixelIdx];

          if (T < 0.001) continue;  // Pixel fully occluded

          // Accumulate 32ch latent features
          // Formula: C += c_i * α_i * T_i
          const weight = pixelAlpha * T;
          for (let c = 0; c < 32; c++) {
            const latentValue = this.latentData[i * 32 + c];
            featureMap[c * height * width + pixelIdx] += latentValue * weight;
          }
          weightMap[pixelIdx] += weight;

          // Update transmittance: T_{i+1} = T_i * (1 - α_i)
          transmittance[pixelIdx] *= (1.0 - pixelAlpha);
        }
      }

      splatCount++;
    }

    // Normalize by accumulated weight (optional, for proper blending)
    for (let pixelIdx = 0; pixelIdx < height * width; pixelIdx++) {
      const w = weightMap[pixelIdx];
      if (w > 0.001) {
        for (let c = 0; c < 32; c++) {
          featureMap[c * height * width + pixelIdx] /= w;
        }
      }
    }

    // Calculate statistics
    let fmMin = Infinity, fmMax = -Infinity, fmSum = 0, fmNonZero = 0;
    for (let i = 0; i < featureMap.length; i++) {
      const v = featureMap[i];
      if (v < fmMin) fmMin = v;
      if (v > fmMax) fmMax = v;
      fmSum += v;
      if (v !== 0) fmNonZero++;
    }

    console.log('[GSViewer]   NDC range:', {
      x: `[${ndcStats.minX.toFixed(3)}, ${ndcStats.maxX.toFixed(3)}]`,
      y: `[${ndcStats.minY.toFixed(3)}, ${ndcStats.maxY.toFixed(3)}]`
    });
    console.log('[GSViewer]   Screen range:', {
      x: `[${screenStats.minX}, ${screenStats.maxX}]`,
      y: `[${screenStats.minY}, ${screenStats.maxY}]`,
      coverageX: ((screenStats.maxX - screenStats.minX) / width * 100).toFixed(1) + '%',
      coverageY: ((screenStats.maxY - screenStats.minY) / height * 100).toFixed(1) + '%'
    });

    console.log('[GSViewer] ✅ Coarse feature map rendered:', {
      resolution: `${width}×${height}`,
      channels: 32,
      splattedGaussians: splatCount,
      depthSkipped,
      wSkipped,
      outOfBounds,
      format: 'CHW',
      nonZeroRatio: (fmNonZero / featureMap.length * 100).toFixed(1) + '%',
      min: fmMin.toFixed(4),
      max: fmMax.toFixed(4),
      mean: (fmSum / featureMap.length).toFixed(6)
    });

    return featureMap;
  }

  private createDefaultProjection(width: number, height: number): THREE.Matrix4 {
    // Default camera: looking at origin from z=5
    const fov = 45;
    const aspect = width / height;
    const near = 0.1;
    const far = 100;

    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.set(0, 1.2, 5);
    camera.lookAt(0, 1.2, 0);
    camera.updateMatrixWorld();

    return new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
  }

  /**
   * Debug: Extract RGB channels from coarse feature map
   * 論文準拠: 最初の3チャンネルはRGB (sigmoid適用済み)
   */
  public extractRGBFromCoarseFeatures(featureMap: Float32Array, width: number, height: number): Float32Array {
    const rgb = new Float32Array(height * width * 3);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = y * width + x;
        const dstIdx = pixelIdx * 3;

        // CHW形式から取得: featureMap[c * H * W + y * W + x]
        rgb[dstIdx + 0] = featureMap[0 * height * width + pixelIdx];  // R
        rgb[dstIdx + 1] = featureMap[1 * height * width + pixelIdx];  // G
        rgb[dstIdx + 2] = featureMap[2 * height * width + pixelIdx];  // B
      }
    }

    // Clamp to [0, 1]
    for (let i = 0; i < rgb.length; i++) {
      rgb[i] = Math.max(0, Math.min(1, rgb[i]));
    }

    // Stats
    let rSum = 0, gSum = 0, bSum = 0;
    for (let i = 0; i < height * width; i++) {
      rSum += rgb[i * 3 + 0];
      gSum += rgb[i * 3 + 1];
      bSum += rgb[i * 3 + 2];
    }
    const pixelCount = height * width;
    console.log('[GSViewer] RGB channel means from coarse features:', {
      R: (rSum / pixelCount).toFixed(4),
      G: (gSum / pixelCount).toFixed(4),
      B: (bSum / pixelCount).toFixed(4)
    });

    return rgb;
  }

  public dispose() {
    this.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    console.log('[GSViewer] Disposed');
  }
}