// src/gvrm-format/gs.ts
// Gaussian Splatting Renderer - GUAVA論文準拠
// Template Decoderの出力 (latent32ch, opacity, scale, rotation) を使用
// 🔍 診断機能追加版

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
    // スケール値を適切な範囲にマップ（学習済みスケールは通常-5～2程度）
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

  uniform bool debugMode;  // 🔍 デバッグモード切り替え

  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);

    // 円の外側は破棄
    if (dist > 0.5) discard;

    // 🔍 デバッグモード: opacity値を赤色で可視化
    if (debugMode) {
      // opacity を [0,1] にマップ
      float normalizedOpacity = 1.0 / (1.0 + exp(-vOpacity));
      // Gaussian falloffを適用
      float sigma = 0.25;
      float gaussian = exp(-dist * dist / (2.0 * sigma * sigma));
      float alpha = normalizedOpacity * gaussian;
      
      // 赤色でopacity値を表示（明るさ = opacity）
      gl_FragColor = vec4(normalizedOpacity, 0.0, 0.0, max(alpha, 0.3));
      return;
    }

    // 通常モード: Gaussian falloff
    float sigma = 0.25;
    float gaussian = exp(-dist * dist / (2.0 * sigma * sigma));

    // 不透明度を適用（sigmoid活性化されたopacity）
    float alpha = 1.0 / (1.0 + exp(-vOpacity));
    alpha *= gaussian;

    // α < 0.01 はスキップ
    if (alpha < 0.01) discard;

    // 特徴量にαを掛けて出力（alpha blending準備）
    gl_FragColor = vec4(vFeature.rgb * alpha, alpha);
  }
`;

export class GSViewer {
  public mesh: THREE.Points;
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
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

    // 🔍 詳細な統計情報
    this.logAttributeStats('opacity', this.opacityData, 1);
    this.logAttributeStats('scale', this.scaleData, 3);
    this.logAttributeStats('latents (first tile)', this.latentData.slice(0, this.vertexCount * 4), 4);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    this.geometry.setAttribute('boneIndices', new THREE.BufferAttribute(data.boneIndices, 4));
    this.geometry.setAttribute('boneWeights', new THREE.BufferAttribute(data.boneWeights, 4));

    // Gaussian属性を設定
    this.geometry.setAttribute('opacity', new THREE.BufferAttribute(this.opacityData, 1));
    this.geometry.setAttribute('gaussianScale', new THREE.BufferAttribute(this.scaleData, 3));

    // 初期状態として最初の4chをセット
    this.updateLatentTile(0);

    // boneMatricesを初期化（ボーン0を単位行列に設定）
    const boneMatricesData = new Float32Array(16 * 64);
    // 単位行列: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] を全ボーンに設定
    for (let bone = 0; bone < 64; bone++) {
      const offset = bone * 16;
      boneMatricesData[offset + 0] = 1;  // m[0][0]
      boneMatricesData[offset + 5] = 1;  // m[1][1]
      boneMatricesData[offset + 10] = 1; // m[2][2]
      boneMatricesData[offset + 15] = 1; // m[3][3]
    }

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        boneMatrices: { value: boneMatricesData },
        basePointSize: { value: 15.0 },
        debugMode: { value: false }  // 🔍 デフォルトはOFF
      },
      depthTest: true,
      depthWrite: false,  // Alpha blending用
      transparent: true,
      blending: THREE.NormalBlending
    });

    this.mesh = new THREE.Points(this.geometry, this.material);
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
    let nanCount = 0, zeroCount = 0, nonZeroCount = 0;

    for (let i = 0; i < this.vertexCount; i++) {
      const baseIdx = i * 32;

      for (let c = 0; c < 4; c++) {
        const srcIdx = baseIdx + startCh + c;
        const dstIdx = i * 4 + c;

        if (srcIdx >= this.latentData.length) {
          tile[dstIdx] = 0;
          zeroCount++;
          continue;
        }

        let value = this.latentData[srcIdx];

        if (isNaN(value)) {
          nanCount++;
          value = 0;
        } else if (!isFinite(value)) {
          value = 0;
        }

        if (Math.abs(value) < 0.001) {
          zeroCount++;
        } else {
          nonZeroCount++;
        }

        tile[dstIdx] = value;

        if (isFinite(value)) {
          minVal = Math.min(minVal, value);
          maxVal = Math.max(maxVal, value);
        }
      }
    }

    const totalValues = this.vertexCount * 4;

    // 🔍 最初のタイルのみ詳細ログ
    if (tileIndex === 0) {
      console.log(`[GSViewer] Tile ${tileIndex}:`, {
        nonZeros: nonZeroCount,
        min: minVal === Infinity ? '0.0000' : minVal.toFixed(4),
        max: maxVal === -Infinity ? '0.0000' : maxVal.toFixed(4),
        nanCount,
        nonZeroRatio: ((nonZeroCount / totalValues) * 100).toFixed(1) + '%'
      });
    }

    this.geometry.setAttribute('latentTile', new THREE.BufferAttribute(tile, 4));
  }

  public updateBones(matrices: Float32Array) {
    this.material.uniforms.boneMatrices.value.set(matrices);
  }

  public setPointSize(size: number) {
    this.material.uniforms.basePointSize.value = size;
    console.log(`[GSViewer] Point size set to: ${size}`);
  }

  /**
   * 🔍 デバッグモード切り替え
   * @param enabled true = opacity値を赤色で可視化, false = 通常レンダリング
   */
  public setDebugMode(enabled: boolean) {
    this.material.uniforms.debugMode.value = enabled;
    console.log(`[GSViewer] Debug mode: ${enabled ? 'ON (opacity visualization)' : 'OFF (normal rendering)'}`);
  }

  /**
   * 🔍 診断情報を出力
   */
  public diagnose(): void {
    console.log('[GSViewer] === Diagnostic Information ===');
    
    // Geometry attributes
    const attrs = this.geometry.attributes;
    console.log('Geometry attributes:', Object.keys(attrs));
    
    // Position check
    const pos = attrs.position.array as Float32Array;
    console.log('Position:', {
      count: pos.length / 3,
      first: [pos[0].toFixed(4), pos[1].toFixed(4), pos[2].toFixed(4)],
      min: [
        Math.min(...Array.from(pos).filter((_, i) => i % 3 === 0)).toFixed(4),
        Math.min(...Array.from(pos).filter((_, i) => i % 3 === 1)).toFixed(4),
        Math.min(...Array.from(pos).filter((_, i) => i % 3 === 2)).toFixed(4)
      ],
      max: [
        Math.max(...Array.from(pos).filter((_, i) => i % 3 === 0)).toFixed(4),
        Math.max(...Array.from(pos).filter((_, i) => i % 3 === 1)).toFixed(4),
        Math.max(...Array.from(pos).filter((_, i) => i % 3 === 2)).toFixed(4)
      ]
    });
    
    // Opacity check
    const opacity = attrs.opacity.array as Float32Array;
    const opacitySample = Array.from(opacity.slice(0, 100));
    console.log('Opacity (sigmoid activated):', {
      sample: opacitySample.slice(0, 10).map(v => v.toFixed(4)),
      linearValues: opacitySample.slice(0, 10).map(v => (1 / (1 + Math.exp(-v))).toFixed(4))
    });
    
    // Scale check
    const scale = attrs.gaussianScale.array as Float32Array;
    const scaleSample = Array.from(scale.slice(0, 30)); // 10 vertices × 3
    console.log('Scale (log space):', {
      sample: scaleSample.slice(0, 9).map(v => v.toFixed(4)),
      linearValues: scaleSample.slice(0, 9).map(v => Math.exp(v).toFixed(4))
    });
    
    // LatentTile check
    const latentTile = attrs.latentTile.array as Float32Array;
    const latentSample = Array.from(latentTile.slice(0, 40)); // 10 vertices × 4
    console.log('LatentTile (current):', {
      sample: latentSample.slice(0, 12).map(v => v.toFixed(4))
    });
    
    console.log('[GSViewer] ============================');
  }

  public dispose() {
    this.geometry.dispose();
    this.material.dispose();
    console.log('[GSViewer] Disposed');
  }
}