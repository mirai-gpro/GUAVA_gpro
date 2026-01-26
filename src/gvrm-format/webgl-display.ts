// webgl-display.ts
// Neural Refiner出力をWebGLで直接表示
// 技術仕様書準拠: Canvas 2D排除、gamma/sRGBはシェーダー内で制御

import * as THREE from 'three';

export class WebGLDisplay {
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;
  private renderer: THREE.WebGLRenderer;
  private texture: THREE.DataTexture;
  private quad: THREE.Mesh;
  private shaderMaterial: THREE.ShaderMaterial;

  constructor(container: HTMLElement, width: number = 512, height: number = 512) {
    console.log('[WebGLDisplay] Initializing (WebGL unified rendering)...');

    // シーン setup
    this.scene = new THREE.Scene();

    // 正投影カメラ（2D描画用）
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // WebGLレンダラー（Canvas 2D不使用）
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true // WebGL内でのデータ保持
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // スタイル設定
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.renderer.domElement.style.position = 'absolute';
    this.renderer.domElement.style.top = '0';
    this.renderer.domElement.style.left = '0';
    this.renderer.domElement.style.zIndex = '10';
    this.renderer.domElement.style.objectFit = 'contain';
    container.appendChild(this.renderer.domElement);

    // DataTexture作成（Float32で精度を維持）
    const emptyData = new Float32Array(width * height * 4);
    this.texture = new THREE.DataTexture(
      emptyData,
      width,
      height,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;

    // フルスクリーンクアッド + シェーダー
    const geometry = new THREE.PlaneGeometry(2, 2);
    this.shaderMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: this.texture },
        uGamma: { value: 2.2 },        // sRGBガンマ値
        uExposure: { value: 1.0 },     // 露出調整
        uContrast: { value: 1.0 }      // コントラスト
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform float uGamma;
        uniform float uExposure;
        uniform float uContrast;
        varying vec2 vUv;

        // リニア → sRGB変換（シェーダー内で制御）
        vec3 linearToSRGB(vec3 linear) {
          vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
          vec3 lower = linear * vec3(12.92);
          return mix(lower, higher, step(vec3(0.0031308), linear));
        }

        // トーンマッピング（ACES近似）
        vec3 ACESFilm(vec3 x) {
          float a = 2.51;
          float b = 0.03;
          float c = 2.43;
          float d = 0.59;
          float e = 0.14;
          return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }

        void main() {
          // Y軸反転（テクスチャ座標系）
          vec2 uv = vec2(vUv.x, 1.0 - vUv.y);
          vec4 texColor = texture2D(tDiffuse, uv);
          vec3 color = texColor.rgb;

          // 露出調整
          color = color * uExposure;

          // コントラスト調整（中心0.5基準）
          color = (color - 0.5) * uContrast + 0.5;

          // トーンマッピング（HDR→LDR）
          color = ACESFilm(color);

          // リニア → sRGB変換
          color = linearToSRGB(color);

          // 最終クランプ
          color = clamp(color, 0.0, 1.0);

          gl_FragColor = vec4(color, 1.0);
        }
      `,
      depthTest: false,
      depthWrite: false
    });

    this.quad = new THREE.Mesh(geometry, this.shaderMaterial);
    this.scene.add(this.quad);

    console.log('[WebGLDisplay] ✅ Initialized (WebGL unified, no Canvas 2D)');
  }

  /**
   * Neural Refiner出力を表示
   * @param data HWC形式のRGBデータ [H*W*3]
   * @param frameCount フレーム番号（デバッグ用）
   */
  public display(data: Float32Array, frameCount: number = 0): void {
    const width = 512;
    const height = 512;
    const expectedLength = width * height * 3;

    if (data.length !== expectedLength) {
      console.error(`[WebGLDisplay] Invalid data length: ${data.length}, expected: ${expectedLength}`);
      return;
    }

    // v80: GLOBAL min/max を計算（全RGB共通）
    // これにより色差を保持しながらコントラストを改善
    let globalMin = Infinity, globalMax = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < globalMin) globalMin = v;
      if (v > globalMax) globalMax = v;
    }
    const globalRange = globalMax - globalMin || 1;

    // HWC → RGBA変換 + GLOBAL contrast stretch
    // 重要: 全RGB同じ係数でストレッチするため、色差が保持される
    const pixels = new Float32Array(width * height * 4);

    for (let i = 0; i < width * height; i++) {
      const srcIdx = i * 3;
      // GLOBAL stretch: [globalMin, globalMax] → [0, 1]
      // 全チャンネル同じmin/maxを使うので色の比率が保持される
      pixels[i * 4 + 0] = (data[srcIdx + 0] - globalMin) / globalRange;
      pixels[i * 4 + 1] = (data[srcIdx + 1] - globalMin) / globalRange;
      pixels[i * 4 + 2] = (data[srcIdx + 2] - globalMin) / globalRange;
      pixels[i * 4 + 3] = 1.0;  // A
    }

    // 初回フレームのみ統計情報を出力
    if (frameCount === 1) {
      console.log('[WebGLDisplay] First frame stats:', {
        originalMin: globalMin.toFixed(4),
        originalMax: globalMax.toFixed(4),
        range: globalRange.toFixed(4)
      });
      console.log('[WebGLDisplay] 🔧 v80: GLOBAL contrast stretch (same factor for RGB → preserves color)');

      // ======== 🔍🔍🔍 RGB CROSS-CHANNEL ANALYSIS IN DISPLAY INPUT ========
      console.log('[WebGLDisplay] 🔍🔍🔍 Input RGB cross-channel analysis:');

      // Sample first 15 non-black pixels
      const samplePixels: {i: number, r: number, g: number, b: number}[] = [];
      for (let i = 0; i < width * height && samplePixels.length < 15; i++) {
        const r = data[i * 3 + 0];
        const g = data[i * 3 + 1];
        const b = data[i * 3 + 2];
        if (r > 0.01 || g > 0.01 || b > 0.01) {
          samplePixels.push({ i, r, g, b });
        }
      }

      console.log('[WebGLDisplay]   Sample input pixels (before global stretch):');
      for (const sp of samplePixels.slice(0, 8)) {
        const diffRG = (sp.r - sp.g).toFixed(4);
        const diffGB = (sp.g - sp.b).toFixed(4);
        console.log(`[WebGLDisplay]     px ${sp.i}: R=${sp.r.toFixed(4)}, G=${sp.g.toFixed(4)}, B=${sp.b.toFixed(4)} | R-G=${diffRG}, G-B=${diffGB}`);
      }

      // Compute overall R-G, G-B variance
      let sumDiffRG = 0, sumDiffGB = 0;
      let sumDiffRG2 = 0, sumDiffGB2 = 0;
      let countNonBg = 0;

      for (let i = 0; i < width * height; i++) {
        const r = data[i * 3 + 0];
        const g = data[i * 3 + 1];
        const b = data[i * 3 + 2];
        if (r > 0.001 || g > 0.001 || b > 0.001) {
          const dRG = r - g;
          const dGB = g - b;
          sumDiffRG += dRG;
          sumDiffGB += dGB;
          sumDiffRG2 += dRG * dRG;
          sumDiffGB2 += dGB * dGB;
          countNonBg++;
        }
      }

      if (countNonBg > 0) {
        const meanDiffRG = sumDiffRG / countNonBg;
        const meanDiffGB = sumDiffGB / countNonBg;
        const varDiffRG = sumDiffRG2 / countNonBg - meanDiffRG * meanDiffRG;
        const varDiffGB = sumDiffGB2 / countNonBg - meanDiffGB * meanDiffGB;

        console.log(`[WebGLDisplay]   Input RGB cross-channel (${countNonBg} pixels):`);
        console.log(`[WebGLDisplay]     R-G: mean=${meanDiffRG.toFixed(6)}, σ=${Math.sqrt(varDiffRG).toFixed(6)}`);
        console.log(`[WebGLDisplay]     G-B: mean=${meanDiffGB.toFixed(6)}, σ=${Math.sqrt(varDiffGB).toFixed(6)}`);

        if (Math.sqrt(varDiffRG) < 0.01 && Math.sqrt(varDiffGB) < 0.01) {
          console.log(`[WebGLDisplay]   ⚠️⚠️⚠️ Input colors have R≈G≈B - grayscale-ish input!`);
        } else {
          console.log(`[WebGLDisplay]   ✅ Input has RGB diversity`);
        }
      }
      // ======== END RGB ANALYSIS ========

      // ストレッチ後は露出調整不要
      this.shaderMaterial.uniforms.uExposure.value = 1.0;
      this.shaderMaterial.uniforms.uContrast.value = 1.0;
    }

    // テクスチャ更新（GPU直接転送）
    this.texture.image.data = pixels;
    this.texture.needsUpdate = true;

    // WebGLレンダリング
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * 露出調整
   */
  public setExposure(value: number): void {
    this.shaderMaterial.uniforms.uExposure.value = value;
  }

  /**
   * コントラスト調整
   */
  public setContrast(value: number): void {
    this.shaderMaterial.uniforms.uContrast.value = value;
  }

  /**
   * ガンマ値調整
   */
  public setGamma(value: number): void {
    this.shaderMaterial.uniforms.uGamma.value = value;
  }

  /**
   * リサイズ
   */
  public resize(width: number, height: number): void {
    this.renderer.setSize(width, height);
  }

  /**
   * リソース解放
   */
  public dispose(): void {
    this.texture.dispose();
    this.quad.geometry.dispose();
    this.shaderMaterial.dispose();
    this.renderer.dispose();
    console.log('[WebGLDisplay] Disposed');
  }
}
