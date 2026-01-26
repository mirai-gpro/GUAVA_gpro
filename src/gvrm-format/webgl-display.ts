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

    // HWC → RGBA変換（WebGLテクスチャ用）
    const pixels = new Float32Array(width * height * 4);

    for (let i = 0; i < width * height; i++) {
      const srcIdx = i * 3;
      pixels[i * 4 + 0] = data[srcIdx + 0]; // R
      pixels[i * 4 + 1] = data[srcIdx + 1]; // G
      pixels[i * 4 + 2] = data[srcIdx + 2]; // B
      pixels[i * 4 + 3] = 1.0;              // A
    }

    // 初回フレームのみ統計情報を出力
    if (frameCount === 1) {
      const sampleSize = Math.min(3000, width * height);
      let min = Infinity, max = -Infinity, sum = 0;
      for (let i = 0; i < sampleSize * 4; i++) {
        if (i % 4 === 3) continue; // Alpha skip
        const v = pixels[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
      }
      const avg = sum / (sampleSize * 3);

      console.log('[WebGLDisplay] First frame stats:', {
        min: min.toFixed(4),
        max: max.toFixed(4),
        avg: avg.toFixed(4)
      });

      // 自動露出調整
      if (max > 0.01) {
        const autoExposure = 0.8 / max;
        this.shaderMaterial.uniforms.uExposure.value = Math.min(autoExposure, 3.0);
        console.log('[WebGLDisplay] Auto exposure:', this.shaderMaterial.uniforms.uExposure.value.toFixed(2));
      }
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
