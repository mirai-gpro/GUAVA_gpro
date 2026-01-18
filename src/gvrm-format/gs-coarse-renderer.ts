// gs-coarse-renderer.ts
// GS Coarse Pass - WebGL統一レンダリング
// 技術仕様書準拠: WebGLは描画ではなく計算基盤

import * as THREE from 'three';
import { GSViewer } from './gs';

/**
 * GS Coarse Pass Renderer
 * WebGLでGaussian Splattingをレンダリングし、32ch feature mapを生成
 */
export class GSCoarseRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderTargets: THREE.WebGLRenderTarget[];
  private gsViewer: GSViewer;

  private readonly FEATURE_MAP_SIZE = 512;
  private readonly NUM_TILES = 8;  // 32ch / 4ch per tile

  constructor(gsViewer: GSViewer, cameraConfig: {
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
    aspect: number;
    near: number;
    far: number;
  }) {
    console.log('[GSCoarseRenderer] Initializing WebGL coarse pass...');

    this.gsViewer = gsViewer;

    // WebGLRenderer (オフスクリーン、Canvas非表示)
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(this.FEATURE_MAP_SIZE, this.FEATURE_MAP_SIZE);
    this.renderer.setPixelRatio(1);  // 精度維持
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;  // sRGB変換なし

    // Scene
    this.scene = new THREE.Scene();
    this.scene.add(gsViewer.mesh);

    // Camera (source_camera.json準拠)
    this.camera = new THREE.PerspectiveCamera(
      cameraConfig.fov,
      cameraConfig.aspect,
      cameraConfig.near,
      cameraConfig.far
    );
    this.camera.position.set(...cameraConfig.position);
    this.camera.lookAt(new THREE.Vector3(...cameraConfig.target));

    // RenderTargets (8タイル分、各RGBA=4ch)
    // WebGL2 + FloatType でHDR精度を維持
    this.renderTargets = [];
    for (let i = 0; i < this.NUM_TILES; i++) {
      const rt = new THREE.WebGLRenderTarget(this.FEATURE_MAP_SIZE, this.FEATURE_MAP_SIZE, {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,  // 32bit float精度
        depthBuffer: true,
        stencilBuffer: false
      });
      this.renderTargets.push(rt);
    }

    console.log('[GSCoarseRenderer] ✅ Initialized:', {
      resolution: `${this.FEATURE_MAP_SIZE}×${this.FEATURE_MAP_SIZE}`,
      tiles: this.NUM_TILES,
      totalChannels: this.NUM_TILES * 4
    });
  }

  /**
   * GS Coarse Pass 実行
   * 8パスで32ch feature mapを生成
   * @returns Float32Array [32, 512, 512] NCHW format
   */
  public renderCoarseFeatureMap(): Float32Array {
    const size = this.FEATURE_MAP_SIZE;
    const channels = this.NUM_TILES * 4;  // 32ch
    const featureMap = new Float32Array(channels * size * size);

    console.log('[GSCoarseRenderer] Starting 8-pass rendering...');

    // 一時バッファ（各パスのRGBA読み取り用）
    const pixelBuffer = new Float32Array(size * size * 4);

    for (let tile = 0; tile < this.NUM_TILES; tile++) {
      // 1. latentTileを更新（4ch単位）
      this.gsViewer.updateLatentTile(tile);

      // 2. RenderTargetにレンダリング
      this.renderer.setRenderTarget(this.renderTargets[tile]);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);

      // 3. Framebufferから読み取り
      this.renderer.readRenderTargetPixels(
        this.renderTargets[tile],
        0, 0, size, size,
        pixelBuffer
      );

      // 4. NCHW形式に変換して格納
      // pixelBuffer: [H, W, 4] (HWC) → featureMap: [C, H, W] (NCHW)
      const baseChannel = tile * 4;
      for (let c = 0; c < 4; c++) {
        const channelIdx = baseChannel + c;
        const channelOffset = channelIdx * size * size;

        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            // WebGLは左下原点、Neural Refinerは左上原点なのでY反転
            const srcY = size - 1 - y;
            const srcIdx = (srcY * size + x) * 4 + c;
            const dstIdx = channelOffset + y * size + x;
            featureMap[dstIdx] = pixelBuffer[srcIdx];
          }
        }
      }

      if (tile === 0) {
        // 最初のタイルの統計情報
        let min = Infinity, max = -Infinity, nonZero = 0;
        for (let i = 0; i < size * size * 4; i++) {
          const v = pixelBuffer[i];
          if (v !== 0) nonZero++;
          if (v < min) min = v;
          if (v > max) max = v;
        }
        console.log(`[GSCoarseRenderer] Tile 0 stats:`, {
          min: min.toFixed(4),
          max: max.toFixed(4),
          nonZeroRatio: (nonZero / (size * size * 4) * 100).toFixed(1) + '%'
        });
      }
    }

    // レンダーターゲットをリセット
    this.renderer.setRenderTarget(null);

    // 最終統計
    let finalMin = Infinity, finalMax = -Infinity, finalNonZero = 0;
    for (let i = 0; i < featureMap.length; i++) {
      const v = featureMap[i];
      if (v !== 0) finalNonZero++;
      if (v < finalMin) finalMin = v;
      if (v > finalMax) finalMax = v;
    }

    console.log('[GSCoarseRenderer] ✅ Coarse pass complete:', {
      shape: `[32, ${size}, ${size}]`,
      totalSize: featureMap.length.toLocaleString(),
      valueRange: `[${finalMin.toFixed(4)}, ${finalMax.toFixed(4)}]`,
      nonZeroRatio: (finalNonZero / featureMap.length * 100).toFixed(1) + '%'
    });

    return featureMap;
  }

  /**
   * カメラ更新
   */
  public updateCamera(config: {
    position: [number, number, number];
    target: [number, number, number];
    fov?: number;
  }): void {
    this.camera.position.set(...config.position);
    this.camera.lookAt(new THREE.Vector3(...config.target));
    if (config.fov !== undefined) {
      this.camera.fov = config.fov;
      this.camera.updateProjectionMatrix();
    }
  }

  /**
   * リソース解放
   */
  public dispose(): void {
    for (const rt of this.renderTargets) {
      rt.dispose();
    }
    this.renderer.dispose();
    console.log('[GSCoarseRenderer] Disposed');
  }
}
