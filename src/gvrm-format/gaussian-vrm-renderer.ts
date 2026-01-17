// gaussian-vrm-renderer.ts
// Gaussian Splatting + VRM統合レンダラー

import * as THREE from 'three';
import { gaussianVertexShader, gaussianFragmentShader } from './gaussian-splatting-shader';

export interface GaussianVRMData {
  positions: Float32Array;      // 頂点位置 [N*3]
  rotations: Float32Array;      // 回転 [N*4] quaternion
  scales: Float32Array;         // スケール [N*3]
  opacities: Float32Array;      // 不透明度 [N]
  latents: Float32Array;        // latent features [N*32]
  boneIndices: Float32Array;    // ボーンインデックス [N*4]
  boneWeights: Float32Array;    // ボーンウェイト [N*4]
  vertexCount: number;
}

export class GaussianVRMRenderer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private renderTarget: THREE.WebGLRenderTarget;
  
  private gaussianMaterial: THREE.ShaderMaterial | null = null;
  private gaussianGeometry: THREE.BufferGeometry | null = null;
  private gaussianPoints: THREE.Points | null = null;
  
  private data: GaussianVRMData | null = null;
  
  // ボーン変形用
  private boneMatrices: Float32Array = new Float32Array(16 * 32); // 最大32ボーン
  
  constructor(
    container: HTMLElement,
    width: number = 256,
    height: number = 256
  ) {
    // Scene
    this.scene = new THREE.Scene();
    
    // Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    this.camera.position.set(0, 1.4, 0.8);
    this.camera.lookAt(0, 1.4, 0);
    
    // Renderer（hidden）
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.domElement.style.display = 'none';
    container.appendChild(this.renderer.domElement);
    
    // Render Target（32ch feature map用）
    this.renderTarget = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter
    });
    
    console.log('[GaussianVRM] Initialized');
  }
  
  /**
   * Gaussianデータをロード
   */
  loadGaussians(data: GaussianVRMData): void {
    this.data = data;
    
    // Geometry作成
    this.gaussianGeometry = new THREE.BufferGeometry();
    
    // 頂点属性を設定
    this.gaussianGeometry.setAttribute('position', 
      new THREE.BufferAttribute(data.positions, 3));
    this.gaussianGeometry.setAttribute('rotation', 
      new THREE.BufferAttribute(data.rotations, 4));
    this.gaussianGeometry.setAttribute('scale', 
      new THREE.BufferAttribute(data.scales, 3));
    this.gaussianGeometry.setAttribute('opacity', 
      new THREE.BufferAttribute(data.opacities, 1));
    
    // Latent featuresを8つの属性に分割（32ch → 8×4ch）
    for (let i = 0; i < 8; i++) {
      const latentData = new Float32Array(data.vertexCount * 4);
      for (let v = 0; v < data.vertexCount; v++) {
        for (let c = 0; c < 4; c++) {
          latentData[v * 4 + c] = data.latents[v * 32 + i * 4 + c];
        }
      }
      this.gaussianGeometry.setAttribute(`latent${i}`, 
        new THREE.BufferAttribute(latentData, 4));
    }
    
    // Material作成
    this.gaussianMaterial = new THREE.ShaderMaterial({
      vertexShader: gaussianVertexShader,
      fragmentShader: gaussianFragmentShader,
      uniforms: {
        modelViewMatrix: { value: new THREE.Matrix4() },
        projectionMatrix: { value: this.camera.projectionMatrix },
        tileIndex: { value: 0 }
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: true,
      depthWrite: false
    });
    
    // Points作成
    this.gaussianPoints = new THREE.Points(this.gaussianGeometry, this.gaussianMaterial);
    this.scene.add(this.gaussianPoints);
    
    console.log('[GaussianVRM] Gaussians loaded:', data.vertexCount);
  }
  
  /**
   * ボーン変形を更新
   */
  updateBones(boneTransforms: Float32Array): void {
    if (boneTransforms.length !== this.boneMatrices.length) {
      console.warn('[GaussianVRM] Bone matrix size mismatch');
      return;
    }
    this.boneMatrices.set(boneTransforms);
    
    // ここで実際のボーンスキニングを実装
    // 現在は省略（VRMManagerから受け取った変形を適用）
  }
  
  /**
   * Coarse Feature Mapを生成（8パス）
   */
  renderCoarseFeatureMap(): Float32Array {
    if (!this.gaussianPoints || !this.gaussianMaterial) {
      throw new Error('[GaussianVRM] Gaussians not loaded');
    }
    
    const width = this.renderTarget.width;
    const height = this.renderTarget.height;
    const coarseFM = new Float32Array(32 * width * height);
    const pixels = new Float32Array(width * height * 4);
    
    console.log('[GaussianVRM] Rendering coarse feature map (8 passes)...');
    
    // 8パスでレンダリング（各パス4チャンネル = 合計32チャンネル）
    for (let tileIdx = 0; tileIdx < 8; tileIdx++) {
      // タイルインデックスを更新
      this.gaussianMaterial.uniforms.tileIndex.value = tileIdx;
      
      // レンダリング
      this.renderer.setRenderTarget(this.renderTarget);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);
      
      // ピクセルデータを読み取り
      this.renderer.readRenderTargetPixels(
        this.renderTarget, 0, 0, width, height, pixels
      );
      
      // 4チャンネルを32チャンネル配列に格納
      const baseOffset = tileIdx * 4 * width * height;
      for (let p = 0; p < width * height; p++) {
        coarseFM[baseOffset + p] = pixels[p * 4 + 0];
        coarseFM[baseOffset + width * height + p] = pixels[p * 4 + 1];
        coarseFM[baseOffset + width * height * 2 + p] = pixels[p * 4 + 2];
        coarseFM[baseOffset + width * height * 3 + p] = pixels[p * 4 + 3];
      }
      
      if (tileIdx === 0) {
        // デバッグ：最初のパスの統計
        const sample = Array.from(pixels.slice(0, 100));
        console.log('[GaussianVRM] Pass 0 sample:', {
          min: Math.min(...sample).toFixed(4),
          max: Math.max(...sample).toFixed(4),
          nonZero: sample.filter(v => Math.abs(v) > 0.001).length
        });
      }
    }
    
    // レンダーターゲットをリセット
    this.renderer.setRenderTarget(null);
    
    console.log('[GaussianVRM] Coarse feature map generated:', {
      shape: `[1, 32, ${width}, ${height}]`,
      size: coarseFM.length
    });
    
    return coarseFM;
  }
  
  /**
   * クリーンアップ
   */
  dispose(): void {
    if (this.gaussianGeometry) {
      this.gaussianGeometry.dispose();
    }
    if (this.gaussianMaterial) {
      this.gaussianMaterial.dispose();
    }
    if (this.renderTarget) {
      this.renderTarget.dispose();
    }
    this.renderer.dispose();
    console.log('[GaussianVRM] Disposed');
  }
}