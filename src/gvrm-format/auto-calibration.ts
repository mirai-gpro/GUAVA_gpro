import * as THREE from 'three';

/**
 * GUAVAとgaussian-vrmの統合のための自動キャリブレーションシステム
 * 
 * 参考:
 * - GUAVA: カメラパラメータとスケールの自動推定
 * - GAGAvatar: FLAME Priorを使った最適化
 * - gaussian-vrm: VRM規格に準拠したリギング
 */
export class AutoCalibration {
    
    /**
     * PLYデータからバウンディングボックスを計算
     */
    static computeBoundingBox(positions: Float32Array): {
        min: THREE.Vector3;
        max: THREE.Vector3;
        center: THREE.Vector3;
        size: THREE.Vector3;
    } {
        const min = new THREE.Vector3(Infinity, Infinity, Infinity);
        const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];
            
            min.x = Math.min(min.x, x);
            min.y = Math.min(min.y, y);
            min.z = Math.min(min.z, z);
            
            max.x = Math.max(max.x, x);
            max.y = Math.max(max.y, y);
            max.z = Math.max(max.z, z);
        }
        
        const center = new THREE.Vector3(
            (min.x + max.x) / 2,
            (min.y + max.y) / 2,
            (min.z + max.z) / 2
        );
        
        const size = new THREE.Vector3(
            max.x - min.x,
            max.y - min.y,
            max.z - min.z
        );
        
        return { min, max, center, size };
    }
    
    /**
     * 自動カメラ位置の計算
     * 
     * GUAVAの手法: バウンディングボックスから最適な視点を計算
     * - 上半身全体が視野に収まるように配置
     * - FOVとアスペクト比を考慮
     */
    static computeCameraPosition(
        bbox: ReturnType<typeof AutoCalibration.computeBoundingBox>,
        fov: number,
        aspect: number
    ): {
        position: THREE.Vector3;
        target: THREE.Vector3;
        distance: number;
    } {
        // アバターの中心（胸のあたり）を注視点に
        const targetY = bbox.min.y + bbox.size.y * 0.7;
        const target = new THREE.Vector3(
            bbox.center.x,
            targetY,
            bbox.center.z
        );
        
        // 視野に収めるための距離を計算
        // GUAVA論文: 上半身が画面の70-80%を占めるように配置
        const verticalSize = bbox.size.y * 0.9; // 上半身の高さ
        const horizontalSize = bbox.size.x * 1.2; // 幅（余裕を持たせる）
        
        // FOVから必要な距離を計算
        const fovRad = (fov * Math.PI) / 180;
        const distanceVertical = verticalSize / (2 * Math.tan(fovRad / 2));
        const distanceHorizontal = horizontalSize / (2 * Math.tan(fovRad / 2) * aspect);
        
        // 大きい方を採用（全体が収まるように）
        const distance = Math.max(distanceVertical, distanceHorizontal) * 1.1; // 10%のマージン
        
        // カメラ位置: 正面やや上から
        const position = new THREE.Vector3(
            target.x,
            target.y + bbox.size.y * 0.1, // 少し上から
            target.z + distance
        );
        
        return { position, target, distance };
    }
    
    /**
     * 適応的スケール係数の計算（修正版）
     * 
     * GAGAvatar/gaussian-vrmの手法:
     * - シンプルな身長正規化のみ
     * - ピクセル変換は不要（3D座標系内で完結）
     * 
     * @param bbox バウンディングボックス
     * @param targetHeight 目標身長（メートル単位）デフォルト1.7m
     */
    static computeAdaptiveScale(
        bbox: ReturnType<typeof AutoCalibration.computeBoundingBox>,
        targetHeight: number = 1.70
    ): {
        scaleFactor: number;
        normalizedHeight: number;
    } {
        // 元の高さ
        const rawHeight = bbox.size.y;
        
        // シンプルな正規化: 目標身長 / 実測身長
        // 例: 1.70 / 1.75 = 0.971
        const scaleFactor = targetHeight / rawHeight;
        
        // 正規化後は常に目標身長
        const normalizedHeight = targetHeight;
        
        return { scaleFactor, normalizedHeight };
    }
    
    /**
     * テクスチャ投影のためのカメラ行列を計算
     * 
     * GUAVA論文 Sec 3.2: Projection Sampling
     * - source.pngの撮影カメラパラメータを推定
     * - ガウシアンへの投影マッピングに使用
     */
    static computeProjectionMatrix(
        imageWidth: number,
        imageHeight: number,
        fov: number = 45
    ): {
        projectionMatrix: THREE.Matrix4;
        viewMatrix: THREE.Matrix4;
    } {
        const aspect = imageWidth / imageHeight;
        const near = 0.01;
        const far = 100;
        
        // 透視投影行列
        const projectionMatrix = new THREE.Matrix4();
        projectionMatrix.makePerspective(
            THREE.MathUtils.degToRad(fov),
            aspect,
            near,
            far
        );
        
        // ビュー行列（原点を見る単位行列）
        const viewMatrix = new THREE.Matrix4();
        viewMatrix.makeTranslation(0, 0, 0);
        
        return { projectionMatrix, viewMatrix };
    }
    
    /**
     * 完全な自動キャリブレーション
     * 
     * 使用例:
     * const calib = AutoCalibration.fullCalibration(plyData, 512, 512);
     * camera.position.copy(calib.camera.position);
     * camera.lookAt(calib.camera.target);
     */
    static fullCalibration(
        positions: Float32Array,
        imageWidth: number,
        imageHeight: number,
        containerWidth: number,
        containerHeight: number
    ) {
        console.log('[AutoCalib] Starting full calibration...');
        
        // 1. バウンディングボックス計算
        const bbox = this.computeBoundingBox(positions);
        console.log('[AutoCalib] Bounding box:', {
            center: bbox.center,
            size: bbox.size
        });
        
        // 2. カメラ位置の自動計算
        const aspect = containerWidth / containerHeight;
        const fov = 35; // GUAVA推奨値
        const camera = this.computeCameraPosition(bbox, fov, aspect);
        console.log('[AutoCalib] Camera position:', camera);
        
        // 3. スケール係数の自動計算（修正版: シンプルな正規化のみ）
        const scale = this.computeAdaptiveScale(bbox, 1.70);
        console.log('[AutoCalib] Scale factor:', scale);
        
        // 4. 投影行列の計算
        const projection = this.computeProjectionMatrix(imageWidth, imageHeight, fov);
        
        return {
            bbox,
            camera: {
                position: camera.position,
                target: camera.target,
                distance: camera.distance,
                fov
            },
            scale: {
                factor: scale.scaleFactor,
                normalizedHeight: scale.normalizedHeight
            },
            projection
        };
    }
}

/**
 * 使用例:
 * 
 * // PLYLoader内で使用
 * const calib = AutoCalibration.fullCalibration(
 *     data.positions, 
 *     512, 512,  // source.pngのサイズ
 *     container.clientWidth,
 *     container.clientHeight
 * );
 * 
 * // 正規化時にスケール係数を適用
 * const scaleFactor = calib.scale.factor;
 * x = (x - calib.bbox.center.x) * scaleFactor;
 * y = (y - calib.bbox.min.y) * scaleFactor;
 * z = (z - calib.bbox.center.z) * scaleFactor;
 * 
 * // GVRM初期化時にカメラを設定
 * camera.position.copy(calib.camera.position);
 * camera.lookAt(calib.camera.target);
 * camera.fov = calib.camera.fov;
 * camera.updateProjectionMatrix();
 */