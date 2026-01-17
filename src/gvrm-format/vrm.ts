import * as THREE from 'three';

// テンプレート(ply.ts)の定義に合わせたピボット位置
const SKELETON_POS: Record<number, THREE.Vector3> = {
    0:  new THREE.Vector3(0.0, 1.05, 0.0),   // Root
    3:  new THREE.Vector3(0.0, 1.25, 0.0),   // Spine1
    6:  new THREE.Vector3(0.0, 1.35, 0.0),   // Spine2
    9:  new THREE.Vector3(0.0, 1.45, 0.0),   // Spine3
    12: new THREE.Vector3(0.0, 1.55, 0.0),   // Neck
    15: new THREE.Vector3(0.0, 1.65, 0.05),  // Head
    
    // 【修正】肩の位置をテンプレートの始点(1.42m)に合わせる
    // これで腕を回したときに肩が脱臼して見えるのを防ぎます
    16: new THREE.Vector3(0.22, 1.42, 0.0),   // L_Shoulder
    17: new THREE.Vector3(-0.22, 1.42, 0.0),  // R_Shoulder
    
    22: new THREE.Vector3(0.0, 1.60, 0.08),  // Jaw
};

const HIERARCHY: Record<number, number> = {
    3: 0, 6: 3, 9: 6, 12: 9, 15: 12, 22: 15, 16: 9, 17: 9
};

export class VRMManager {
    public matrices: Float32Array;
    private clock: THREE.Clock;
    private _lipSyncLevel: number = 0;
    private localRotations: Record<number, THREE.Euler> = {};

    constructor() {
        this.matrices = new Float32Array(16 * 64);
        this.clock = new THREE.Clock();
        this.resetPose();
    }

    private resetPose() {
        const identity = new THREE.Matrix4();
        for (let i = 0; i < 64; i++) {
            identity.toArray(this.matrices, i * 16);
            this.localRotations[i] = new THREE.Euler(0, 0, 0);
        }
    }

    public setLipSync(level: number) {
        this._lipSyncLevel = level;
    }

    private getPivotMatrix(boneIndex: number, rotation: THREE.Euler): THREE.Matrix4 {
        const pivot = SKELETON_POS[boneIndex] || new THREE.Vector3(0, 0, 0);
        const m = new THREE.Matrix4();
        m.makeTranslation(pivot.x, pivot.y, pivot.z);
        const r = new THREE.Matrix4();
        r.makeRotationFromEuler(rotation);
        m.multiply(r);
        const tInv = new THREE.Matrix4();
        tInv.makeTranslation(-pivot.x, -pivot.y, -pivot.z);
        m.multiply(tInv);
        return m;
    }

    private solveHierarchy() {
        const globalMatrices: Record<number, THREE.Matrix4> = {};
        [0, 3, 6, 9, 12, 15, 16, 17, 22].forEach(idx => {
            const rot = this.localRotations[idx] || new THREE.Euler(0, 0, 0);
            const localMat = this.getPivotMatrix(idx, rot);
            const parentIdx = HIERARCHY[idx];
            
            if (parentIdx !== undefined && globalMatrices[parentIdx]) {
                globalMatrices[idx] = globalMatrices[parentIdx].clone().multiply(localMat);
            } else {
                globalMatrices[idx] = localMat;
            }
            globalMatrices[idx].toArray(this.matrices, idx * 16);
        });
    }

    public update(): Float32Array {
        const time = this.clock.getElapsedTime();

        // 1. Root: 固定
        this.localRotations[0] = new THREE.Euler(0, 0, 0);

        // 2. Spine3 (胸): 微細な呼吸
        this.localRotations[9] = new THREE.Euler(Math.sin(time * 1.5) * 0.015, 0, 0);

        // 3. Head (頭): 微細なゆらぎ
        this.localRotations[15] = new THREE.Euler(
             Math.sin(time * 0.4) * 0.01, 
             Math.sin(time * 0.25) * 0.01, 
             0
        );

        // 4. Jaw (顎): リップシンク (テンプレート吸着法なら吸い付きが良いので0.5で十分)
        this.localRotations[22] = new THREE.Euler(this._lipSyncLevel * 0.5, 0, 0);

        // 5. 腕: 自然なAポーズ
        // 直立(1.35) から 少し広げた状態(1.2) に戻します
        // X: 0.1 (自然に前へ)
        this.localRotations[16] = new THREE.Euler(0.1, 0, -1.2); // Left
        this.localRotations[17] = new THREE.Euler(0.1, 0,  1.2); // Right

        this.solveHierarchy();
        return this.matrices.slice(); 
    }
}