import * as THREE from 'three';
import { AutoCalibration } from './auto-calibration';

// テンプレート点の型定義
type TemplatePoint = { x: number, y: number, z: number, bone: number };

// ボーン検出結果の型定義
type BonePosition = {
    position: THREE.Vector3;
    radius: number;
};

export class PLYLoader {
    
    /**
     * 点群からボーン位置を自動検出
     */
    private static detectBonePositions(
        positions: Float32Array,
        bbox: ReturnType<typeof AutoCalibration.computeBoundingBox>
    ): Record<number, BonePosition> {
        const points: THREE.Vector3[] = [];
        for (let i = 0; i < positions.length; i += 3) {
            points.push(new THREE.Vector3(
                positions[i],
                positions[i + 1],
                positions[i + 2]
            ));
        }

        const height = bbox.size.y;
        const result: Record<number, BonePosition> = {};

        // Y座標でグループ分け
        const getPointsInRange = (minY: number, maxY: number, minZ?: number, maxZ?: number) => {
            return points.filter(p => {
                const inY = p.y >= minY && p.y <= maxY;
                if (minZ !== undefined && maxZ !== undefined) {
                    return inY && p.z >= minZ && p.z <= maxZ;
                }
                return inY;
            });
        };

        // 重心計算
        const computeCentroid = (pts: THREE.Vector3[]) => {
            if (pts.length === 0) return new THREE.Vector3(0, 0, 0);
            const sum = pts.reduce((acc, p) => acc.add(p.clone()), new THREE.Vector3(0, 0, 0));
            return sum.divideScalar(pts.length);
        };

        // 標準偏差（幅）計算
        const computeStdDev = (pts: THREE.Vector3[], axis: 'x' | 'y' | 'z') => {
            if (pts.length === 0) return 0.1;
            const mean = pts.reduce((sum, p) => sum + p[axis], 0) / pts.length;
            const variance = pts.reduce((sum, p) => sum + Math.pow(p[axis] - mean, 2), 0) / pts.length;
            return Math.sqrt(variance);
        };

        console.log('[BoneDetect] Starting automatic bone detection...');

        // --- 1. Hips (Root) ---
        const hipsPoints = getPointsInRange(0.50 * height, 0.65 * height);
        result[0] = {
            position: computeCentroid(hipsPoints),
            radius: Math.max(computeStdDev(hipsPoints, 'x'), 0.12)
        };

        // --- 2. Spine1 ---
        const spine1Points = getPointsInRange(0.60 * height, 0.75 * height);
        result[3] = {
            position: computeCentroid(spine1Points),
            radius: Math.max(computeStdDev(spine1Points, 'x'), 0.14)
        };

        // --- 3. Chest (Spine3) ---
        const chestPoints = getPointsInRange(0.75 * height, 0.88 * height);
        result[9] = {
            position: computeCentroid(chestPoints),
            radius: Math.max(computeStdDev(chestPoints, 'x'), 0.15)
        };

        // --- 4. Neck ---
        const neckPoints = getPointsInRange(0.88 * height, 0.93 * height);
        result[12] = {
            position: computeCentroid(neckPoints),
            radius: Math.max(computeStdDev(neckPoints, 'x'), 0.06)
        };

        // --- 5. Head ---
        const headPoints = getPointsInRange(0.90 * height, 1.0 * height);
        const headCenter = computeCentroid(headPoints);
        result[15] = {
            position: headCenter,
            radius: Math.max(computeStdDev(headPoints, 'x'), 0.10)
        };

        // --- 6. Jaw（顎）の精密検出 ---
        const neckCenter = result[12].position;
        
        const jawRegionPoints = points.filter(p => {
            const isInYRange = p.y >= neckCenter.y && p.y <= neckCenter.y + 0.08;
            const isInFront = p.z > neckCenter.z + 0.02;
            const isNearCenter = Math.abs(p.x - neckCenter.x) < 0.08;
            return isInYRange && isInFront && isNearCenter;
        });

        const sortedJawPoints = jawRegionPoints.sort((a, b) => a.y - b.y);
        const jawCandidates = sortedJawPoints.slice(0, Math.max(10, Math.floor(sortedJawPoints.length * 0.2)));
        
        const jawCenter = computeCentroid(jawCandidates);
        
        if (jawCandidates.length < 5 || jawCenter.y < neckCenter.y) {
            result[22] = {
                position: new THREE.Vector3(
                    neckCenter.x,
                    neckCenter.y + 0.05,
                    neckCenter.z + 0.08
                ),
                radius: 0.04
            };
            console.warn('[BoneDetect] Jaw detection fallback activated');
        } else {
            result[22] = {
                position: jawCenter,
                radius: 0.04
            };
        }

        console.log('[BoneDetect] Jaw detected:', {
            position: result[22].position,
            candidateCount: jawCandidates.length,
            totalRegionPoints: jawRegionPoints.length
        });

        // --- 7. 腕（Left / Right Shoulder）---
        const shoulderHeight = result[9].position.y;
        const shoulderPoints = getPointsInRange(
            shoulderHeight - 0.05,
            shoulderHeight + 0.05
        );

        const sortedByX = [...shoulderPoints].sort((a, b) => a.x - b.x);
        
        const leftShoulderPoints = sortedByX.slice(Math.floor(sortedByX.length * 0.8));
        result[16] = {
            position: computeCentroid(leftShoulderPoints),
            radius: 0.06
        };

        const rightShoulderPoints = sortedByX.slice(0, Math.floor(sortedByX.length * 0.2));
        result[17] = {
            position: computeCentroid(rightShoulderPoints),
            radius: 0.06
        };

        console.log('[BoneDetect] Detected bone positions:', result);
        return result;
    }

    /**
     * 検出されたボーン位置からテンプレート点を生成
     */
    private static generateTemplatePoints(
        bonePositions: Record<number, BonePosition>
    ): TemplatePoint[] {
        const templates: TemplatePoint[] = [];

        const addSphere = (boneIdx: number, center: THREE.Vector3, radius: number, density: number) => {
            if (boneIdx === 22) {
                // 顎専用: 前方に向いた半球
                for (let i = 0; i <= density; i++) {
                    for (let j = 0; j <= density; j++) {
                        const phi = (i / density) * Math.PI - Math.PI / 2;
                        const theta = (j / density) * Math.PI - Math.PI / 2;
                        
                        const x = center.x + radius * Math.cos(phi) * Math.sin(theta + Math.PI / 2);
                        const y = center.y + radius * Math.sin(phi);
                        const z = center.z + radius * Math.cos(phi) * Math.cos(theta + Math.PI / 2);
                        
                        templates.push({ x, y, z, bone: boneIdx });
                    }
                }
            } else {
                // 他のボーンは球状
                const phi = Math.PI / 4;
                const theta = Math.PI / 3;
                
                for (let i = 0; i <= density; i++) {
                    for (let j = 0; j <= density; j++) {
                        const p = i / density * phi;
                        const t = j / density * theta * 2;
                        
                        const x = center.x + radius * Math.sin(p) * Math.cos(t);
                        const y = center.y + radius * Math.cos(p);
                        const z = center.z + radius * Math.sin(p) * Math.sin(t);
                        
                        templates.push({ x, y, z, bone: boneIdx });
                    }
                }
            }
        };

        Object.entries(bonePositions).forEach(([boneIdxStr, bone]) => {
            const boneIdx = parseInt(boneIdxStr);
            
            let density = 4;
            if (boneIdx === 22) {
                density = 8;
            } else if (boneIdx === 15) {
                density = 5;
            }
            
            addSphere(boneIdx, bone.position, bone.radius, density);
        });

        console.log(`[BoneDetect] Generated ${templates.length} template points`);
        return templates;
    }

    static async load(url: string, imageWidth: number = 512, imageHeight: number = 512) {
        console.log("[GVRM] PLYLoader: Start Fetching " + url);
        const res = await fetch(url);
        const buffer = await res.arrayBuffer();
        
        // ヘッダー解析
        const headerText = new TextDecoder().decode(buffer.slice(0, 5000));
        const headerEndIdx = headerText.indexOf("end_header") + 10;
        const vertexCount = parseInt(headerText.match(/element vertex (\d+)/)![1]);
        const props = headerText.split('\n').filter(l => l.startsWith('property float')).map(l => l.split(' ').pop()?.trim() || "");
        const stride = props.length * 4;
        const dataView = new DataView(buffer, headerEndIdx + 1);

        const data = {
            positions: new Float32Array(vertexCount * 3),
            normals: new Float32Array(vertexCount * 3),      // ← 法線配列を追加
            colors: new Float32Array(vertexCount * 3),
            boneIndices: new Float32Array(vertexCount * 4),
            boneWeights: new Float32Array(vertexCount * 4),
            scales: new Float32Array(vertexCount * 3)
        };

        // --- Pass 1: 生データの読み込み ---
        const rawPositions = new Float32Array(vertexCount * 3);
        
        for (let i = 0; i < vertexCount; i++) {
            const b = i * stride;
            const x = dataView.getFloat32(b + props.indexOf('x') * 4, true);
            const y = dataView.getFloat32(b + props.indexOf('y') * 4, true);
            const z = dataView.getFloat32(b + props.indexOf('z') * 4, true);
            
            rawPositions[i * 3] = x;
            rawPositions[i * 3 + 1] = y;
            rawPositions[i * 3 + 2] = z;
        }

        // 自動キャリブレーション
        const bbox = AutoCalibration.computeBoundingBox(rawPositions);
        const scaleInfo = AutoCalibration.computeAdaptiveScale(bbox, 1.70);
        const scaleFactor = scaleInfo.scaleFactor;
        const normalizedHeight = scaleInfo.normalizedHeight;

        console.log(`[GVRM] Auto-scaling... Raw height: ${bbox.size.y.toFixed(3)}m -> Normalized: ${normalizedHeight.toFixed(3)}m (factor: ${scaleFactor.toFixed(3)})`);

        // --- Pass 2: 正規化とボーン位置検出 ---
        const normalizedPositions = new Float32Array(vertexCount * 3);
        
        for (let i = 0; i < vertexCount; i++) {
            const x = (rawPositions[i * 3] - bbox.center.x) * scaleFactor;
            const y = (rawPositions[i * 3 + 1] - bbox.min.y) * scaleFactor;
            const z = (rawPositions[i * 3 + 2] - bbox.center.z) * scaleFactor;
            
            normalizedPositions[i * 3] = x;
            normalizedPositions[i * 3 + 1] = y;
            normalizedPositions[i * 3 + 2] = z;
        }

        const normalizedBbox = AutoCalibration.computeBoundingBox(normalizedPositions);
        const bonePositions = this.detectBonePositions(normalizedPositions, normalizedBbox);
        const templatePoints = this.generateTemplatePoints(bonePositions);

        // --- Pass 3: 最近傍探索とリギング ---
        const boneStats: Record<number, number> = {};
        [0, 3, 9, 12, 15, 16, 17, 22].forEach(idx => boneStats[idx] = 0);

        for (let i = 0; i < vertexCount; i++) {
            const b = i * stride;
            const read = (n: string) => {
                const idx = props.indexOf(n);
                return idx === -1 ? 0 : dataView.getFloat32(b + idx * 4, true);
            };

            const x = normalizedPositions[i * 3];
            const y = normalizedPositions[i * 3 + 1];
            const z = normalizedPositions[i * 3 + 2];

            data.positions.set([x, y, z], i * 3);
            data.normals.set([read('nx'), read('ny'), read('nz')], i * 3);    // ← 法線を読み込み
            data.colors.set([read('f_dc_0'), read('f_dc_1'), read('f_dc_2')], i * 3);
            data.scales.set([
                read('scale_0') * scaleFactor,
                read('scale_1') * scaleFactor,
                read('scale_2') * scaleFactor
            ], i * 3);

            // 最近傍探索
            let bestBone = 0;
            let minDistSq = Infinity;

            for (let j = 0; j < templatePoints.length; j++) {
                const tp = templatePoints[j];
                const distSq = (x - tp.x)**2 + (y - tp.y)**2 + (z - tp.z)**2;
                
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    bestBone = tp.bone;
                }
            }

            boneStats[bestBone] = (boneStats[bestBone] || 0) + 1;

            data.boneIndices.set([bestBone, 0, 0, 0], i * 4);
            data.boneWeights.set([1.0, 0.0, 0.0, 0.0], i * 4);
        }

        console.log('[GVRM] Rigging Complete (Automatic Bone Detection).');
        console.log('[GVRM] Bone assignment stats:', boneStats);
        console.log(`[GVRM] Jaw points: ${boneStats[22] || 0}`);

        return {
            ...data,
            calibration: { bbox, scaleInfo },
            boneStats
        };
    }
}
