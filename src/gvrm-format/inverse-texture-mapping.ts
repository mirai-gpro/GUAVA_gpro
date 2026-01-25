// inverse-texture-mapping.ts
// 公式実装の正確な移植 (ubody_gaussian.py lines 85-114)

/**
 * GUAVA Inverse Texture Mapping
 * 
 * 公式実装: convert_pixel_feature_to_uv
 * 
 * 処理フロー:
 * 1. UV map上の各ピクセルに対応する三角形と重心座標を取得
 * 2. 重心座標で3D位置を補間
 * 3. 3D → Camera space → Image space へ変換
 * 4. Image featuresをgrid samplingでUV spaceにマッピング
 * 5. UV maskを適用
 */

// Canonical camera (data_loader.py)
const CANONICAL_W2C = {
    R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T: [0.0, 0.6, 22.0]
};
const INV_TAN_FOV = 24.0;

export interface EHMMeshData {
    vertices: Float32Array;        // [V, 3] deformed vertices
    faces: Uint32Array;            // [F, 3] triangle vertex indices
    uvCoords: Float32Array;        // [V, 2] UV coordinates per vertex
    
    // UV mapping data (from SMPLX)
    uvmap_f_idx: Int32Array;       // [H_uv, W_uv] triangle index for each UV pixel
    uvmap_f_bary: Float32Array;    // [H_uv, W_uv, 3] barycentric coords
    uvmap_mask: Uint8Array;        // [H_uv, W_uv] valid region mask
    
    numVertices: number;
    numFaces: number;
    uvmapSize: number;             // Typically 512
}

export interface ImageFeatures {
    data: Float32Array;     // [C, H, W] CHW format
    channels: number;       // C
    height: number;         // H
    width: number;          // W
}

export class InverseTextureMapper {
    private uvmapSize: number;
    private initialized: boolean = false;
    
    // EHM mesh data
    private vertices: Float32Array | null = null;
    private faces: Uint32Array | null = null;
    private uvmap_f_idx: Int32Array | null = null;
    private uvmap_f_bary: Float32Array | null = null;
    private uvmap_mask: Uint8Array | null = null;

    constructor(uvmapSize: number = 512) {
        this.uvmapSize = uvmapSize;
        console.log('[InverseTextureMapper] Created');
        console.log(`  UV map size: ${uvmapSize}×${uvmapSize}`);
    }

    /**
     * Initialize with EHM mesh data
     * Must be called before mapImageToUV
     */
    initialize(meshData: EHMMeshData): void {
        console.log('[InverseTextureMapper] Initializing...');
        
        // Validate input
        if (!meshData.vertices || !meshData.faces) {
            throw new Error('Invalid mesh data: missing vertices or faces');
        }
        
        if (!meshData.uvmap_f_idx || !meshData.uvmap_f_bary || !meshData.uvmap_mask) {
            throw new Error('Invalid mesh data: missing UV mapping data');
        }
        
        if (meshData.uvmapSize !== this.uvmapSize) {
            console.warn(`[InverseTextureMapper] UV map size mismatch: expected ${this.uvmapSize}, got ${meshData.uvmapSize}`);
            this.uvmapSize = meshData.uvmapSize;
        }
        
        // Store mesh data
        this.vertices = meshData.vertices;
        this.faces = meshData.faces;
        this.uvmap_f_idx = meshData.uvmap_f_idx;
        this.uvmap_f_bary = meshData.uvmap_f_bary;
        this.uvmap_mask = meshData.uvmap_mask;
        
        this.initialized = true;
        
        console.log('[InverseTextureMapper] ✅ Initialized');
        console.log(`  Vertices: ${meshData.numVertices}`);
        console.log(`  Faces: ${meshData.numFaces}`);
        console.log(`  UV map: ${this.uvmapSize}×${this.uvmapSize}`);
    }

    /**
     * Map image features to UV space
     * 
     * Python実装 (ubody_gaussian.py lines 85-114):
     * ```python
     * def convert_pixel_feature_to_uv(self, img_features, deformed_vertices, w2c_cam):
     *     # Get UV mapping data
     *     uvmap_f_idx = self.smplx.uvmap_f_idx      # [H, W]
     *     uvmap_f_bary = self.smplx.uvmap_f_bary    # [H, W, 3]
     *     
     *     # Get triangle vertices
     *     uv_vertex_id = faces[uvmap_f_idx]         # [H, W, 3]
     *     uv_vertex = deformed_vertices[..., uv_vertex_id, :]  # [H, W, 3, 3]
     *     
     *     # Barycentric interpolation
     *     uv_vertex = torch.einsum('hwk,hwkn->hwn', uvmap_f_bary, uv_vertex)  # [H, W, 3]
     *     
     *     # Transform to camera space
     *     uv_vertex_homo = torch.cat([uv_vertex, torch.ones_like(uv_vertex[..., :1])], dim=-1)
     *     uv_vertex_cam = torch.einsum('ij,hwj->hwi', w2c_cam, uv_vertex_homo)[..., :3]
     *     
     *     # Project to image space
     *     vertices_img = uv_vertex_cam * invtanfov / (uv_vertex_cam[..., 2:] + 1e-7)
     *     
     *     # Grid sample
     *     uv_features = F.grid_sample(img_features, vertices_img[..., :2], 
     *                                mode='bilinear', padding_mode='zeros', 
     *                                align_corners=False)
     *     
     *     # Apply mask
     *     uv_features = uv_features * uvmap_mask[None, None]
     *     return uv_features
     * ```
     */
    mapImageToUV(
        imageFeatures: ImageFeatures,
        deformedVertices?: Float32Array  // Optional: use stored vertices if not provided
    ): Float32Array {
        if (!this.initialized) {
            throw new Error('Not initialized. Call initialize() first.');
        }
        
        const vertices = deformedVertices || this.vertices!;
        const { data: imgData, channels: C, height: H_img, width: W_img } = imageFeatures;
        const H_uv = this.uvmapSize;
        const W_uv = this.uvmapSize;
        
        console.log('[InverseTextureMapper] Mapping image to UV...');
        console.log(`  Image features: ${C}ch × ${H_img}×${W_img}`);
        console.log(`  Target UV: ${C}ch × ${H_uv}×${W_uv}`);
        
        // Output: [C, H_uv, W_uv]
        const uvFeatures = new Float32Array(C * H_uv * W_uv);
        
        // Canonical camera (source imageは常にcanonical view)
        const R = CANONICAL_W2C.R;
        const T = CANONICAL_W2C.T;
        const invtanfov = INV_TAN_FOV;
        
        let validPixels = 0;
        let invalidTriangles = 0;
        
        // Process each UV pixel
        for (let v = 0; v < H_uv; v++) {
            for (let u = 0; u < W_uv; u++) {
                const uvIdx = v * W_uv + u;
                
                // 1. Get triangle index and barycentric coordinates
                const faceIdx = this.uvmap_f_idx![uvIdx];
                
                if (faceIdx < 0 || faceIdx >= this.faces!.length / 3) {
                    invalidTriangles++;
                    continue;  // Invalid pixel (outside UV layout)
                }
                
                const bary = [
                    this.uvmap_f_bary![uvIdx * 3 + 0],
                    this.uvmap_f_bary![uvIdx * 3 + 1],
                    this.uvmap_f_bary![uvIdx * 3 + 2]
                ];
                
                // 2. Get triangle vertex indices
                const v0_idx = this.faces![faceIdx * 3 + 0];
                const v1_idx = this.faces![faceIdx * 3 + 1];
                const v2_idx = this.faces![faceIdx * 3 + 2];
                
                // 3. Get vertex positions
                const v0 = [
                    vertices[v0_idx * 3 + 0],
                    vertices[v0_idx * 3 + 1],
                    vertices[v0_idx * 3 + 2]
                ];
                const v1 = [
                    vertices[v1_idx * 3 + 0],
                    vertices[v1_idx * 3 + 1],
                    vertices[v1_idx * 3 + 2]
                ];
                const v2 = [
                    vertices[v2_idx * 3 + 0],
                    vertices[v2_idx * 3 + 1],
                    vertices[v2_idx * 3 + 2]
                ];
                
                // 4. Barycentric interpolation
                const pos3D = [
                    bary[0] * v0[0] + bary[1] * v1[0] + bary[2] * v2[0],
                    bary[0] * v0[1] + bary[1] * v1[1] + bary[2] * v2[1],
                    bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]
                ];
                
                // 5. World to Camera transform
                // cam_pos = R * world_pos + T
                const camX = R[0][0] * pos3D[0] + R[0][1] * pos3D[1] + R[0][2] * pos3D[2] + T[0];
                const camY = R[1][0] * pos3D[0] + R[1][1] * pos3D[1] + R[1][2] * pos3D[2] + T[1];
                const camZ = R[2][0] * pos3D[0] + R[2][1] * pos3D[1] + R[2][2] * pos3D[2] + T[2];
                
                // 6. Perspective projection
                // img_pos = cam_pos * invtanfov / z
                const divZ = camZ + 1e-7;
                const imgX = (camX * invtanfov) / divZ;
                const imgY = (camY * invtanfov) / divZ;
                
                // 7. Grid sample coordinates (align_corners=False)
                // Convert from NDC space [-1, 1] to pixel coordinates [0, W-1]
                const pixelX = ((imgX + 1.0) * W_img - 1.0) / 2.0;
                const pixelY = ((imgY + 1.0) * H_img - 1.0) / 2.0;
                
                // 8. Bilinear sampling (padding_mode='zeros')
                if (pixelX < 0 || pixelX > W_img - 1 || pixelY < 0 || pixelY > H_img - 1) {
                    // Out of bounds → zeros (padding_mode='zeros')
                    continue;
                }
                
                validPixels++;
                
                const x0 = Math.floor(pixelX);
                const y0 = Math.floor(pixelY);
                const x1 = Math.min(x0 + 1, W_img - 1);
                const y1 = Math.min(y0 + 1, H_img - 1);
                
                const wx = pixelX - x0;
                const wy = pixelY - y0;
                
                // Sample all channels
                for (let c = 0; c < C; c++) {
                    const plane = c * H_img * W_img;
                    
                    const v00 = imgData[plane + y0 * W_img + x0];
                    const v10 = imgData[plane + y0 * W_img + x1];
                    const v01 = imgData[plane + y1 * W_img + x0];
                    const v11 = imgData[plane + y1 * W_img + x1];
                    
                    const val = 
                        v00 * (1 - wx) * (1 - wy) +
                        v10 * wx * (1 - wy) +
                        v01 * (1 - wx) * wy +
                        v11 * wx * wy;
                    
                    // Write to UV map
                    const outIdx = c * H_uv * W_uv + uvIdx;
                    uvFeatures[outIdx] = val;
                }
            }
        }
        
        // 9. Apply UV mask
        // uvmap_mask is boolean: 1 for valid, 0 for invalid
        for (let v = 0; v < H_uv; v++) {
            for (let u = 0; u < W_uv; u++) {
                const uvIdx = v * W_uv + u;
                
                if (!this.uvmap_mask![uvIdx]) {
                    // Invalid region → zero out all channels
                    for (let c = 0; c < C; c++) {
                        uvFeatures[c * H_uv * W_uv + uvIdx] = 0;
                    }
                }
            }
        }
        
        console.log('[InverseTextureMapper] ✅ Mapping complete');
        console.log(`  Valid pixels: ${validPixels}/${H_uv * W_uv} (${((validPixels / (H_uv * W_uv)) * 100).toFixed(1)}%)`);
        console.log(`  Invalid triangles: ${invalidTriangles}`);
        
        return uvFeatures;
    }

    /**
     * Helper: Transform point from world to camera space
     */
    private transformWorldToCamera(worldPos: number[], w2c_R: number[][], w2c_T: number[]): number[] {
        return [
            w2c_R[0][0] * worldPos[0] + w2c_R[0][1] * worldPos[1] + w2c_R[0][2] * worldPos[2] + w2c_T[0],
            w2c_R[1][0] * worldPos[0] + w2c_R[1][1] * worldPos[1] + w2c_R[1][2] * worldPos[2] + w2c_T[1],
            w2c_R[2][0] * worldPos[0] + w2c_R[2][1] * worldPos[1] + w2c_R[2][2] * worldPos[2] + w2c_T[2]
        ];
    }

    /**
     * Get UV map statistics for debugging
     */
    getUVMapStats(): { totalPixels: number; validPixels: number; coverage: number } {
        if (!this.initialized || !this.uvmap_mask) {
            return { totalPixels: 0, validPixels: 0, coverage: 0 };
        }
        
        const totalPixels = this.uvmapSize * this.uvmapSize;
        let validPixels = 0;
        
        for (let i = 0; i < totalPixels; i++) {
            if (this.uvmap_mask[i]) validPixels++;
        }
        
        return {
            totalPixels,
            validPixels,
            coverage: validPixels / totalPixels
        };
    }

    dispose(): void {
        this.vertices = null;
        this.faces = null;
        this.uvmap_f_idx = null;
        this.uvmap_f_bary = null;
        this.uvmap_mask = null;
        this.initialized = false;
        console.log('[InverseTextureMapper] Disposed');
    }
}

/**
 * Utility: Convert HWC image to CHW for compatibility
 */
export function imageHWCtoCHW(
    hwcData: Float32Array,
    height: number,
    width: number,
    channels: number
): ImageFeatures {
    const chwData = new Float32Array(channels * height * width);
    
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            for (let c = 0; c < channels; c++) {
                const hwcIdx = h * width * channels + w * channels + c;
                const chwIdx = c * height * width + h * width + w;
                chwData[chwIdx] = hwcData[hwcIdx];
            }
        }
    }
    
    return {
        data: chwData,
        channels,
        height,
        width
    };
}