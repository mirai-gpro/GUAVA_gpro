// src/gvrm-format/uv-triangle-mapping.ts
// UV Triangle Mapping の読み込みと管理

/**
 * UV Triangle Mapping データ構造
 * 各UVピクセルに対応する三角形と重心座標
 */
export interface UVTriangleMapping {
  width: number;
  height: number;
  numValid: number;
  
  // 各有効ピクセルのデータ
  uvCoords: Uint16Array;        // [numValid * 2] (u, v) ピクセル座標
  triangleIndices: Uint32Array;  // [numValid] 三角形インデックス
  barycentricCoords: Float32Array; // [numValid * 3] 重心座標 (u, v, w)
}

/**
 * UV Triangle Mapping ローダー
 */
export class UVTriangleMappingLoader {
  /**
   * バイナリファイルからUV Triangle Mappingをロード
   * 
   * バイナリフォーマット:
   *   Header (20 bytes):
   *     - magic: uint32 (0x5554524D = "UTRM")
   *     - version: uint32
   *     - width: uint32
   *     - height: uint32
   *     - num_valid: uint32
   *   
   *   Data (各有効ピクセル、20 bytes):
   *     - u: uint16 (ピクセルx座標)
   *     - v: uint16 (ピクセルy座標)
   *     - triangle_idx: uint32
   *     - bary_u: float32
   *     - bary_v: float32
   *     - bary_w: float32
   */
  static async load(url: string): Promise<UVTriangleMapping> {
    console.log('[UVTriangleMapping] Loading from:', url);
    
    try {
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
      }
      
      const buffer = await response.arrayBuffer();
      const view = new DataView(buffer);
      
      // ヘッダーを読み込み
      let offset = 0;
      
      const magic = view.getUint32(offset, true);
      offset += 4;
      
      if (magic !== 0x5554524D) {
        throw new Error(`Invalid magic number: 0x${magic.toString(16)}`);
      }
      
      const version = view.getUint32(offset, true);
      offset += 4;
      
      const width = view.getUint32(offset, true);
      offset += 4;
      
      const height = view.getUint32(offset, true);
      offset += 4;
      
      const numValid = view.getUint32(offset, true);
      offset += 4;
      
      console.log('[UVTriangleMapping] Header:', {
        version,
        width,
        height,
        numValid: numValid.toLocaleString(),
        coverage: ((numValid / (width * height)) * 100).toFixed(1) + '%'
      });
      
      // データ配列を確保
      const uvCoords = new Uint16Array(numValid * 2);
      const triangleIndices = new Uint32Array(numValid);
      const barycentricCoords = new Float32Array(numValid * 3);
      
      // データを読み込み
      for (let i = 0; i < numValid; i++) {
        // UV座標 (uint16 x 2)
        uvCoords[i * 2 + 0] = view.getUint16(offset, true);
        offset += 2;
        uvCoords[i * 2 + 1] = view.getUint16(offset, true);
        offset += 2;
        
        // 三角形インデックス (uint32)
        triangleIndices[i] = view.getUint32(offset, true);
        offset += 4;
        
        // 重心座標 (float32 x 3)
        barycentricCoords[i * 3 + 0] = view.getFloat32(offset, true);
        offset += 4;
        barycentricCoords[i * 3 + 1] = view.getFloat32(offset, true);
        offset += 4;
        barycentricCoords[i * 3 + 2] = view.getFloat32(offset, true);
        offset += 4;
      }
      
      console.log('[UVTriangleMapping] ✅ Loaded successfully');
      console.log('[UVTriangleMapping] Data size:', {
        uvCoords: `${(uvCoords.byteLength / 1024).toFixed(1)} KB`,
        triangleIndices: `${(triangleIndices.byteLength / 1024).toFixed(1)} KB`,
        barycentricCoords: `${(barycentricCoords.byteLength / 1024).toFixed(1)} KB`,
        total: `${(buffer.byteLength / (1024 * 1024)).toFixed(2)} MB`
      });
      
      // サンプルデータを表示（デバッグ用）
      if (numValid > 0) {
        console.log('[UVTriangleMapping] Sample data (first pixel):', {
          uv: [uvCoords[0], uvCoords[1]],
          triangleIdx: triangleIndices[0],
          barycentric: [
            barycentricCoords[0].toFixed(3),
            barycentricCoords[1].toFixed(3),
            barycentricCoords[2].toFixed(3)
          ]
        });
      }
      
      return {
        width,
        height,
        numValid,
        uvCoords,
        triangleIndices,
        barycentricCoords
      };
      
    } catch (error) {
      console.error('[UVTriangleMapping] ❌ Failed to load:', error);
      throw error;
    }
  }
  
  /**
   * マッピングの統計情報を取得
   */
  static getStats(mapping: UVTriangleMapping): {
    coverage: number;
    memoryUsage: number;
    triangleRange: [number, number];
  } {
    const coverage = (mapping.numValid / (mapping.width * mapping.height)) * 100;
    
    const memoryUsage = 
      mapping.uvCoords.byteLength +
      mapping.triangleIndices.byteLength +
      mapping.barycentricCoords.byteLength;
    
    let minTriIdx = Infinity;
    let maxTriIdx = -Infinity;
    
    for (let i = 0; i < mapping.numValid; i++) {
      const idx = mapping.triangleIndices[i];
      if (idx < minTriIdx) minTriIdx = idx;
      if (idx > maxTriIdx) maxTriIdx = idx;
    }
    
    return {
      coverage,
      memoryUsage,
      triangleRange: [minTriIdx, maxTriIdx]
    };
  }
}