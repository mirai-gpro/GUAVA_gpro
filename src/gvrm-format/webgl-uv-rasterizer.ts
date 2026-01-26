/**
 * webgl-uv-rasterizer.ts
 *
 * UV Triangle Mapping のインターフェースとバイナリローダー
 *
 * ファイル形式 (uv_triangle_mapping.bin):
 *   - Header (16 bytes):
 *     - width: uint32 (4 bytes)
 *     - height: uint32 (4 bytes)
 *     - numValid: uint32 (4 bytes)
 *     - reserved: uint32 (4 bytes)
 *   - Data:
 *     - triangleIndices: int32[numValid] (4 * numValid bytes)
 *     - barycentricCoords: float32[numValid * 3] (12 * numValid bytes)
 *     - uvCoords: float32[numValid * 2] (8 * numValid bytes)
 */

export interface UVTriangleMapping {
  width: number;
  height: number;
  numValid: number;
  triangleIndices: Uint32Array;     // [numValid] 各UV pixel が属する三角形のインデックス
  barycentricCoords: Float32Array;  // [numValid, 3] 重心座標
  uvCoords: Float32Array;           // [numValid, 2] UV座標
}

/**
 * uv_triangle_mapping.bin からマッピングデータを読み込む
 */
export async function loadUVTriangleMapping(path: string): Promise<UVTriangleMapping> {
  console.log('[UVTriangleMapping] Loading from:', path);

  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load UV triangle mapping: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  const totalBytes = buffer.byteLength;
  console.log(`[UVTriangleMapping] File size: ${(totalBytes / 1024 / 1024).toFixed(2)} MB`);

  // ヘッダーを読み込み
  const headerView = new DataView(buffer, 0, 16);
  const width = headerView.getUint32(0, true);      // little-endian
  const height = headerView.getUint32(4, true);
  const numValid = headerView.getUint32(8, true);
  const reserved = headerView.getUint32(12, true);

  console.log(`[UVTriangleMapping] Header: ${width}x${height}, numValid=${numValid.toLocaleString()}`);

  // データサイズを計算
  const triangleIndicesSize = numValid * 4;         // int32
  const barycentricCoordsSize = numValid * 3 * 4;   // float32 * 3
  const uvCoordsSize = numValid * 2 * 4;            // float32 * 2
  const expectedSize = 16 + triangleIndicesSize + barycentricCoordsSize + uvCoordsSize;

  console.log(`[UVTriangleMapping] Expected size: ${(expectedSize / 1024 / 1024).toFixed(2)} MB`);

  if (totalBytes < expectedSize) {
    // 代替フォーマットを試す: ヘッダーなし、直接データ
    console.log('[UVTriangleMapping] ⚠️ File smaller than expected, trying alternative format...');
    return loadUVTriangleMappingAlternative(buffer);
  }

  // データを読み込み
  let offset = 16;

  const triangleIndices = new Uint32Array(buffer, offset, numValid);
  offset += triangleIndicesSize;

  const barycentricCoords = new Float32Array(buffer, offset, numValid * 3);
  offset += barycentricCoordsSize;

  const uvCoords = new Float32Array(buffer, offset, numValid * 2);

  // 検証
  let validTriangles = 0;
  let maxTriIdx = 0;
  for (let i = 0; i < Math.min(numValid, 10000); i++) {
    if (triangleIndices[i] < 100000) validTriangles++;
    if (triangleIndices[i] > maxTriIdx) maxTriIdx = triangleIndices[i];
  }

  console.log(`[UVTriangleMapping] ✅ Loaded successfully`);
  console.log(`[UVTriangleMapping]   Resolution: ${width}x${height}`);
  console.log(`[UVTriangleMapping]   Valid pixels: ${numValid.toLocaleString()} (${(numValid / (width * height) * 100).toFixed(1)}%)`);
  console.log(`[UVTriangleMapping]   Max triangle index: ${maxTriIdx}`);

  return {
    width,
    height,
    numValid,
    triangleIndices,
    barycentricCoords,
    uvCoords
  };
}

/**
 * 代替フォーマット: numpy の直接出力形式
 * [triangleIndices, barycentricCoords, uvCoords] を連結したもの
 */
function loadUVTriangleMappingAlternative(buffer: ArrayBuffer): UVTriangleMapping {
  // 512x512 UV マップを仮定
  const width = 512;
  const height = 512;
  const totalPixels = width * height;

  // ファイル全体を float32 として解釈してサイズを推測
  const totalFloats = buffer.byteLength / 4;
  console.log(`[UVTriangleMapping] Alternative format: ${totalFloats} floats`);

  // 想定: [H, W, 6] = triangleIdx(1) + bary(3) + uv(2) = 6 values per pixel
  const expectedFloatsPerPixel = 6;
  const expectedTotalFloats = totalPixels * expectedFloatsPerPixel;

  if (Math.abs(totalFloats - expectedTotalFloats) < 1000) {
    console.log('[UVTriangleMapping] Format: [H, W, 6] per-pixel data');

    const data = new Float32Array(buffer);
    const triangleIndices = new Uint32Array(totalPixels);
    const barycentricCoords = new Float32Array(totalPixels * 3);
    const uvCoords = new Float32Array(totalPixels * 2);

    let numValid = 0;
    for (let i = 0; i < totalPixels; i++) {
      const baseIdx = i * 6;
      const triIdx = Math.round(data[baseIdx]);

      if (triIdx >= 0 && triIdx < 100000) {
        triangleIndices[numValid] = triIdx;
        barycentricCoords[numValid * 3 + 0] = data[baseIdx + 1];
        barycentricCoords[numValid * 3 + 1] = data[baseIdx + 2];
        barycentricCoords[numValid * 3 + 2] = data[baseIdx + 3];
        uvCoords[numValid * 2 + 0] = data[baseIdx + 4];
        uvCoords[numValid * 2 + 1] = data[baseIdx + 5];
        numValid++;
      }
    }

    console.log(`[UVTriangleMapping] ✅ Parsed ${numValid.toLocaleString()} valid pixels`);

    return {
      width,
      height,
      numValid,
      triangleIndices: triangleIndices.slice(0, numValid),
      barycentricCoords: barycentricCoords.slice(0, numValid * 3),
      uvCoords: uvCoords.slice(0, numValid * 2)
    };
  }

  throw new Error(`Unknown UV triangle mapping format: ${totalFloats} floats`);
}

/**
 * WebGL を使ってリアルタイムで UV Triangle Mapping を生成
 * (将来の実装用プレースホルダー)
 */
export class WebGLUVRasterizer {
  // TODO: メッシュデータから UV mapping を GPU で生成
  static async generate(
    vertices: Float32Array,
    faces: Uint32Array,
    uvCoords: Float32Array,
    width: number,
    height: number
  ): Promise<UVTriangleMapping> {
    throw new Error('WebGL UV Rasterizer not yet implemented. Use loadUVTriangleMapping() with pre-computed data.');
  }
}
