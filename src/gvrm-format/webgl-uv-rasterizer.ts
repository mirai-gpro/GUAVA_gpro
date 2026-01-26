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
  const totalFloats = totalBytes / 4;
  console.log(`[UVTriangleMapping] File size: ${(totalBytes / 1024 / 1024).toFixed(2)} MB (${totalFloats.toLocaleString()} floats)`);

  // ファイルサイズから形式を推定
  // 20MB = 5,242,880 floats = 1024 * 1024 * 5 → 1024x1024 UV map with 5 values/pixel
  // 6MB = 1,572,864 floats = 512 * 512 * 6 → 512x512 UV map with 6 values/pixel

  // 1024x1024 x 5 チェック (triIdx + bary3 + pixelIdx or similar)
  if (Math.abs(totalFloats - 1024 * 1024 * 5) < 1000) {
    console.log('[UVTriangleMapping] Detected format: 1024x1024 x 5 (per-pixel)');
    return loadPerPixelFormat(buffer, 1024, 1024, 5);
  }

  // 512x512 x 6 チェック
  if (Math.abs(totalFloats - 512 * 512 * 6) < 1000) {
    console.log('[UVTriangleMapping] Detected format: 512x512 x 6 (per-pixel)');
    return loadPerPixelFormat(buffer, 512, 512, 6);
  }

  // 512x512 x 5 チェック
  if (Math.abs(totalFloats - 512 * 512 * 5) < 1000) {
    console.log('[UVTriangleMapping] Detected format: 512x512 x 5 (per-pixel)');
    return loadPerPixelFormat(buffer, 512, 512, 5);
  }

  // ヘッダー付きフォーマットを試す
  const headerView = new DataView(buffer, 0, 16);
  const width = headerView.getUint32(0, true);
  const height = headerView.getUint32(4, true);
  const numValid = headerView.getUint32(8, true);

  // ヘッダーが妥当かチェック
  if (width > 0 && width <= 4096 && height > 0 && height <= 4096 && numValid > 0 && numValid <= width * height) {
    const expectedSize = 16 + numValid * (4 + 12 + 8);
    if (Math.abs(totalBytes - expectedSize) < 1000) {
      console.log(`[UVTriangleMapping] Detected format: Header + packed data`);
      console.log(`[UVTriangleMapping] Header: ${width}x${height}, numValid=${numValid.toLocaleString()}`);
      return loadHeaderFormat(buffer, width, height, numValid);
    }
  }

  // Numpy sparse format: 有効ピクセルのみを保存
  // numValid pixels * (y, x, triIdx, bary0, bary1, bary2) = 6 values per valid pixel
  const possibleNumValid = Math.floor(totalFloats / 6);
  if (possibleNumValid > 10000 && possibleNumValid < 1000000) {
    console.log(`[UVTriangleMapping] Trying sparse format: ~${possibleNumValid.toLocaleString()} valid pixels`);
    return loadSparseFormat(buffer, 1024, 1024);
  }

  throw new Error(`Unknown UV triangle mapping format: ${totalFloats.toLocaleString()} floats, ${totalBytes} bytes`);
}

/**
 * Per-pixel format: [H, W, N] where each pixel has N values
 */
function loadPerPixelFormat(buffer: ArrayBuffer, width: number, height: number, valuesPerPixel: number): UVTriangleMapping {
  const data = new Float32Array(buffer);
  const totalPixels = width * height;

  const triangleIndices = new Uint32Array(totalPixels);
  const barycentricCoords = new Float32Array(totalPixels * 3);
  const uvCoords = new Float32Array(totalPixels * 2);

  let numValid = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIdx = y * width + x;
      const baseIdx = pixelIdx * valuesPerPixel;

      // First value is triangle index (-1 or negative = invalid)
      const triIdx = data[baseIdx];

      if (triIdx >= 0 && triIdx < 100000) {
        triangleIndices[numValid] = Math.round(triIdx);

        // Barycentric coordinates (next 3 values)
        barycentricCoords[numValid * 3 + 0] = data[baseIdx + 1];
        barycentricCoords[numValid * 3 + 1] = data[baseIdx + 2];
        barycentricCoords[numValid * 3 + 2] = data[baseIdx + 3];

        // UV coords (pixel position normalized or from data)
        if (valuesPerPixel >= 6) {
          uvCoords[numValid * 2 + 0] = data[baseIdx + 4];
          uvCoords[numValid * 2 + 1] = data[baseIdx + 5];
        } else {
          // Compute UV from pixel position
          uvCoords[numValid * 2 + 0] = x;
          uvCoords[numValid * 2 + 1] = y;
        }

        numValid++;
      }
    }
  }

  console.log(`[UVTriangleMapping] ✅ Loaded ${numValid.toLocaleString()} valid pixels from ${totalPixels.toLocaleString()} total`);
  console.log(`[UVTriangleMapping]   Coverage: ${(numValid / totalPixels * 100).toFixed(1)}%`);

  // サンプル表示
  if (numValid > 0) {
    console.log(`[UVTriangleMapping]   Sample: tri[0]=${triangleIndices[0]}, bary=[${barycentricCoords[0].toFixed(3)}, ${barycentricCoords[1].toFixed(3)}, ${barycentricCoords[2].toFixed(3)}]`);
  }

  return {
    width,
    height,
    numValid,
    triangleIndices: triangleIndices.slice(0, numValid),
    barycentricCoords: barycentricCoords.slice(0, numValid * 3),
    uvCoords: uvCoords.slice(0, numValid * 2)
  };
}

/**
 * Header format: 16-byte header followed by packed data
 */
function loadHeaderFormat(buffer: ArrayBuffer, width: number, height: number, numValid: number): UVTriangleMapping {
  let offset = 16;

  const triangleIndices = new Uint32Array(buffer, offset, numValid);
  offset += numValid * 4;

  const barycentricCoords = new Float32Array(buffer, offset, numValid * 3);
  offset += numValid * 12;

  const uvCoords = new Float32Array(buffer, offset, numValid * 2);

  console.log(`[UVTriangleMapping] ✅ Loaded with header format`);

  return { width, height, numValid, triangleIndices, barycentricCoords, uvCoords };
}

/**
 * Sparse format: only valid pixels stored as [y, x, triIdx, bary0, bary1, bary2]
 */
function loadSparseFormat(buffer: ArrayBuffer, defaultWidth: number, defaultHeight: number): UVTriangleMapping {
  const data = new Float32Array(buffer);
  const numValid = Math.floor(data.length / 6);

  const triangleIndices = new Uint32Array(numValid);
  const barycentricCoords = new Float32Array(numValid * 3);
  const uvCoords = new Float32Array(numValid * 2);

  let maxX = 0, maxY = 0;
  let validCount = 0;

  for (let i = 0; i < numValid; i++) {
    const baseIdx = i * 6;
    const y = data[baseIdx + 0];
    const x = data[baseIdx + 1];
    const triIdx = data[baseIdx + 2];
    const bary0 = data[baseIdx + 3];
    const bary1 = data[baseIdx + 4];
    const bary2 = data[baseIdx + 5];

    if (triIdx >= 0 && triIdx < 100000 && x >= 0 && y >= 0) {
      triangleIndices[validCount] = Math.round(triIdx);
      barycentricCoords[validCount * 3 + 0] = bary0;
      barycentricCoords[validCount * 3 + 1] = bary1;
      barycentricCoords[validCount * 3 + 2] = bary2;
      uvCoords[validCount * 2 + 0] = x;
      uvCoords[validCount * 2 + 1] = y;

      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      validCount++;
    }
  }

  const width = Math.max(defaultWidth, Math.ceil(maxX) + 1);
  const height = Math.max(defaultHeight, Math.ceil(maxY) + 1);

  console.log(`[UVTriangleMapping] ✅ Loaded sparse format: ${validCount.toLocaleString()} valid pixels`);
  console.log(`[UVTriangleMapping]   Detected resolution: ${width}x${height}`);

  return {
    width,
    height,
    numValid: validCount,
    triangleIndices: triangleIndices.slice(0, validCount),
    barycentricCoords: barycentricCoords.slice(0, validCount * 3),
    uvCoords: uvCoords.slice(0, validCount * 2)
  };
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
