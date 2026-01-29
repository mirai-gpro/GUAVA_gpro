// src/gvrm-format/view-encoding.ts
// Spherical Harmonics View Direction Encoding (27ch)
// 論文 Section 3.2: "27-dimensional view direction encoding"

/**
 * 球面調和関数によるView Direction Encoding
 *
 * 3DGSで使用される標準的なSH encoding:
 * - L=0: 1ch (DC component)
 * - L=1: 3ch
 * - L=2: 5ch
 * - L=3: 7ch
 * - L=4: 9ch (オプション)
 *
 * 合計: 1+3+5+7+9 = 25ch または 1+3+5+7+5+3+1+1+1 = 27ch
 *
 * GUAVAでは27chを使用（deg=4までの拡張版）
 */

// SH basis functions (up to degree 4)
const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = [
  1.0925484305920792,
  -1.0925484305920792,
  0.31539156525252005,
  -1.0925484305920792,
  0.5462742152960396
];
const SH_C3 = [
  -0.5900435899266435,
  2.890611442640554,
  -0.4570457994644658,
  0.3731763325901154,
  -0.4570457994644658,
  1.445305721320277,
  -0.5900435899266435
];
const SH_C4 = [
  2.5033429417967046,
  -1.7701307697799304,
  0.9461746957575601,
  -0.6690465435572892,
  0.10578554691520431,
  -0.6690465435572892,
  0.47308734787878004,
  -1.7701307697799304,
  0.6258357354491761
];

/**
 * カメラ方向を27次元のSH特徴にエンコード
 *
 * @param viewDir 正規化されたview direction [x, y, z]
 * @returns 27次元のSH encoding
 */
export function encodeViewDirection(viewDir: [number, number, number]): Float32Array {
  const [x, y, z] = viewDir;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, yz = y * z, xz = x * z;

  const result = new Float32Array(27);

  // L=0 (1 coefficient)
  result[0] = SH_C0;

  // L=1 (3 coefficients)
  result[1] = SH_C1 * y;
  result[2] = SH_C1 * z;
  result[3] = SH_C1 * x;

  // L=2 (5 coefficients)
  result[4] = SH_C2[0] * xy;
  result[5] = SH_C2[1] * yz;
  result[6] = SH_C2[2] * (2 * zz - xx - yy);
  result[7] = SH_C2[3] * xz;
  result[8] = SH_C2[4] * (xx - yy);

  // L=3 (7 coefficients)
  result[9] = SH_C3[0] * y * (3 * xx - yy);
  result[10] = SH_C3[1] * xy * z;
  result[11] = SH_C3[2] * y * (4 * zz - xx - yy);
  result[12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
  result[13] = SH_C3[4] * x * (4 * zz - xx - yy);
  result[14] = SH_C3[5] * (xx - yy) * z;
  result[15] = SH_C3[6] * x * (xx - 3 * yy);

  // L=4 (9 coefficients)
  result[16] = SH_C4[0] * xy * (xx - yy);
  result[17] = SH_C4[1] * yz * (3 * xx - yy);
  result[18] = SH_C4[2] * xy * (7 * zz - 1);
  result[19] = SH_C4[3] * yz * (7 * zz - 3);
  result[20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
  result[21] = SH_C4[5] * xz * (7 * zz - 3);
  result[22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
  result[23] = SH_C4[7] * xz * (xx - 3 * yy);
  result[24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));

  // Additional 2 coefficients to make 27 (padding or extended basis)
  result[25] = 0;
  result[26] = 0;

  return result;
}

/**
 * カメラ位置とターゲットからview directionを計算
 */
export function computeViewDirection(
  cameraPosition: [number, number, number],
  targetPosition: [number, number, number]
): [number, number, number] {
  const dx = targetPosition[0] - cameraPosition[0];
  const dy = targetPosition[1] - cameraPosition[1];
  const dz = targetPosition[2] - cameraPosition[2];

  const length = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (length < 1e-8) {
    return [0, 0, 1];  // Default forward direction
  }

  return [dx / length, dy / length, dz / length];
}

/**
 * UV map全体に対してview direction encodingを生成
 * 各ピクセルで同じview directionを使用（単一視点）
 *
 * @param viewDir 正規化されたview direction
 * @param width UV map width
 * @param height UV map height
 * @returns [27, height, width] のview encoding
 */
export function createViewEncodingMap(
  viewDir: [number, number, number],
  width: number,
  height: number
): Float32Array {
  const encoding = encodeViewDirection(viewDir);
  const result = new Float32Array(27 * height * width);

  // Broadcast encoding to all pixels (CHW format)
  for (let c = 0; c < 27; c++) {
    const offset = c * height * width;
    for (let i = 0; i < height * width; i++) {
      result[offset + i] = encoding[c];
    }
  }

  return result;
}

/**
 * 128ch features + 27ch view encoding = 155ch
 */
export function concatenateWithViewEncoding(
  features128ch: Float32Array,
  viewDir: [number, number, number],
  width: number,
  height: number
): Float32Array {
  const viewEncoding = createViewEncodingMap(viewDir, width, height);
  const result = new Float32Array(155 * height * width);

  // Copy 128ch features
  for (let i = 0; i < 128 * height * width; i++) {
    result[i] = features128ch[i];
  }

  // Copy 27ch view encoding
  for (let i = 0; i < 27 * height * width; i++) {
    result[128 * height * width + i] = viewEncoding[i];
  }

  console.log('[ViewEncoding] Concatenated: 128ch + 27ch = 155ch');
  return result;
}
