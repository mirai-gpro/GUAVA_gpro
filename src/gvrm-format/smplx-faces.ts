// src/gvrm-format/smplx-faces.ts
// SMPLX標準三角形構造
// 
// SMPLXモデルは固定のトポロジーを持つ
// 頂点数: 10,475
// 三角形数: 20,908
//
// 参考: https://github.com/vchoutas/smplx

/**
 * SMPLX Body の標準三角形構造
 * 
 * 注意: これは簡略版です
 * 実際の実装では完全なSMPLX facesデータが必要
 */
export const SMPLX_FACES_SUBSET = new Uint32Array([
    // 顔の三角形（サンプル）
    0, 1, 2,
    1, 3, 2,
    2, 3, 4,
    
    // 胴体の三角形（サンプル）
    100, 101, 102,
    101, 103, 102,
    
    // ... 実際には20,908個の三角形が必要
    // これは /assets/smplx_faces.bin として提供すべき
]);

/**
 * SMPLX Facesをロードする
 * 
 * 実際の実装では:
 * 1. /assets/smplx_faces.bin をロード
 * 2. または smplx_faces.json をロード
 * 3. geometry_data.bin に統合
 */
export async function loadSMPLXFaces(basePath: string = '/assets'): Promise<Uint32Array> {
    try {
        // オプション1: バイナリファイル
        const response = await fetch(`${basePath}/smplx_faces.bin`);
        if (response.ok) {
            const buffer = await response.arrayBuffer();
            return new Uint32Array(buffer);
        }
    } catch (e) {
        console.warn('[SMPLX] Failed to load smplx_faces.bin');
    }
    
    try {
        // オプション2: JSONファイル
        const response = await fetch(`${basePath}/smplx_faces.json`);
        if (response.ok) {
            const data = await response.json();
            return new Uint32Array(data.faces);
        }
    } catch (e) {
        console.warn('[SMPLX] Failed to load smplx_faces.json');
    }
    
    // フォールバック: 最小限のダミーデータ
    console.error('[SMPLX] No faces data found! Using minimal dummy data');
    return SMPLX_FACES_SUBSET;
}

/**
 * SMPLX UV座標をロードする
 * 
 * SMPLXモデルには標準のUV座標が含まれている
 */
export async function loadSMPLXUVCoords(basePath: string = '/assets'): Promise<Float32Array> {
    try {
        // オプション1: バイナリファイル
        const response = await fetch(`${basePath}/smplx_uv_coords.bin`);
        if (response.ok) {
            const buffer = await response.arrayBuffer();
            return new Float32Array(buffer);
        }
    } catch (e) {
        console.warn('[SMPLX] Failed to load smplx_uv_coords.bin');
    }
    
    try {
        // オプション2: JSONファイル
        const response = await fetch(`${basePath}/smplx_uv_coords.json`);
        if (response.ok) {
            const data = await response.json();
            return new Float32Array(data.uv_coords);
        }
    } catch (e) {
        console.warn('[SMPLX] Failed to load smplx_uv_coords.json');
    }
    
    // フォールバック: デフォルトUV（平面展開）
    console.error('[SMPLX] No UV coords found! Using planar projection fallback');
    const uvCoords = new Float32Array(10475 * 2);  // SMPLX頂点数
    
    // 簡易的な平面投影（Y-Z平面）
    // これは暫定的なもので、実際のSMPLX UVを使うべき
    for (let i = 0; i < 10475; i++) {
        // デフォルト値（0.5, 0.5）に設定
        uvCoords[i * 2 + 0] = 0.5;
        uvCoords[i * 2 + 1] = 0.5;
    }
    
    return uvCoords;
}
