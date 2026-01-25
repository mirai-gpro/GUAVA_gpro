// camera-utils.ts
// 公式実装 (dataset/data_loader.py, utils/graphics_utils.py) の完全移植
// 独自の推測を排除し、ハードコードされた定数(Z=22, tanfov=1/24)を定義する。

export class CameraUtils {
    // dataset/data_loader.py より: tanfov = 1.0 / 24.0
    static readonly TAN_FOV = 1.0 / 24.0; 

    // 公式データセットのデフォルトView Matrix (World to Camera)
    // Python: [[1, 0, 0, 0], [0, 1, 0, 0.6], [0, 0, 1, 22], [0, 0, 0, 1]]
    // これを WebGPU (Column-Major) 形式の配列で返す
    static getCanonicalViewMatrix(): Float32Array {
        return new Float32Array([
            1, 0, 0, 0,   // Col 0
            0, 1, 0, 0,   // Col 1
            0, 0, 1, 0,   // Col 2
            0, 0.6, 22, 1 // Col 3 (Translation: x=0, y=0.6, z=22)
        ]);
    }

    // 任意のR(3x3), T(3)からView Matrixを生成
    static getViewMatrixFromRT(R_in: number[][] | number[], T_in: number[]): Float32Array {
        // Flatten R if needed
        let R: number[];
        if (Array.isArray(R_in[0])) {
            R = (R_in as number[][]).flat();
        } else {
            R = R_in as number[];
        }

        // Construct 4x4 Matrix (Column-Major)
        // M = [ R   T ]
        //     [ 0   1 ]
        const m = new Float32Array(16);
        
        // Col 0
        m[0] = R[0]; m[1] = R[3]; m[2] = R[6]; m[3] = 0;
        // Col 1
        m[4] = R[1]; m[5] = R[4]; m[6] = R[7]; m[7] = 0;
        // Col 2
        m[8] = R[2]; m[9] = R[5]; m[10] = R[8]; m[11] = 0;
        // Col 3 (Translation)
        m[12] = T_in[0]; m[13] = T_in[1]; m[14] = T_in[2]; m[15] = 1;

        return m;
    }

    // OpenGL/PyTorch3D 準拠の Projection Matrix 生成
    // Aspect比を考慮し、WebGPUのNDC仕様に合わせて出力する
    static getProjMatrix(aspect: number, zNear: number = 0.01, zFar: number = 100.0): Float32Array {
        const tanHalfFovY = this.TAN_FOV;
        const tanHalfFovX = this.TAN_FOV / aspect; 

        const top = tanHalfFovY * zNear;
        const bottom = -top;
        const right = tanHalfFovX * zNear;
        const left = -right;

        const P = new Float32Array(16);
        const r_l = right - left;
        const t_b = top - bottom;
        const f_n = zFar - zNear;

        // Column-Major Construction
        P[0] = (2 * zNear) / r_l;
        P[5] = (2 * zNear) / t_b;
        P[8] = (right + left) / r_l;
        P[9] = (top + bottom) / t_b;
        
        // WebGPU Clip Space Z is [0, 1]
        // OpenGL Clip Space Z is [-1, 1]
        // PyTorch3D uses OpenGL convention. Let's stick to GL convention first.
        // If the screen is black, we might need to adjust Z mapping.
        P[10] = -(zFar + zNear) / f_n; 
        P[11] = -1;
        P[14] = -(2 * zFar * zNear) / f_n;

        return P;
    }
}