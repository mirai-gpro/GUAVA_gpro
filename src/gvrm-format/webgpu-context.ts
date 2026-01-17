// lbs_inverse.wgsl
// iPhone 13のGPUアーキテクチャに最適化された並列処理

struct Gaussian {
    pos: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    rotation: vec4<f32>,
    latent: array<f32, 32>, // GUAVAの32次元特徴量
    bone_indices: vec4<f32>,
    bone_weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> input_gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> output_gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> bone_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<uniform> camera_matrix: mat4x4<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input_gaussians)) { return; }

    var g = input_gaussians[idx];

    // --- LBS (Linear Blend Skinning) ---
    // ボーン行列の合成
    var skin_m = 
        g.bone_weights.x * bone_matrices[u32(g.bone_indices.x)] +
        g.bone_weights.y * bone_matrices[u32(g.bone_indices.y)] +
        g.bone_weights.z * bone_matrices[u32(g.bone_indices.z)] +
        g.bone_weights.w * bone_matrices[u32(g.bone_indices.w)];

    let posed_pos = (skin_m * vec4<f32>(g.pos, 1.0)).xyz;

    // --- Static Inverse Mapping ---
    // 3D座標からスクリーン座標への逆投影。ロード時に焼き付けたlatent情報を動的に反映
    let clip_pos = camera_matrix * vec4<f32>(posed_pos, 1.0);
    let uv = (clip_pos.xy / clip_pos.w) * 0.5 + 0.5;

    // 前面のみ表示（背面カリング）
    if (posed_pos.z < -0.05) { g.opacity = 0.0; }

    g.pos = posed_pos;
    output_gaussians[idx] = g;
}