// gaussian-splatting-shader.ts
// Gaussian Splatting用のWebGLシェーダー

export const gaussianVertexShader = `
precision highp float;

// 頂点属性
attribute vec3 position;      // Gaussian中心位置
attribute vec4 rotation;      // 回転（quaternion）
attribute vec3 scale;         // スケール
attribute float opacity;      // 不透明度
attribute vec4 latent0;       // latent feature 0-3
attribute vec4 latent1;       // latent feature 4-7
attribute vec4 latent2;       // latent feature 8-11
attribute vec4 latent3;       // latent feature 12-15
attribute vec4 latent4;       // latent feature 16-19
attribute vec4 latent5;       // latent feature 20-23
attribute vec4 latent6;       // latent feature 24-27
attribute vec4 latent7;       // latent feature 28-31

// Uniforms
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform int tileIndex;        // 現在のタイル（0-7）

// Varyings
varying float vOpacity;
varying vec4 vLatent0;
varying vec4 vLatent1;

void main() {
    // Gaussianの中心をスクリーン空間に投影
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // スケールをビューポートサイズに変換
    vec2 screenScale = scale.xy * 100.0; // 適切なスケーリング
    gl_PointSize = max(screenScale.x, screenScale.y);
    
    // 不透明度を渡す
    vOpacity = opacity;
    
    // タイルインデックスに応じてlatent featureを選択
    // 8タイルで32チャンネルをカバー（各タイル4チャンネル）
    if (tileIndex == 0) {
        vLatent0 = latent0;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 1) {
        vLatent0 = latent1;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 2) {
        vLatent0 = latent2;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 3) {
        vLatent0 = latent3;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 4) {
        vLatent0 = latent4;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 5) {
        vLatent0 = latent5;
        vLatent1 = vec4(0.0);
    } else if (tileIndex == 6) {
        vLatent0 = latent6;
        vLatent1 = vec4(0.0);
    } else {
        vLatent0 = latent7;
        vLatent1 = vec4(0.0);
    }
}
`;

export const gaussianFragmentShader = `
precision highp float;

varying float vOpacity;
varying vec4 vLatent0;
varying vec4 vLatent1;

void main() {
    // Gaussianの形状（円形）
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Gaussian falloff
    float alpha = exp(-4.0 * dist * dist) * vOpacity;
    
    if (alpha < 0.01) discard;
    
    // latent featuresを出力（RGBA = 4チャンネル）
    gl_FragColor = vec4(vLatent0.rgb, alpha);
}
`;