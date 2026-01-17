// debug-visualizer.ts
// GUAVA/gaussian-vrm統合のデバッグ可視化ツール

export class DebugVisualizer {
  private container: HTMLElement;

  constructor(containerId: string = 'debug-panel') {
    let container = document.getElementById(containerId);
    if (!container) {
      container = document.createElement('div');
      container.id = containerId;
      container.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        width: 300px;
        max-height: 80vh;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.8);
        color: #0f0;
        padding: 15px;
        font-family: monospace;
        font-size: 12px;
        border-radius: 8px;
        z-index: 9999;
      `;
      document.body.appendChild(container);
    }
    this.container = container;
  }

  /**
   * 32chのcoarse feature mapを8枚の画像として可視化
   */
  visualizeCoarseFeatureMap(coarseFM: Float32Array, width: number = 256, height: number = 256) {
    console.log('[DebugViz] Visualizing coarse feature map...');

    const section = document.createElement('div');
    section.innerHTML = '<h3 style="color:#0ff; margin-top:0;">Coarse Feature Map (32ch)</h3>';

    for (let tile = 0; tile < 8; tile++) {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      canvas.style.cssText = `
        width: 128px;
        height: 128px;
        border: 1px solid #0f0;
        margin: 5px;
        display: inline-block;
      `;

      const ctx = canvas.getContext('2d')!;
      const imageData = ctx.createImageData(width, height);
      const pixels = imageData.data;

      // 4チャンネル分を可視化
      const baseOffset = tile * 4 * width * height;

      let minVal = Infinity;
      let maxVal = -Infinity;

      // 値域を取得
      for (let i = 0; i < width * height; i++) {
        for (let c = 0; c < 3; c++) {
          const val = coarseFM[baseOffset + c * width * height + i];
          minVal = Math.min(minVal, val);
          maxVal = Math.max(maxVal, val);
        }
      }

      console.log(`[DebugViz] Tile ${tile}: min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}`);

      // 正規化して描画
      const range = maxVal - minVal;
      for (let i = 0; i < width * height; i++) {
        const r = coarseFM[baseOffset + 0 * width * height + i];
        const g = coarseFM[baseOffset + 1 * width * height + i];
        const b = coarseFM[baseOffset + 2 * width * height + i];

        pixels[i * 4 + 0] = range > 0 ? ((r - minVal) / range) * 255 : 128;
        pixels[i * 4 + 1] = range > 0 ? ((g - minVal) / range) * 255 : 128;
        pixels[i * 4 + 2] = range > 0 ? ((b - minVal) / range) * 255 : 128;
        pixels[i * 4 + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);

      const label = document.createElement('div');
      label.textContent = `Tile ${tile} [${minVal.toFixed(2)}, ${maxVal.toFixed(2)}]`;
      label.style.cssText = 'color: #0f0; font-size: 10px; text-align: center;';

      const wrapper = document.createElement('div');
      wrapper.style.cssText = 'display: inline-block; margin: 5px;';
      wrapper.appendChild(canvas);
      wrapper.appendChild(label);

      section.appendChild(wrapper);
    }

    this.container.appendChild(section);
  }

  /**
   * Template Decoder出力の統計情報を表示
   */
  visualizeTemplateOutput(output: {
    latent32ch: Float32Array;
    opacity: Float32Array;
    scale: Float32Array;
    rotation: Float32Array;
  }) {
    console.log('[DebugViz] Visualizing template output...');

    const section = document.createElement('div');
    section.innerHTML = '<h3 style="color:#0ff;">Template Decoder Output</h3>';

    const stats = [
      { name: 'Latent (32ch)', data: output.latent32ch, channels: 32 },
      { name: 'Opacity', data: output.opacity, channels: 1 },
      { name: 'Scale', data: output.scale, channels: 3 },
      { name: 'Rotation', data: output.rotation, channels: 4 }
    ];

    stats.forEach(({ name, data, channels }) => {
      const numVertices = data.length / channels;
      
      let min = Infinity;
      let max = -Infinity;
      let sum = 0;

      for (let i = 0; i < data.length; i++) {
        const val = data[i];
        min = Math.min(min, val);
        max = Math.max(max, val);
        sum += val;
      }

      const mean = sum / data.length;
      
      let variance = 0;
      for (let i = 0; i < data.length; i++) {
        variance += Math.pow(data[i] - mean, 2);
      }
      const std = Math.sqrt(variance / data.length);

      const div = document.createElement('div');
      div.style.cssText = 'margin: 10px 0; padding: 10px; background: rgba(0,255,0,0.1); border-radius: 4px;';
      div.innerHTML = `
        <div style="color:#0ff; font-weight:bold;">${name}</div>
        <div>Vertices: ${numVertices}</div>
        <div>Channels: ${channels}</div>
        <div>Min: ${min.toFixed(4)}</div>
        <div>Max: ${max.toFixed(4)}</div>
        <div>Mean: ${mean.toFixed(4)}</div>
        <div>Std: ${std.toFixed(4)}</div>
      `;

      section.appendChild(div);
    });

    this.container.appendChild(section);
  }

  /**
   * ボーン割り当て統計を可視化
   */
  visualizeBoneAssignment(boneStats: Record<number, number>, totalVertices: number) {
    console.log('[DebugViz] Visualizing bone assignment...');

    const section = document.createElement('div');
    section.innerHTML = '<h3 style="color:#0ff;">Bone Assignment Stats</h3>';

    const boneNames: Record<number, string> = {
      0: 'Hips',
      3: 'Spine1',
      9: 'Chest',
      12: 'Neck',
      15: 'Head',
      16: 'L_Shoulder',
      17: 'R_Shoulder',
      22: 'Jaw'
    };

    const canvas = document.createElement('canvas');
    canvas.width = 280;
    canvas.height = 200;
    canvas.style.cssText = 'border: 1px solid #0f0; margin: 10px 0;';

    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const maxCount = Math.max(...Object.values(boneStats));
    const barWidth = 30;
    const spacing = 35;
    const startX = 10;

    Object.entries(boneStats).forEach(([boneIdxStr, count], i) => {
      const boneIdx = parseInt(boneIdxStr);
      const barHeight = (count / maxCount) * 150;
      const x = startX + i * spacing;
      const y = 180 - barHeight;

      // バー
      ctx.fillStyle = boneIdx === 22 ? '#f0f' : '#0f0';
      ctx.fillRect(x, y, barWidth, barHeight);

      // ラベル
      ctx.fillStyle = '#fff';
      ctx.font = '10px monospace';
      ctx.save();
      ctx.translate(x + 15, 195);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(boneNames[boneIdx] || `B${boneIdx}`, 0, 0);
      ctx.restore();

      // 数値
      ctx.fillStyle = '#0ff';
      ctx.fillText(count.toString(), x, y - 5);
    });

    section.appendChild(canvas);

    // 詳細テキスト
    const details = document.createElement('div');
    details.style.cssText = 'margin-top: 10px;';
    
    Object.entries(boneStats).forEach(([boneIdxStr, count]) => {
      const boneIdx = parseInt(boneIdxStr);
      const percentage = ((count / totalVertices) * 100).toFixed(1);
      const div = document.createElement('div');
      div.style.cssText = `color: ${boneIdx === 22 ? '#f0f' : '#0f0'};`;
      div.textContent = `${boneNames[boneIdx]}: ${count} (${percentage}%)`;
      details.appendChild(div);
    });

    section.appendChild(details);
    this.container.appendChild(section);
  }

  /**
   * フレームごとのパフォーマンス統計
   */
  logFrameStats(frameCount: number, refinedRgb: Float32Array | null) {
    if (frameCount > 5) return; // 最初の5フレームのみ

    const section = document.createElement('div');
    section.style.cssText = 'margin: 10px 0; padding: 10px; background: rgba(255,255,0,0.1); border-radius: 4px;';
    
    if (refinedRgb) {
      const sample = Array.from(refinedRgb.slice(0, 100));
      const min = Math.min(...sample);
      const max = Math.max(...sample);
      const avg = sample.reduce((a, b) => a + b, 0) / sample.length;

      section.innerHTML = `
        <div style="color:#ff0; font-weight:bold;">Frame ${frameCount}</div>
        <div>Refined RGB: ${refinedRgb.length} values</div>
        <div>Min: ${min.toFixed(4)}</div>
        <div>Max: ${max.toFixed(4)}</div>
        <div>Avg: ${avg.toFixed(4)}</div>
      `;
    } else {
      section.innerHTML = `
        <div style="color:#f00; font-weight:bold;">Frame ${frameCount}</div>
        <div>⚠️ No refined RGB data</div>
      `;
    }

    this.container.appendChild(section);
  }

  clear() {
    this.container.innerHTML = '';
  }
}

// 使用例:
// const debugViz = new DebugVisualizer();
// 
// // Template Decoder出力の可視化
// debugViz.visualizeTemplateOutput(templateOutput);
//
// // Coarse feature mapの可視化
// debugViz.visualizeCoarseFeatureMap(coarseFM);
//
// // ボーン割り当ての可視化
// debugViz.visualizeBoneAssignment(boneStats, totalVertices);
//
// // フレーム統計
// debugViz.logFrameStats(frameCount, refinedRgb);