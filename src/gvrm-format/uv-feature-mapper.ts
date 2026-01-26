/**
 * uv-feature-mapper.ts
 *
 * Maps image-space appearance features to UV space
 *
 * Pipeline:
 * 1. For each valid UV pixel, get 3D position via barycentric interpolation
 * 2. Project 3D position to image space using camera matrix
 * 3. Sample appearance map at projected coordinates
 * 4. Output UV feature map [C, H_uv, W_uv]
 */

import type { UVTriangleMapping } from './webgl-uv-rasterizer';

export interface UVFeatureMapperConfig {
  uvWidth: number;
  uvHeight: number;
  imageWidth: number;
  imageHeight: number;
}

export interface CameraParams {
  // Intrinsics
  focalLength: number;  // or fx, fy
  principalPoint: [number, number];  // cx, cy

  // Extrinsics (world to camera)
  rotation: Float32Array;  // 3x3
  translation: Float32Array;  // 3
}

/**
 * UV Feature Mapper
 * Maps appearance features from image space to UV space
 */
export class UVFeatureMapper {
  private config: UVFeatureMapperConfig;

  constructor(config: Partial<UVFeatureMapperConfig> = {}) {
    this.config = {
      uvWidth: config.uvWidth ?? 512,
      uvHeight: config.uvHeight ?? 512,
      imageWidth: config.imageWidth ?? 518,
      imageHeight: config.imageHeight ?? 518,
    };
  }

  /**
   * Map appearance features to UV space
   *
   * @param appearanceMap Image features [C, H, W] in CHW format
   * @param vertices Mesh vertices [V, 3]
   * @param faces Triangle indices [F, 3]
   * @param uvMapping Pre-computed UV triangle mapping
   * @param camera Camera parameters (optional, uses canonical if not provided)
   * @returns UV feature map [C, H_uv, W_uv]
   */
  mapToUV(
    appearanceMap: Float32Array,
    vertices: Float32Array,
    faces: Uint32Array,
    uvMapping: UVTriangleMapping,
    camera?: CameraParams
  ): Float32Array {
    const { uvWidth, uvHeight, imageWidth, imageHeight } = this.config;

    // Determine number of channels from appearance map
    const totalPixels = imageWidth * imageHeight;
    const numChannels = Math.round(appearanceMap.length / totalPixels);

    console.log('[UVFeatureMapper] Mapping to UV space...');
    console.log(`[UVFeatureMapper]   Input: [${numChannels}, ${imageHeight}, ${imageWidth}]`);
    console.log(`[UVFeatureMapper]   Output: [${numChannels}, ${uvHeight}, ${uvWidth}]`);
    console.log(`[UVFeatureMapper]   Valid UV pixels: ${uvMapping.numValid.toLocaleString()}`);

    // Output UV feature map [C, H_uv, W_uv]
    const uvFeatures = new Float32Array(numChannels * uvHeight * uvWidth);

    // Camera matrix for projection (canonical view if not provided)
    const focalLength = camera?.focalLength ?? 24.0;  // invtanfov from Python
    const cx = camera?.principalPoint?.[0] ?? 0;
    const cy = camera?.principalPoint?.[1] ?? 0;

    // For each valid UV pixel
    let validCount = 0;
    let outOfBoundsCount = 0;

    for (let i = 0; i < uvMapping.numValid; i++) {
      const triIdx = uvMapping.triangleIndices[i];
      const bary0 = uvMapping.barycentricCoords[i * 3 + 0];
      const bary1 = uvMapping.barycentricCoords[i * 3 + 1];
      const bary2 = uvMapping.barycentricCoords[i * 3 + 2];
      const uvU = uvMapping.uvCoords[i * 2 + 0];
      const uvV = uvMapping.uvCoords[i * 2 + 1];

      // Get triangle vertex indices
      const v0Idx = faces[triIdx * 3 + 0];
      const v1Idx = faces[triIdx * 3 + 1];
      const v2Idx = faces[triIdx * 3 + 2];

      // Validate indices
      if (v0Idx * 3 + 2 >= vertices.length ||
          v1Idx * 3 + 2 >= vertices.length ||
          v2Idx * 3 + 2 >= vertices.length) {
        continue;
      }

      // Get vertex positions
      const v0x = vertices[v0Idx * 3 + 0];
      const v0y = vertices[v0Idx * 3 + 1];
      const v0z = vertices[v0Idx * 3 + 2];

      const v1x = vertices[v1Idx * 3 + 0];
      const v1y = vertices[v1Idx * 3 + 1];
      const v1z = vertices[v1Idx * 3 + 2];

      const v2x = vertices[v2Idx * 3 + 0];
      const v2y = vertices[v2Idx * 3 + 1];
      const v2z = vertices[v2Idx * 3 + 2];

      // Barycentric interpolation to get 3D position
      const px = bary0 * v0x + bary1 * v1x + bary2 * v2x;
      const py = bary0 * v0y + bary1 * v1y + bary2 * v2y;
      const pz = bary0 * v0z + bary1 * v1z + bary2 * v2z;

      // Project to image space (canonical camera: looking at +Z)
      // Assuming vertices are already in camera space or need minimal transform
      const depth = pz + 22.0;  // T[2] offset from Python
      if (depth < 0.001) continue;

      // Perspective projection
      const imgX = (px * focalLength / depth) + cx;
      const imgY = (py * focalLength / depth) + cy;

      // Convert to pixel coordinates [0, imageWidth-1], [0, imageHeight-1]
      // NDC range is [-1, 1] -> pixel range [0, size-1]
      const pixelX = (imgX + 1) * 0.5 * (imageWidth - 1);
      const pixelY = (imgY + 1) * 0.5 * (imageHeight - 1);

      // Bounds check
      if (pixelX < 0 || pixelX >= imageWidth - 1 ||
          pixelY < 0 || pixelY >= imageHeight - 1) {
        outOfBoundsCount++;
        continue;
      }

      // Bilinear interpolation
      const x0 = Math.floor(pixelX);
      const y0 = Math.floor(pixelY);
      const x1 = x0 + 1;
      const y1 = y0 + 1;
      const fx = pixelX - x0;
      const fy = pixelY - y0;

      // UV pixel coordinates
      const uvPixelX = Math.round(uvU * (uvWidth - 1));
      const uvPixelY = Math.round(uvV * (uvHeight - 1));

      if (uvPixelX < 0 || uvPixelX >= uvWidth ||
          uvPixelY < 0 || uvPixelY >= uvHeight) {
        continue;
      }

      // Sample each channel with bilinear interpolation
      for (let c = 0; c < numChannels; c++) {
        const chOffset = c * imageHeight * imageWidth;

        // Get 4 corner values
        const v00 = appearanceMap[chOffset + y0 * imageWidth + x0];
        const v01 = appearanceMap[chOffset + y0 * imageWidth + x1];
        const v10 = appearanceMap[chOffset + y1 * imageWidth + x0];
        const v11 = appearanceMap[chOffset + y1 * imageWidth + x1];

        // Bilinear interpolation
        const value = (1 - fx) * (1 - fy) * v00 +
                      fx * (1 - fy) * v01 +
                      (1 - fx) * fy * v10 +
                      fx * fy * v11;

        // Write to UV feature map
        const uvOffset = c * uvHeight * uvWidth + uvPixelY * uvWidth + uvPixelX;
        uvFeatures[uvOffset] = value;
      }

      validCount++;
    }

    console.log(`[UVFeatureMapper]   Mapped: ${validCount.toLocaleString()} pixels`);
    console.log(`[UVFeatureMapper]   Out of bounds: ${outOfBoundsCount.toLocaleString()}`);

    // Statistics
    let min = Infinity, max = -Infinity, sum = 0, nonZero = 0;
    for (let i = 0; i < uvFeatures.length; i++) {
      const v = uvFeatures[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      if (Math.abs(v) > 0.001) nonZero++;
    }
    console.log(`[UVFeatureMapper]   Output stats: [${min.toFixed(4)}, ${max.toFixed(4)}], nonZero=${nonZero.toLocaleString()}`);

    return uvFeatures;
  }

  /**
   * Add view direction embedding to UV features
   *
   * @param uvFeatures UV feature map [128, H, W]
   * @param viewDir View direction [3] (normalized)
   * @returns UV features with view embedding [155, H, W]
   */
  addViewEmbedding(
    uvFeatures: Float32Array,
    viewDir: [number, number, number] = [0, 0, 1]
  ): Float32Array {
    const { uvWidth, uvHeight } = this.config;
    const inputChannels = 128;
    const viewChannels = 27;  // 3 raw + 12 sin + 12 cos (4 frequencies)
    const outputChannels = inputChannels + viewChannels;

    console.log('[UVFeatureMapper] Adding view direction embedding...');
    console.log(`[UVFeatureMapper]   View direction: (${viewDir.join(', ')})`);

    // Compute harmonic embedding
    const embedding = this.computeHarmonicEmbedding(viewDir);
    console.log(`[UVFeatureMapper]   Embedding: ${embedding.length} dims`);

    // Output: [155, H, W]
    const output = new Float32Array(outputChannels * uvHeight * uvWidth);
    const pixelCount = uvHeight * uvWidth;

    // Copy original features [0:128]
    for (let c = 0; c < inputChannels; c++) {
      const srcOffset = c * pixelCount;
      const dstOffset = c * pixelCount;
      for (let i = 0; i < pixelCount; i++) {
        output[dstOffset + i] = uvFeatures[srcOffset + i];
      }
    }

    // Add view embedding [128:155] - same for all pixels
    for (let c = 0; c < viewChannels; c++) {
      const dstOffset = (inputChannels + c) * pixelCount;
      const value = embedding[c];
      for (let i = 0; i < pixelCount; i++) {
        output[dstOffset + i] = value;
      }
    }

    return output;
  }

  /**
   * Compute harmonic embedding for view direction
   * Same as Python: raw + sin(2^k * x) + cos(2^k * x) for k=0..3
   */
  private computeHarmonicEmbedding(dir: [number, number, number]): Float32Array {
    const numFreqs = 4;
    const embedding = new Float32Array(3 + numFreqs * 6);  // 3 + 4*6 = 27

    // Raw direction
    embedding[0] = dir[0];
    embedding[1] = dir[1];
    embedding[2] = dir[2];

    // Harmonic encoding
    let idx = 3;
    for (let freq = 0; freq < numFreqs; freq++) {
      const scale = Math.pow(2, freq);
      // sin
      embedding[idx++] = Math.sin(scale * dir[0]);
      embedding[idx++] = Math.sin(scale * dir[1]);
      embedding[idx++] = Math.sin(scale * dir[2]);
      // cos
      embedding[idx++] = Math.cos(scale * dir[0]);
      embedding[idx++] = Math.cos(scale * dir[1]);
      embedding[idx++] = Math.cos(scale * dir[2]);
    }

    return embedding;
  }
}
