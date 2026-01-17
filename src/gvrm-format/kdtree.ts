// kdtree.ts
// KD-Tree による高速最近傍探索
// O(N×M) → O(N×log M) に削減

interface KDNode {
  point: Float32Array; // [x, y, z]
  index: number;
  left: KDNode | null;
  right: KDNode | null;
  axis: number;
}

export class KDTree {
  private root: KDNode | null = null;
  private points: Float32Array;
  private numPoints: number;

  constructor(points: Float32Array, pointDim: number = 3) {
    this.points = points;
    this.numPoints = points.length / pointDim;
    
    console.log('[KDTree] Building tree for', this.numPoints, 'points...');
    const startTime = performance.now();
    
    // インデックス配列を作成
    const indices = Array.from({ length: this.numPoints }, (_, i) => i);
    
    // ツリーを構築
    this.root = this.buildTree(indices, 0);
    
    const elapsed = performance.now() - startTime;
    console.log(`[KDTree] ✅ Tree built in ${elapsed.toFixed(2)}ms`);
  }

  private buildTree(indices: number[], depth: number): KDNode | null {
    if (indices.length === 0) return null;
    
    if (indices.length === 1) {
      const idx = indices[0];
      return {
        point: this.points.slice(idx * 3, idx * 3 + 3),
        index: idx,
        left: null,
        right: null,
        axis: depth % 3
      };
    }

    // 現在の軸（x, y, z を循環）
    const axis = depth % 3;

    // 中央値で分割
    indices.sort((a, b) => {
      const valA = this.points[a * 3 + axis];
      const valB = this.points[b * 3 + axis];
      return valA - valB;
    });

    const medianIdx = Math.floor(indices.length / 2);
    const nodeIdx = indices[medianIdx];

    return {
      point: this.points.slice(nodeIdx * 3, nodeIdx * 3 + 3),
      index: nodeIdx,
      left: this.buildTree(indices.slice(0, medianIdx), depth + 1),
      right: this.buildTree(indices.slice(medianIdx + 1), depth + 1),
      axis
    };
  }

  /**
   * 最近傍探索
   * @param query [x, y, z]
   * @returns 最も近い点のインデックス
   */
  public nearestNeighbor(query: number[]): number {
    if (!this.root) return -1;

    let best: { node: KDNode | null; distSq: number } = {
      node: null,
      distSq: Infinity
    };

    this.searchNearest(this.root, query, best);

    return best.node ? best.node.index : -1;
  }

  private searchNearest(
    node: KDNode | null,
    query: number[],
    best: { node: KDNode | null; distSq: number }
  ): void {
    if (!node) return;

    // 現在のノードとの距離を計算
    const distSq = 
      (node.point[0] - query[0]) ** 2 +
      (node.point[1] - query[1]) ** 2 +
      (node.point[2] - query[2]) ** 2;

    // ベストを更新
    if (distSq < best.distSq) {
      best.node = node;
      best.distSq = distSq;
    }

    // 探索する方向を決定
    const axis = node.axis;
    const diff = query[axis] - node.point[axis];
    
    const nearNode = diff < 0 ? node.left : node.right;
    const farNode = diff < 0 ? node.right : node.left;

    // 近い方を先に探索
    this.searchNearest(nearNode, query, best);

    // 遠い方も探索する必要があるかチェック
    if (diff * diff < best.distSq) {
      this.searchNearest(farNode, query, best);
    }
  }

  /**
   * バッチ最近傍探索（複数のクエリを一度に処理）
   * @param queries Float32Array [x0, y0, z0, x1, y1, z1, ...]
   * @param numQueries クエリの数
   * @returns Int32Array [idx0, idx1, idx2, ...]
   */
  public batchNearestNeighbor(queries: Float32Array, numQueries: number): Int32Array {
    console.log(`[KDTree] Batch nearest neighbor search for ${numQueries} queries...`);
    const startTime = performance.now();
    
    const results = new Int32Array(numQueries);
    
    // 進捗表示用
    const progressInterval = Math.floor(numQueries / 10);
    
    for (let i = 0; i < numQueries; i++) {
      const query = [
        queries[i * 3],
        queries[i * 3 + 1],
        queries[i * 3 + 2]
      ];
      
      results[i] = this.nearestNeighbor(query);
      
      // 10%ごとに進捗表示
      if (i > 0 && i % progressInterval === 0) {
        const progress = ((i / numQueries) * 100).toFixed(0);
        console.log(`[KDTree] Progress: ${progress}%`);
      }
    }
    
    const elapsed = performance.now() - startTime;
    console.log(`[KDTree] ✅ Batch search completed in ${elapsed.toFixed(2)}ms`);
    console.log(`[KDTree] Average: ${(elapsed / numQueries).toFixed(3)}ms per query`);
    
    return results;
  }
}