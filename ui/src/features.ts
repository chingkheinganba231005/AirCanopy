// Which feature (building, road, tree) each triangle of a merged mesh belongs to.
//
// All buildings of one kind are merged into one mesh, so the graphics card draws them in one go.
// Each vertex carries a feature number (the glTF attribute _FEATURE_ID); the mesh's extras list
// the feature ids in the same order (see src/aircanopy/site/mesh.py).

export class FeatureIndex {
  /** Triangle numbers of each feature. */
  readonly triangles = new Map<number, number[]>();

  constructor(
    private readonly featureOfVertex: ArrayLike<number>,
    private readonly index: ArrayLike<number> | null,
    triangleCount: number,
  ) {
    for (let t = 0; t < triangleCount; t++) {
      const f = this.featureOfTriangle(t);
      let list = this.triangles.get(f);
      if (!list) this.triangles.set(f, (list = []));
      list.push(t);
    }
  }

  private corner(t: number, k: number): number {
    const i = 3 * t + k;
    return this.index ? (this.index[i] ?? 0) : i;
  }

  featureOfTriangle(t: number): number {
    return Math.round(this.featureOfVertex[this.corner(t, 0)] ?? -1);
  }

  /** x, y, z of every corner of the feature's triangles, ready for a new geometry. */
  positions(feature: number, xyz: ArrayLike<number>): Float32Array {
    const tris = this.triangles.get(feature) ?? [];
    const out = new Float32Array(tris.length * 9);
    tris.forEach((t, n) => {
      for (let k = 0; k < 3; k++) {
        const v = this.corner(t, k);
        for (let c = 0; c < 3; c++) out[9 * n + 3 * k + c] = xyz[3 * v + c] ?? 0;
      }
    });
    return out;
  }

  /** Centre of the feature's highest points (its roof, for a building), in the mesh's axes. */
  topCentre(feature: number, xyz: ArrayLike<number>, upAxis = 1): [number, number, number] | null {
    const p = this.positions(feature, xyz);
    if (p.length === 0) return null;
    let top = -Infinity;
    for (let i = upAxis; i < p.length; i += 3) top = Math.max(top, p[i] ?? -Infinity);
    const sum = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < p.length; i += 3) {
      if (Math.abs((p[i + upAxis] ?? 0) - top) > 1e-3) continue;
      for (let c = 0; c < 3; c++) sum[c]! += p[i + c] ?? 0;
      n++;
    }
    return [sum[0]! / n, sum[1]! / n, sum[2]! / n];
  }
}
