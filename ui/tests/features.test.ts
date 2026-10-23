import { describe, expect, it } from "vitest";
import { FeatureIndex } from "../src/features";

// Two features in one mesh: a unit square (two triangles) at height 0 for feature 0, and one
// triangle at height 5 for feature 1. Axes as in glTF: y is up.
const positions = [
  0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
  // feature 1
  2, 5, 0, 3, 5, 0, 2, 5, 1,
];
const featureOfVertex = [0, 0, 0, 0, 1, 1, 1];
const index = [0, 1, 2, 0, 2, 3, 4, 5, 6];

describe("FeatureIndex", () => {
  const fi = new FeatureIndex(featureOfVertex, index, 3);

  it("knows which feature each triangle belongs to", () => {
    expect([0, 1, 2].map((t) => fi.featureOfTriangle(t))).toEqual([0, 0, 1]);
    expect(fi.triangles.get(0)).toEqual([0, 1]);
    expect(fi.triangles.get(1)).toEqual([2]);
  });

  it("copies out the corners of one feature", () => {
    const p = fi.positions(1, positions);
    expect(Array.from(p)).toEqual([2, 5, 0, 3, 5, 0, 2, 5, 1]);
    expect(fi.positions(7, positions).length).toBe(0);
  });

  it("finds the centre of a feature's top", () => {
    const top = fi.topCentre(1, positions);
    expect(top?.[1]).toBe(5);
    expect(top?.[0]).toBeCloseTo(7 / 3);
    expect(fi.topCentre(9, positions)).toBeNull();
  });

  it("works without an index buffer", () => {
    const flat = new FeatureIndex([3, 3, 3, 4, 4, 4], null, 2);
    expect(flat.featureOfTriangle(1)).toBe(4);
  });
});
