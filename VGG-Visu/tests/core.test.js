import { test } from "node:test";
import assert from "node:assert/strict";
import {
  preprocessRGB,
  layers,
  stats,
  intensity,
  activationBytes,
} from "../src/core.js";
test("Keras RGB → BGR, sans division par 255", () => {
  const a = preprocessRGB(new Uint8Array([10, 20, 30]));
  [30 - 103.939, 20 - 116.779, 10 - 123.68].forEach((v, i) =>
    assert.ok(Math.abs(a[i] - v) < 1e-5),
  );
});
test("18 dimensions et volume exact", () => {
  assert.equal(layers.length, 18);
  assert.deepEqual(layers.at(-1), {
    name: "block5_pool",
    type: "Max-pooling · maximum 2 × 2",
    size: 7,
    channels: 512,
    block: 5,
  });
  assert.equal(activationBytes, 60311552);
});
test("canaux NHWC et statistiques avant normalisation", () => {
  assert.deepEqual(stats(new Float32Array([1, 20, 3, 40]), 2, 1), {
    min: 20,
    max: 40,
    mean: 30,
    count: 2,
    invalid: 0,
  });
});
test("constantes, zéro et non-finis", () => {
  assert.equal(intensity(4, 4, 4), 0);
  assert.equal(intensity(0, 0, 0), 0);
  assert.equal(intensity(NaN, 0, 1), 0);
  assert.equal(stats([0, Infinity, NaN]).invalid, 2);
  assert.equal(intensity(5, 0, 10), 0.5);
});
