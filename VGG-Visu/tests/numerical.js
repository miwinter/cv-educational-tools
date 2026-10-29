import * as tf from "@tensorflow/tfjs";
import { preprocessRGB, layers } from "../src/core.js";
window.runNumerical = async () => {
  await tf.setBackend("webgl");
  await tf.ready();
  tf.env().set("WEBGL_FORCE_F16_TEXTURES", false);
  const get = async (path) => {
    const r = await fetch("/tests/reference/" + path);
    if (!r.ok) throw Error(path);
    return r.arrayBuffer();
  };
  const rgb = new Uint8Array(await get("input.bin")),
    x = preprocessRGB(rgb),
    refx = new Float32Array(await get("preprocessed.bin"));
  let preprocessingMax = 0;
  x.forEach(
    (v, i) =>
      (preprocessingMax = Math.max(preprocessingMax, Math.abs(v - refx[i]))),
  );
  const model = await tf.loadLayersModel("/model/model.json"),
    multi = tf.model({
      inputs: model.inputs,
      outputs: layers.map((l) => model.getLayer(l.name).output),
    }),
    input = tf.tensor4d(x, [1, 224, 224, 3]);
  const start = performance.now(),
    ys = multi.predict(input),
    results = [];
  for (let j = 0; j < ys.length; j++) {
    const actual = await ys[j].data(),
      expected = new Float32Array(await get(layers[j].name + ".bin"));
    let maxAbs = 0,
      sum = 0,
      refMax = 0,
      failures = 0;
    for (let i = 0; i < actual.length; i++) {
      const d = Math.abs(actual[i] - expected[i]);
      maxAbs = Math.max(maxAbs, d);
      sum += d;
      refMax = Math.max(refMax, Math.abs(expected[i]));
      if (!Number.isFinite(actual[i]) || d > 0.1 + 0.01 * Math.abs(expected[i]))
        failures++;
    }
    results.push({
      name: layers[j].name,
      shape: ys[j].shape,
      maxAbs,
      meanAbs: sum / actual.length,
      relativeToPeak: maxAbs / refMax,
      failures,
    });
    ys[j].dispose();
  }
  input.dispose();
  return {
    backend: tf.getBackend(),
    webglVersion: tf.env().get("WEBGL_VERSION"),
    preprocessingMax,
    ms: performance.now() - start,
    results,
  };
};
