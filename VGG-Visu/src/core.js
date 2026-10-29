export const layers = [2, 2, 3, 3, 3].flatMap((n, b) => {
  const size = 224 / 2 ** b,
    channels = [64, 128, 256, 512, 512][b];
  return [
    ...Array.from({ length: n }, (_, i) => ({
      name: `block${b + 1}_conv${i + 1}`,
      type: "Convolution · après ReLU",
      size,
      channels,
      block: b + 1,
    })),
    {
      name: `block${b + 1}_pool`,
      type: "Max-pooling · maximum 2 × 2",
      size: size / 2,
      channels,
      block: b + 1,
    },
  ];
});
export function preprocessRGB(rgb) {
  const out = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i += 3) {
    out[i] = rgb[i + 2] - Math.fround(103.939);
    out[i + 1] = rgb[i + 1] - Math.fround(116.779);
    out[i + 2] = rgb[i] - Math.fround(123.68);
  }
  return out;
}
export function stats(data, stride = 1, offset = 0) {
  let min = Infinity,
    max = -Infinity,
    sum = 0,
    count = 0,
    invalid = 0;
  for (let i = offset; i < data.length; i += stride) {
    const v = data[i];
    if (!Number.isFinite(v)) {
      invalid++;
      continue;
    }
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
    count++;
  }
  return {
    min: count ? min : 0,
    max: count ? max : 0,
    mean: count ? sum / count : 0,
    invalid,
    count,
  };
}
export function intensity(v, min, max) {
  return Number.isFinite(v) && max > min
    ? Math.max(0, Math.min(1, (v - min) / (max - min)))
    : 0;
}
export function color(t, heat) {
  if (!heat) return [t * 255, t * 255, t * 255];
  return [
    255 * Math.min(1, t * 3),
    255 * Math.max(0, Math.min(1, t * 3 - 1)),
    255 * Math.max(0, t * 3 - 2),
  ];
}
export const activationBytes = layers.reduce(
  (s, l) => s + l.size * l.size * l.channels * 4,
  0,
);
