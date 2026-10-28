// The exported weights are those of the original Keras Dense layers, in order.
export class Autoencoder {
  constructor(tf, encoder, decoder, manifest) {
    Object.assign(this, { tf, encoder, decoder, manifest });
  }

  static async load(tf, manifestURL) {
    const manifestResponse = await fetch(manifestURL);
    if (!manifestResponse.ok) throw new Error('Le fichier du modèle est introuvable.');
    const manifest = await manifestResponse.json();
    const weightsResponse = await fetch(new URL(manifest.weightsFile, manifestURL));
    if (!weightsResponse.ok) throw new Error('Les poids du modèle sont introuvables.');
    return Autoencoder.fromWeights(tf, manifest, await weightsResponse.arrayBuffer());
  }

  static fromWeights(tf, manifest, buffer) {
    if (manifest.format !== 'ae-gui-dense-v1' || ![2, 128].includes(manifest.latentDimensions)) {
      throw new Error('Format de modèle incompatible.');
    }
    if (buffer.byteLength !== manifest.parameterCount * 4) {
      throw new Error('Le fichier de poids est incomplet.');
    }
    const floats = new Float32Array(buffer);
    function build(layers, inputSize) {
      const model = tf.sequential();
      for (const [index, layer] of layers.entries()) {
        model.add(tf.layers.dense({
          units: layer.units, activation: layer.activation, name: layer.name,
          ...(index === 0 ? { inputShape: [inputSize] } : {}),
        }));
        const tensors = layer.weights.map(weight => tf.tensor(
          floats.subarray(weight.offset, weight.offset + weight.length), weight.shape,
        ));
        model.layers[index].setWeights(tensors);
        tensors.forEach(tensor => tensor.dispose());
      }
      return model;
    }
    const encoder = build(manifest.layers.slice(0, manifest.splitAfter), 784);
    const decoder = build(manifest.layers.slice(manifest.splitAfter), manifest.latentDimensions);
    return new Autoencoder(tf, encoder, decoder, manifest);
  }

  reconstruct(input) {
    if (input.length !== 784) throw new Error('Une entrée doit contenir 28 × 28 pixels.');
    return this.tf.tidy(() => {
      const encoded = this.encoder.predict(this.tf.tensor2d(input, [1, 784]));
      const z = Array.from(encoded.dataSync());
      const pixels = Float32Array.from(this.decoder.predict(encoded).dataSync());
      const { delta, mse } = reconstructionError(input, pixels);
      return { z, pixels, delta, mse };
    });
  }

  decode(z) {
    const dimensions = this.manifest.latentDimensions;
    if (z.length !== dimensions || !z.every(Number.isFinite)) throw new Error('Coordonnées invalides.');
    return this.tf.tidy(() => Float32Array.from(
      this.decoder.predict(this.tf.tensor2d(z, [1, dimensions])).dataSync(),
    ));
  }

  dispose() {
    this.encoder.dispose();
    this.decoder.dispose();
  }
}

export function reconstructionError(input, output) {
  const delta = Float32Array.from(input, (value, i) => Math.abs(value - output[i]));
  const mse = delta.reduce((sum, value) => sum + value * value, 0) / delta.length;
  return { delta, mse };
}
