import * as tf from "@tensorflow/tfjs";
import {
  layers,
  preprocessRGB,
  stats,
  intensity,
  color,
  activationBytes,
} from "./core.js";
import "./style.css";
import { networkMarkup } from "./network.js";
const $ = (id) => document.getElementById(id),
  base = import.meta.env.BASE_URL;
let model,
  multi,
  singles = [],
  cache = new Map(),
  pixels,
  layerIndex = 0,
  channel = 0,
  page = 0,
  version = 0,
  running = false,
  inferences = 0,
  lastMs = 0;
const status = (s, error = false) => {
  $("status").textContent = s;
  $("status").classList.toggle("error", error);
};
function network() {
  const host = $("network");
  host.innerHTML = networkMarkup(
    layerIndex,
    pixels ? $("input").toDataURL("image/png") : null,
  );
  host.querySelectorAll("[data-layer]").forEach((el) => {
    const select = (index, focus = false) => {
      layerIndex = index;
      channel = 0;
      page = 0;
      network();
      render();
      schedule();
      if (focus) {
        const target = host.querySelector(`[data-layer="${index}"]`);
        target.focus({ preventScroll: true });
        target.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    };
    el.onclick = () => select(Number(el.dataset.layer));
    el.onkeydown = (e) => {
      const index = Number(el.dataset.layer);
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select(index, true);
      }
      if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) {
        e.preventDefault();
        select(
          e.key === "Home"
            ? 0
            : e.key === "End"
              ? 17
              : Math.max(
                  0,
                  Math.min(17, index + (e.key === "ArrowRight" ? 1 : -1)),
                ),
          true,
        );
      }
    };
  });
}
function paint(canvas, entry, c, big = false) {
  const { size, channels } = entry.layer,
    s = stats(entry.data, channels, c),
    range = $("normalization").value === "layer" ? entry.range : s;
  const tiny = document.createElement("canvas");
  tiny.width = tiny.height = size;
  const ctx = tiny.getContext("2d"),
    im = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = entry.data[i * channels + c],
      rgb = color(
        intensity(v, range.min, range.max),
        $("palette").value === "heat",
      );
    im.data.set(Number.isFinite(v) ? [...rgb, 255] : [255, 0, 255, 255], i * 4);
  }
  ctx.putImageData(im, 0, 0);
  canvas.width = canvas.height = big ? 448 : size;
  const dest = canvas.getContext("2d");
  dest.imageSmoothingEnabled = false;
  dest.drawImage(tiny, 0, 0, canvas.width, canvas.height);
  return s;
}
function render() {
  const l = layers[layerIndex],
    entry = cache.get(l.name);
  $("layer-title").textContent = l.name;
  $("layer-info").textContent =
    `Couche ${layerIndex + 1} / 18 · ${l.type} · ${l.size} × ${l.size} · ${l.channels} canaux`;
  $("channel").max = l.channels;
  $("channel").value = channel + 1;
  $("mosaic").replaceChildren();
  $("page-info").textContent = `${page + 1} / ${Math.ceil(l.channels / 32)}`;
  $("page-prev").disabled = page === 0;
  $("page-next").disabled = (page + 1) * 32 >= l.channels;
  $("prev").disabled = channel === 0;
  $("next").disabled = channel === l.channels - 1;
  if (!entry) {
    $("detail").getContext("2d").clearRect(0, 0, 448, 448);
    $("statistics").textContent =
      "Activations non disponibles pour cette couche.";
    return;
  }
  for (let c = page * 32; c < Math.min((page + 1) * 32, l.channels); c++) {
    const btn = document.createElement("button");
    btn.setAttribute("aria-label", `Sélectionner le canal ${c + 1}`);
    btn.setAttribute("aria-pressed", String(c === channel));
    if (c === channel) btn.className = "selected";
    const canvas = document.createElement("canvas");
    paint(canvas, entry, c);
    btn.append(canvas, document.createTextNode(`${c + 1}`));
    btn.onclick = () => {
      channel = c;
      render();
    };
    $("mosaic").append(btn);
  }
  const s = paint($("detail"), entry, channel, true);
  $("statistics").textContent =
    `Canal ${channel + 1} · résolution native ${l.size} × ${l.size} · affichage 448 × 448. Min ${s.min.toPrecision(6)} · Max ${s.max.toPrecision(6)} · Moyenne ${s.mean.toPrecision(6)}.${s.invalid ? ` ${s.invalid} valeurs non finies (magenta).` : ""}${s.max === s.min ? " Carte constante : contraste nul." : ""}`;
}
async function schedule() {
  if (running || !model || !pixels) return;
  const requested = version,
    selected = layerIndex,
    mode = $("cache").value;
  if (
    (mode === "all" && cache.size === 18) ||
    (mode === "layer" && cache.has(layers[selected].name))
  )
    return;
  running = true;
  status("Calcul des activations…");
  const start = performance.now();
  let input,
    outputs = [];
  try {
    await tf.nextFrame();
    if (requested !== version) return;
    input = tf.tensor4d(preprocessRGB(pixels), [1, 224, 224, 3]);
    const result = (mode === "all" ? multi : singles[selected]).predict(input);
    outputs = Array.isArray(result) ? result : [result];
    inferences++;
    const next = new Map();
    for (let i = 0; i < outputs.length; i++) {
      const l = layers[mode === "all" ? i : selected],
        shape = outputs[i].shape;
      if (shape.join(",") !== [1, l.size, l.size, l.channels].join(","))
        throw Error(`Dimensions inattendues : ${l.name}`);
      const data = await outputs[i].data();
      next.set(l.name, { layer: l, data, range: stats(data) });
    }
    if (requested !== version || (mode === "layer" && selected !== layerIndex))
      return;
    cache = next;
    lastMs = performance.now() - start;
    status("Préparation de la mosaïque…");
    await tf.nextFrame();
    if (requested !== version) return;
    render();
    status("Prêt · activations réelles en cache");
    $("duration").textContent =
      `Dernier calcul : ${(lastMs / 1000).toFixed(2)} s · cache ${(Array.from(cache.values()).reduce((s, e) => s + e.data.byteLength, 0) / 1048576).toFixed(1)} Mio`;
  } catch (e) {
    if (requested === version)
      status(
        `Calcul impossible : ${e.message}. Essayez le cache une couche ou rechargez la page.`,
        true,
      );
  } finally {
    input?.dispose();
    outputs.forEach((t) => t.dispose());
    running = false;
    if (requested !== version || (mode === "layer" && selected !== layerIndex))
      schedule();
  }
}
async function importImage(blob) {
  const ticket = ++version;
  cache.clear();
  pixels = null;
  render();
  status("Décodage et recadrage de l’image…");
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
    if (ticket !== version) return;
    const canvas = $("input"),
      ctx = canvas.getContext("2d"),
      side = Math.min(bitmap.width, bitmap.height);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, 224, 224);
    ctx.drawImage(
      bitmap,
      (bitmap.width - side) / 2,
      (bitmap.height - side) / 2,
      side,
      side,
      0,
      0,
      224,
      224,
    );
    const rgba = ctx.getImageData(0, 0, 224, 224).data;
    pixels = new Uint8Array(224 * 224 * 3);
    for (let p = 0; p < 224 * 224; p++)
      pixels.set(rgba.subarray(p * 4, p * 4 + 3), p * 3);
    network();
    if (model) schedule();
    else status("Image prête ; chargement du modèle en cours…");
  } catch {
    if (ticket === version)
      status(
        "Image illisible. Choisissez un fichier JPEG ou PNG valide.",
        true,
      );
  } finally {
    bitmap?.close();
  }
}
// Vite bundles local examples with relative URLs, including on GitHub Pages.
const exampleImages = import.meta.glob(
  "/images/*.{jpg,jpeg,JPG,JPEG,png,PNG}",
  { eager: true, query: "?url", import: "default" },
);
const exampleLabels = {
  electric_guitar: "Guitare électrique",
  jeep: "Jeep",
  agaric: "Champignon",
  ringlet: "Papillon",
  cairn: "Terrier cairn",
  trombone: "Trombone",
  thunder_snake: "Serpent",
  broccoli: "Brocoli",
  television: "Téléviseur",
};
for (const [path, url] of Object.entries(exampleImages)) {
  const key = path
    .split("/")
    .pop()
    .replace(/^n\d+_/, "")
    .replace(/\.[^.]+$/, "");
  const name = exampleLabels[key] || key.replaceAll("_", " "),
    button = document.createElement("button"),
    thumb = document.createElement("img");
  button.type = "button";
  button.className = "example-choice";
  button.setAttribute("aria-label", `Analyser l’image : ${name}`);
  thumb.src = url;
  thumb.alt = "";
  thumb.loading = "lazy";
  thumb.width = 96;
  thumb.height = 72;
  button.append(thumb, document.createTextNode(name));
  button.onclick = async () => {
    const ticket = ++version;
    pixels = null;
    cache.clear();
    render();
    status(`Chargement de l’image : ${name}…`);
    try {
      const response = await fetch(url);
      if (!response.ok) throw Error();
      const blob = await response.blob();
      if (ticket !== version) return;
      await importImage(blob);
    } catch {
      if (ticket === version)
        status(`Impossible de charger l’image « ${name} ».`, true);
    }
  };
  $("examples").append(button);
}
$("file").onchange = (e) => {
  if (e.target.files[0]) importImage(e.target.files[0]);
  e.target.value = "";
};
document.querySelector(".input").ondragover = (e) => e.preventDefault();
document.querySelector(".input").ondrop = (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) importImage(e.dataTransfer.files[0]);
};
$("example").onclick = async () => {
  try {
    const r = await fetch(base + "example.svg");
    if (!r.ok) throw Error();
    const svg = await r.blob();
    const img = new Image(),
      url = URL.createObjectURL(svg);
    try {
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });
      const c = document.createElement("canvas");
      c.width = 640;
      c.height = 400;
      c.getContext("2d").drawImage(img, 0, 0);
      await new Promise((resolve) =>
        c.toBlob(async (b) => {
          await importImage(b);
          resolve();
        }, "image/png"),
      );
    } finally {
      URL.revokeObjectURL(url);
    }
  } catch {
    status("Impossible de charger l’exemple.", true);
  }
};
for (const id of ["palette", "normalization"]) $(id).onchange = render;
$("cache").onchange = () => {
  version++;
  cache.clear();
  render();
  schedule();
};
function choose(c) {
  channel = Math.max(0, Math.min(layers[layerIndex].channels - 1, c));
  page = Math.floor(channel / 32);
  render();
}
$("prev").onclick = () => choose(channel - 1);
$("next").onclick = () => choose(channel + 1);
$("channel").onchange = () => choose((Number($("channel").value) || 1) - 1);
$("page-prev").onclick = () => {
  page--;
  render();
};
$("page-next").onclick = () => {
  page++;
  render();
};
// Diagnostic read-only surface for reproducible browser tests.
window.vggDiagnostics = () => ({
  version,
  inferences,
  lastMs,
  cached: Array.from(cache.keys()),
  activationBytes,
  tf: tf.memory(),
  backend: tf.getBackend(),
  running,
});
network();
render();
(async () => {
  try {
    if (!(await tf.setBackend("webgl"))) throw Error("WebGL indisponible");
    await tf.ready();
    if (!tf.env().get("WEBGL_RENDER_FLOAT32_CAPABLE"))
      throw Error("WebGL float32 indisponible");
    tf.env().set("WEBGL_FORCE_F16_TEXTURES", false);
    status("Chargement des poids ImageNet…");
    model = await tf.loadLayersModel(base + "model/model.json", {
      onProgress: (f) =>
        status(`Chargement du modèle : ${Math.round(f * 100)} %`),
    });
    multi = tf.model({
      inputs: model.inputs,
      outputs: layers.map((l) => model.getLayer(l.name).output),
    });
    singles = layers.map((l) =>
      tf.model({
        inputs: model.inputs,
        outputs: model.getLayer(l.name).output,
      }),
    );
    status("Modèle prêt. Importez une image ou chargez l’exemple.");
    schedule();
  } catch (e) {
    status(
      `Modèle indisponible : ${e.message}. Vérifiez WebGL et les fichiers dans model/ ; voir le README.`,
      true,
    );
  }
})();
