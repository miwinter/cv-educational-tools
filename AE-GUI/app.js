import { Autoencoder } from './model-runtime.js';

const $ = id => document.getElementById(id);
const palette = ['#aa7bc9', '#6ca8d4', '#64b9b2', '#ddb86b', '#df8e9e', '#8d9ed8', '#b1b871', '#d6a279', '#9b8bbf', '#82ad8b'];
const inputCanvas = $('input-canvas');
// Keep smooth pointer strokes internally, and display only the 28 × 28 input
// actually sent to the encoder, enlarged with nearest-neighbor rendering.
const drawingCanvas = document.createElement('canvas');
drawingCanvas.width = drawingCanvas.height = 280;
const drawingContext = drawingCanvas.getContext('2d');
const brushRadius = 16;
const brushCanvas = document.createElement('canvas');
brushCanvas.width = brushCanvas.height = brushRadius * 2;
const brushContext = brushCanvas.getContext('2d');
const brushGradient = brushContext.createRadialGradient(brushRadius, brushRadius, 0, brushRadius, brushRadius, brushRadius);
brushGradient.addColorStop(0, '#fff');
brushGradient.addColorStop(.3, '#fff');
brushGradient.addColorStop(.6, '#bfbfbf');
brushGradient.addColorStop(.82, '#555');
brushGradient.addColorStop(1, '#000');
brushContext.fillStyle = brushGradient;
brushContext.fillRect(0, 0, brushCanvas.width, brushCanvas.height);
const plotCanvas = $('latent-canvas');
const plotContext = plotCanvas.getContext('2d');
const tinyCanvas = document.createElement('canvas');
tinyCanvas.width = tinyCanvas.height = 28;
const tinyContext = tinyCanvas.getContext('2d', { willReadFrequently: true });
const state = {
  ready: false, sample: null, modified: false, free: false,
  input: new Float32Array(784), z: [0, 0], result: null,
  drawing: false, dragging: false, plotSize: [0, 0], pendingFrame: null,
  strokeFrom: null,
  plotStartPoint: null, plotMoved: false,
  bounds: null,
  showProjections: true,
};
let network, dataset, imageBytes;
const modelCache = new Map();
const latentControlIds = ['restore-button', 'show-projections', 'zoom-in', 'zoom-out', 'zoom-reset', 'z1', 'z2'];
const format = new Intl.NumberFormat('fr-FR', { minimumFractionDigits: 5, maximumFractionDigits: 5 });
const coordinateFormat = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 2 });

function announce(message) { $('announcement').textContent = message; }

function is2D() { return network?.manifest.latentDimensions === 2; }

function updateControls() {
  for (const id of ['model-select', 'random-button', 'clear-button', 'noise-button']) $(id).disabled = !state.ready;
  document.querySelectorAll('.sample-button').forEach(button => { button.disabled = !state.ready; });
  for (const id of latentControlIds) $(id).disabled = !state.ready || !is2D();
  plotCanvas.tabIndex = state.ready && is2D() ? 0 : -1;
  $('latent-body').inert = !state.ready || !is2D();
}

function updateModelPresentation() {
  const dimensions = network.manifest.latentDimensions;
  $('model-select').value = String(dimensions);
  $('latent-badge').textContent = `${dimensions} dimensions`;
  $('latent-panel').classList.toggle('latent-unavailable', !is2D());
  $('latent-body').setAttribute('aria-hidden', String(!is2D()));
  $('latent-explanation').hidden = is2D();
  $('plot-tooltip').hidden = true;
  updateControls();
}

async function switchModel(dimensions) {
  if (!state.ready || dimensions === network.manifest.latentDimensions) return;
  cancelPending();
  const previous = network;
  state.ready = false;
  updateControls();
  announce(`Chargement du modèle ${dimensions}D…`);
  $('error-banner').hidden = true;
  try {
    if (!modelCache.has(dimensions)) {
      const path = dimensions === 2 ? './assets/model/model.json' : './assets/model-128/model.json';
      modelCache.set(dimensions, await Autoencoder.load(globalThis.tf, new URL(path, import.meta.url)));
    }
    network = modelCache.get(dimensions);
    state.ready = true;
    updateModelPresentation();
    // Always compare reconstructions of the same reference input, even when
    // switching away from a freely chosen point in the 2D decoder.
    reconstruct();
    announce(`Modèle ${dimensions}D chargé, entrée conservée. MSE : ${format.format(state.result.mse)}.`);
  } catch (error) {
    console.error(error);
    network = previous;
    state.ready = true;
    updateModelPresentation();
    reconstruct();
    $('error-banner').textContent = `Le modèle ${dimensions}D n’a pas pu être chargé. ${error.message} Le modèle précédent reste disponible ; vous pouvez réessayer.`;
    $('error-banner').hidden = false;
  }
}

function fatal(error) {
  console.error(error);
  state.ready = false;
  updateControls();
  $('workspace').setAttribute('aria-busy', 'false');
  const banner = $('error-banner');
  banner.textContent = `L’application n’a pas pu démarrer. ${error.message} Rechargez la page. Pour un essai local, ouvrez-la depuis un serveur HTTP statique (voir le README).`;
  banner.hidden = false;
}

// Display values retain their original [0,1] scale; there is no per-image contrast stretch.
function paintPixels(canvas, values, heatmap = false) {
  const context = canvas.getContext('2d');
  const image = context.createImageData(28, 28);
  const stops = [[17, 19, 29], [99, 52, 38], [186, 93, 49], [246, 162, 88], [255, 237, 179]];
  for (let i = 0; i < 784; i++) {
    const value = Math.max(0, Math.min(1, values[i]));
    let rgb;
    if (heatmap) {
      const position = value * (stops.length - 1);
      const lo = Math.min(stops.length - 2, Math.floor(position));
      const fraction = position - lo;
      rgb = stops[lo].map((channel, c) => Math.round(channel + fraction * (stops[lo + 1][c] - channel)));
    } else {
      rgb = [17, 19, 29].map(channel => Math.round(channel + value * (255 - channel)));
    }
    image.data.set([...rgb, 255], i * 4);
  }
  context.putImageData(image, 0, 0);
}

function sampleInput(sample) {
  return Float32Array.from(imageBytes.subarray(sample.id * 784, (sample.id + 1) * 784), value => value / 255);
}

function updateLibrary() {
  document.querySelectorAll('.sample-button').forEach(button => {
    button.setAttribute('aria-pressed', String(!state.modified && !state.free && Number(button.dataset.sample) === state.sample?.id));
  });
}

function renderResult() {
  const { result, free } = state;
  $('input-placeholder').hidden = !free;
  $('error-placeholder').hidden = !free;
  $('restore-button').hidden = !free;
  $('plot-mode').textContent = free ? 'Exploration du décodeur' : 'Position de l’entrée';
  $('input-caption').textContent = free ? 'Aucune image associée à cette position' : state.sample
    ? `Chiffre ${state.sample.label} · ${state.modified ? 'image modifiée' : `test #${state.sample.testIndex}`}`
    : 'Image libre · 28 × 28 pixels';
  $('output-caption').textContent = free ? 'Image générée depuis (z₁, z₂)' : 'Sortie du décodeur';
  $('mse-value').textContent = free ? '—' : format.format(result.mse);
  $('metric-explanation').innerHTML = free
    ? 'Choisissez une entrée pour mesurer<br>l’erreur de reconstruction.'
    : 'Une valeur proche de 0 indique<br>une reconstruction fidèle à l’entrée.';
  $('z1').value = is2D() ? state.z[0].toFixed(3) : '';
  $('z2').value = is2D() ? state.z[1].toFixed(3) : '';
  paintPixels($('output-canvas'), result.pixels);
  if (!free) paintPixels($('error-canvas'), result.delta, true);
  updateLibrary();
  drawPlot();
}

function reconstruct() {
  state.free = false;
  state.result = network.reconstruct(state.input);
  state.z = state.result.z;
  keepCursorVisible();
  renderResult();
}

// Seed the editable surface with exact grayscale pixels, not the display palette.
// All sources (MNIST, noise and free drawing) then share the same drawing surface.
function seedDrawing() {
  const image = tinyContext.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const gray = Math.round(state.input[i] * 255);
    image.data.set([gray, gray, gray, 255], i * 4);
  }
  tinyContext.putImageData(image, 0, 0);
  drawingContext.imageSmoothingEnabled = false;
  drawingContext.drawImage(tinyCanvas, 0, 0, 280, 280);
  paintPixels(inputCanvas, state.input);
}

function selectSample(sample) {
  cancelPending();
  state.sample = sample;
  state.modified = false;
  state.input = sampleInput(sample);
  seedDrawing();
  $('draw-hint').hidden = true;
  reconstruct();
  announce(`Chiffre ${sample.label} sélectionné. MSE : ${format.format(state.result.mse)}.`);
}

function randomSample() {
  const others = dataset.samples.filter(sample => sample.id !== state.sample?.id);
  selectSample(others[Math.floor(Math.random() * others.length)]);
}

function clearDrawing() {
  cancelPending();
  state.sample = null;
  state.modified = false;
  state.input = new Float32Array(784);
  seedDrawing();
  $('draw-hint').hidden = false;
  reconstruct();
  announce('Zone de dessin effacée.');
}

function addNoise() {
  cancelPending();
  // Independent zero-mean Gaussian noise, sigma 0.15 on normalized pixels.
  // Quantize to uint8 so subsequent strokes retain exactly the same baseline.
  state.input = Float32Array.from(state.input, value => {
    const noise = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    return Math.round(Math.max(0, Math.min(1, value + .15 * noise)) * 255) / 255;
  });
  state.modified = true;
  seedDrawing();
  $('draw-hint').hidden = true;
  reconstruct();
  announce(`Bruit gaussien ajouté à l’image. MSE : ${format.format(state.result.mse)}.`);
}

function readDrawing() {
  tinyContext.imageSmoothingEnabled = true;
  tinyContext.drawImage(drawingCanvas, 0, 0, 28, 28);
  const rgba = tinyContext.getImageData(0, 0, 28, 28).data;
  state.input = Float32Array.from({ length: 784 }, (_, i) => rgba[i * 4] / 255);
  state.modified = true;
  paintPixels(inputCanvas, state.input);
  reconstruct();
}

function cancelPending() {
  if (state.pendingFrame !== null) cancelAnimationFrame(state.pendingFrame);
  state.pendingFrame = null;
}

function schedule(work) {
  cancelPending();
  state.pendingFrame = requestAnimationFrame(() => {
    state.pendingFrame = null;
    try { work(); } catch (error) { fatal(error); }
  });
}

function explore(z) {
  if (!state.ready || !is2D()) return;
  state.free = true;
  state.z = z;
  state.result = { pixels: network.decode(z) };
  keepCursorVisible();
  renderResult();
}

function buildLibrary() {
  const fragment = document.createDocumentFragment();
  for (const sample of dataset.samples) {
    const button = document.createElement('button');
    button.className = 'sample-button';
    button.dataset.sample = sample.id;
    button.setAttribute('aria-label', `Chiffre ${sample.label}, exemple ${sample.id % 5 + 1}, test MNIST ${sample.testIndex}`);
    button.setAttribute('aria-pressed', 'false');
    button.title = `Chiffre ${sample.label} · test #${sample.testIndex}`;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 28;
    canvas.setAttribute('aria-hidden', 'true');
    paintPixels(canvas, sampleInput(sample));
    const label = document.createElement('span');
    label.textContent = sample.label;
    label.setAttribute('aria-hidden', 'true');
    button.append(canvas, label);
    button.addEventListener('click', () => selectSample(sample));
    fragment.append(button);
  }
  $('sample-grid').append(fragment);
  palette.forEach((color, digit) => {
    const key = document.createElement('span');
    key.className = 'digit-key';
    key.innerHTML = `<i style="background:${color}"></i>${digit}`;
    $('digit-legend').append(key);
  });
}

const margins = { left: 41, right: 17, top: 17, bottom: 35 };
function keepCursorVisible() {
  if (!state.bounds || !is2D()) return;
  state.bounds = state.bounds.map(([min, max], axis) => {
    const center = state.z[axis];
    return center < min || center > max ? [center - (max - min) / 2, center + (max - min) / 2] : [min, max];
  });
}

function zoomPlot(factor) {
  if (!state.ready || !is2D()) return;
  cancelPending();
  state.bounds = state.bounds.map(([min, max], axis) => {
    const halfSpan = Math.max(.02, Math.min(100000, (max - min) * factor / 2));
    return [state.z[axis] - halfSpan, state.z[axis] + halfSpan];
  });
  drawPlot();
}

function plotPosition(z) {
  const [[xmin, xmax], [ymin, ymax]] = state.bounds;
  const [width, height] = state.plotSize;
  return [margins.left + (z[0] - xmin) / (xmax - xmin) * (width - margins.left - margins.right),
    height - margins.bottom - (z[1] - ymin) / (ymax - ymin) * (height - margins.top - margins.bottom)];
}

function latentPosition(point) {
  const [[xmin, xmax], [ymin, ymax]] = state.bounds;
  const [width, height] = state.plotSize;
  const x = Math.max(margins.left, Math.min(width - margins.right, point[0]));
  const y = Math.max(margins.top, Math.min(height - margins.bottom, point[1]));
  return [xmin + (x - margins.left) / (width - margins.left - margins.right) * (xmax - xmin),
    ymin + (height - margins.bottom - y) / (height - margins.top - margins.bottom) * (ymax - ymin)];
}

function niceTicks(min, max) {
  const rough = (max - min) / 5;
  const exponent = 10 ** Math.floor(Math.log10(rough));
  const factor = rough / exponent;
  const step = (factor <= 1 ? 1 : factor <= 2 ? 2 : factor <= 5 ? 5 : 10) * exponent;
  const ticks = [];
  for (let value = Math.ceil(min / step) * step; value <= max; value += step) {
    ticks.push(Math.abs(value) < step * 1e-10 ? 0 : value);
  }
  return ticks;
}

function drawPlot() {
  if (!dataset) return;
  const context = plotContext;
  const [width, height] = state.plotSize;
  if (!width || !height) return;
  context.clearRect(0, 0, width, height);
  if (!is2D()) return;
  const [[xmin, xmax], [ymin, ymax]] = state.bounds;
  const left = margins.left, right = width - margins.right;
  const top = margins.top, bottom = height - margins.bottom;
  context.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
  context.lineWidth = 1;
  const xTicks = niceTicks(xmin, xmax), yTicks = niceTicks(ymin, ymax);
  for (const tick of xTicks) {
    const [x] = plotPosition([tick, 0]);
    context.strokeStyle = tick === 0 ? '#e1dcea' : '#f0edf4';
    context.beginPath(); context.moveTo(x, top); context.lineTo(x, bottom); context.stroke();
    context.fillStyle = '#b1a9bc'; context.textAlign = 'center';
    context.fillText(coordinateFormat.format(tick), x, bottom + 17);
  }
  for (const tick of yTicks) {
    const [, y] = plotPosition([0, tick]);
    context.strokeStyle = tick === 0 ? '#e1dcea' : '#f0edf4';
    context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
    context.fillStyle = '#b1a9bc'; context.textAlign = 'right';
    context.fillText(coordinateFormat.format(tick), left - 10, y + 3);
  }
  context.fillStyle = '#a296b1'; context.font = 'italic 12px Georgia, serif';
  context.textAlign = 'right'; context.fillText('z₁', right + 3, bottom + 30);
  context.textAlign = 'left'; context.fillText('z₂', left - 28, top - 4);
  context.save();
  context.beginPath(); context.rect(left, top, right - left, bottom - top); context.clip();
  if (state.showProjections) {
    context.globalAlpha = .65;
    for (const point of dataset.cloud) {
      const [x, y] = plotPosition(point.z);
      context.fillStyle = palette[point.label];
      context.beginPath(); context.arc(x, y, 1.8, 0, Math.PI * 2); context.fill();
    }
    context.globalAlpha = 1;
  }
  if (state.result) {
    const [x, y] = plotPosition(state.z);
    const inBounds = x >= left && x <= right && y >= top && y <= bottom;
    if (inBounds) {
      context.strokeStyle = '#9271c967'; context.lineWidth = 1; context.setLineDash([3, 4]);
      context.beginPath(); context.moveTo(left, y); context.lineTo(x, y); context.lineTo(x, bottom); context.stroke();
      context.setLineDash([]);
      context.fillStyle = '#7050db20';
      context.beginPath(); context.arc(x, y, 13, 0, Math.PI * 2); context.fill();
      context.fillStyle = '#7050db'; context.strokeStyle = '#fff'; context.lineWidth = 2.5;
      context.beginPath(); context.arc(x, y, 6, 0, Math.PI * 2); context.fill(); context.stroke();
      context.fillStyle = '#fff'; context.beginPath(); context.arc(x, y, 1.5, 0, Math.PI * 2); context.fill();
    }
  }
  context.restore();
}

function resizePlot() {
  const rect = plotCanvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  plotCanvas.width = Math.round(rect.width * ratio);
  plotCanvas.height = Math.round(rect.height * ratio);
  plotContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  state.plotSize = [rect.width, rect.height];
  drawPlot();
}

function plotEventPoint(event) {
  const rect = plotCanvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function nearestSample(point) {
  if (!is2D() || !state.showProjections) return null;
  let nearest = null, distance = 9;
  for (const sample of dataset.samples) {
    const pixel = plotPosition(sample.z);
    const d = Math.hypot(pixel[0] - point[0], pixel[1] - point[1]);
    if (d < distance) { nearest = sample; distance = d; }
  }
  return nearest;
}

function inPlot(point) {
  return point[0] >= margins.left && point[0] <= state.plotSize[0] - margins.right &&
    point[1] >= margins.top && point[1] <= state.plotSize[1] - margins.bottom;
}

function drawingPoint(event) {
  const rect = inputCanvas.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width * 280, (event.clientY - rect.top) / rect.height * 280];
}

function stroke(point, from) {
  const start = from || point;
  const distance = Math.hypot(point[0] - start[0], point[1] - start[1]);
  const steps = Math.max(1, Math.ceil(distance / 1.5));
  drawingContext.save();
  // A feathered grayscale stamp softens the edge. Taking the lighter value
  // keeps its profile stable regardless of pointer speed or stamp overlap,
  // and preserves the existing MNIST image and noise beneath the stroke.
  drawingContext.globalCompositeOperation = 'lighten';
  drawingContext.imageSmoothingEnabled = true;
  for (let i = 0; i <= steps; i++) {
    const fraction = i / steps;
    const x = start[0] + (point[0] - start[0]) * fraction;
    const y = start[1] + (point[1] - start[1]) * fraction;
    drawingContext.drawImage(brushCanvas, x - brushRadius, y - brushRadius);
  }
  drawingContext.restore();
  state.strokeFrom = point;
}

function bindEvents() {
  $('model-select').addEventListener('change', event => switchModel(Number(event.target.value)));
  $('show-projections').addEventListener('change', event => {
    state.showProjections = event.target.checked;
    $('plot-tooltip').hidden = true;
    plotCanvas.style.cursor = 'crosshair';
    $('digit-legend').style.visibility = state.showProjections ? 'visible' : 'hidden';
    $('projection-summary').textContent = state.showProjections
      ? '2 000 exemples · 50 sélectionnables' : 'Projections masquées';
    $('plot-help-text').textContent = state.showProjections
      ? 'Survolez les points pour repérer les images sélectionnables. Cliquez dans le plan ou faites glisser le curseur pour explorer.'
      : 'Cliquez dans le plan ou faites glisser le curseur pour explorer le décodeur.';
    plotCanvas.setAttribute('aria-label', state.showProjections
      ? 'Espace latent 2D : points MNIST colorés de 0 à 9. Survolez les points pour repérer les images sélectionnables ou déplacez le curseur. Les champs z1 et z2 permettent aussi de choisir une position au clavier.'
      : 'Espace latent 2D : projections MNIST masquées. Déplacez le curseur ou saisissez les coordonnées z1 et z2 pour explorer le décodeur.');
    drawPlot();
    announce(state.showProjections ? 'Projections MNIST affichées.' : 'Projections MNIST masquées.');
  });
  $('random-button').addEventListener('click', randomSample);
  $('clear-button').addEventListener('click', clearDrawing);
  $('noise-button').addEventListener('click', addNoise);
  $('zoom-in').addEventListener('click', () => zoomPlot(.5));
  $('zoom-out').addEventListener('click', () => zoomPlot(2));
  $('zoom-reset').addEventListener('click', () => {
    cancelPending(); state.bounds = dataset.bounds.map(range => [...range]); drawPlot();
  });
  $('restore-button').addEventListener('click', () => {
    cancelPending(); reconstruct();
    announce('Retour à la reconstruction de l’entrée.');
  });
  for (const id of ['z1', 'z2']) {
    $(id).addEventListener('change', () => {
      const z = [$('z1').valueAsNumber, $('z2').valueAsNumber];
      if (!z.every(Number.isFinite)) {
        $('z1').value = state.z[0].toFixed(3); $('z2').value = state.z[1].toFixed(3);
        announce('Saisissez deux coordonnées numériques.'); return;
      }
      cancelPending(); explore(z);
      announce(`Position libre : z1 ${coordinateFormat.format(z[0])}, z2 ${coordinateFormat.format(z[1])}.`);
    });
  }
  inputCanvas.addEventListener('pointerdown', event => {
    if (!state.ready || event.button !== 0 || state.drawing) return;
    event.preventDefault(); cancelPending();
    if (state.free) reconstruct();
    state.drawing = true;
    inputCanvas.setPointerCapture(event.pointerId);
    $('draw-hint').hidden = true;
    stroke(drawingPoint(event), null);
    schedule(readDrawing);
  });
  inputCanvas.addEventListener('pointermove', event => {
    if (!state.drawing) return;
    const samples = event.getCoalescedEvents?.();
    for (const sample of samples?.length ? samples : [event]) {
      stroke(drawingPoint(sample), state.strokeFrom);
    }
    schedule(readDrawing);
  });
  function finishDrawing() {
    if (!state.drawing) return;
    state.drawing = false; state.strokeFrom = null;
    cancelPending(); readDrawing();
    announce(`Dessin encodé. MSE : ${format.format(state.result.mse)}.`);
  }
  inputCanvas.addEventListener('pointerup', finishDrawing);
  inputCanvas.addEventListener('pointercancel', finishDrawing);
  inputCanvas.addEventListener('lostpointercapture', finishDrawing);

  plotCanvas.addEventListener('pointerdown', event => {
    if (!state.ready || !is2D() || event.button !== 0 || state.dragging) return;
    const point = plotEventPoint(event);
    if (!inPlot(point)) return;
    event.preventDefault(); cancelPending();
    $('plot-tooltip').hidden = true;
    const sample = nearestSample(point);
    if (sample) selectSample(sample);
    else explore(latentPosition(point));
    state.dragging = true;
    state.plotMoved = false;
    state.plotStartPoint = point;
    plotCanvas.setPointerCapture(event.pointerId);
  });
  plotCanvas.addEventListener('pointermove', event => {
    if (!state.ready || !is2D()) return;
    const point = plotEventPoint(event);
    if (state.dragging) {
      if (Math.hypot(...point.map((value, i) => value - state.plotStartPoint[i])) > 3) state.plotMoved = true;
      if (state.plotMoved) schedule(() => explore(latentPosition(point)));
      return;
    }
    const sample = inPlot(point) ? nearestSample(point) : null;
    const tooltip = $('plot-tooltip');
    tooltip.hidden = !sample;
    plotCanvas.style.cursor = sample ? 'pointer' : 'crosshair';
    if (sample) {
      tooltip.textContent = `Chiffre ${sample.label} · sélectionner`;
      tooltip.style.left = `${Math.min(point[0] + 12, state.plotSize[0] - 145)}px`;
      tooltip.style.top = `${Math.max(0, point[1] - 33)}px`;
    }
  });
  plotCanvas.addEventListener('pointerleave', () => {
    $('plot-tooltip').hidden = true;
  });
  function finishPlot(event) {
    if (!state.dragging) return;
    state.dragging = false;
    cancelPending();
    if (state.plotMoved) explore(latentPosition(plotEventPoint(event)));
    if (state.free) announce('Exploration libre. Aucune entrée de référence pour calculer l’erreur.');
  }
  plotCanvas.addEventListener('pointerup', finishPlot);
  plotCanvas.addEventListener('pointercancel', () => { state.dragging = false; cancelPending(); });
  plotCanvas.addEventListener('lostpointercapture', () => { state.dragging = false; });
  plotCanvas.addEventListener('keydown', event => {
    if (!state.ready || !is2D()) return;
    const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
    if (!directions[event.key]) return;
    event.preventDefault(); cancelPending();
    const step = event.shiftKey ? .5 : .1;
    explore(state.z.map((value, i) => value + directions[event.key][i] * step));
  });
  new ResizeObserver(resizePlot).observe($('plot-container'));
}

async function start() {
  bindEvents();
  if (!globalThis.tf) throw new Error('TensorFlow.js n’a pas été chargé.');
  const tf = globalThis.tf;
  // WebGL is preferred, with CPU fallback for browsers without GPU support.
  try {
    if (!await tf.setBackend('webgl') || !tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
      await tf.setBackend('cpu');
    }
  }
  catch { await tf.setBackend('cpu'); }
  await tf.ready();
  const [loadedNetwork, response] = await Promise.all([
    Autoencoder.load(tf, new URL('./assets/model/model.json', import.meta.url)),
    fetch(new URL('./assets/data/mnist.json', import.meta.url)),
  ]);
  network = loadedNetwork;
  modelCache.set(2, network);
  if (!response.ok) throw new Error('Les exemples MNIST sont introuvables.');
  dataset = await response.json();
  state.bounds = dataset.bounds.map(range => [...range]);
  const imagesResponse = await fetch(new URL(`./assets/data/${dataset.imagesFile}`, import.meta.url));
  if (!imagesResponse.ok) throw new Error('Les images MNIST sont introuvables.');
  imageBytes = new Uint8Array(await imagesResponse.arrayBuffer());
  if (imageBytes.length !== dataset.samples.length * 784) throw new Error('Les images sont incomplètes.');
  buildLibrary(); resizePlot();
  // First inference also warms up WebGL before enabling the controls.
  selectSample(dataset.samples.find(sample => sample.label === 3));
  state.ready = true;
  updateModelPresentation();
  $('workspace').setAttribute('aria-busy', 'false');
  announce('Modèle 2D prêt. Vous pouvez modifier l’entrée ou changer de modèle.');
}

start().catch(fatal);
