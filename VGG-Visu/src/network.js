import { layers } from "./core.js";

// Spatial sizes are compressed for readability; depth suggests channel count.
// Each feature volume represents one output; classifier volumes are informational only.
export function networkMarkup(active, image) {
  const starts = [175, 365, 555, 785, 1015];
  const heights = { 224: 220, 112: 150, 56: 100, 28: 68, 14: 46, 7: 30 };
  const volume = (x, h, w, d) => {
    const y = 200 - h / 2;
    return `<path class="volume-top" d="M${x},${y} l${d},-${d} h${w} l-${d},${d} Z"/>
   <path class="volume-side" d="M${x + w},${y} l${d},-${d} v${h} l-${d},${d} Z"/>
   <path class="volume-front" d="M${x},${y} h${w} v${h} h-${w} Z"/>`;
  };
  const current = layers[active];
  return `<svg viewBox="0 0 1650 355" role="group" aria-label="VGG16 complet : entrée, cinq blocs convolutionnels explorables, puis classifieur représenté sans calcul">
 <defs><linearGradient id="input-background" x2="0" y2="1"><stop stop-color="#d8e9ee"/><stop offset="1" stop-color="#a4bb76"/></linearGradient><marker id="flow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#a4b6ac"/></marker></defs>
 <rect class="classifier-background" x="1250" y="5" width="390" height="335" rx="12"/>
 <path class="network-flow" d="M70 200H1630" marker-end="url(#flow-arrow)"/>
 <text class="classifier-heading" x="1270" y="22">CLASSIFIEUR · NON CALCULÉ</text>
 <text class="classifier-text" x="1270" y="45">Repères non sélectionnables</text>
 <text class="block-label" x="30" y="22">ENTRÉE</text><text class="shape-label" x="30" y="45">224 × 224 × 3</text>
 <g aria-label="Image d’entrée 224 par 224, 3 canaux"><path fill="url(#input-background)" stroke="#637f72" d="M35 310V90L115 58V278Z"/>
 ${image ? `<image href="${image}" width="224" height="224" transform="matrix(.357143 -.142857 0 .982143 35 90)" preserveAspectRatio="none"/>` : ""}</g>
 ${[1, 2, 3, 4, 5].map((b) => `<text class="block-label" x="${starts[b - 1]}" y="22">BLOC ${b}</text><text class="shape-label" x="${starts[b - 1]}" y="45">${224 / 2 ** (b - 1)} × ${224 / 2 ** (b - 1)} × ${[64, 128, 256, 512, 512][b - 1]}</text>`).join("")}
 ${layers
   .map((l, i) => {
     const j = layers.slice(0, i).filter((x) => x.block === l.block).length;
     const x = starts[l.block - 1] + j * 48,
       h = heights[l.size],
       d = 20,
       w = l.name.endsWith("pool") ? 14 : 22;
     return `<g class="network-layer ${l.name.endsWith("pool") ? "pool" : "conv"} ${i === active ? "active" : ""}" role="button" tabindex="0" data-layer="${i}" aria-pressed="${i === active}" aria-label="Couche ${i + 1} : ${l.name}, ${l.type}, ${l.size} par ${l.size}, ${l.channels} canaux">
  <title>${l.name} — ${l.type} — ${l.size} × ${l.size} × ${l.channels}</title>
  <rect class="layer-hit" x="${x - 4}" y="${Math.min(200 - h / 2 - d - 4, 258)}" width="${w + d + 8}" height="${330 - Math.min(200 - h / 2 - d - 4, 258)}"/>
  ${volume(x, h, w, d)}
  <path class="layer-guide" d="M${x + w / 2} ${200 + h / 2 + 5} V305"/>
  <circle class="layer-number" cx="${x + w / 2}" cy="320" r="12"/><text class="number-label" x="${x + w / 2}" y="324" text-anchor="middle">${i + 1}</text></g>`;
   })
   .join("")}
 <text class="shape-label" x="1152" y="253">7 × 7 × 512</text>
 <g class="classifier-layer" role="img" aria-label="Aplatissement : 7 par 7 par 512 devient 25 088 valeurs, non calculé">
 <title>Flatten — 25 088 valeurs — non calculé</title>
 <text class="classifier-text" x="1270" y="175">25 088</text><text class="classifier-text" x="1270" y="248">Flatten</text></g>
 ${[
   {
     x: 1350,
     height: 156,
     name: "fc1",
     units: "4 096",
     activation: "Dense + ReLU",
   },
   {
     x: 1450,
     height: 156,
     name: "fc2",
     units: "4 096",
     activation: "Dense + ReLU",
   },
   {
     x: 1560,
     height: 96,
     name: "predictions",
     units: "1 000",
     activation: "Dense + softmax",
   },
 ]
   .map(
     (l) =>
       `<g class="classifier-layer" role="img" aria-label="${l.name}, ${l.units} unités, ${l.activation}, non calculé et non sélectionnable"><title>${l.name} — ${l.units} unités — ${l.activation} — non calculé</title>${volume(l.x, l.height, 12, 7)}<text class="classifier-text" x="${l.x + 10}" y="105" text-anchor="middle">1 × 1 × ${l.units}</text><text class="classifier-name" x="${l.x + 10}" y="300" text-anchor="middle">${l.name}</text><text class="classifier-text" x="${l.x + 10}" y="320" text-anchor="middle">${l.activation}</text></g>`,
   )
   .join("")}
 </svg><div class="network-caption"><div class="network-legend"><span><i class="legend-conv"></i>Convolution + ReLU</span><span><i class="legend-pool"></i>Max-pooling</span><span><i class="legend-active"></i>Couche sélectionnée</span><span><i class="legend-classifier"></i>Classifieur non calculé</span></div><p><strong>${current.name}</strong> · ${current.size} × ${current.size} × ${current.channels} · couche ${active + 1}/18</p></div><p class="muted network-note">Lecture de gauche à droite. Un volume = une couche ; les numéros identifient les 18 sorties explorables. Le classifieur en gris complète l’architecture VGG16 ; ses activations ne sont pas calculées. Hauteurs et profondeurs schématiques pour préserver la lisibilité. Sélection au clic, avec Entrée/Espace ou avec les flèches du clavier.</p>`;
}
