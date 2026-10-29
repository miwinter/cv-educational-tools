# VGG16 — Explorateur pédagogique

Application statique en français : image locale → VGG16 ImageNet réel → 13 sorties après ReLU et 5 sorties de max-pooling. Aucun classifieur, serveur de calcul, service d’inférence ou CDN à l’exécution. Les poids sont inclus dans `public/model/`.

## Installation et lancement

Node.js 22 ou 24, npm. Depuis ce dossier :

```sh
npm ci
npm test
npm run dev
```

Ouvrir l’URL affichée (normalement http://127.0.0.1:5173). Le double-clic sur HTML ne fonctionne pas. WebGL avec textures float32 est requis ; une erreur explicite est affichée sinon. Aucun repli CPU automatique.

## Préparer et reproduire les poids

Python 3.9–3.11, TensorFlow 2.15.1 / Keras 2.15, NumPy 1.26.4, h5py 3.10.0. Sur Apple Silicon, utiliser une version de Python ARM64. Les versions transitives de l’environnement utilisé sont dans `requirements-model.lock.txt`.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-model.txt
.venv/bin/python scripts/prepare_model.py
```

Le script instancie `tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))`. Il exporte la topologie Keras 2 compatible LayersModel et un manifeste TFJS standard, directement, pour éviter la chaîne de dépendances du convertisseur généraliste. Chaque poids est enregistré en float32 little endian, sans quantification ; les morceaux font au plus 4 Mio. Cette conversion est vérifiée par comparaison complète des sorties. `provenance.json` contient la version source et le SHA256 des poids concaténés. Les références de tests sont régénérées, jamais utilisées par l’application.

## Utilisation

Choisir une image JPEG/PNG, la déposer dans la zone d’entrée, cliquer sur une des neuf miniatures fournies, ou charger le paysage d’exemple. Les fichiers JPEG/PNG du dossier `images/` sont automatiquement intégrés par Vite au lancement et au build ; leurs URLs restent compatibles avec un sous-répertoire GitHub Pages. Cliquer sur une miniature utilise le même prétraitement et le même calcul réel qu’un import local. Le synoptique SVG représente les sorties sous forme de volumes en perspective, regroupés dans les cinq blocs : gris pour convolution + ReLU, orange pour max-pooling, vert pour la sélection. Le classifieur complet est aussi dessiné en gris et non sélectionnable : Flatten (25 088 valeurs), fc1 et fc2 (4 096 unités, ReLU), puis predictions (1 000 unités, softmax). Il est fourni à titre de repère et ne fait pas partie du modèle calculé. Les dimensions indiquées sont exactes ; les proportions des volumes sont schématiques. L’entrée reprend l’image analysée. Sélectionner une couche au clic ou avec Tab puis Entrée/Espace ; les flèches gauche/droite sélectionnent la couche voisine, Début/Fin la première/dernière. Chaque bouton de la mosaïque est **un canal**, numéroté à partir de **1**. Les 512 canaux sont accessibles par pages de 32 ou directement par numéro. L’agrandissement sans lissage conserve les cellules natives.

Les minima, maxima et moyennes proviennent des activations avant mise en couleur. La normalisation par carte révèle les structures sans permettre une comparaison d’intensité entre canaux. La normalisation commune utilise le minimum et maximum de toute la couche. En normalisation par carte, les cartes constantes, y compris nulles, sont noires et signalées ; en normalisation commune, elles gardent leur intensité relative uniforme ; les valeurs non finies sont exclues des statistiques et affichées en magenta.

## Prétraitement exact

Décodage avec orientation EXIF par `createImageBitmap(..., {imageOrientation:'from-image'})`. Recadrage carré central et redimensionnement à 224 × 224 par Canvas (équivalent géométrique à redimensionner le côté court puis recadrer). Composition alpha sur blanc. L’aperçu montre ces pixels RGB, distincts du tenseur float32. Conversion RGB → BGR puis soustraction de `[103.939, 116.779, 123.68]`, sans autre normalisation. Référence : [documentation officielle Keras VGG16](https://keras.io/api/applications/vgg/vgg_models/).

## Calcul et mémoire

Un modèle à 18 sorties calcule une fois par image. Les sorties sont lues en tableaux CPU NHWC ; les tenseurs d’entrée et de sortie sont libérés dans `finally`. Le cache contient **60 311 552 octets (57,52 Mio)**. Les poids occupent **58 858 752 octets (56,13 Mio)**. Le pic est supérieur : sorties GPU et CPU simultanées, intermédiaires, allocations et caches WebGL, pixels et Canvas. `tf.memory()` ne mesure pas toute la mémoire du navigateur.

Le mode « une couche » conserve uniquement la sortie choisie (12,25 Mio au maximum) et recalcule le chemin jusqu’à cette couche à sa sélection ; les changements de canal et palette ne recalculent jamais l’inférence. Les wrappers partagent les poids. La mosaïque est limitée à 32 miniatures. Les calculs sont sérialisés et versionnés : un ancien résultat ne peut pas remplacer l’image courante. Un appel WebGL synchrone peut toutefois retarder brièvement l’interface. Le contexte WebGL est libéré par le navigateur à la fermeture de la page.

## Vérifications

```sh
npm test
# Régénérer les références puis lancer Vite dans un autre terminal
.venv/bin/python scripts/prepare_model.py
npm run build
npm run dev
# Dans un autre terminal :
node scripts/browser-check.mjs
node scripts/import-check.mjs
```

Le script navigateur utilise Chrome macOS (`/Applications/Google Chrome.app/...`) et SwiftShader ; adapter `executablePath` pour une autre machine. Il compare toutes les valeurs des 18 sorties à Keras sur une entrée déterministe `RGB[i]=(i*37)%256`. Tolérance par élément : `|TFJS-Keras| ≤ 0.1 + 0.01*|Keras|`, adaptée aux accumulations WebGL ; les écarts maximaux et moyens restent consignés, sans prétendre à une égalité bit à bit. Voir `docs/numerical-results.json`, `docs/browser-results.json` et `docs/verification.md`.

## Build et GitHub Pages

```sh
npm run build
npm run preview
```

`dist/` contient tout, y compris TFJS et les poids. Le build utilise `base=./` ; modèles, fragments et exemple sont relatifs au sous-répertoire. Aucun routage serveur. Dans ce dépôt regroupant plusieurs outils, le workflow est situé à la racine Git : `../.github/workflows/vgg-pages.yml`. Configurer Pages → Source → GitHub Actions puis lancer manuellement le workflow `VGG16 — GitHub Pages`. Il attend les poids versionnés dans `public/model/` ; ne pas placer les fragments sous Git LFS sans gérer leur téléchargement au checkout. Chaque fragment est inférieur à 100 Mo. Aucune publication n’a été effectuée.

Les premières visites nécessitent HTTP(S) et un réseau pour charger le site et ses poids. Aucun service worker ni mode hors ligne complet n’est promis. Une fois les ressources chargées, l’image ne quitte pas le navigateur. La CSP du serveur peut renforcer cette propriété ; le code n’envoie que des GET pour ses ressources locales.

## Provenance et conditions d’utilisation

- Architecture : Simonyan et Zisserman, _Very Deep Convolutional Networks for Large-Scale Image Recognition_, ICLR 2015, [article](https://arxiv.org/abs/1409.1556).
- Poids : distribution officielle Keras, [fichier sans classifieur](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5), MD5 vérifié par Keras `6d6bbae143d832006294945121d1f1fc`. [Code source Keras](https://github.com/keras-team/keras/blob/v2.15.0/keras/applications/vgg16.py) sous Apache-2.0. La licence du code ne constitue pas à elle seule une concession sur le jeu ImageNet : consulter ses [conditions d’accès](https://www.image-net.org/download.php) et celles de la distribution des poids pour une redistribution ou un usage particulier.
- TensorFlow.js : Apache-2.0 ; Vite : MIT. Les avis des dépendances sont conservés par leur distribution npm.
- `images/` : neuf images fournies avec le dossier de travail. Leur provenance et leur licence ne sont pas précisées dans les fichiers fournis ; aucune licence supplémentaire ne leur est attribuée par ce projet.
- `public/example.svg` : paysage géométrique original créé pour ce projet, placé sous CC0-1.0 ; aucune photo tierce. Il illustre le calcul, pas la qualité de classification ImageNet.

## Limites

Cible ordinateur moderne. Pas de garantie de performance mobile, pas de Grad-CAM ou superposition. L’interpolation Canvas peut varier légèrement entre navigateurs ; les tests numériques utilisent directement les pixels déterministes pour isoler le réseau. Les performances SwiftShader ne préjugent pas des performances GPU. Une perte de contexte WebGL demande un rechargement. Les formats EXIF et les petits écrans doivent être vérifiés sur les navigateurs effectivement ciblés avant diffusion large.
