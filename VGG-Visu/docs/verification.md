# Vérifications du 29 octobre 2026

Environnement : macOS Apple Silicon, Node 24.12.0, Python 3.9.6, TensorFlow 2.15.1, TensorFlow.js 4.22.0, Chrome headless, WebGL 2 via SwiftShader. Le GPU matériel et les autres navigateurs n’ont pas été mesurés.

## Résultats exécutés

- Export réel Keras ImageNet : 58 858 752 octets float32, 15 fragments ; chargement effectif confirmé par les calculs dans Chrome.
- Quatre tests Node réussis : prétraitement, dimensions/volume des sorties, indexation NHWC, cartes constantes/nulles/non finies.
- Prétraitement déterministe : écart maximal **0** avec Keras après arrondi des constantes en float32.
- Comparaison de **toutes les valeurs des 18 sorties** : aucune violation de `atol=0.1, rtol=0.01`. Maximum absolu observé **0,017578125** ; maximum relatif au pic d’une couche **1,70 × 10⁻⁶** environ. Les valeurs proches de zéro justifient une tolérance absolue ; la tolérance retenue laisse une marge pour les différences d’accumulation entre moteurs. Cette vérification porte sur une entrée, sans garantie universelle sur tous les moteurs.
- Les 18 formes sont conformes, de `[1,224,224,64]` à `[1,7,7,512]`. Les sorties convolutionnelles du modèle source utilisent ReLU.
- Navigation sur les 18 couches et accès au dernier canal de chacune ; sélection de palette chaleur et normalisation commune : compteur d’inférences inchangé à **1**. L’accès aux canaux intermédiaires repose sur le même index NHWC testé et la pagination, sans test visuel manuel de chaque canal.
- Quatre nouveaux chargements : **26 tenseurs**, **58 858 752 octets actifs** à chaque mesure. Le pool WebGL réserve **530 055 280 octets**, dont **471 179 376** réutilisables. Pas de croissance observée sur cette séquence ; ce n’est pas une mesure exhaustive du tas JS ou du processus Chrome.
- Import local PNG et remplacement pendant le calcul : la version finale reste sélectionnée, 18 sorties disponibles. Le contrôle de version empêche l’ancien cache de remplacer le nouveau.
- Mode une couche : une seule sortie en cache, même nombre de tenseurs actifs.
- Import JPEG d’une image rouge de 320 × 160, recadrage et calcul : succès ; erreur de décodage d’un faux PNG : message français attendu.
- Build chargé depuis **`/depot/`**, y compris les 15 fragments et l’exemple : calcul réussi.
- Requêtes observées : GET locaux et URLs blob locales ; aucun envoi de l’image ou requête d’inférence distante. Le code des images utilise Canvas, sans fetch de téléversement.
- Capture bureau inspectée. À 390 px de largeur, largeur du document = 390 px ; le SVG défile dans son conteneur.
- `npm test`, build Vite 6.4.5 et audit npm : succès, aucune vulnérabilité signalée au moment de l’installation finale.

## Mesures de durée

Premier calcul de l’exemple dans l’application : **3,92 s**. Même calcul au premier chargement du build sous `/depot/` : **3,93 s**. Sur la première séquence, les calculs réchauffés étaient autour de **1,3 s**. Comparaison numérique complète avec lecture des références : **4,46 s**, mesure incluant ces lectures. Ces résultats SwiftShader ne sont pas une estimation des performances mobiles ou d’un GPU matériel.

## Limites de validation

Pas de publication GitHub Pages effective, de test Firefox/Safari, de test EXIF orienté par un appareil photo, de vérification manuelle avec lecteur d’écran, de test de perte de contexte ou de stress mémoire prolongé. La composition sur blanc est implémentée mais n’a pas de fixture PNG transparente dédiée. L’absence de débordement mobile a été mesurée ; les performances mobiles restent inconnues. L’image d’exemple est une illustration originale, pas une photo ImageNet. Aucune attribution Grad-CAM.

Les rapports JSON contiennent les mesures détaillées ; les captures servent uniquement à la vérification de l’interface et ne sont pas utilisées par le site.

## Évolution du cadrage et du cache

Le sélecteur de cache et le mode une couche ont été retirés après ces mesures historiques ; le cache des 18 sorties est conservé. `scripts/framing-check.mjs` vérifie les pixels de marge et du centre sur des rectangles portrait et paysage, le retour au recadrage depuis le fichier original et le calcul des 18 sorties à chaque changement.

## Correction du service GitHub Pages

La page distante servait le HTML source (`/src/main.js`) sans compilation Vite. Le build est maintenant versionné dans `site/` avec une entrée de redirection locale à cet outil. `scripts/pages-check.mjs` a vérifié le chargement sous `/cv-educational-tools/VGG-Visu/`, la redirection vers `site/`, le style calculé en grille, les neuf miniatures, les 18 couches et une inférence réelle, sans ressource manquante ni erreur JavaScript. Aucune configuration commune ou fichier des autres outils n’a été modifié.
