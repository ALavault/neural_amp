# Premier modèle VS100R : prédictions

Voie produit, aucune revendication scientifique. **Écrit le 2026-09-27, avant le démarrage du run**
`vs100r_gc_film_joint_s925`, prévu après G2.2. Mesures faites par
`scripts/product_vs100r_eval.py`, écrit en même temps que ce fichier et testé à blanc sur un
modèle non entraîné.

## Dispositif

- **Modèle** : SSM-WaveNet paramétrique, FiLM avec tête jointe, options du bras A de G2.
  Conditionné par **gain et contour seuls** : graves, médiums et aigus n'atteignent pas le départ
  de boucle (`QC.md`).
- **Données** : entrée = bouclage de chaque prise, cible = départ de boucle, 48 kHz.
  - Entraînement : 31 prises OD2 sans Power Dimension, regroupées en 15 couples gain × contour.
    Le couple (5, 5) compte 15 prises.
  - **test** : 8 prises. Sept ont une autre prise du même couple à l'entraînement ; seule la
    prise 16 (gain 2,5, contour 7,5) est un couple jamais vu.
  - **test_seen** : les 31 prises d'entraînement, sur leurs 30 s de test (audio jamais vu, même
    prise).
- **Évaluation** : ESR sur les 30 s de test de chaque prise, traitées d'un seul tenant depuis un
  état nul. La première demi-seconde est écartée. Médianes par groupe.

## Plafond mesuré avant le run

Deux prises du même couple gain × contour diffèrent de D = 0,019 à 0,028 (prises « centre »), et
jusqu'à 0,049 quand graves, médiums et aigus varient (`repeat.json`). Si chaque prise s'écarte
d'une moyenne commune, l'écart d'une prise à la moyenne de son couple vaut environ D/2, soit 0,01 à
0,025 en ESR.

## Prédictions à 48 kHz

- **P1, plafond** : ESR médian du test ≥ 0,010 ; il devrait tomber entre 0,01 et 0,05. Un
  modèle ne prédit pas la dérive propre à une prise.
- **P2, regroupement** : sur test_seen, les prises du couple (5, 5), qui partagent leur couple
  avec 14 autres, ont un ESR médian au moins 1,5 fois celui des prises seules dans leur couple.
  Le modèle prédit la moyenne du couple, et seules les prises regroupées s'en écartent.
- **P3, faible signal** : les prises au gain 0 (6, 7, 10 et 27) ont un ESR test_seen ≥ 0,3. Leur
  cible est à 40 % du bruit (rapport signal sur bruit d'environ 2 dB). Elles sont rapportées à
  part et hors des médianes.
- **P4** : l'ESR de la prise 16, seul couple jamais vu, est rapporté sans seuil.

## Prédictions : passage de 48 à 96 et 192 kHz

Le modèle entraîné à 48 kHz tourne à 96 et 192 kHz de deux façons : Δ rééchelonné (`delta`), et
relecture FOH puis Δ rééchelonné (`foh`). Il reçoit le bouclage natif à cette fréquence. Sa sortie
est comparée à la vraie capture.

- **P5, qui est le plus proche de l'appareil** : R = ESR(modèle à 192 kHz, ramené à 48 kHz, contre
  l'appareil) / ESR(modèle à 48 kHz contre l'appareil). Médiane sur les 35 prises hors gain 0.

  | R (`foh`) | lecture |
  |---|---|
  | ≤ 0,95 | tourner à 192 kHz rapproche le modèle de l'appareil : le repliement à 48 kHz compte |
  | entre 0,95 et 1,05 | indiscernable |
  | ≥ 1,05 | tourner à 192 kHz ajoute de l'erreur : l'écart entre fréquences est une erreur du modèle, pas le contenu aigu de l'appareil |

  Borne connue d'avance : au gain 5, l'appareil n'a que −42 dB de son énergie au-dessus de
  24 kHz (`hf.json`). Même replié en entier dans la bande, ce contenu ne déplacerait l'ESR
  que d'environ 10⁻⁴, moins de 1 % d'un ESR de 0,02. **A priori : R ≥ 0,95, à 80 %.** Un écart
  de plusieurs pour cent à 1 viendrait donc du comportement du modèle lui-même. C'est la question
  laissée ouverte par le plancher à −20 dB du Marshall (`demo/param/predictions_rate.md`), cette
  fois avec l'appareil comme référence.
- **P6** : l'écart entre le modèle à 192 kHz ramené à 48 kHz et le modèle à 48 kHz,
  |m₁₉₂→₄₈ − m₄₈|² / |m₄₈|², est rapporté par variante. Pour le Marshall, `foh` donnait 0,0097.
- **P7** : la part d'énergie du modèle à 192 kHz au-dessus de 24 kHz est rapportée face à celle de
  l'appareil, sans seuil. Le modèle n'a jamais vu cette bande.

## Ce que cela ne dira pas

- Une seule partition, une seule graine, un jeu d'exploration.
- Le départ de boucle ne contient ni la lampe, ni le correcteur, ni l'ampli de puissance.
- Le bouclage en entrée porte son propre bruit (−97,6 dBFS), que l'ampli n'a pas reçu.
