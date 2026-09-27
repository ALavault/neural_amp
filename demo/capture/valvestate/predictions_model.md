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

## Amendement du 2026-09-27, avant le run (relecture Codex, lecture seule)

Il remplace les passages concernés ci-dessus ; aucun seuil n'a été déplacé.

- **Jeu corrigé.** Le constructeur réparait l'échantillon 37 989 402 de la prise 14 au lieu du
  pic, en 37 989 410, parce qu'il repartait d'un temps arrondi. Le jeu a été reconstruit avec
  l'indice exact : sa cible de test coïncide avec la référence réparée à 1,5·10⁻⁸ près. Ce pic est
  dans le test de la prise 14.
- **Fenêtre notée** : 0,5 s écartée au début et 0,1 s à la fin, soit 29,4 s par prise.
- **P1 et P2.** D/2 est un plafond **supposé** : il suppose des écarts indépendants et comparables
  autour d'une moyenne commune. Le modèle reçoit en plus le bouclage propre à chaque prise, qui
  peut expliquer une partie de l'écart d'une prise à l'autre. P2 compare aussi le couple (5, 5) à
  d'autres couples gain × contour : un rapport de 1,5 ou plus serait compatible avec l'effet du
  regroupement, sans l'isoler.
- **P5, lecture.** R dit quelle fréquence de fonctionnement rapproche le modèle de l'appareil,
  pas pourquoi : bande d'entrée, discrétisation et repliement changent ensemble. La table devient :
  - R ≤ 0,95 : plus proche de l'appareil à 192 kHz, cause non isolée ;
  - entre 0,95 et 1,05 : écart inférieur à la tolérance conventionnelle de ±5 %, sans estimation
    d'incertitude ;
  - R ≥ 1,05 : plus loin de l'appareil à 192 kHz.
- **P5, borne.** Replier une énergie normalisée H sur une erreur E change l'ESR d'au plus
  H + 2√(EH), à cause du terme croisé : pour E = 0,02 et H = 10⁻⁴·², au plus 11,5 % de E. Si le
  contenu replié n'est pas corrélé à l'erreur, la variation tombe à environ 0,3 %. La valeur de
  −42 dB vaut au gain 5, pas à tous les gains. L'a priori R ≥ 0,95 à 80 % est maintenu.
- **P7 homogène.** La part de l'appareil est désormais mesurée sur la même fenêtre de test que le
  modèle, pic réparé, bruit de fond soustrait (silence de la même prise), dans la même bande :
  24–48 kHz pour le modèle à 96 kHz, 24–96 kHz pour le modèle à 192 kHz.

## Relance, écrite le 2026-09-27 avant les sondes

Le run `vs100r_gc_film_joint_s925` n'a rien appris : sa tête de sortie ReLU s'est fermée, et
le modèle sort une constante (ESR de test 1,05 à 1,29, échec consigné dans `demo/RUNS.jsonl`).
Diagnostic : `diag_dead_head.py` et relecture Codex. Rien de ce qui précède n'est mesuré sur lui.

- **Quatre sondes de 600 pas**, toutes lancées, dans cet ordre :
  1. tête LeakyReLU, garde-fou de polarité actif, graine 42 ;
  2. ReLU, garde-fou coupé, graine 42 ;
  3. LeakyReLU, garde-fou coupé, graine 42 ;
  4. ReLU, garde-fou actif, graine 43.
- **Règle, fixée d'avance** : le run complet `vs100r_gc_film_joint_s925_r2` prend le premier bras
  de cet ordre qui quitte le plateau, c'est-à-dire dernière perte de validation < 2,0 et ESR de
  test < 0,9 (`scripts/product_vs100r_probe_check.py`). Si aucun n'en sort, il n'y a pas de run
  complet.
- **P1 à P7 s'appliquent sans changement au run `_r2`**, évalué par
  `product_vs100r_eval.py --run-id vs100r_gc_film_joint_s925_r2`.

## Résultats des sondes (2026-09-27, 600 pas chacune)

| sonde | perte de validation (dernière époque) | ESR de test | issue |
|---|---|---|---|
| 1. LeakyReLU, garde-fou actif, graine 42 | 0,320 | 0,134 | sort du plateau |
| 2. ReLU, garde-fou coupé, graine 42 | 2,497 | 1,057 | reste bloquée |
| 3. LeakyReLU, garde-fou coupé, graine 42 | 0,320 | 0,134 | sort, identique à la 1 |
| 4. ReLU, garde-fou actif, graine 43 | 0,382 | 0,108 | sort du plateau |

**Règle appliquée** : le run `_r2` prend la sonde 1 (LeakyReLU, garde-fou actif, graine 42).

**Lecture, chaque case ne comptant qu'une trajectoire (n = 1), donc « compatible avec » :**

- **Tête** : sonde 1 contre le run d'origine, sonde 3 contre sonde 2. À graine 42, la ReLU reste
  bloquée, avec ou sans garde-fou ; la LeakyReLU apprend.
- **Garde-fou** : sondes 1 et 3 identiques. Le modèle apprenant, le garde-fou ne s'est pas
  déclenché. Couper le garde-fou ne sauve pas la ReLU (sonde 2).
- **Graine** : la sonde 4 apprend avec une tête ReLU. La trajectoire compte donc beaucoup : l'échec
  d'origine combine une tête ReLU fragile et une trajectoire défavorable. La LeakyReLU supprime ce
  risque par construction, car une tête fermée garde un gradient.
