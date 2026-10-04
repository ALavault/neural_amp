# Analyse de Hammerstein de la VS100R et de ses modèles (exploration)

Voie produit, régime exploration, 2026-10-04. Chaque affirmation est marquée **établie** (mesure
citée) ou **supposée**. Instrument : `src/fssr_nam/product/hammerstein.py` ; analyse :
`vs100r_hammerstein.py` → `vs100r_hammerstein.json`.

## Instrument

- Identification d'un modèle de Hammerstein parallèle par sinus glissant exponentiel. La matrice
  qui relie les réponses harmoniques D_k aux filtres G_n est **mesurée** en passant les
  puissances pures de la même entrée dans la même chaîne, au lieu d'être dérivée par
  trigonométrie : phases du sinus glissant non synchronisé, fondus et décimation se compensent.
- **Établi** : sur des systèmes synthétiques aux filtres connus, l'erreur relative médiane est
  d'environ 10⁻³, à 48 kHz comme après décimation depuis 192 kHz (`tests/unit/test_hammerstein.py`).
- Données : les trois sinus glissants de chaque prise VS100R (20 Hz–20 kHz, 5 s, −20, −10 et
  −3 dBFS, à 192 kHz). Référence : l'entrée bouclée de la prise. Analyse à 48 kHz après la
  décimation des données d'entraînement. Modèles : 3 graines à 48 kHz (A48), 3 à 192 kHz
  (B192), alimentés par la même entrée bouclée. 8 prises de test.

## Résultats

**Un modèle de Hammerstein d'ordre 5 ne décrit pas l'OD2 (établi).**
- Les filtres G_n changent fortement avec le niveau : G₁ baisse d'environ 21 dB entre −20 et
  −3 dBFS (prise 17).
- Prédire une autre partie du signal avec les filtres identifiés échoue. Sur le bruit rose
  à −30 dBFS efficace, l'ESR médiane vaut 2,8 (modèle identifié à −20 dBFS), 1,5 (−10) et 1,05
  (−3). Sur la guitare de test, le modèle à −20 dBFS diverge (ESR de l'ordre de 10⁷) et celui à
  −3 dBFS donne 1,3.
- La comparaison utile porte donc sur les **réponses harmoniques D_k** mesurées directement, à
  niveau égal, et non sur les G_n.

**Répétabilité de l'appareil (établi)** : sur les 15 prises du réglage (5 ; 5), l'écart-type
de |D_k| entre prises vaut environ 0,3 dB pour tous les ordres et tous les niveaux (1,1 dB pour
D₄ à −20 dBFS), médiane sur 200 Hz–8 kHz.

**Harmoniques impaires bien reproduites, paires non (établi)** : écart absolu médian entre
modèle et appareil, en dB, sur 200 Hz–8 kHz, 8 prises.

| niveau | bras | D₁ | D₂ | D₃ | D₄ | D₅ |
|---|---|---|---|---|---|---|
| −20 dBFS | A48 | 0,21 | 12,5 | 0,28 | 12,8 | 0,36 |
| −20 dBFS | B192 | 0,35 | 16,6 | 0,57 | 15,9 | 0,75 |
| −10 dBFS | A48 | 0,25 | 4,6 | 0,34 | 5,1 | 0,44 |
| −10 dBFS | B192 | 0,35 | 6,0 | 0,61 | 7,5 | 0,73 |
| −3 dBFS | A48 | 0,27 | 5,0 | 0,41 | 4,7 | 0,52 |
| −3 dBFS | B192 | 0,40 | 7,1 | 0,76 | 5,8 | 0,98 |

- **Sens de l'erreur (établi)** : les modèles produisent **trop** d'harmoniques paires.
  L'appareil a D₂ à −57 dB sous D₁ à −20 dBFS et à environ −35 dB à −10 et −3 dBFS : son
  écrêtage est presque symétrique. Écart signé médian de D₂ : A48 +12,3, +5,0 et +4,7 dB ;
  B192 +16,1, +3,0 et +5,4 dB (−20, −10 et −3 dBFS).
- Les harmoniques impaires sont à 0,2–1 dB de l'appareil, soit du même ordre que sa
  répétabilité. A48 est un peu plus proche que B192 sur tous les ordres.

## Où naissent les harmoniques paires (`gate_harmonics.py`, établi sur un modèle et une prise)

Modèle `vs100r_gc_film_joint_s925_r2`, prise 17, sinus glissants des trois niveaux. Rapport
D₂/D₁ en dB (énergie, 200 Hz–8 kHz), médiane sur les canaux de chaque signal interne.

| niveau | ampli | sorties des portes (blocs 0–7) | somme des skips | après la dernière 1×1 (sortie) |
|---|---|---|---|---|
| −20 dBFS | −53,0 | −32,6 à −9,5 | −23,4 | −40,0 |
| −10 dBFS | −34,5 | −21,1 à −3,4 | −20,8 | −30,9 |
| −3 dBFS | −35,2 | −16,5 à −2,3 | −20,2 | −30,5 |

- **La porte `tanh(a)·sigmoid(b)` crée les harmoniques paires** : dans chaque bloc, D₂/D₁ monte
  de ses entrées a et b à sa sortie, de 4 à 70 dB (bloc 0 : de −67 à −33 dB à −20 dBFS, de −84 à
  −15 dB à −3 dBFS).
- **La tête ne les crée pas** : les LeakyReLU ne changent pas le rapport (à 0,1 dB près).
- **La dernière couche 1×1 les annule en partie** : elle abaisse D₂/D₁ de 11 à 21 dB. L'excès
  mesuré en sortie est le reste d'une annulation imparfaite d'harmoniques paires fabriquées en
  grande quantité à l'intérieur du réseau.

## Ce que cela suggère

- **Établi** : l'excès de D₂ dépend surtout de la prise, donc du réglage, et peu de la graine.
  À −20 dBFS, il va de +5,1 à +26,2 dB selon la prise pour la graine 42, avec le même profil pour
  les graines 43 et 44 : la prise 16 est la plus forte pour les trois (+26 à +30 dB), la prise 30
  la plus faible (+0,3 à +5,6 dB).
- **Supposé** : l'annulation par la dernière couche 1×1 est réglée pour les réglages fréquents
  à l'entraînement et moins bien ailleurs. La prise 16, la plus en excès, est aussi la seule à un
  couple gain–contour non vu. Mesure qui la départagerait : l'excès de D₂ en fonction du nombre
  de prises d'entraînement au même couple, sur toutes les prises (y compris celles d'entraînement).
- **Supposé** : une porte symétrique (impaire) n'aurait rien à annuler. Mesure qui la
  départagerait : un entraînement avec une porte impaire, sur le même protocole. C'est un
  changement de modèle, à préinscrire.
- **Supposé** : poids perceptif faible à bas niveau, plus net aux niveaux de jeu (2ᵉ harmonique
  du modèle vers −30 dB contre −35 dB pour l'ampli). Non vérifié à l'écoute.

## Non fait

- Le bras C (FFT de 288 000 points sans TF32) n'existait pas encore : à ajouter à la fin de sa
  file.
- Aucune analyse par fréquence fine ni par réglage au-delà des médianes.
- L'analyse interne ne porte que sur un modèle et une prise.

## Excès de 2ᵉ harmonique : réglages fréquents ou gain ? (prédictions écrites avant la mesure, 2026-10-04)

Mesure : `d2_vs_familiarity.py`, toutes les prises hors faible signal (entraînement comprises :
les sinus glissants n'ont jamais servi à l'entraînement), 3 niveaux, les 3 modèles A48. Excès =
niveau de D₂ du modèle moins celui de l'appareil, médiane sur 200 Hz–8 kHz.
- **Familiarité** : l'annulation est réglée pour les couples fréquents. Prédiction : au gain 5,
  l'excès médian du couple (5 ; 5), 15 prises d'entraînement, est inférieur d'au moins 5 dB à
  celui des couples (5 ; x) à 1 prise.
- **Gain** : l'excès suit surtout le gain, car l'appareil distord peu à gain faible et le plancher
  de 2ᵉ harmonique du modèle y ressort. Prédiction : à gain fixé, pas d'écart de plus de 3 dB entre
  (5 ; 5) et les (5 ; x) ; et une corrélation de Spearman entre gain et excès inférieure à −0,5.
- Les deux peuvent tenir à la fois ; si aucune ne tient, l'excès dépend d'autre chose (contour,
  prise).

## Antisymétrisation à l'inférence (candidat 1 ; prédictions écrites avant la mesure, 2026-10-04)

Mesure : `antisym.py`. Pour les 6 modèles existants (A48 et B192, graines 42 à 44), sur les 8
prises de test (région notée, 48 kHz) : ESR de f(x) et de y_impair = (f(x) − f(−x))/2. Contrôle :
D₂ de y_impair sur le sinus glissant de −10 dBFS de la prise 17, qui doit être au plancher.
- Si la partie paire du modèle n'est qu'un artefact, l'ESR baisse pour la plupart des prises et des
  modèles. Lecture : « l'antisymétrisation aide » si la médiane sur modèles de la variation de
  la moyenne des ln ESR est ≤ −0,05 et que 5 modèles sur 6 baissent.
- Si la partie paire porte aussi une partie utile (asymétrie réelle de l'ampli, composante
  continue), l'ESR monte ou ne bouge pas.

### Résultats familiarité contre gain (`d2_vs_familiarity.json`)

Excès médian de D₂ (modèle − appareil, dB), 3 modèles A48, toutes les prises hors faible signal.

| niveau | gain 2,5 | gain 5 | gain 7,5 | gain 10 | (5 ; 5), 15 prises | autres (5 ; x), 1 prise | Spearman gain–excès |
|---|---|---|---|---|---|---|---|
| −20 dBFS | 16,2 | 9,8 | 12,2 | 2,8 | 8,9 | 11,4 | −0,43 |
| −10 dBFS | 23,7 | 2,1 | 6,1 | 5,5 | 1,6 | 2,5 | −0,29 |
| −3 dBFS | 24,3 | 4,9 | 4,5 | 4,1 | 4,9 | 3,9 | −0,15 |

- **Familiarité : non soutenue** (écart (5 ; 5) contre (5 ; x) de 2,5, 0,9 et −1,0 dB, sous les
  5 dB prédits, et de signe opposé à −3 dBFS).
- **Gain : en partie.** Pas d'écart de plus de 3 dB à gain fixé (tenu), mais le Spearman n'atteint
  pas −0,5 (non tenu) : la relation n'est pas monotone, l'excès se concentre au gain 2,5.
- **Lecture (établie sur ces niveaux)** : D₂/D₁ de l'appareil vaut −58 à −68 dB au gain 2,5,
  contre −34 à −55 dB aux autres gains ; celui des modèles reste vers −30 à −40 dB partout. Les
  modèles ont un plancher de 2ᵉ harmonique qui ne suit pas l'appareil quand il devient plus
  symétrique, à gain faible.
- L'hypothèse « annulation réglée pour les réglages fréquents » est donc écartée ; la prise 16
  ressortait parce qu'elle est à gain 2,5, pas parce que son couple n'était pas vu.

### Résultats de l'antisymétrisation (`antisym.json`)

| modèle | Δ moyenne ln ESR (impair − tel quel) | D₂/D₁ tel quel → impair (sinus −10 dBFS, prise 17) |
|---|---|---|
| A48 s42 | −0,028 | −30,9 → −52,9 dB |
| A48 s43 | +0,003 | −30,1 → −46,0 dB |
| A48 s44 | −0,021 | −29,6 → −48,5 dB |
| B192 s42 | −0,054 | −35,5 → −81,3 dB |
| B192 s43 | −0,116 | −35,5 → −76,6 dB |
| B192 s44 | −0,074 | −31,1 → −57,4 dB |

Appareil, même sinus : −34,5 dB.

- **Lecture préinscrite : « n'aide pas »**, de justesse. Médiane −0,041, au-dessus du seuil de
  −0,05 ; 5 modèles sur 6 baissent, ce qui tient.
- **Descriptif** : le gain est presque nul pour A48 (0 à 3 % d'ESR) et de 5 à 11 % pour B192
  (jusqu'à 27 % sur une prise). Sur la guitare de test, la partie paire des modèles pèse −25 à
  −39 dB de leur partie impaire : la retirer change peu l'ESR, même si elle est surtout parasite.
- **Conséquence** : pour l'ESR, l'excès d'harmoniques paires est un contributeur mineur ; son
  intérêt est perceptif (une 2ᵉ harmonique que l'ampli n'a pas, surtout à gain faible) et
  d'empreinte. Le contrôle montre que l'antisymétrisation fait bien ce qu'elle doit (D₂ au
  plancher), au prix d'une inférence doublée ; elle retire aussi la petite partie paire réelle
  de l'appareil (−34,5 dB sur ce sinus).
