# Montée en fréquence : maintien appris ou repliement appris ? — prédictions

Voie produit, mesure produit. Écrit le 2026-09-23 **avant** tout chiffre de ces bras.
Hypothèses : `reports/neural_amp-research/2026-09-23/brainstorm_sr_param.md`, pistes A et
B. La loi en τ² (τ = 1/48000 − 1/fs) qui motive A a été relevée **après coup** dans les
résultats de `RESULTS.md` : ce test est sa première confrontation préenregistrée.

## Bras (mêmes checkpoints `last` que `RESULTS.md` : 3 SSM-WaveNet, 3 S4-TF-L-16)

Tous les scores sont calculés sur l'intérieur des segments (0,1 s retirée à chaque bord),
avec rééchantillonnage du fichier entier, comme les contrôles post hoc.

- `delta` : Δ rééchelonné (bras existant).
- `delta_hold` : Δ rééchelonné **et** porte de maintien de 48 kHz conservée. Pour chaque
  mode, le coefficient d'entrée ZOH (e^{λdt′} − 1)/λ devient
  (e^{λdt0} − 1)/λ · dt′/dt0, où dt0 est le pas appris (1/48000 s en temps physique).
  C'est l'identité à 48 kHz, et c'est causal. Implémenté en multipliant C par le rapport
  des deux coefficients ; même formule pour les DSSM de nablafx.
- Montée : à 88,2, 96 et 192 kHz, sortie redescendue à 48 kHz et comparée au modèle à
  48 kHz (`band48_vs_48k`, cohérence dans la bande commune).
- Descente : à 44,1 kHz contre la capture native de l'appareil.
- **Test B** : les mêmes bras à 96 et 192 kHz avec l'entrée multipliée par 0,1 (−20 dB),
  référence comprise (le modèle à 48 kHz sur l'entrée atténuée).

## Prédictions (SSM-WaveNet, jugées graine par graine)

- **H1 (A, le maintien domine)** À 96 et 192 kHz, l'ESR de `delta_hold` ≤ 1/5 de celui de
  `delta`. Réfutée si la réduction est inférieure à ×2 ; partielle entre ×2 et ×5.
- **H2 (A, descente)** À 44,1 kHz contre l'appareil, `delta_hold` à ±2 % de `delta` (τ n'y
  vaut que −1,8 µs).
- **H3 (B, départage)** Rapport ESR(−20 dB) / ESR(0 dB) pour `delta` à 96 et 192 kHz.
  Si le maintien domine : entre 0,5 et 2. Si le repliement appris domine : < 0,2. Entre
  0,2 et 0,5 : mélange, rapporté comme tel.
- **H4 (cohérence)** Si H1 tient, le rapport ESR 192/96 de `delta_hold` s'écarte nettement
  de 2,25 (la loi en τ² disparaît avec sa cause). Pas de valeur prédite.

## Contraste S4-TF-L-16

- **H5** La correction réduit aussi l'écart de S4, mais moins que pour SSM-WaveNet (le
  TFiLM à blocs de 128 échantillons n'est pas corrigé) : réduction entre ×1,5 et ×5.

Aucun checkpoint écarté ; un échec sur une graine est rapporté comme tel.
