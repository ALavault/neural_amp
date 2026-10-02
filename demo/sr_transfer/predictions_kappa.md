# κ : le décalage du réseau au transfert de fréquence d'échantillonnage

Voie produit. **Pré-enregistré le 2026-10-02, avant toute mesure de ce document.** Lecture seule,
CPU, checkpoints existants. Script : `demo/sr_transfer/kappa.py`, sortie `kappa.json`.

## Ce qu'on sait déjà (à ne pas compter comme confirmation)

- En ZOH (convention S4D), l'avance d'un mode vaut exactement 1/s − 1/(eˢ − 1) échantillon,
  avec s = −Re(λΔ) : environ ½ pour un mode lent, moins pour un mode rapide.
- Rééchelonner Δ de 48 kHz à fs′ devrait donc retarder la sortie, ramenée à 48 kHz, de
  κ·(1 − 48 000/fs′)/2 échantillon à 48 kHz, soit 0,25 κ à 96 kHz et 0,375 κ à 192 kHz.
- Une relecture critique du 2026-10-01 a estimé **κ ≈ 1,86** pour le VS100R, de façon
  indirecte : en inversant les courbes ESR(décalage) contre l'appareil (35 prises), post hoc,
  sur un seul checkpoint. La mesure ci-dessous est directe, modèle contre modèle.
- Sur Big Muff, le rapport ESR 192/96 du transfert `delta` (2,08 à 2,14) est proche du carré
  attendu d'un décalage pur (2,25).

## Mesure

- **Checkpoints** : `vs100r_gc_film_joint_s925_r2` et `paired_ssm_wavenet_s925`, `s736` et
  `s688` (SSM-WaveNet, ZOH).
- **Entrées.**
  - VS100R : les 8 prises de test (`split == "test"`) de `product_vs100r_eval.takes()`. Le
    bouclage natif à 192 kHz est rééchantillonné à 48, 96 et 192 kHz, comme dans
    `product_vs100r_eval.py` ; les réglages sont ceux de la prise ; la partie notée est
    `scored()`.
  - Big Muff : les segments de test du banc à 48 kHz, rééchantillonnés en un seul fichier
    (`product_sr_transfer.whole`) vers 96 et 192 kHz ; la partie notée est `interior()`.
- **Comparaison.** La sortie du modèle rééchelonné en Δ (`set_rate`, sans correction), ramenée
  à 48 kHz, est comparée à la sortie du même modèle à 48 kHz.
- **Retard par bande** : méthode de `demo/capture/valvestate/group_delay.py`.
  - Spectres croisés de Welch (Hann, 4096, moitié de recouvrement), bandes d'octave centrées sur
    500 Hz, 1, 2, 4 et 8 kHz, cases de cohérence ≥ 0,5.
  - Le retard est l'opposé de la pente de la phase, pondérée par |spectre croisé|. Il est positif
    quand la sortie à fs′ est en retard sur la sortie à 48 kHz.
- **κ** = retard / ((1 − 48 000/fs′)/2), par bande. Le κ d'un checkpoint à une fréquence est la
  médiane sur les bandes et les prises (ou sur les bandes du fichier entier, pour Big Muff).
- **Blocs seuls (VS100R)** : on rééchelonne le Δ d'un seul bloc (les 7 autres restent à leur Δ de
  48 kHz), pour chacun des 8 blocs, à 96 et 192 kHz. Le retard δ_b est mesuré de la même façon.

## Lectures, seuils fixés ici

Les retards sont en échantillons à 48 kHz.

- **R1, forme (1 − T′/T)** : pour chaque checkpoint, κ(192)/κ(96) dans [0,8 ; 1,25] → forme
  confirmée ; sinon réfutée pour ce checkpoint.
- **R2, accord avec l'estimation indirecte** : |κ_VS100R − 1,86| ≤ 0,3 (médiane des deux
  fréquences) → la mesure directe confirme l'estimation tirée de l'appareil ; sinon elles
  divergent, et c'est rapporté.
- **R3, propriété de l'architecture** : la médiane des κ Big Muff (3 checkpoints, 2 fréquences)
  est à ±20 % du κ du VS100R → propriété de l'architecture ; sinon le κ dépend du checkpoint ou
  de l'appareil.
- **R4, retard pur** : pour un checkpoint et une fréquence, écart-type du retard entre bandes
  ≤ 0,1 échantillon → retard pur ; sinon dispersif.
- **R5, origine (VS100R, à chaque fréquence)** :
  - |Σ_b δ_b − δ_tous| ≤ 0,2·δ_tous → les décalages des blocs **s'additionnent en cascade** ;
  - un bloc seul donne δ_b > (1 − T′/T)/2, plus que la jauge ZOH maximale d'une couche (0,25 à
    96 kHz, 0,375 à 192 kHz) → **compensation à l'intérieur d'une couche** ;
  - les deux, ou aucun : rapporté tel quel.

## Ce que cela ne dira pas

Un checkpoint par appareil ; Big Muff et VS100R diffèrent aussi par le conditionnement (FiLM sur
le VS100R). Le retard est mesuré modèle contre modèle, pas contre l'appareil.
