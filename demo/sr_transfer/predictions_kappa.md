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

## Résultats (2026-10-02, `kappa.json`)

Retards en échantillons à 48 kHz ; VS100R : 8 prises de test (2, 3, 14, 16, 17, 30, 33, 36).

| checkpoint | κ à 96 kHz | κ à 192 kHz | rapport 192/96 | écart-type entre bandes (96 / 192) |
|---|---|---|---|---|
| `vs100r_gc_film_joint_s925_r2` | 1,934 | 1,993 | 1,031 | 0,175 / 0,265 |
| `paired_ssm_wavenet_s925` | 2,525 | 2,529 | 1,001 | 0,033 / 0,050 |
| `paired_ssm_wavenet_s736` | 2,428 | 2,419 | 0,996 | 0,031 / 0,048 |
| `paired_ssm_wavenet_s688` | 1,920 | 1,903 | 0,991 | 0,022 / 0,031 |

**R1, forme (1 − T′/T) : confirmée pour les 4 checkpoints** (rapports de 0,991 à 1,031, dans
[0,8 ; 1,25]).

**R2 : la mesure directe confirme l'estimation indirecte.** κ_VS100R = 1,964 (médiane des deux
fréquences), |1,964 − 1,86| = 0,104 ≤ 0,3.

**R3 : pas une propriété de l'architecture.** Médiane des 6 κ Big Muff = 2,423, à 0,460 du κ du
VS100R, au-delà de 20 % (0,393). Le κ dépend du checkpoint ou de l'appareil. Descriptif : sur le
même appareil, les trois graines Big Muff vont de 1,90 (s688) à 2,53 (s925).

**R4 : retard pur pour les 3 checkpoints Big Muff** à 96 et 192 kHz (écarts-types de 0,022 à
0,050). **Dispersif pour le VS100R** (0,175 et 0,265) : la bande de 500 Hz retarde nettement
moins que les autres (0,144 contre 0,442 à 0,549 à 96 kHz ; 0,225 contre 0,686 à 0,847 à
192 kHz).

**R5 : lecture non interprétable, défaut du protocole.** Les seuils donnent : pas de cascade
(Σ_b δ_b = 6,17 contre δ_tous = 0,483 à 96 kHz ; 67,82 contre 0,748 à 192 kHz) et compensation
dans une couche (δ_b maximal 1,276 > 0,25 à 96 kHz ; 14,65 > 0,375 à 192 kHz). Mais le bras
« bloc seul », tel qu'il est spécifié plus haut et exécuté, fait tourner tout le réseau à fs′ en
laissant les 7 autres blocs à leur Δ de 48 kHz : comptée en échantillons, leur dynamique ne
change pas, donc en temps physique elle va fs′/48 000 fois trop vite (×2, ×4). Le bras ne mesure
donc pas la jauge d'un bloc isolé ; la part de ce désaccord dans des retards qui montent jusqu'à
14,7 échantillons n'est pas établie (sens corrigé après relecture Codex : la première rédaction
disait « ralentie »). Ce défaut a été vu en lisant
les chiffres ; les seuils ne sont pas modifiés, et **aucune conclusion sur l'origine de κ n'est
tirée de R5**. Retards par bloc (blocs 0 à 7) : 96 kHz 0,705, 1,027, 1,276, 1,210, 1,275, 0,768,
−0,082, −0,004 ; 192 kHz 14,65, 9,17, 10,33, 10,30, 8,87, 5,89, 8,56, 0,04. Une mesure qui isole
un bloc (par exemple le réseau à 48 kHz, ce seul bloc à fs′ avec rééchantillonnage autour de lui)
demanderait une nouvelle préinscription.

**Ce qui se dit.** Rééchelonner Δ sans correction retarde la sortie de κ (1 − 48 000/fs′)/2
échantillon, avec la forme prévue sur les deux appareils. κ vaut 1,9 à 2,5 selon le checkpoint ;
il n'est pas fixé par l'architecture. Le κ du VS100R mesuré modèle contre modèle rejoint
l'estimation tirée de l'appareil. κ > 1 partout : d'après T1 (`THEORY.md`), un mode réel seul ne
dépasse pas 1 ; l'origine (cascade, modes complexes, signes mêlés) reste ouverte.
