# D'où vient le plancher à −37 dB de la montée corrigée ?

Voie produit. Écrit le 2026-09-24, **avant tout calcul**.

**Constat** (`RESULTS_hold.md`, H7 réfutée ; `FIX_foh.md`). Corrigé par deux coefficients par
mode, ce qui revient à la relecture FOH, SSM-WaveNet reste à une ESR d'environ 2·10⁻⁴ (−37 dB) de
son modèle à 48 kHz, à 88,2, 96 et 192 kHz. Ce plancher dépend à peine de la fréquence cible :
rapport 192/96 de 1,19 à 1,36.

**Quatre explications.**

- **R, rééchantillonnage** : le chemin 44,1 → f → 48 kHz et le chemin 44,1 → 48 kHz de
  référence utilisent le filtre par défaut de torchaudio (largeur 6). Leur différence dans la
  bande 20–24 kHz pèse dans l'ESR.
- **P, précision** : le modèle tourne en float32 et ses convolutions FFT portent sur 2¹⁹ points.
- **Q, discrétisation à 48 kHz** : le modèle à 48 kHz est une discrétisation FOH du système
  continu, avec une erreur relative en (ωT)², forte vers 20 kHz. À 88,2, 96 et 192 kHz, les
  modèles sont proches du continu. L'écart mesuré serait alors surtout l'erreur propre du modèle
  à 48 kHz, ce qui expliquerait qu'il ne dépende presque pas de la fréquence cible.
- **N, repliement des non-linéarités** : les portes tanh·sigmoïde produisent des harmoniques qui
  se replient différemment à 48 kHz et au-dessus.

**Mesures** (graine 925, puis 736 et 688 si le résultat est net ; relecture FOH ; bras 96 kHz ;
ESR sur l'intérieur des segments, bande de 48 kHz, comme `two_tap_test`) :

- V0 : reproduction du chiffre enregistré (2,22·10⁻⁴).
- V1 : tous les rééchantillonnages en haute qualité (fenêtre de Kaiser, largeur 64), y compris
  la référence à 48 kHz, tirée du fichier à 44,1 kHz et non plus du chargeur segment par segment.
- V2 : V1 en float64 de bout en bout.
- V3 : V2 à −20 dB en entrée.
- V4 : part de l'énergie d'erreur de V2 au-dessus de 10 kHz, et ESR de V2 après un passe-bas à
  10 kHz des deux sorties.

**Prédictions** :

- R si V1 ≤ 0,5 × V0 ;
- P si V2 ≤ 0,5 × V1 ;
- si ni R ni P, le plancher est une différence réelle entre modèles, et :
  - Q ou N si la part au-dessus de 10 kHz est ≥ 70 % (V4) ;
  - N plutôt que Q si V3 ≤ 0,5 × V2, puisque le repliement non linéaire baisse avec le niveau ;
    Q plutôt que N si V3 ≥ 0,8 × V2, puisque l'erreur de discrétisation linéaire, rapportée au
    signal, ne dépend pas du niveau.

Plusieurs explications peuvent contribuer : chaque mesure est rapportée avec son chiffre, et la
conclusion reste « non départagé » si les seuils ne tranchent pas.
