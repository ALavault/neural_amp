# FFT de 288 000 points et TF32 : vitesse, mémoire et précision des gradients

Voie produit (D-PRODUCT-001), aucune revendication scientifique. 2026-10-04, mesures sur le GPU
(NVIDIA RTX PRO 4000 Blackwell), aucun entraînement.

## Ce qui change

- `DiagonalSSMLayer.fft_size` (`src/fssr_nam/models/ssm_wavenet.py`) : `"pow2"` par défaut, la
  puissance de 2 suivant 2L − 1 (524 288 pour L = 144 000), identique au bit aux runs
  existants (`tests/regression/test_ssm_wavenet_bitwise.py`). `"fast"` prend la plus petite
  longueur 2-3-5-lisse au moins égale à 2L − 1 (288 000). La convolution reste linéaire et
  causale (`tests/numerical/test_ssm_fft_size.py`).
- Banc : `--fft-size {pow2,fast}` et `--no-tf32` (matmuls et convolutions cuDNN en float32
  complet), tous deux inactifs par défaut et inscrits au rapport du run.

## Mesures (`bench_fft.py` → `bench_fft.json` ; `grad_precision.py` → `grad_precision.json`)

Modèle VS100R à 48 kHz (`vs100r_gc_film_joint_s925_r2`, poids entraînés), un lot réel de
16 × 144 000, perte du banc (0,5 L1 + 0,5 MR-STFT), mode déterministe. Un pas mesuré =
forward, perte, backward et copie des gradients, sans pas d'optimiseur. Moyenne de deux
médianes de 20 pas, passes alternées.

| réglage | temps d'un pas | pic alloué par PyTorch |
|---|---|---|
| FFT pow2, TF32 (réglage actuel des runs) | 0,931 s | 19,4 Gio |
| FFT fast, TF32 | 0,718 s | 15,6 Gio |
| FFT pow2, sans TF32 | 0,983 s | 19,4 Gio |
| FFT fast, sans TF32 | 0,785 s | 15,6 Gio |

- FFT fast contre pow2 : −22,8 % de temps avec TF32, −20,1 % sans.
- Désactiver le TF32 : +5,6 % (pow2) et +9,3 % (fast).
- FFT fast sans TF32 contre le réglage actuel : −15,6 % de temps, −3,8 Gio.
- Sortie, FFT fast contre pow2 (TF32) : erreur relative en norme 1,9·10⁻⁴, écart absolu
  maximal 4,7·10⁻⁴.

**Précision des gradients** contre une référence en float64 (même modèle, FFT pow2, 2 premiers
segments du lot) :

| écart relatif au float64 | TF32 | sans TF32 |
|---|---|---|
| gradient complet, FFT pow2 | 21 % | 0,06 % |
| gradient complet, FFT fast | 37 % | 0,47 % |
| `log_dt` du bloc 2, FFT pow2 | 100 % | 1,5 % |
| `log_dt` du bloc 2, FFT fast | 927 % | 14 % |

En float64, les deux longueurs de FFT donnent les mêmes gradients à 6·10⁻¹³ : l'écart est un
effet de l'arrondi en float32, pas une différence de calcul.

## Lecture

- **Sur ce lot, avec le réglage TF32 des runs actuels, le gradient est faux d'environ 21 %**
  par rapport au float64. cuDNN active le TF32 par défaut pour les convolutions, et le banc fixe
  `set_float32_matmul_precision("high")`. Le désactiver coûte 6 à 9 % de temps.
- **La FFT fast gagne 20 à 23 % de temps et 3,8 Gio**, mais rend environ 10 fois moins précis le
  gradient des pôles lents (bloc 2), avec ou sans TF32. La cause (radix 3 et 5 de cuFFT en
  float32, ou autre) n'est pas établie.
- **FFT fast sans TF32** est plus rapide (−15,6 %), moins gourmand (−3,8 Gio) et plus précis
  (0,47 % contre 21 %) que le réglage actuel, sur ce lot.

## Ce qui n'est pas établi

- L'effet sur un entraînement complet : la précision du gradient d'un lot ne dit pas ce que
  devient la qualité finale. Une comparaison à plusieurs graines est nécessaire avant tout
  changement de réglage par défaut.
- La généralité : un seul modèle (VS100R, 48 kHz), un seul lot, un seul GPU.
- `DiagonalSSMLayer.fft_size` est un attribut de classe, non sauvegardé dans les checkpoints :
  un modèle entraîné en fast est évalué en pow2 par les scripts existants, ce qui ne change sa
  sortie qu'à l'arrondi près.
