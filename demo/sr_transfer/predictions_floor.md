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

## Résultats (2026-09-24, `floor.py`, `floor_925.json` et `floor_736_688.json`)

| graine | V0 défaut | V1 haute qualité | V2 float64 | part > 10 kHz | ESR passe-bas 10 kHz | V3 −20 dB | V3 / V2 |
|---|---|---|---|---|---|---|---|
| 925 | 2,224·10⁻⁴ | 2,350·10⁻⁴ | 2,351·10⁻⁴ | 48 % | 1,22·10⁻⁴ | 4,37·10⁻⁵ | 0,19 |
| 736 | 2,263·10⁻⁴ | 2,395·10⁻⁴ | 2,395·10⁻⁴ | 44 % | 1,36·10⁻⁴ | 5,65·10⁻⁵ | 0,24 |
| 688 | 1,970·10⁻⁴ | 2,044·10⁻⁴ | 2,045·10⁻⁴ | 50 % | 1,05·10⁻⁴ | 3,95·10⁻⁵ | 0,19 |

V0 reproduit au dernier chiffre les valeurs `--as-foh` enregistrées (par exemple 2,22357·10⁻⁴ pour
925).

**Verdicts, selon les seuils écrits.**

- **R, rééchantillonnage : réfutée.** Le filtre de haute qualité ne baisse pas le plancher ; il le
  relève de 4 à 6 %.
- **P, précision : réfutée.** Le float64 ne change rien, écart ≤ 0,1 %.
- **Concentration dans l'aigu : non.** 44 à 50 % de l'erreur au-dessus de 10 kHz, sous le seuil de
  70 %. Le critère « Q ou N » n'est donc pas rempli tel qu'écrit.
- **N plutôt que Q : oui.** À −20 dB, l'ESR tombe à 0,19–0,24 fois sa valeur à 0 dB, sous le
  seuil de 0,5. Une erreur de discrétisation linéaire, rapportée au signal, ne dépendrait pas du
  niveau.

**Lecture.** Le plancher de −37 dB n'est pas un artefact de mesure : c'est une différence réelle
entre le modèle à 48 kHz et le même modèle à 96 kHz. Elle dépend du niveau, ce qui désigne les
non-linéarités, dont les harmoniques se replient différemment selon la fréquence
d'échantillonnage. Comme ce repliement retombe dans toute la bande, l'erreur n'est pas concentrée
dans l'aigu, ce que le seuil de 70 % supposait à tort. Environ 20 % du plancher ne suit pas le
niveau ; il reste compatible avec Q, sans être départagé.

**Conséquence produit.** Le modèle à 96 kHz est vraisemblablement **plus** fidèle au système
continu appris que le modèle à 48 kHz, puisqu'il replie moins. « −37 dB du modèle 48 kHz » n'est
donc pas une erreur de transfert, mais l'écart entre deux fréquences d'échantillonnage d'un même
modèle non linéaire. Le vérifier demanderait une cible native à 96 kHz, qui n'existe pas pour Big
Muff.
