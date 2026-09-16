# DIAGNOSIS — Fulltone : écart résiduel S3 à 33 taps (0,140) contre A2 Full (0,064)

Annexe : `diagnosis/fulltone_s3t33_residual_vs_a2/` (`hypotheses.md` horodaté avant mesure, `measure_windows.py`, `class_bounds.py`, `level_dependence.py`, sorties JSON, `decomposition.json`). Lecture seule, CPU.

## 1. Écart

ESR test (`error_to_signal_ratio`, sans alignement ni gain), fichier test Fulltone V100_T050_O050_B000, trois seeds par condition (seeds 0 / 1 / 2, médiane) :

- S3 t33 (1 276 paramètres) : test 0,1404 / 0,1445 / 0,1333 (**0,1404**) ; fenêtres d'entraînement 101–200 : 0,1041 / 0,1146 / 0,1127 (0,1127).
- A2 Full (12 145 paramètres) : test 0,0668 / 0,0639 / 0,0409 (**0,0639**) ; fenêtres 101–200 : 0,0387 / 0,0320 / 0,0166 (0,0320).

Écart test par paire de seeds 0,074 / 0,081 / 0,092 (médiane 0,081) contre une étendue inter-seeds de 0,011 (S3) et 0,026 (A2) : **non attribuable au bruit**. Le même écart existe sur les fenêtres d'entraînement : 0,065 / 0,083 / 0,096 (médiane 0,083).

## 2. Comparabilité

- Même fichier test, même métrique : ESR recalculé depuis `predictions/test_prediction.f32` égal à `metrics.json` (1e-10) pour les six runs.
- Mêmes fenêtres (rng rejoué) : le harnais reproduit 0,1041 (S3 seed 0) et 0,0387 (A2 seed 0). **Correction** : l'ancien diagnostic donnait A2 à 0,062 sur ces fenêtres ; la valeur est 0,039 (médiane 0,032).
- Même optimiseur, perte, données vues ; budget de paramètres déséquilibré (1 276 / 12 145), assumé par le protocole. Un enregistrement de 30 s par split ; l'écart est réparti sur les tranches, les bandes et les seeds (§4).
- Perte de transfert : S3 +0,036 / +0,030 / +0,021 ; A2 +0,028 / +0,032 / +0,024. Identiques : l'écart se juge sur les fenêtres d'entraînement.
- S3 t33 sur test : résidu ≤ 6e-5 d'énergie, modulation lente drive × [0,96 ; 1,03], gain × [0,95 ; 1,01] : fonctionnellement la cascade FIR33 → spline 17 nœuds → FIR33.
- Aucun défaut : les sections 3–5 portent sur le modèle et son optimisation.

## 3. Hypothèses (écrites avant mesure)

- **H1 optimisation** : 200 pas n'amènent pas la cascade à son optimum. Mesure : FIR LS 65 taps sur les 400 fenêtres vues ; moindres carrés alternés post-FIR ↔ spline et Gauss-Newton sur le pré-FIR depuis le checkpoint. Prédiction : ≥ 0,02 gagné sur les fenêtres 101–200, ≥ 0,015 en test.
- **H2 classe, filtre dépendant du niveau** : topologie Tube-Screamer (écrêteur dans la contre-réaction), la forme du filtre change avec le niveau, hors de portée d'une non-linéarité statique entre deux FIR fixes. Mesure : FIR LS 65 taps par tranche d'enveloppe causale (10 ms) sur cible, sortie A2, sortie S3 ; banc commuté contre FIR unique. Prédiction : ≥ 3 dB d'écart de forme, banc ≥ 0,02 meilleur en test, A2 le reproduit, S3 non.
- **H3 régime de niveau** : ≥ 50 % de l'excès sous une enveloppe de 0,1 ; un gain par segment de 50 ms retire ≥ 0,02.
- **H4 transfert/sélection** : écart test ≥ écart fenêtres + 0,03.
- **H5 spectre** (descriptif) : l'excès suit l'énergie cible (≈ 60 % en 100–300 Hz).
- **H6 mémoire linéaire** : FIR LS 257 taps x → résidu de S3 retire ≥ 30 % de l'énergie du résidu en test.

## 4. Mesures et verdict

**H1 — soutenue (un tiers de l'écart).** Fenêtres 101–200, seed 0 : FIR LS 65 taps ajusté sur les 400 fenêtres vues 0,0955 ; S3 t33 0,1041 alors qu'il contient cette sous-classe (spline identité) : optimisation incomplète, prouvée sans refit. Refit dans la classe propre (MSE, 400 fenêtres) : fenêtres 101–200 médiane **0,113 → 0,086** (−0,022 à −0,027 selon le seed) ; test 0,140 → 0,132 (−0,001 à −0,020) : non régularisé, le refit sur-apprend la guitare d'entraînement (pentes de spline jusqu'à 3,6 aux nœuds peu peuplés). Le Gauss-Newton a multiplié l'échelle du pré-FIR par ≈ 5 pour étaler le signal sur les nœuds : au checkpoint, 83 % des entrées de spline tombent dans [−0,25 ; 0,25], deux intervalles de nœuds, alors que le genou de compression de la cible est vers 0,03–0,07 (H2). Sous-cause : grille (± 2, pas 0,25) et drive (1,006) mal adaptés à l'amplitude du signal (rms 0,09).

**H2 — soutenue.** FIR LS 65 taps par tranche d'enveloppe sur la cible (train), tranches < 0,02 / 0,02–0,05 / 0,05–0,1 / 0,1–0,2 / > 0,2 : gain à 200 Hz +11,3 / +4,3 / +1,5 / −0,4 / −1,9 dB (13 dB de compression) ; **forme** relative à 200 Hz, du niveau le plus bas au plus haut : 50 Hz −11,3 → −2,2 dB, 100 Hz −5,5 → −1,6 dB, 500 Hz +4,0 → +2,0 dB : la réponse s'aplatit quand l'écrêteur conduit. Banc commuté contre FIR unique : train 0,096 → 0,062, test 0,125 → **0,093**. Hammerstein 63 taps (x, tanh 3x, tanh 6x) test 0,095 ; Hammerstein commuté 0,085 (fenêtres 101–200 seed 0 : 0,062 → 0,049). Sortie d'A2 : 11 dB de compression, 8 dB d'excursion de forme à 50 Hz : reproduit. Sortie de S3 t33 : 3,8 dB, forme < 0,5 dB : ni l'une ni l'autre.

**H3 — réfutée comme mécanisme de gain.** Excès test par tranche (médianes) : 0 / 11 / 41 / 45 / 2 % : il suit l'énergie (51 % sous 0,1). ESR local S3/A2 de 3,8 (0,02–0,05) à 1,6 (> 0,2), cohérent avec H2. Gain par segment de 50 ms : 0,140 → 0,129.

**H4 — réfutée** : écart test 0,081, écart fenêtres 0,083.

**H5** : excès test 100–300 Hz 44 %, 300–1 000 Hz 39 %, 1–3 kHz 10 %, 0–100 Hz 7 % ; sur les fenêtres 300–1 000 Hz 46 %, 100–300 Hz 31 %, 1–3 kHz 21 % (bande à 0,8 % de l'énergie cible : harmoniques fausses).

**H6 — réfutée** : FIR 257 taps x → résidu de S3 retire 15 % en test (65 taps : 12 %), 0,140 → 0,119. Le résidu d'A2 est plus linéaire (28 %, 0,067 → 0,048 : gain non convergé).

## 5. Contribution retenue

**Prédicat.** L'écart est entièrement présent sur les fenêtres d'entraînement (0,083 contre 0,081 en test), réparti sur les tranches de niveau au prorata de l'énergie avec une erreur relative 2 à 4 fois pire sous une enveloppe de 0,1, à 83 % en 100–1 000 Hz en test et 21 % en 1–3 kHz sur train ; identique pour les trois seeds.

**Mécanisme, fenêtres 101–200 (écart médian 0,081) :**

| Étape | ESR | Part |
|---|---:|---:|
| S3 t33 checkpoint | 0,113 | |
| optimum local de la classe S3 (refit) : **optimisation**, dont échelle d'entrée de la spline | 0,086 | 0,027 (33 %) |
| Hammerstein parallèle 63 taps (seed 0) : **structure cascade et résolution de spline**, borne supérieure | 0,062 | ≤ 0,024 (≤ 30 %) |
| Hammerstein commuté par niveau : **filtre dépendant du niveau (Tube-Screamer)**, borne inférieure | 0,049 | ≥ 0,013 (≥ 16 %) |
| A2 (médiane) : dynamique non décomposée | 0,032 | 0,017 (21 %) |

Le Hammerstein parallèle produit déjà une forme dépendante du niveau (ses branches tanh saturent) : une partie du pas 0,086 → 0,062 appartient à H2, d'où les bornes. En test : 0,140 → refit 0,132 → Hammerstein 0,095 → commuté 0,085 → A2 0,064 ; la part « optimisation » ne se transfère que régularisée (courbure).

**Prédiction chiffrée pour un correctif (test Fulltone, protocole M4, médiane de 3 seeds).**
- Optimisation seule (drive ou grille adaptés à rms 0,09, soit échelle d'entrée de spline × 5, ou plus de pas) : fenêtres 101–200 ≤ **0,090**, test ≤ **0,125**, transfert ≤ +0,035. Sans changement de la grille effective : ΔESR < 0,01.
- Classe : filtre du chemin principal dont la forme dépend de l'enveloppe (≥ 8 dB d'excursion à 50 Hz relatif à 200 Hz, 13 dB de compression à 200 Hz entre enveloppes 0,02 et 0,2) : fenêtres ≤ **0,055**, test ≤ **0,090** ; contrôle : le FIR LS par tranche sur la sortie du modèle reproduit la variation de forme de la cible à ± 2 dB.
- Test falsifiable de H2 : un correctif qui garde la NL statique entre deux filtres fixes et descend sous **0,085** en test réfute H2 (référence LS statique : 0,095).

## 6. Non expliqué, non vérifié

- 0,017 (21 %) entre Hammerstein commuté et A2 sur train, 0,021 en test : dynamique non linéaire à plus d'un état, ou limite des trois branches tanh du banc LS.
- Le refit est un optimum local (bloc-coordonnées depuis le checkpoint, MSE seul) : la part « optimisation » est un minimum ; la part propre de la grille de spline n'est pas isolée ; la frontière structure / dépendance au niveau entre les deux échelons Hammerstein n'est pas mesurée.
- Bruit de sélection du checkpoint (un seul sauvé ; validation 5 s d'une autre guitare, 5 points, ± 0,03 entre évaluations pour S3 comme A2) : non quantifiable.
- A2 seed 2 descend encore au pas 200 et 28 % de son résidu est un défaut de gain linéaire : référence non convergée, écart réel plus grand.
- Réponses par tranche au-dessus de 1 kHz mal conditionnées (< 1 % de l'énergie cible).

Correctif : voir skill controlled-fix.
