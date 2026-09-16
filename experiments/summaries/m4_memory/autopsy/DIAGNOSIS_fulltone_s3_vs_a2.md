# DIAGNOSIS — Fulltone : S3 (FSSR) contre A2 Full (NAM), M4

## 1. Écart

- Métrique : ESR test (`error_to_signal_ratio`, somme des erreurs carrées / énergie cible, sans alignement ni correction de gain), fichier test Fulltone V100_T050_O050_B000, 1 440 000 échantillons (30 s, guitare `idmt-gtr4-ibanez_2820`).
- Paire à paramètres équivalents (seed 0, n = 1 chacun) : S3 largeur 31 (12 192 paramètres, `m4_recovery_fulltone_s3w31_seed0_v1`) **0.2312** contre A2 Full (12 145 paramètres, `m4_fulltone_b0_seed0_v1`) **0.0668**. Écart absolu 0.164, rapport 3.5.
- Référence de dispersion inter-seeds (largeur 8, 1 244 paramètres, 3 seeds) : S3 0.204 / 0.211 / 0.232 (médiane 0.211) ; A2 0.041 / 0.064 / 0.067 (médiane 0.064). Étendue ≈ 0.03 de chaque côté. S3 largeur 31 (0.2312) est identique à S3 largeur 8 seed 0 (0.2319), ce qui justifie d'utiliser la dispersion de largeur 8 comme proxy.
- L'écart (0.14 à 0.19 selon les paires de seeds) vaut 5 fois l'étendue inter-seeds : **non attribuable au bruit**. La section 1 ne clôt pas le diagnostic.
- Trajectoire de validation (240 000 premiers échantillons de validation, guitare SG) : S3 part de l'ESR **0.1552 à l'initialisation** (identité) et son meilleur checkpoint atteint 0.1479 (pas 120) ; les 5 points de validation oscillent entre 0.130 et 0.193 sur les 3 seeds. A2 part de 1.04 et atteint 0.0585 (seed 0), 0.028 (seed 2, encore en descente au pas 200).

## 2. Comparabilité

- Même fichier test : `datasets/raw/internal_m4/fulltone_full_drive_2/test_{input,target}.wav`, sha256 vérifiés dans `dataset-manifest.json` de chaque run (input `e53e4360…`, cible `a174b877…`).
- Même métrique : `fssr_nam.metrics.time.error_to_signal_ratio`, appliquée par `scripts/run_m4_smoke.py` à la prédiction causale fichier entier (`causal_predict`, état remis à zéro une fois au début). A2 utilise un contexte de 6 346 échantillons rempli à gauche, S3 utilise `stream` par blocs de 8 192 ; les deux sont causaux et couvrent le fichier entier. Latence déclarée 0 pour les deux (pas de décalage de cible).
- Mêmes données vues : `rng = np.random.default_rng(seed)` tire les mêmes `starts` pour tous les modèles d'un même seed ; 200 pas × 2 fenêtres × 8 192 = 3 276 800 échantillons de sortie pour chaque run (`training_output_samples_seen` identique). Les pertes du pas 1 de S3 largeur 8 et largeur 31 sont identiques (0.000767806), preuve que le tirage est le même.
- Même optimiseur et même perte : Adam lr 0.004, weight decay 3.17e-7, clip 1.0, MSE + 5e-4·MR-STFT. **Ce recette est la recette NAM par défaut** (lr et weight decay d'A2), appliquée telle quelle à S3 ; S3 porte en plus deux pénalités (courbure 1e-5, énergie résiduelle 1e-3). Asymétrie de budget de mise au point, pas un défaut d'évaluation.
- Budget de paramètres : la paire largeur 31 est équilibrée (12 192 contre 12 145, +0.39 %). La matrice principale (1 244 contre 12 145) ne l'est pas ; elle sert seulement de référence de dispersion.
- Taille du jeu d'évaluation face à l'écart : un seul enregistrement de 30 s par split, guitares différentes (train `gtr2`, validation `gtr4-sg`, test `gtr4-ib`). Amplitude crête des entrées : train 0.50, validation 0.86, test 0.86. À vérifier en section 4 que l'écart est réparti sur le fichier et non concentré sur un segment.
- Différence entraînement/évaluation propre à S3 : à l'entraînement, `forward_components` (état froid au début de chaque fenêtre de 14 538 échantillons) ; à l'évaluation, `stream` (état chaud porté sur le fichier). A vérifier en section 4 (parité forward/stream sur le fichier test).
- Aucun défaut de comparabilité identifié à ce stade : les sections 3 à 5 portent sur le modèle et son entraînement.

## 3. Hypothèses (écrites avant toute mesure)

**H1 — Optimisation gelée à l'initialisation identité.** Mécanisme : S3 est initialisé à l'identité (FIR = delta, spline y = x, gains 1, projections lente et résiduelle à zéro) ; l'ESR de validation initial 0.155 n'est amélioré que de 0.007 en 200 pas, et la trajectoire oscille. Avec lr 0.004 et clip 1.0, les paramètres ne bougent presque pas ; le modèle final est essentiellement l'identité. Mesure : (a) ESR test et gain_error de l'entrée brute prise comme prédiction ; (b) distance paramètre par paramètre entre le checkpoint et une instance fraîche ; (c) écart échantillon par échantillon entre prédictions largeur 8 et largeur 31. Prédiction si vraie : ESR test de l'identité ≈ 0.23 (proche de 0.2319), deltas de paramètres petits devant leur échelle, prédictions w8 et w31 quasi identiques.

**H2 — Mémoire linéaire structurellement insuffisante.** Mécanisme : la mémoire totale de S3 est 17 + 17 + 31 = 63 échantillons (1.3 ms) hors état lent ; la pédale a des filtres linéaires (couplage, tone stack) à mémoire bien plus longue ; A2 dispose de 6 347 échantillons. Mesure : (a) plancher ESR d'un FIR linéaire aux moindres carrés entrée→cible, ajusté sur train, évalué sur test, pour 17, 63, 256, 1 024, 4 096 taps ; (b) énergie d'erreur par bande (< 100, 100–300, 300–1k, 1k–3k, 3k–10k, > 10k Hz) pour S3 et A2. Prédiction si vraie : le plancher linéaire chute nettement au-delà de 63 taps, et l'excès d'erreur de S3 se concentre dans les basses fréquences (< 300 Hz).

**H3 — Décalage de niveau train→test non couvert par la spline.** Mécanisme : l'entrée d'entraînement culmine à 0.50, l'entrée test à 0.86 ; les nœuds de la spline au-delà de |x| > 0.5 ne reçoivent aucun gradient et restent à l'identité (linéaire), alors que la pédale compresse les crêtes. Mesure : (a) énergie d'erreur par tranche d'enveloppe causale de l'entrée (|x| < 0.1, 0.1–0.25, 0.25–0.5, > 0.5) pour S3 et A2 ; (b) valeurs et pentes des nœuds de spline à |x| > 0.5 dans le checkpoint contre l'init. Prédiction si vraie : la majorité de l'excès d'erreur de S3 est dans la tranche > 0.5 ; nœuds extérieurs inchangés.

**H4 — Erreur de gain global.** Mécanisme : gain_error −0.23 pour S3 contre −0.19 (seed 0) et −0.05/−0.06 (seeds 1, 2) pour A2 ; une partie de l'ESR est une simple erreur d'échelle. Mesure : ESR après gain scalaire optimal (1 − ⟨p,t⟩²/(‖p‖²‖t‖²)) pour S3 et A2. Prédiction si vraie : l'ESR corrigé de S3 rejoint celui d'A2 ; sinon la correction de gain n'explique qu'une fraction chiffrée.

**H5 — Résidu rapide inactif par construction.** Mécanisme : projection de sortie initialisée à zéro, échelle 0.5·σ(logit) = 0.025 à l'init, pénalité d'énergie 1e-3 : le résidu n'a ni signal de départ ni marge pour croître en 200 pas ; passer à 31 canaux ne change rien. Mesure : norme de la projection de sortie et échelle dans les checkpoints w8 et w31 ; ratio d'énergie résiduelle (déjà mesuré : 3.4e-6 et 2.4e-5). Prédiction si vraie : projection de sortie de norme ≪ 1, échelle ≈ 0.025, ratio résiduel < 1e-4. (Sous-cas de H1 pour la branche résiduelle.)

## 4. Mesures et verdict

Scripts (lecture seule, `uv run --no-sync`) et sorties JSON dans `scratchpad/green/` : `analyze_predictions.py`, `inspect_checkpoints.py`, `linear_structure.py`, `shift_scan.py`, `train_windows_and_response.py`. Sauf mention, chiffres = ESR sur le fichier test complet, seed 0.

Contrôles de comparabilité annoncés en section 2 :
- Parité forward/stream du checkpoint S3 sur le fichier test : écart max 3.0e-8 ; l'ESR recalculé (0.231868) reproduit la prédiction sauvée. Écarté.
- Répartition : ESR par seconde, S3 − A2 ≥ 0.098 sur chacune des 30 secondes (médiane 0.17, max 0.30). L'écart n'est pas un segment isolé.

**H1 — Optimisation gelée à l'identité.**
- (a) Identité (entrée prise comme prédiction) : ESR test **0.2367**, gain_error −0.307. S3 largeur 8 : 0.2319 ; largeur 31 : 0.2312 ; seeds 1, 2 : 0.2044, 0.2112. Une instance fraîche de S3 reproduit exactement 0.2367.
- (b) Deltas de paramètres : **non négligeables**. Spline : valeurs jusqu'à ±0.45 (nœuds |x| ≥ 1), pentes jusqu'à 0.42 ; FIR pré ‖Δ‖ = 0.19 (taps de 0.03 à 0.06 ajoutés), FIR post 0.10 ; GRU lent ‖Δ‖ > norme initiale ; drive 1.010, offset 0.013, gain 0.972.
- (c) Prédictions largeur 8 contre largeur 31 : écart rms 0.0005 (cible rms 0.139) : identiques. S3 contre identité : écart rms 0.024 (ESR de S3 par rapport à l'identité 0.049), soit une sortie modifiée de ~22 % en rms relatif pour un gain d'ESR test de 0.005.
- (d) Sur les fenêtres d'entraînement exactes (rng seedé rejoué ; preuve : MSE identité au pas 1 = 1.700165e-5, MSE enregistrée de S3 au pas 1 = 1.700165e-5), pas 101–200 : identité 0.197, FIR LS 17 taps 0.155, S3 **0.142**, S3 largeur 31 0.143, FIR LS 33 taps (borne de classe du cœur) **0.123**, 48 taps 0.104, 63 taps 0.096, A2 0.062. En test : S3 0.232, FIR 33 taps 0.173, A2 0.067. Perte de transfert train→test : S3 +0.090, FIR 17 +0.074, FIR 33 +0.050, A2 +0.005.
- Verdict : **réfutée dans sa forme stricte** (les paramètres bougent, S3 apprend sur train et n'est qu'à 0.019 de sa borne de classe sur les fenêtres vues), **soutenue fonctionnellement** : en test S3 vaut identité − 0.005, parce que ce qu'un filtre court apprend sur la guitare d'entraînement ne se transfère pas (même comportement pour les FIR LS 17 et 33 taps). Voir H2 pour la raison.

**H2 — Mémoire linéaire insuffisante.**
- (a) Plancher FIR linéaire aux moindres carrés (ajusté sur train, évalué sur test) selon la mémoire : 1 tap 0.229 ; **17 taps 0.229** ; 20 : 0.216 ; 24 : 0.198 ; 28 : 0.184 ; **33 : 0.173** ; 40 : 0.152 ; 48 : 0.136 ; **63 : 0.123** ; 96 : 0.120 ; 128 : 0.124 ; 256 : 0.128 ; 1 024 : 0.124 ; 4 096 : 0.130. Le plancher sature à 63–96 échantillons (1.3–2 ms) ; au-delà, l'ajustement sur-apprend. Un FIR de 17 taps n'apporte rien par rapport à un gain scalaire.
- Le FIR LS de 33 taps se factorise en deux FIR de 17 taps (erreur relative 2.3e-4, ESR 0.173) : la cascade pré/post de S3 peut structurellement atteindre 0.173, pas mieux.
- Non-linéarité statique + mémoire (Hammerstein LS, entrées x, tanh 3x, tanh 6x, un FIR de 63 taps chacune) : test **0.091** (train 0.059).
- (b) Erreur par bande (part de l'ESR ; énergie cible : 59 % en 100–300 Hz, 38 % en 300–1 000 Hz, < 1 % au-dessus de 1 kHz) :

| Bande | Identité | S3 w8 | S3 w31 | FIR LS 63 | A2 s0 |
|---|---:|---:|---:|---:|---:|
| 0–100 Hz | 0.016 | 0.021 | 0.021 | 0.011 | 0.010 |
| 100–300 Hz | 0.134 | **0.134** | 0.134 | 0.047 | **0.027** |
| 300–1 000 Hz | 0.078 | 0.065 | 0.064 | 0.054 | 0.025 |
| 1–3 kHz | 0.008 | 0.012 | 0.012 | 0.008 | 0.005 |

  Excès S3 − A2 = 0.165 : 65 % en 100–300 Hz, 24 % en 300–1 000 Hz, 7 % en 0–100 Hz, 5 % en 1–3 kHz.
- Réponse du FIR LS 63 (indicative au-dessous de 1 kHz ; mal conditionnée au-dessus, où la cible n'a pas d'énergie) : −5.7 dB à 30 Hz, −3.5 dB à 100 Hz, 0 dB à 200 Hz, +2.5 dB à 400–500 Hz, avance de phase +25° à 100–200 Hz. Pic d'intercorrélation entrée→cible à −4 échantillons (la cible « précède ») : signature d'une avance de phase de filtre passe-haut, pas d'un retard — décaler l'identité de −4 ne donne que 0.216 et aucune fenêtre de 17 taps décalée (0 à 64) ne descend sous 0.218.
- Verdict : **soutenue, reformulée**. La mémoire linéaire nécessaire est de 48 à 96 échantillons (1–2 ms), pas des milliers ; la cascade 17 + 17 de S3 (33 échantillons, résolution fréquentielle ≈ 1.5 kHz) ne peut pas façonner la bande 100–500 Hz qui porte 97 % de l'énergie cible.

**H3 — Décalage de niveau train→test.**
- Échantillons test avec |x| ≥ 0.5 : 1 239 sur 1 440 000. Part de l'erreur de S3 dans la tranche d'enveloppe > 0.5 : 0.0055 / 0.2319 = **2.4 %** ; ESR local de cette tranche 0.12, le plus bas de toutes (0.52 pour < 0.05, 0.39 pour 0.05–0.1, 0.26 pour 0.1–0.25, 0.20 pour 0.25–0.5). A2 est plat (0.05–0.09 selon la tranche).
- Les nœuds extérieurs de la spline ont bougé (−0.45 à x = 2) mais la région ne pèse rien. Un Hammerstein polynomial (x, x³, x⁵) explose en test (0.68 contre 0.062 en train) : le décalage de niveau ne nuit qu'aux extrapolateurs non bornés ; la spline de S3 extrapole linéairement.
- Verdict : **réfutée**.

**H4 — Erreur de gain global.**
- ESR après gain scalaire optimal : S3 0.2319 → **0.2319** (Δ < 1e-4 ; corrélation 0.876, 1 − 0.876² = 0.232 : erreur de forme pure). Identité 0.2367 → 0.2289. A2 s0 0.0668 → 0.0464 ; A2 s1 0.0639 → 0.0638.
- Verdict : **réfutée** (une correction de gain aiderait A2, pas S3).

**H5 — Résidu rapide inactif par construction.**
- Projection de sortie du résidu : norme 0.027 (w8), 0.097 (w31), contre 1.6–3.2 pour les couches internes ; échelle 0.023 ; sortie résiduelle rms 2.2e-4 (w8), 6.0e-4 (w31) ; ratio d'énergie 3.4e-6, 2.4e-5.
- Ablations sur le checkpoint : sans résidu 0.23186 (inchangé) ; sans résidu ni modulation lente 0.2311. Modulation lente : drive × [0.978, 1.020], gain × [0.971, 0.999], offset ± 0.018.
- Verdict : **soutenue**. Ces deux branches sont inertes ; elles expliquent pourquoi passer de 1 244 à 12 192 paramètres ne change rien (les 11 000 paramètres ajoutés alimentent une sortie 0.023·tanh(·) initialisée à zéro), mais elles ne créent pas l'écart : toute la sortie de S3 est le cœur S0.

## 5. Contribution retenue

**Prédicat.** L'écart est présent sur chaque seconde du fichier test (S3 − A2 ≥ 0.098), à tous les niveaux d'entrée < 0.5 (97.6 % de l'erreur de S3), concentré à 89 % dans la bande 100–1 000 Hz (65 % en 100–300 Hz), identique pour largeur 8 et 31 (écart rms 0.0005), déjà visible sur les fenêtres d'entraînement (S3 0.142 contre A2 0.062) et amplifié par le transfert de guitare (S3 +0.090, A2 +0.005).

**Mécanisme.** La Fulltone à ce réglage est, au premier ordre, un réseau linéaire de mise en forme (coupe-bas −3.5 dB à 100 Hz, +2.5 dB à 400–500 Hz, avance de phase 25° à 100–200 Hz) suivi d'une saturation modérée. Le reproduire demande 48 à 96 échantillons de mémoire linéaire (1–2 ms). Le cœur de S3 dispose de deux FIR de 17 taps (33 échantillons en cascade, 0.7 ms) ; ses branches lente et résiduelle sont inertes. Borne de classe du cœur linéaire : 0.173 ; borne à 63 taps : 0.123 ; avec non-linéarité statique bornée et 63 taps : 0.091. S3 ne rejoint même pas sa borne de classe (0.232 ≈ plancher 17 taps 0.229) : un filtre court mal spécifié ajusté au spectre de la guitare d'entraînement (0.142 sur train) ne se transfère pas (0.232 en test), exactement comme le FIR LS 17 taps (train 0.155 → test 0.229). Le champ réceptif d'A2 (6 347 échantillons) couvre trivialement 2 ms et transfère (0.062 → 0.067). Le nombre de paramètres n'est pas la variable : c'est la mémoire linéaire du chemin principal.

Décomposition de l'écart seed 0 (0.231 → 0.067 = 0.164), par bornes LS emboîtées (ordres de grandeur, pas des attributions exactes ; le plancher 0.091 est un Hammerstein parallèle à trois branches, pas la cascade pré→spline→post de S3) :
- 0.231 → 0.173 : S3 sous la borne de sa propre classe linéaire (0.058, 35 %) ; sur les fenêtres d'entraînement, seul 0.019 de ce manque est visible (S3 0.142 contre borne 0.123), le reste est le non-transfert d'un filtre court mal spécifié (le FIR LS 33 taps perd lui-même +0.050 de train à test) ;
- 0.173 → 0.123 : mémoire 33 → 63 échantillons (0.050, 30 %) ;
- 0.123 → 0.091 : non-linéarité statique bornée avec cette mémoire (0.032, 20 %) ;
- 0.091 → 0.067 : reste vers A2 seed 0 (0.024, 15 %) ; A2 seed 2 atteint 0.041.

**Prédiction chiffrée pour un correctif (seed 0, test Fulltone, protocole M4 inchangé).**
- Tout correctif qui laisse la mémoire linéaire effective du cœur à 33 échantillons (optimiseur, lr, pas, largeur, pénalités, seeds) ne peut pas descendre sous **0.17** (borne 0.173). Largeur 31 (0.231) et seeds 1–2 (0.204–0.211) le confirment.
- Un correctif qui donne au chemin principal ≥ 63 échantillons de mémoire linéaire effective dans la bande 100–500 Hz et converge réellement doit atteindre : ESR test ≤ **0.125** (plancher linéaire 63 taps 0.123, cible dure) ; erreur de bande 100–300 Hz ≤ **0.05** (contre 0.134) ; perte de transfert train→test ≤ **+0.03** (contre +0.09). Référence, non exigée : une non-linéarité statique bornée avec cette mémoire atteint 0.091 en Hammerstein parallèle ; une cascade à FIR longs peut se situer entre 0.123 et 0.091. Il reste alors 0.03–0.06 d'écart avec A2 seed 0, hors de portée d'une structure statique à mémoire courte.
- Routes qui ajoutent cette mémoire : FIR pré/post plus longs ; sections IIR/biquad en entrée et sortie ; ou activation du résidu existant (init non nulle de la projection de sortie, échelle initiale plus grande), car le TCN a un champ réceptif de 31 sur [entrée, cœur] et le cœur porte déjà 33 échantillons, soit ≈ 63 de mémoire effective, et le run Big Muff largeur 31 montre qu'il peut s'activer (17.5 % d'énergie résiduelle).
- Un correctif portant sur le résidu ou l'état lent se juge conditionnellement : résidu resté inerte (ratio d'énergie < 1e-4) → ΔESR < 0.01 ; résidu activé (ratio ≥ 1e-2) → mêmes cibles que ci-dessus (≤ 0.125, bande 100–300 Hz ≤ 0.05, transfert ≤ +0.03).

## 6. Non expliqué, non vérifié

- Pourquoi S3 reste à 0.019 de sa borne de classe sur les fenêtres d'entraînement (0.142 contre 0.123) : la cohérence de signe des gradients sur 200 pas à 2 fenêtres n'est pas dans les artefacts et aucun entraînement n'est autorisé. Le reste du 0.058 (non-transfert) est expliqué par la mauvaise spécification, pas mesuré par branche de S3.
- Le reste 0.024 (A2 seed 0) à 0.050 (A2 seed 2) au-delà du plancher Hammerstein 63 taps : non décomposé (dynamique non linéaire à mémoire longue, contre non-convergence d'A2 à 200 pas — A2 seed 2 descend encore au pas 200, la référence n'est donc pas convergée et l'écart réel est probablement plus grand).
- Réponses des FIR LS au-dessus de 1 kHz : mal conditionnées (énergie cible < 1 %) ; seuls les chiffres sous 1 kHz sont indicatifs.
- Dérive des nœuds extérieurs de la spline (±0.45 pour |x| ≥ 1) par Adam via la pénalité de courbure : observée, non reliée à l'écart (région portant 2.4 % de l'erreur), non vérifiée comme cause.
- Bruit de sélection du checkpoint (validation ±0.02 sur guitare SG) et différence de guitare validation/test : non quantifiés au-delà de l'ESR identité par split (train 0.197, validation 0.218, test 0.237).
- Big Muff (effondrement d'énergie, ESR 0.97) et S4 : non couverts ; l'autopsie M4 y décrit un mécanisme différent.

Correctif : voir skill controlled-fix.
