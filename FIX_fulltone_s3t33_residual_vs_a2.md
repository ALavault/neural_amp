# FIX — Fulltone S3 t33 : grille de spline adaptée à l'amplitude du signal

Diagnostic d'entrée : `DIAGNOSIS_fulltone_s3t33_residual_vs_a2.md` (annexe
`diagnosis/fulltone_s3t33_residual_vs_a2/`). Point de départ : lignée M4_MEMORY
(S3 avec FIR 33 taps, `configs/training/m4_memory.yaml`, runs
`m4_memory_{fulltone,bigmuff}_s3t33_seed{0,1,2}_v1`). Route retenue : candidat
« optimisation seule » du diagnostic §5. Les sections 1 à 5 sont écrites avant
tout code et ne sont plus modifiées ensuite ; la section 6 est remplie après les
runs.

## 1. Spécification

**Prédicat** (diagnostic §5) : l'écart S3 t33 − A2 est entièrement présent sur les
fenêtres d'entraînement 101–200 (0,083 contre 0,081 en test), identique pour les
trois seeds ; la part « optimisation » vaut 0,027 sur ces fenêtres (0,113 → 0,086
par refit dans la classe propre), et sa sous-cause désignée est une grille de
spline (17 nœuds sur ± 2, pas 0,25) et un drive (1,006) inadaptés à l'amplitude
de l'entrée de spline : 83 % des entrées tombent dans deux intervalles de nœuds.

**Cible sur le prédicat** — test Fulltone V100_T050_O050_B000, protocole M4
inchangé (200 pas, mêmes fenêtres par seed), S3 t33, seeds 0 / 1 / 2, **médiane
sur les seeds**, ESR `error_to_signal_ratio` sans alignement ni gain, fenêtres
101–200 rejouées par le résumeur :

| Métrique | Valeurs actuelles (seeds 0 / 1 / 2, médiane) | Confirmé si | Réfuté si |
|---|---|---|---|
| ESR fenêtres 101–200 | 0,1041 / 0,1146 / 0,1127 (**0,1127**) | ≤ **0,090** | > 0,1027 (baisse < 0,01 : la grille effective n'était pas le levier) |
| ESR test | 0,1404 / 0,1445 / 0,1333 (**0,1404**) | ≤ **0,125** | — |
| Transfert (test − fenêtres) | +0,0363 / +0,0299 / +0,0205 (**+0,0299**) | ≤ **+0,035** | — |
| Erreur de bande 100–300 Hz (informatif) | 0,0545 / 0,0643 / 0,0522 (0,0545) | — | — |

Verdict : *confirmé* si les trois conditions « confirmé » tiennent ; *réfuté* si
la médiane des fenêtres baisse de moins de 0,01 ; *partiel* sinon. Les bornes
rappelées comme seuils, pas comme gains attendus : optimum local de la classe S3
par refit 0,086 (fenêtres, médiane), 0,132 (test) ; FIR LS 65 taps sur les
400 fenêtres 0,0955 (seed 0) — un S3 correctement optimisé doit au moins l'égaler.

**Invariants hors prédicat**, valeur mesurée sur les runs existants et tolérance :

| Invariant | Valeur actuelle | Tolérance |
|---|---|---|
| Big Muff S3 t33, ESR test (seeds 0 / 1 / 2, médiane) | 0,9444 / 0,9443 / 0,9450 (**0,9444**) | hausse de la médiane ≤ 0,05 (règle de `m4_memory.yaml`) |
| Nombre de paramètres S3 t33 | 1 276 | égal |
| Initialisation identité (sortie = entrée à l'init, amplitude jusqu'à 0,5) | vraie (≤ 3e-6) | inchangée, y compris hors grille |
| Parité bloc (stream par blocs = forward complet) | < 2e-6 | inchangée |
| Prédiction finie, `alternate_block_max_abs` | vraie, 0,0 | inchangées |
| Fenêtres, données, split, perte, optimiseur, 200 pas, politique de checkpoint | protocole M4 | identiques (`fixed_dimensions`) |

## 2. Correctif

**Une seule intervention** : les 17 nœuds de la spline du cœur S3 couvrent
[−0,4 ; +0,4] au lieu de [−2 ; +2] (pas 0,05 au lieu de 0,25, échelle × 5). Tout le
reste est fixé : 33 taps, 17 nœuds, drive initial 1, largeur résiduelle 8,
perte, optimiseur, budget de pas, sélection de checkpoint.

Justification de la borne, mesurée sur le checkpoint `m4_memory_fulltone_s3t33_seed0_v1`
(entrée de spline u = drive·mod·pre(x) + offset, fichier train Fulltone) : rms de u
0,143 ; percentiles de |u| : 50 % 0,046, 83 % 0,194, 95 % 0,322, 99 % 0,44,
99,9 % 0,58, max 0,75 ; **98,1 % des échantillons train et 94,9 % des échantillons
test tombent dans ± 0,4** (89 % / 83 % dans ± 0,25, l'intervalle que couvraient
deux nœuds). Avec le pas 0,05, le genou de compression de la cible (0,03–0,07,
diagnostic H2) reçoit un nœud propre (0,05) et la zone où vit 95 % du signal en
compte seize intervalles au lieu de deux.

Ce qui change de classe : les nœuds à |u| ∈ {0,5 ; … ; 2} disparaissent ; au-delà
de ± 0,4 la spline est affine avec pentes d'extrémité apprenables (une seule
pente de part et d'autre). C'est une restriction dans le régime d'écrêtage dur
(1,9 % des échantillons train, 5,1 % en test), assumée : la cible y est déjà
saturée et une pente affine faible la représente. Ce point est une hypothèse
fragile du correctif, pas un acquis.

Candidats écartés (ordonnés, chacun aurait son propre FIX) : (b) drive initial × 5
— rompt l'initialisation identité sauf compensation par `output_gain` ou par les
valeurs de spline, soit deux changements ; (c) plus de pas — le diagnostic prédit
ΔESR < 0,01 sans changement de la grille effective, et `optimizer_steps` est une
dimension fixée du protocole.

## 3. Argument de non-régression

- **Initialisation identité** : la spline est initialisée avec valeurs = nœuds et
  pentes = 1 ; l'extrapolation linéaire donne `values[0] + 1·(u − knots[0]) = u` à
  gauche et symétriquement à droite : la spline est exactement l'identité sur tout
  ℝ quelle que soit la grille. Le modèle à l'init est donc le même qu'avant
  (sortie = entrée, à l'arithmétique flottante près). Le correctif est l'identité
  sur cet invariant. Vérifié par test unitaire, amplitude 0,5 > 0,4.
- **Paramètres, parité bloc, causalité, finitude** : le nombre de paramètres ne
  dépend pas de la grille (17 valeurs, 17 pentes) ; la spline est sans état, la
  parité bloc et la causalité ne dépendent que des FIR, du résidu et du
  contrôleur lent, non touchés ; l'entrée est vérifiée finie comme avant. Identité
  sur ces invariants.
- **Effet de bord nommé, non compensé** : `curvature_penalty` est exprimée en
  unités de nœuds (différences secondes des valeurs, différences premières des
  pentes). Pour une même courbure physique, le terme des valeurs est 25 × plus
  petit et celui des pentes 5 × plus petit sur la grille fine : le poids 1e-5
  régularise moins fort. On ne le compense pas (ce serait une seconde
  intervention). Le diagnostic indique que la part « optimisation » ne se
  transfère que régularisée : c'est la colonne **transfert ≤ +0,035** qui
  surveille cet effet ; un transfert > 0,035 avec fenêtres ≤ 0,090 signalerait
  ce mécanisme.
- **Tous les runs S3 et S4 existants** : la clé `spline_range` est absente de
  leur `config-resolved.yaml` ; le défaut 2,0 reproduit la grille ± 2 ; le résumeur
  M4_MEMORY rebâtit ses modèles à partir de `fssr_model` et reste inchangé.
- **Big Muff (contrôle hors prédicat)** : au checkpoint t33 seed 0, l'entrée de
  spline a un rms de 0,030 et |u| ≤ 0,32 (train et test) : tout le signal tombe
  dans la nouvelle grille, le correctif n'y est pas l'identité (la spline y gagne
  aussi en résolution). **L'argument ne se ferme pas : contrôle empirique
  obligatoire** (runs Big Muff S3 t33 grille fine, seeds 0–2, tolérance +0,05 sur
  la médiane).
- **Régime hors grille (|u| > 0,4, Fulltone)** : la classe y est restreinte à une
  affine ; l'argument ne se ferme pas par construction, il est couvert par le
  prédicat lui-même (ESR test et fenêtres, où ces échantillons pèsent).

## 4. Contrôle

Niveau retenu : **invariants scientifiques + relance des runs de référence hors
prédicat aux mêmes seeds**, le coût le permettant (six runs de ~3,5 min sur le
GPU partagé, plus un préflight).

1. Invariants (toujours) : test unitaire permanent
   `test_m4_grid_diagnostic_keeps_identity_init_and_block_parity` — configuration
   cohérente (taps 33, 17 nœuds, grille ± 0,4, pas 0,05, modèle S3 seul),
   1 276 paramètres, identité à l'init sur un signal d'amplitude 0,5 (au-delà de
   la grille), parité bloc < 2e-6, et grille ± 2 quand la clé est absente. À
   l'exécution : `checks.finite_prediction`, `alternate_block_max_abs` de chaque
   run, ESR test recalculé depuis la prédiction sauvée = `metrics.json`.
2. Relance de référence hors prédicat : Big Muff S3 t33 grille fine, seeds 0 / 1 / 2,
   médiane de l'ESR test comparée à 0,9444 ; passe si hausse ≤ 0,05.
3. Non couvert : S4 (le shaper suréchantillonné construit sa propre spline et
   n'est pas modifié ; la lignée le déclare hors périmètre) ; les runs à 17 taps
   (non relancés : l'intervention porte sur la lignée t33) ; le régime |u| > 0,4
   n'a pas de mesure séparée.

## 5. Exécution

Nouvelle lignée préinscrite **M4_GRID** (post-freeze, dans la lignée M4_MEMORY),
`PROTOCOL_LOCK.yaml` (M4-P4-v1) et verdict NO-GO inchangés ; décision D-M4-011.

Patch (calqué sur D-M4-010) :
- `configs/training/m4_grid.yaml` : hypothèse, intervention, seuils ci-dessus.
- `src/fssr_nam/models/structured.py`, `fssr.py`, `training/m3.py` : paramètre
  `spline_range` (défaut 2,0) transmis à `SmoothHermiteSpline(minimum=−r, maximum=r)`.
- `scripts/campaigns/run_m4_smoke.py` : option `--grid-range` (exclusive de `--memory-taps`
  et `--recovery-wide`), écrit `fssr_model.spline_range` et `grid` dans
  `config-resolved.yaml`, phase `M4_GRID` / `M4_GRID_PREFLIGHT`.
- `scripts/campaigns/run_m4_grid.py` (pilote, `--seeds`/`--devices` pour lancer un run par
  appel), `scripts/summarize_m4_grid.py` (réutilise l'évaluation de
  `summarize_m4_memory.py`, qui apprend à étiqueter la phase `M4_GRID`).
- `tests/unit/test_m4_training.py` : le test de la section 4.
- `.codex_campaign/DECISIONS.md` : D-M4-011.

Registre des runs (jamais d'écrasement ; les identifiants n'existent pas) :

| Run | Rôle |
|---|---|
| `m4_grid_preflight_fulltone_s3t33_seed0_v1` | préflight 2 pas, exclu des agrégats |
| `m4_grid_fulltone_s3t33_seed{0,1,2}_v1` | prédicat |
| `m4_grid_bigmuff_s3t33_seed{0,1,2}_v1` | contrôle hors prédicat |

Commandes :

```
make lint && uv run pytest tests/unit/test_m4_training.py -q && make test
uv run python scripts/campaigns/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 0   # préflight + seed 0
uv run python scripts/campaigns/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 1
uv run python scripts/campaigns/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 2
uv run python scripts/campaigns/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 0
uv run python scripts/campaigns/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 1
uv run python scripts/campaigns/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 2
uv run python scripts/summarize_m4_grid.py   # experiments/summaries/m4_grid/metrics.json, reports/M4_GRID.md
```

Un run par appel (≈ 210 s seul, GPU partagé), au premier plan ; la relance tient
dans la session, pas de `/goal`.

## 6. Rapport

Runs : `m4_grid_preflight_fulltone_s3t33_seed0_v1` (2 pas, 13 s),
`m4_grid_{fulltone,bigmuff}_s3t33_seed{0,1,2}_v1` (215–230 s chacun), tous
`completed`, code au commit `a8443d4` (les runs Big Muff seeds 1–2 enregistrent
`dde7419` comme HEAD : commit d'un autre agent sur `DIAGNOSIS_seeds.md`,
`diagnosis/seeds/` et `scripts/product_nablafx_queue.sh`, sans effet sur cette
lignée). Résumé : `experiments/summaries/m4_grid/metrics.json`, `reports/M4_GRID.md`.

**Prédiction tenue — verdict `confirmed`** (Fulltone, S3 t33, seeds 0 / 1 / 2, médiane) :

| Métrique | Avant (grille ± 2) | Après (grille ± 0,4) | Seuil | |
|---|---|---|---|---|
| ESR fenêtres 101–200 | 0,1041 / 0,1146 / 0,1127 (0,1127) | 0,0464 / 0,0543 / 0,0529 (**0,0529**) | ≤ 0,090 | tenu ; baisse 0,060 ≫ 0,01 |
| ESR test | 0,1404 / 0,1445 / 0,1333 (0,1404) | 0,0628 / 0,0791 / 0,0857 (**0,0791**) | ≤ 0,125 | tenu |
| Transfert | +0,0363 / +0,0299 / +0,0205 (+0,0299) | +0,0164 / +0,0248 / +0,0328 (**+0,0248**) | ≤ +0,035 | tenu |
| Bande 100–300 Hz | 0,0545 / 0,0643 / 0,0522 (0,0545) | 0,0222 / 0,0324 / 0,0378 (0,0324) | informatif | |

Position face aux bornes de la section 1 : le nouveau S3 passe **sous** l'optimum
local par refit (0,086 fenêtres / 0,132 test) et sous le FIR LS 65 taps
(0,0955) ; il passe aussi sous la borne « Hammerstein parallèle » du diagnostic
(0,062, seed 0 : ici 0,046) et rejoint la borne « Hammerstein commuté » (0,049).
En test, la médiane 0,079 se situe à 0,015 d'A2 (0,064) ; seed 0 (0,063) est sous
A2 seed 0 (0,067). Sur l'écart test S3 t33 − A2 (0,077), 80 % sont fermés par
cette seule intervention, contre une part « optimisation » estimée à un tiers
par le diagnostic : la grille pesait plus que ce que le refit bloc-coordonnées
(un optimum local, section 6 du diagnostic) laissait voir.

**Test falsifiable de H2 du diagnostic (§5)** : « un correctif qui garde la NL
statique entre deux filtres fixes et descend sous 0,085 en test réfute H2 ». Sur
les trois checkpoints, contrôleur lent et résidu sont inertes (drive × [0,96 ;
0,99], gain × [0,96 ; 0,99], rapport d'énergie du résidu ≤ 3,5e-6) ; la cascade
statique seule (FIR33 → spline → FIR33, sans contrôleur ni résidu) donne en test
0,0640 / 0,0809 / 0,0844 (**0,0809**), sous 0,085 pour les trois seeds. Par le
critère préinscrit du diagnostic, H2 comme mécanisme *nécessaire* est réfutée :
une non-linéarité statique correctement résolue entre deux FIR appris atteint le
niveau que la référence LS statique (0,095) laissait croire hors de portée. Les
mesures de H2 sur la cible (13 dB de compression, forme dépendante du niveau)
restent des descriptions valides ; c'est leur attribution causale à l'écart qui
tombe. Marge faible (0,081 contre 0,085 ; seed 2 à 0,084) : à confirmer avant
d'en faire une conclusion de rapport.

**Contrôles face aux tolérances (section 4) :**

| Contrôle | Résultat | Tolérance | |
|---|---|---|---|
| Test unitaire `test_m4_grid_diagnostic_keeps_identity_init_and_block_parity` | passe (identité ≤ 3e-6 à amplitude 0,5 ; parité < 2e-6 ; 1 276 paramètres ; grille ± 2 sans la clé) | — | tenu |
| `make lint`, `make test` | passent, 704 tests | — | tenu |
| ESR de validation initiale | 0,155239 (Fulltone), 1,755279 (Big Muff), égaux aux runs mémoire à 1e-7 | identité à l'init | tenu |
| `finite_prediction` / `alternate_block_max_abs` (7 runs) | vrai / 0,0 (seed 2 Fulltone : 1,5e-8) | vrai / ≈ 0 | tenu |
| ESR test recalculé depuis `predictions/test_prediction.f32` | écart max 4,7e-10 sur les 18 lignes | — | tenu |
| Paramètres | 1 276 | égal | tenu |
| **Big Muff S3 t33, ESR test médiane** | 1,0353 / 1,0003 / 1,0265 (**1,0265**) contre 0,9444 : **+0,082** | hausse ≤ 0,05 | **non tenu** |

Le contrôle hors prédicat échoue : +0,082 sur la médiane, +0,032 au-delà de la
tolérance, et la hausse est aussi présente sur les fenêtres d'entraînement
(0,9511 → 0,9835 en médiane), donc ce n'est pas un effet de transfert. Contexte,
qui ne l'annule pas : dans les deux lignées, S3 n'apprend pas la Big Muff
(erreur de gain −0,90, corrélation 0,22–0,25 : la sortie vaut ≈ 10 % de la cible,
l'ESR ≈ 1 est le plancher « prédire zéro ») ; le checkpoint retenu est le pas 40
pour deux seeds sur trois dans chaque lignée. Sur cet appareil, l'entrée de
spline reste dans ± 0,15 (p99) même sur la grille fine, la spline apprise reste
à moins de 0,14 de l'identité et les pentes d'extrémité descendent à 0,48–0,86 :
le modèle réduit son gain plutôt que d'écrêter. La grille fine n'y est donc pas
neutre et rend légèrement plus mauvais un modèle déjà au plancher. Ce résultat
est enregistré tel quel : le correctif est validé sur son prédicat, pas comme
réglage universel de S3 ; la Big Muff exigerait sa propre lignée (grille ou drive
adaptés à un rms d'entrée de spline de 0,04, ou une autre classe).

**Effet de bord de la section 3 (régularisation de courbure)** : le transfert
médian baisse (+0,030 → +0,025) au lieu de monter ; pentes maximales de spline
1,15–1,36 (contre 3,6 pour le refit non régularisé) ; l'effet redouté ne s'est pas
matérialisé à 200 pas.

**Régime hors grille** : au checkpoint, le pré-FIR s'est contracté (rms de l'entrée
de spline 0,10–0,11 contre 0,14 ; 0,4–1,0 % des échantillons train au-delà de
± 0,4, 0,15–0,4 % en test) : l'optimiseur a ramené le signal dans la grille plutôt
que d'exploiter les pentes affines.

**Incertitudes restantes.**
- Un seul niveau de grille (± 0,4) testé ; la dépendance à la borne (± 0,3, ± 0,6)
  et l'interaction avec le nombre de nœuds ne sont pas mesurées ; le
  meilleur réglage n'est pas connu.
- Sélection de checkpoint sur 5 points de validation (± 0,03 entre évaluations) :
  l'étendue inter-seeds en test (0,063–0,086) en contient une part non quantifiée.
- La réfutation de H2 repose sur le seuil préinscrit 0,085 et une marge de 0,004 ;
  elle ne dit pas que la cible n'a pas de filtre dépendant du niveau, seulement
  que ce n'est pas ce qui séparait S3 t33 d'A2 à ce niveau d'ESR.
- L'écart restant à A2 (0,015 en test, 0,021 sur les fenêtres) n'est pas décomposé.
- Régression Big Muff : cause non isolée (grille trop large pour un rms de 0,04,
  ou interaction avec la sélection précoce au pas 40) ; un seul contrôle, trois seeds.
- Les seuils de `m4_memory.yaml` (test ≤ 0,125, bande ≤ 0,05, transfert ≤ 0,03)
  sont tous atteints par cette lignée (0,079 / 0,032 / +0,025), mais elle est
  post-freeze : `PROTOCOL_LOCK` et le verdict NO-GO restent inchangés ; rouvrir
  quoi que ce soit demande une décision séparée.
