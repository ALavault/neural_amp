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
- `scripts/run_m4_smoke.py` : option `--grid-range` (exclusive de `--memory-taps`
  et `--recovery-wide`), écrit `fssr_model.spline_range` et `grid` dans
  `config-resolved.yaml`, phase `M4_GRID` / `M4_GRID_PREFLIGHT`.
- `scripts/run_m4_grid.py` (pilote, `--seeds`/`--devices` pour lancer un run par
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
uv run python scripts/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 0   # préflight + seed 0
uv run python scripts/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 1
uv run python scripts/run_m4_grid.py --devices fulltone_full_drive_2 --seeds 2
uv run python scripts/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 0
uv run python scripts/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 1
uv run python scripts/run_m4_grid.py --devices electro_harmonix_big_muff --seeds 2
uv run python scripts/summarize_m4_grid.py   # experiments/summaries/m4_grid/metrics.json, reports/M4_GRID.md
```

Un run par appel (≈ 210 s seul, GPU partagé), au premier plan ; la relance tient
dans la session, pas de `/goal`.

## 6. Rapport

À remplir après la relance.
