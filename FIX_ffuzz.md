# FIX_ffuzz : compenser la latence de 4 échantillons des cibles F-Fuzz

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Écrit le 2026-09-24, **avant tout
changement de code**. Les sections 1 à 4 ne sont plus modifiées après les résultats. Diagnostic :
`DIAGNOSIS_ffuzz.md`.

## 1. Spécification

**Prédicat et prédiction du diagnostic.** Tous les segments, fichiers, réglages et les deux
architectures. Les cibles F-Fuzz rendues arrivent 4 échantillons après ce que les modèles
produisent, et aucun des deux ne l'apprend.

Critères, en avançant les cibles de 4 échantillons à l'entraînement comme au test, avec le même
protocole et les mêmes graines :

- **ESR de test moyenne ≤ 0,01** pour SSM-WaveNet-FiLM comme pour S4-param, contre 0,1366 et 0,1377
  (`param_*_ffuzz_r2`) ;
- **l'écart entre architectures reparaît** : ESR(S4) ≥ 1,5 × ESR(SSM). Le réalignement a posteriori
  donne 0,0032 contre 0,0013 ;
- **rapport F100/F000 de l'ESR de SSM ≤ 10**, contre 6 après réalignement a posteriori.

Les bornes du réalignement a posteriori (0,0013 et 0,0032) sont des seuils de comparaison, pas des
gains attendus : ce réalignement choisissait le retard après avoir vu la sortie.

**Invariants hors prédicat**, avec leur valeur actuelle et leur tolérance :

- Big Muff et Marshall ne sont pas touchés. L'option ne s'applique qu'aux fichiers cibles de
  `--device ffuzz`, et elle est désactivée par défaut. Le code sans l'option reste le même ; aucun
  run existant n'est relancé.
- **Question de la MR-STFT des réglages non vus** (0,300 et 0,264 contre 0,134 à 0,140 pour leurs
  voisins vus, pour SSM) : elle est aveugle à la phase, donc à la latence. Elle doit rester du même
  ordre, à un facteur 1,5 près. Sinon, la latence jouait aussi sur l'apprentissage spectral,
  ce qui serait une surprise à rapporter.

## 2. Correctif

Une seule intervention : `--target-advance N` du banc, pour `--device ffuzz` seulement. Chaque
fichier cible est lu à partir de l'échantillon N, puis complété de N zéros à la fin, pour garder
sa longueur. Cela vaut pour l'entraînement, la validation et le test. N = 4, la valeur mesurée par
le diagnostic, identique sur les sept fichiers.

## 3. Argument de non-régression

- Hors F-Fuzz, l'option ne s'applique pas ; sans l'option, `_load` est inchangé. C'est l'identité
  par construction.
- Sur F-Fuzz, les N derniers échantillons de chaque fichier deviennent des zéros. Seul le dernier
  segment de chaque fichier est touché, sur 4 échantillons à 48 kHz, soit < 0,1 ms. Les
  segments de test font 5 s, et le dernier segment complet d'un fichier ne contient ces zéros que
  si la fin du fichier tombe exactement sur une frontière de segment. Effet négligeable, et
  borné par ce nombre.
- L'argument ne se ferme pas pour la dynamique d'entraînement : la cible change. **Contrôle
  empirique obligatoire**, par les deux runs de la section 5.

## 4. Contrôle

- **Invariant** : un test vérifie que `_load` avec l'option rend la cible décalée de N échantillons,
  complétée de N zéros et de même longueur, et qu'il ne touche pas un fichier d'entrée.
- **Relance** : les deux runs de `param_*_ffuzz_r2`, avec les mêmes drapeaux plus
  `--target-advance 4` : `param_ssm_wavenet_ffuzz_adv4` et `param_s4_ffuzz_adv4`. C'est le jeu de
  référence complet, pas un jeu réduit.
- **Mesure croisée** : `diagnosis/ffuzz/measure.py` sur les nouveaux checkpoints, avec la cible
  décalée. Le retard optimal doit être 0 dans la grande majorité des segments.

## 5. Exécution

Deux runs d'environ 4,5 h chacune, dans une file après G2.1 et le pilote C « fixe »
(`scripts/product_ffuzz_fix_queue.sh`), depuis un worktree figé. Nouveaux run_id, jamais
d'écrasement ; enregistrement dans `demo/RUNS.jsonl`.

**Écart déclaré avant relance (2026-09-28).** `param_ssm_wavenet_ffuzz_adv4` a été tué par
SIGTERM au redémarrage de 21:46, après `apt full-upgrade`, et après 3 h 26 de calcul. Il est
consigné `killed` dans `demo/RUNS.jsonl`. `param_s4_ffuzz_adv4` n'a jamais démarré.

- La cause est externe. Aucune mesure du run tué n'est exploitée. Une seule ligne de son
  `metrics.csv` (époque 334) a été affichée par inadvertance lors de l'inventaire des
  artefacts, après la décision de relance.
- **Une seule relance**, sous `param_ssm_wavenet_ffuzz_adv4_r2`, avec les mêmes drapeaux et le
  même code (runner `neural_amp_runner_g2` @ a0a4a4f). Elle repart de zéro, sans `--resume`.
  `param_s4_ffuzz_adv4` garde son id.
- **Le pilote NVIDIA a changé** (595.84 → 595.91.07, même noyau), alors que les références
  `param_*_ffuzz_r2` ont été entraînées sous l'ancien. Une porte de reproductibilité au bit
  précède les deux runs (`scripts/product_chain_after_reboot.sh`, trois chemins de
  `diagnosis/seeds/repro_gate.md`). Elle couvre le chemin SSM-WaveNet-FiLM (Marshall), pas
  S4-param, que la porte du 2026-09-24 ne couvrait déjà pas. S4-param partage la classe `S4`
  de nablafx et ses blocs (8 × 32 × 16) avec le chemin S4-TF, que la porte couvre. Ce qui lui est
  propre (FiLM statique, pas de `--honor-optim`) ne l'est pas : pour lui, le changement de pilote
  reste un facteur non mesuré.

  Résultat de la porte (2026-09-28, 22:59–23:31, a0a4a4f, pilote 595.91.07) : les trois chemins
  sont identiques au bit à leurs références (865, 881 et 348 valeurs, 0 écart ; lignes
  `repro_gate_drv59591_*` de `demo/RUNS.jsonl`).

## 6. Rapport

Rempli le 2026-09-29 après les deux runs. Sources :

- enregistrements `demo/nablafx_bench/param_ssm_wavenet_ffuzz_adv4_r2.json` et
  `demo/nablafx_bench/param_s4_ffuzz_adv4.json` (ESR de test du dernier checkpoint, ESR et
  MR-STFT par réglage) ;
- mesure croisée `diagnosis/ffuzz/segments_adv4.json`, produite par `diagnosis/ffuzz/measure.py
  --runs param_ssm_wavenet_ffuzz_adv4_r2 param_s4_ffuzz_adv4 --target-advance 4 --out
  segments_adv4.json`.

Pour la mesure, `measure.py` a reçu trois arguments (`--runs`, `--target-advance`, `--out`).
Avec leurs valeurs par défaut, sa relance sur `param_*_ffuzz_r2` redonne `segments.json` au bit
(même SHA-256).

**Runs.**

| run | pas | fin | commit |
|---|---|---|---|
| `param_ssm_wavenet_ffuzz_adv4_r2` | 15 000 | pas d'arrêt anticipé | a0a4a4f |
| `param_s4_ffuzz_adv4` | 14 944 | arrêt anticipé à l'époque 466 (meilleur checkpoint à l'époque 365) | a0a4a4f |

L'arrêt anticipé est actif dans les deux runs, comme dans les références `_r2`. Chaque run a été
écrit en une seule session, sans reprise. Les deux ont tourné sous le pilote 595.91.07, après la
porte au bit décrite en section 5.

**Critères de la section 1 : tous tenus.**

| critère | seuil | avant (`_r2`) | après (`adv4`) | issue |
|---|---|---|---|---|
| ESR de test moyenne, SSM-WaveNet-FiLM | ≤ 0,01 | 0,1366 | **0,00164** | tenu |
| ESR de test moyenne, S4-param | ≤ 0,01 | 0,1377 | **0,00344** | tenu |
| l'écart entre architectures reparaît : ESR(S4) / ESR(SSM) | ≥ 1,5 | 1,01 | **2,10** | tenu |
| F100 / F000 de l'ESR de SSM | ≤ 10 | 1,62 | **5,49** (0,00455 / 0,00083) | tenu |

Comparaison avec les bornes du réalignement a posteriori, qui sont des seuils et non des gains
attendus : l'ESR intérieure à retard 0 (`segments_adv4.json`, moyenne des segments) vaut
**0,00109** pour SSM, contre 0,0013 réalignée a posteriori, et **0,00284** pour S4, contre
0,0032. Les modèles entraînés sur les cibles avancées font donc un peu mieux que le réalignement
après coup des modèles `_r2`. Pour S4 à F100, l'ESR vaut 0,01181, soit plus que 0,01, mais le
critère porte sur la moyenne des réglages.

**Invariant, MR-STFT des réglages non vus (SSM)** : dans le facteur 1,5, tenu.

| réglage non vu | avant | après | rapport |
|---|---|---|---|
| F030 | 0,300 | 0,307 | ×1,02 |
| F070 | 0,264 | 0,316 | ×1,20 |

L'invariant pré-enregistré est respecté, ce qui ne démontre pas une absence d'effet (+2 % et
+20 %). Descriptif,
hors critère : les voisins vus, eux, baissent (F020, F040, F060 et F080 : de 0,135–0,163 à
0,104–0,128). L'écart de MR-STFT entre réglages vus et non vus s'élargit donc, d'environ ×2 à
×2,5 à ×3 : c'est la question, distincte, de `demo/param/RESULTS.md`.

**Invariant, Big Muff et Marshall intouchés** : tenu par construction, puisque l'option
n'agit que sur `--device ffuzz`. C'est aussi vérifié empiriquement : au code a0a4a4f, qui
contient l'option, la porte pilote de la section 5 reproduit au bit les chemins SSM Big Muff,
S4-TF Big Muff et SSM-FiLM Marshall.

**Mesure croisée de la section 4** : le retard optimal vaut **0 dans 1 463 segments sur 1 463**,
pour chacun des deux modèles. Avant le correctif (`diagnosis/ffuzz/segments.json`), il valait +4
dans 1 463 segments pour SSM, et dans 1 459 pour S4 (+3 dans les 4 autres).
L'ESR intérieure au meilleur retard est égale à celle à retard 0 (0,00109 et 0,00284).

**Invariant de code : tenu, après ajout du test prévu.** `tests/unit/test_product_advance.py`
vérifie `advance()` seul : décalage de N échantillons, complément de N zéros, longueur
conservée. Le test de `_load` prévu en section 4 manquait (relevé à la relecture Codex du
rapport). Il est ajouté après les runs : `tests/integration/test_bench_target_advance.py`. Avec
`TARGET_ADVANCE = 4`, il vérifie que `_load` rend un fichier `.target.` décalé de 4 et complété
de 4 zéros, de même longueur, et qu'il rend un fichier `.input.` intact. Le banc y est importé
dans un sous-processus. Contrôle du test lui-même : une copie dont la sonde règle `bench.TARGET_ADVANCE = 0`
au lieu de 4 échoue. Le test ne change rien aux
runs, qui ont tourné au code a0a4a4f, dont `_load` est le même.

**Correction d'une prémisse de la section 1 (2026-10-02).** La section 1 dit la MR-STFT « aveugle à
la phase, donc à la latence ». C'est inexact : sur les cibles de test F-Fuzz, un retard de 4
échantillons coûte 0,06 à 0,07 en MR-STFT, linéairement en d. Le retard est visible, mais faible
devant la MR-STFT de test des modèles (0,18 pour SSM et 0,29 pour S4 en moyenne, jusqu'à 0,40 selon le
réglage). L'invariant de la section 1 et son issue ne changent
pas.

**Incertitudes restantes.**

- Une graine et une partition (925) par architecture. Le rapport 2,10 entre architectures
  n'a pas de barre d'erreur. Sur Big Muff, le résidu apparié valait 0,048 en log
  (`diagnosis/seeds/paired_protocol.md`), mais rien ne dit qu'il vaut pour F-Fuzz.
- Pour S4-param, le changement de pilote n'est contrôlé que par les blocs S4 que la porte
  couvre, pas par son FiLM statique (section 5).
- Le doublement de la MR-STFT sur les réglages non vus reste entier : ce correctif ne le visait
  pas.
