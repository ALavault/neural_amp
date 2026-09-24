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

## 6. Rapport

À remplir après les runs.
