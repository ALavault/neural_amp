# Modèle paramétrique — prédictions

Voie produit (D-PRODUCT-001), mesure produit, aucune revendication scientifique. Écrit le
2026-09-23 **avant** tout entraînement paramétrique ; aucun chiffre de ces runs n'existe.
Code : `scripts/product_nablafx_bench.py` (`--device`, modèle `s4-param`), FiLM de
`src/fssr_nam/models/ssm_wavenet.py`, test `tests/numerical/test_ssm_wavenet_film.py`.

## Cellules (un run par cellule, rien d'autre ne sera compté)

| run_id | modèle | appareil | paramètres |
|---|---|---|---|
| `param_ssm_wavenet_marshall` | SSM-WaveNet-FiLM | Marshall JVM410H OD1 | 28 385 |
| `param_s4_marshall` | S4 paramétrique (nablafx, `cond_type=film`) | Marshall | 29 713 |
| `param_ssm_wavenet_ffuzz` | SSM-WaveNet-FiLM | F-Fuzz | 28 289 |
| `param_s4_ffuzz` | S4 paramétrique | F-Fuzz | 29 617 |

Réglages communs : `--seed 42 --split-seed 925 --deterministic`, pertes L1 0,5 + MR-STFT 0,5
(configs paramétriques de nablafx) pour les deux modèles, lr 0,01, clip 1, 15 000 pas au
plus, arrêt précoce patience 50, PolarityGuard. SSM-WaveNet : 8 blocs, 16 canaux, état 4,
`--discretization zoh --honor-optim`. FiLM : MLP réglages → 32 → (échelle, décalage) de la
couche 1×1 de chaque bloc, initialisé à zéro. Ordre : Marshall SSM, Marshall S4, F-Fuzz
SSM, F-Fuzz S4, un à la fois. Résultat principal : `last.ckpt`, comme le banc.
Un run qui échoue reste enregistré ; une relance prend un nouveau run_id.

## Découpages

- **Marshall** : découpage ToneTwist tel quel. 29 réglages en entraînement/validation
  (90/10 par segment de 3 s), 9 réglages de test **disjoints**, tous à gain 5. Réglages
  vus = segments de validation, par réglage : **même DI que l'entraînement**, donc mesure
  optimiste. Deux scénarios nommés d'après Miklánek et al. (DAFx-23) :
  interpolation `B010_M030_T070_G050`, combinaison extrême `B100_M000_T000_G050`.
- **F-Fuzz** : les 11 réglages de fuzz existent en entraînement et en test ; `F030_V100`
  et `F070_V100` sont **retirés de l'entraînement** (vue par liens symboliques). 10 % des
  segments d'entraînement (config nablafx). Test complet : réglages vus = 9 réglages sur le
  matériau de test, non vus = F030, F070. Le matériau de test diffère de l'entraînement :
  c'est la comparaison propre.

Métriques : ESR et MR-STFT par réglage (`test_last_by_setting`, `val_last_by_setting`),
moyenne sur segments de 5 s (test) et 3 s (validation).

## Dispersion attendue sans effet

Un seul run par cellule. Sur Big Muff, l'écart-type du log-ESR entre graines valait 0,27
à 0,53 (DIAGNOSIS_seeds.md). Donc : **un rapport d'ESR entre deux modèles inférieur à 1,5
n'est pas interprétable** et sera déclaré « indistinguable ».

## Prédictions

- **Q1 (F-Fuzz, interpolation)** Pour chaque modèle, l'ESR de F030 ≤ 1,5 × la moyenne de
  F020 et F040, et celui de F070 ≤ 1,5 × la moyenne de F060 et F080. Réfutée au-delà de
  2 ×, partielle entre les deux. Motif : un bouton unique, grille de 10 %, FiLM lisse.
- **Q2 (Marshall, extrême contre interpolation)** Pour chaque modèle, l'ESR de
  `B100_M000_T000` ≥ 2 × celui de `B010_M030_T070` (Miklánek : la boîte noire s'effondre
  sur la combinaison extrême, pas sur l'interpolation). Réfutée si < 1,2 ×.
- **Q3 (Marshall, vus contre non vus)** Pour chaque modèle, la moyenne des 9 réglages de
  test ≥ 1,5 × la moyenne des réglages vus en validation. Réfutée si < 1,1 ×.
- **Q4 (piège d'étiquette)** Si le nom ToneTwist `B030_M070_T100` est faux (l'original dit
  T1), le modèle reçoit un aigu à 1,0 pour un son à 0,1 : ce réglage sera le pire ou le
  deuxième pire des 9 pour les deux modèles. S'il est dans la moyenne pour les deux, rien
  n'indique d'erreur d'étiquette (ce n'est pas une preuve qu'elle est juste).
- **Q5 (SSM-WaveNet-FiLM contre S4)** Sur les réglages vus de F-Fuzz, SSM-WaveNet-FiLM
  n'est pas pire que S4 au-delà du seuil d'indistinguabilité (rapport S4/SSM ≥ 1/1,5),
  comme sur Big Muff où il le battait. Sur Marshall, pas de sens prédit.
- **Q6 (non vus, les deux modèles)** Sur F-Fuzz, l'écart relatif non vus/vus de
  SSM-WaveNet-FiLM est ≤ celui de S4 × 1,5 : un FiLM statique sur des couches ponctuelles
  n'interpole pas moins bien qu'un FiLM statique sur S4.

Chaque prédiction est jugée modèle par modèle ; un échec partiel est rapporté comme tel.
