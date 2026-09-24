# Modèle paramétrique : résultats

Voie produit (D-PRODUCT-001), mesure produit, aucune revendication scientifique.
Prédictions : `predictions.md` (commit 29a869c, avant le premier entraînement). Chiffres :
`demo/nablafx_bench/param_*.json` (`test_last_by_setting`, `val_last_by_setting`),
tableau complet par `uv run python scripts/product_param_summary.py`. Écart au
protocole : `DEVIATION.md` (entrées sèches de F-Fuzz). Aucun run écarté.

## Runs

| run_id | modèle | appareil | pas | arrêt précoce | ESR test global | statut |
|---|---|---|---|---|---|---|
| `param_ssm_wavenet_marshall` | SSM-WaveNet-FiLM (28 385 p.) | Marshall | 15 000 | non | 0,1008 | valide |
| `param_s4_marshall` | S4 paramétrique (29 713 p.) | Marshall | 15 000 | non | 0,2833 | valide |
| `param_ssm_wavenet_ffuzz` | SSM-WaveNet-FiLM | F-Fuzz | 14 720 | époque 459 | 1,5904 | **invalide** (données désalignées) |
| `param_s4_ffuzz` | S4 paramétrique | F-Fuzz | 6 080 | époque 189 | 1,5060 | **invalide** (données désalignées) |
| `param_ssm_wavenet_ffuzz_r2` | SSM-WaveNet-FiLM | F-Fuzz | en cours | | | relance |
| `param_s4_ffuzz_r2` | S4 paramétrique | F-Fuzz | à venir | | | relance |

Les deux premiers runs F-Fuzz ont été entraînés sur un DRY sans marqueurs, décalé de 2 à
3 s par rapport aux cibles (`DEVIATION.md`) : ESR 1,5 à 1,6 sur tous les réglages, vus
et non vus, pour les deux modèles. Ils restent enregistrés comme échecs.

## Marshall JVM410H OD1

ESR et MR-STFT par réglage. **Non vus** = les 9 réglages de test (DI de test).
**Vus** = segments de validation des 29 réglages d'entraînement, pris dans les mêmes
fichiers DI que l'entraînement : mesure optimiste.

| réglage de test (non vu) | SSM ESR | SSM MR-STFT | S4 ESR | S4 MR-STFT |
|---|---|---|---|---|
| B000_M000_T100_G050 | 0,1212 | 0,832 | 0,2966 | 1,525 |
| B000_M100_T000_G050 | 0,0418 | 0,741 | 0,1663 | 1,152 |
| B000_M100_T100_G050 | 0,0763 | 0,792 | 0,3487 | 1,504 |
| **B010_M030_T070_G050** (interpolation, scénario 1) | 0,0475 | 0,668 | 0,1869 | 1,288 |
| B030_M070_T100_G050 (étiquette suspecte) | 0,3708 | 1,528 | 0,6315 | 1,700 |
| B065_M085_T035_G050 | 0,0151 | 0,603 | 0,1314 | 1,211 |
| **B100_M000_T000_G050** (combinaison extrême, scénario 2) | 0,1729 | 2,298 | 0,4583 | 2,517 |
| B100_M000_T100_G050 | 0,0442 | 0,706 | 0,2034 | 1,548 |
| B100_M100_T000_G050 | 0,0177 | 0,656 | 0,1263 | 1,281 |

| synthèse | SSM-WaveNet-FiLM | S4 paramétrique |
|---|---|---|
| vus, ESR moyen (règle préenregistrée) | 0,1193 | 0,7882 |
| vus, MR-STFT moyen | 0,341 | 1,232 |
| non vus, ESR moyen | 0,1008 | 0,2833 |
| non vus, MR-STFT moyen | 0,980 | 1,525 |
| vus, ESR médian (post hoc) | 0,0053 | 0,1702 |
| non vus, ESR médian (post hoc) | 0,0475 | 0,2034 |
| vus, ESR moyen sans G000 (post hoc) | 0,0058 | 0,1611 |

Note (relecture Codex, 2026-09-24) : la MR-STFT d'auraloss contient un rapport de
normes calculé sur tout un lot ; les MR-STFT rapportées sont des moyennes de scores de
lots de 8 segments pondérées par la taille, pas des moyennes de scores individuels comme
l'annonçait `predictions.md`. Elles restent comparables entre modèles (mêmes lots) ;
les verdicts, fondés sur l'ESR, n'en dépendent pas.

Le réglage vu `B050_M050_T050_G000` (gain 0, sortie presque silencieuse) a un ESR de
3,18 (SSM) et 17,7 (S4) : l'ESR divise par l'énergie de la cible et explose sur un
quasi-silence. Il domine à lui seul la moyenne des réglages vus.

### Verdicts Marshall

**Q2** (ESR `B100_M000_T000` ≥ 2 × `B010_M030_T070`) — SSM 0,1729 / 0,0475 = 3,64 ;
S4 0,4583 / 0,1869 = 2,45. **Tenue** pour les deux modèles : la combinaison extrême
jamais vue coûte bien plus que l'interpolation, comme chez Miklánek et al.

**Q3** (moyenne des non vus ≥ 1,5 × moyenne des vus) — SSM 0,845 ; S4 0,359.
**Réfutée** pour les deux selon la règle préenregistrée, à cause du réglage G000 (ci-
dessus). Post hoc, non prédit : sur les médianes, 9,0 (SSM) et 1,19 (S4) ; sur les
moyennes sans G000, 17,5 et 1,76. Le sens attendu apparaît dès que le quasi-silence est
écarté, fortement pour SSM, faiblement pour S4 dont l'erreur est déjà élevée sur les
réglages vus. Une règle d'agrégation robuste aurait dû être fixée d'avance.

**Q4** (`B030_M070_T100` pire ou deuxième pire des 9) — rang 1 pour les deux modèles
(SSM 0,3708, soit 2,1 fois le deuxième ; S4 0,6315, 1,4 fois le deuxième). **Tenue** : compatible avec une étiquette
fausse (l'original nomme ce réglage T1, pas T10), sans en être la preuve.

**Écart entre modèles** (non prédit sur Marshall) : SSM-WaveNet-FiLM a un ESR de test 2,8
fois plus bas que S4 (0,101 contre 0,283) et 32 fois plus bas en médiane sur les réglages
vus. S4 paramétrique reste à ESR 0,15 à 0,21 sur presque tous les réglages vus : il
sous-apprend dans ce protocole (15 000 pas, pertes 0,5/0,5).

## F-Fuzz

En attente des relances `_r2`.
