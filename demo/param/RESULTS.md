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
| `param_ssm_wavenet_ffuzz_r2` | SSM-WaveNet-FiLM | F-Fuzz | 15 000 | non | 0,1366 | valide (relance) |
| `param_s4_ffuzz_r2` | S4 paramétrique | F-Fuzz | 15 000 | non | 0,1377 | valide (relance) |

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

## F-Fuzz (relances `_r2`, entrées `DRY-with-markers`)

Réglages de test sur le DI de test (133 segments de 5 s chacun). **Non vus** = F030 et
F070, retirés de l'entraînement ; **vus** = les 9 autres, sur le même matériau de test
(comparaison propre, contrairement au Marshall).

| réglage | SSM ESR | SSM MR-STFT | S4 ESR | S4 MR-STFT |
|---|---|---|---|---|
| F000 | 0,1129 | 0,139 | 0,1155 | 0,255 |
| F010 | 0,1153 | 0,139 | 0,1147 | 0,285 |
| F020 | 0,1202 | 0,135 | 0,1192 | 0,232 |
| **F030 (non vu)** | 0,1226 | **0,300** | 0,1240 | **0,346** |
| F040 | 0,1281 | 0,137 | 0,1283 | 0,258 |
| F050 | 0,1346 | 0,138 | 0,1335 | 0,239 |
| F060 | 0,1371 | 0,140 | 0,1372 | 0,221 |
| **F070 (non vu)** | 0,1458 | **0,264** | 0,1444 | **0,404** |
| F080 | 0,1473 | 0,163 | 0,1510 | 0,272 |
| F090 | 0,1565 | 0,193 | 0,1593 | 0,312 |
| F100 | 0,1827 | 0,234 | 0,1875 | 0,375 |

| synthèse (test) | SSM-WaveNet-FiLM | S4 paramétrique |
|---|---|---|
| vus, ESR moyen | 0,1372 | 0,1385 |
| vus, MR-STFT moyen | 0,157 | 0,272 |
| non vus, ESR moyen | 0,1342 | 0,1342 |
| non vus, MR-STFT moyen | 0,282 | 0,375 |

### Verdicts F-Fuzz

**Q1** (ESR de F030 ≤ 1,5 × moyenne F020/F040, et F070 ≤ 1,5 × moyenne F060/F080) —
SSM 0,988 et 1,025 ; S4 1,002 et 1,002. **Tenue** pour les deux modèles : en ESR,
interpoler un bouton de fuzz ne coûte rien.

**Q5** (sur les réglages vus, S4/SSM ≥ 1/1,5) — 1,009. **Tenue** : indistinguables en
ESR.

**Q6** (écart non vus/vus de SSM ≤ 1,5 × celui de S4) — 0,978 contre 0,969, rapport
1,010. **Tenue** : les deux FiLM statiques interpolent aussi bien.

### Ce que l'ESR ne montre pas (post hoc, non prédit)

- **La MR-STFT double sur les réglages non vus.** SSM : 0,300 (F030) et 0,264 (F070)
  contre 0,134 à 0,140 pour leurs voisins vus ; S4 : 0,346 et 0,404 contre 0,221 à 0,258.
  L'ESR des mêmes réglages suit la pente de ses voisins. L'erreur d'interpolation est
  spectrale (timbre), pas d'énergie de forme d'onde : l'ESR seul aurait conclu à tort que
  l'interpolation est gratuite.
- **SSM-WaveNet-FiLM a une MR-STFT 1,7 fois plus basse que S4** sur les réglages vus
  (0,157 contre 0,272), à ESR égal.
- **Les deux modèles ont presque exactement le même ESR réglage par réglage** (écarts
  < 3 %), qui croît régulièrement de F000 à F100. Un tel accord entre deux architectures
  suggère un plancher dû aux données (latence résiduelle du rendu numérique, bruit) plutôt
  qu'aux modèles ; non vérifié.

## Conclusion

- Les deux FiLM statiques (SSM-WaveNet sur ses couches 1×1, S4 nablafx) interpolent un
  bouton continu sans perte d'ESR (F-Fuzz), mais pas sans perte de timbre (MR-STFT ×2).
- Sur le Marshall, la combinaison extrême jamais vue coûte ×2,5 à ×3,6 par rapport à
  l'interpolation, comme Miklánek et al. l'ont montré à l'écoute ; SSM-WaveNet-FiLM y est
  2,8 fois meilleur que S4 paramétrique, qui sous-apprend dans ce protocole.
- Le FiLM est sans mémoire : SSM-WaveNet-FiLM garde le transfert de fréquence par Δ
  (phase 1). Son test par modèle entraîné n'est pas fait.
- Réserves : un run par cellule (graine 42) ; sur Big Muff, l'écart-type du log-ESR entre
  graines valait 0,27 à 0,53, donc seuls les rapports supérieurs à 1,5 sont interprétés.
