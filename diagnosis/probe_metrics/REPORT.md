# Sondes d'analyse comme métriques de fidélité : rapport de validation

Voie produit, aucune revendication scientifique ni perceptive. Protocole, écarts et règles de
lecture : `NOTE.md`, qui date tout ce qui est préinscrit avant les chiffres. Tableau principal :
`table_test.csv` (**métrique × défaut × sévérité × condition d'excitation**). Synthèse :
`summary_test.json`. Chaque chiffre cité ici est recalculé par `report_numbers.py` dans
`report_numbers.json`. Figures : `figures/`. Coût : `cost.json`.

**Question.** Quelles différences entre une sortie de référence et une sortie évaluée (même
entrée) chaque sonde détecte-t-elle, avec quelle sensibilité, quelle robustesse et quels angles
morts ?

## Ce que recouvre chaque type de résultat

| Type | Contenu | Fichiers |
|---|---|---|
| Exécuté, synthétique | 13 défauts DSP connus × 3 sévérités, 6 familles d'excitation (bruit coloré, multisinus, sinus glissant synchronisé, deux tons, plucks, DI réelle) × 3 niveaux × 3 graines (DI : 1 réalisation) ; sondes FIR, Hammerstein 3 et 5, Volterra–Laguerre 2 et 3, Wiener–Hammerstein ; ESR, MR-STFT, IMD, signature harmonique | `calibration_test_*.csv` |
| Exécuté, réel | VS100R (envoi FX) : 3 répétitions, 12 contrôles négatifs, 8 changements réels ; JVM410H : 26 réglages d'un bouton × 2 sorties | `real_pairs_*.csv` |
| Hypothèses | lectures marquées « post hoc », « exploratoire » ou « hypothèse » ci-dessous | — |
| Non mesurable sans nouvelles captures | BLA stricte sur le réel ; phase harmonique synchronisée sur la VS100R ; niveaux hors des 3 sinus glissants ; plancher apparié dans le temps et répétabilité d'un jour à l'autre ; > 20/24 kHz ; sortie VS100R après l'étage de tonalité | `manifest.json` |

## Contrôles mathématiques

- **Borne** |d_audio − d_probe| ≤ r_ref + r_eval : aucune violation, sur toutes les lignes
  d'étalonnage de chaque sonde (de 120 lignes pour WH à 1 920 pour FIR et Hammerstein) et sur
  toutes les paires réelles.
- **Linéarité.** Pour les sondes linéaires en paramètres, fit(Δy) = Δθ et
  d_probe² = Δθᵀ G_B Δθ sont vérifiés numériquement (tests unitaires). Cela suppose que la
  normalisation et la régularisation ne dépendent que de x_A.
- **Contre-exemple F = x + 0,1 x³ / G = x + 0,2 x³.** L'écart entre les erreurs de fit est
  inférieur à 1e-6 en relatif, tandis que d_probe = 0,1 RMS(x_B³) à 1e-4 près, porté par la
  seule branche d'ordre 3.
- **Wiener–Hammerstein.** Deux reparamétrisations exactes (signe, gain) rendent la même sortie
  au bit près, avec des paramètres éloignés. Entre deux départs, les paramètres diffèrent environ
  30 fois plus que les rendus. Aucune métrique ne compare donc des paramètres.
- **Emboîtement Hammerstein 5 / Volterra–Laguerre 3 : absent dans les deux sens.** Une colonne
  xᵏ retardée est de 13 à 96 % hors de l'espace Volterra ; une colonne Volterra est en médiane
  à 66 % hors de l'espace Hammerstein. Un écart de score entre ces deux sondes ne se lit donc
  pas comme un écart de capacité.

## Défauts connus

**Lecture préinscrite.** Rapport de détection R = score / plancher. Le plancher est le score
obtenu entre deux mesures de la même chaîne, chacune avec un bruit à −80 dB. « Détecté » si
R ≥ 10, « non détecté » si R < 3. L'ESR et l'IMD sont lues en amplitude.

**Lectures ajoutées.** Elles sont signalées comme telles :
- la **part captée** d_probe / d_audio ;
- la **part non représentée** ;
- les **profils relatifs à d_audio**, qui retirent l'échelle de sévérité.

### 1. Quels défauts chaque métrique détecte-t-elle ?

Comptes à sévérité maximale et niveau médian, sur toutes les familles (une cellule = un défaut
× une famille) :

| Métrique | Détectés | Limites | Non détectés |
|---|---|---|---|
| FIR | 76 | 2 | 0 |
| Hammerstein 3 et 5 | 76 | 1 | 1 |
| ESR (= d_audio en amplitude) | 72 | 1 | 5 |
| MR-STFT | 8 | 7 | 63 |
| IMD (deux tons) | 13 | 0 | 0 |
| Signature harmonique totale (sinus glissant) | 12 | 1 | 0 |

**Le seuil seul ne départage pas les sondes.** Elles ne représentent pas le bruit blanc, donc
leur plancher est très bas, et elles « détectent » presque tout. Ce qu'elles représentent
vraiment se lit dans la part captée (médiane sur les familles, sévérité maximale, niveau
médian ; part non représentée entre parenthèses) :

| Défaut | FIR | H3 | H5 | VL2 | VL3 |
|---|---|---|---|---|---|
| gain | 0,99 (0,10) | 1,00 (0,05) | 1,00 (0,05) | 0,98 (0,19) | 0,99 (0,08) |
| coloration, préfiltre, redistribution à THD égal | 0,99 | 1,00 | 1,00 | 0,97–0,99 | 0,99–1,00 |
| retard fractionnaire | 0,97 (0,18) | 1,00 (0,06) | 1,00 (0,05) | 0,97 (0,23) | 0,89 (0,45) |
| passe-tout | 0,97 (0,18) | 1,00 (0,06) | 1,00 (0,05) | 0,97 (0,25) | 0,93 (0,36) |
| relâchement d'enveloppe | 0,95 (0,22) | 0,97 (0,15) | 0,97 (0,15) | 0,95 (0,25) | 0,96 (0,21) |
| cubique (après passe-haut) | 0,79 (0,60) | 0,99 (0,30) | 0,99 (0,30) | 0,79 (0,61) | 0,99 (0,12) |
| quadratique (après passe-haut) | 0,64 (0,77) | 0,96 (0,42) | 0,96 (0,42) | 0,98 (0,17) | 0,98 (0,15) |
| interaction u[n]·u[n−5] | 0,45 (0,90) | 0,85 (0,58) | 0,86 (0,57) | 0,97 (0,26) | 0,99 (0,12) |
| terme u⁷ (au-delà du degré) | 0,35 (0,94) | 0,81 (0,72) | 0,98 (0,52) | 0,39 (0,95) | 0,97 (0,65) |
| écho à 30 ms (au-delà de la mémoire) | 0,16 (0,99) | 0,19 (1,00) | 0,21 (1,00) | 0,11 (0,99) | 0,15 (0,99) |
| bruit additif | 0,02 (1,00) | 0,06 (1,00) | 0,08 (1,00) | 0,01 (1,00) | 0,03 (1,00) |

VL2 et VL3 = Volterra–Laguerre d'ordre 2 et 3 ; la DI est comprise pour FIR et Hammerstein.
Une part captée et une part non représentée ne s'additionnent pas : rien ne les rend
orthogonales hors échantillon.

- **Sensibilité sous le bruit brut.** À la plus faible sévérité, les sondes voient le cubique
  sur toutes les familles, alors que l'ESR ne le voit sur aucune ; même chose pour le
  quadratique (ESR : 3 familles sur 6). La sonde moyenne sur toute l'estimation et rejette ce
  qui sort de son espace, comme une détection synchrone. Le gain dépend de sa capacité et d'un
  plancher de bruit blanc (voir le réel ci-dessous).
- **Signature harmonique D_k, spécifique par ordre.** Le quadratique ne touche que l'ordre 2 ;
  le cubique touche les ordres 1 et 3 ; l'interaction croisée l'ordre 2. Elle est quasi
  déterministe d'une réalisation à l'autre et exige un sinus glissant.
- **Complémentarité Hammerstein / Volterra–Laguerre.** Volterra–Laguerre, avec ses produits de x
  filtré, représente les non-linéarités qui suivent un filtre et les interactions temporelles.
  Hammerstein, avec ses 256 taps, représente mieux la mémoire linéaire longue.
- **FIR.** Il ne voit les changements non linéaires que par la réponse linéaire équivalente au
  niveau de l'excitation.

### 2. Quels défauts manque-t-elle ?

- **Bruit additif et écho hors mémoire** : non représentés par aucune sonde. Pour l'écho, cela
  dépend du contenu : il est capté à ~0,5 sur des notes périodiques (plucks), où il ressemble à
  un filtre en peigne, et à ~0,2 sur la DI.
- **Non-linéarité préfiltrée** : Hammerstein laisse 0,30 à 0,42 non représenté ; Volterra–Laguerre
  3 seulement 0,12 à 0,15.
- **Interaction croisée** : Hammerstein l'attribue à l'ordre 2 quand l'excitation est corrélée
  sur 5 échantillons (bruit coloré, plucks). Sa décomposition par branche ne distingue alors
  pas « carré » et « produit retardé ».
- **Dynamique d'enveloppe sur jeu réel** : sur la DI, le relâchement n'est capté qu'à 0,60–0,66,
  contre 0,97 sur bruit stationnaire. Ce manque n'apparaît qu'avec une excitation réaliste.
- **Terme u⁷ à faible sévérité** : vu par presque aucune métrique (FIR 2 familles sur 6).
- **MR-STFT de `metrics/spectral.py`** (epsilon 1e-7) : aveugle au passe-tout et au retard
  (magnitude seule) et dominée par un bruit à −80 dB dans les bandes où la référence est faible.
  Constat propre à cette implémentation.

### 3. L'extraction est-elle plus stable que les écarts recherchés ?

Oui, sauf pour Wiener–Hammerstein. Entre graines indépendantes, l'écart-type du log du score a
une médiane de 0,0016 à 0,007 pour toutes les métriques sauf l'IMD (0,026). La signature
harmonique est quasi déterministe.

**Réserve.** Les sinus glissants de test sont bit à bit ceux du développement : la durée y est
tirée de graine % 3 (voir `NOTE.md`). Sur cette famille, la stabilité ne vaut donc pas
généralisation à de nouvelles réalisations.

Wiener–Hammerstein : voir la section dédiée.

### 4. Que révèlent les sondes au-delà des métriques existantes ?

- **La nature de l'écart.** La part captée et la part non représentée disent si l'écart est
  linéaire, une non-linéarité statique, une interaction temporelle, ou hors de toute sonde
  (bruit, écho long). La décomposition par branche et la signature harmonique donnent l'ordre.
- **La robustesse au bruit de mesure.** Le bruit additif n'entre presque pas dans les scores de
  sonde (part captée de 0,01 à 0,08), alors que l'ESR le compte entièrement.
- **La robustesse aux défauts d'acquisition sur le réel.** Chaque sonde est ajustée sur sa propre
  entrée enregistrée. Exemple, le contrôle négatif aigus 10, dont la section contient un
  glissement : ESR à 3,7 planchers, Hammerstein 3 sous 1.
- **Une limite.** À entrée commune, les sondes ne séparent pas mieux que l'ESR des changements
  réels (voir le réel ci-dessous).
- **Un signal de non-identifiabilité (exploratoire).** Quand la somme des contributions par
  branche dépasse 2,5 fois le score, les branches se compensent. Le seuil a été choisi après
  l'étalonnage : 1,2 à 1,8 pour les défauts linéaires et statiques, 3,7 à 8,7 pour le bruit et
  l'écho. Il est à valider ailleurs.

### 5. Quand faut-il refuser de conclure ?

- Part non représentée ou rapport r / RMS(y) élevé : un faible score n'est pas une garantie de
  fidélité.
- Branches qui se compensent : la décomposition par ordre n'est pas lisible.
- Dispersion entre initialisations comparable à l'écart mesuré (Wiener–Hammerstein).
- Référence quasi silencieuse, ou signature harmonique sans fondamental : « non mesurable ».
- Lecture hors du domaine excité. Exemple corrigé en cours de campagne : les deux tons
  d'estimation et de lecture n'avaient pas les mêmes fréquences.
- Comparaison directe d'enregistrements à entrée non commune.
- Plancher réel non apparié dans le temps (voir la VS100R).
- Score de sonde au-dessus de d_audio (hors échantillon, une branche peut extrapoler : 1,10 à
  1,14 sur la DI pour les termes pairs) : écart surestimé.

### 6. Ensemble minimal de métriques complémentaires

- **FIR** : repère linéaire, très bon marché (0,1 s).
- **Hammerstein 3** : H5 lui est redondant ici. Il voit les non-linéarités statiques et garde la
  mémoire linéaire longue.
- **Volterra–Laguerre 3** : il voit les non-linéarités préfiltrées et les interactions
  temporelles que Hammerstein manque ; plus coûteux (6 s).
- **Signature harmonique par ordre**, quand un sinus glissant est disponible : elle sépare les
  défauts pairs et impairs.
- **ESR (ou d_audio)** : elle voit ce qu'aucune sonde ne représente, bruit et écho long.
- **La part non représentée** de chaque sonde, comme indicateur de couverture.
- **Écartés.** Wiener–Hammerstein : peu fiable ici et très coûteux. MR-STFT telle qu'implémentée :
  aveugle à la phase et sensible au bruit.

## Wiener–Hammerstein

**À compléter** avec l'étalonnage corrigé (relecture Codex : la renormalisation de h_in annulait
le pas de Gauss–Newton ; h_out était ajusté sur une autre région que les autres étapes).
Développement corrigé : capacité 0,186, dispersion médiane entre départs 0,092, sous le seuil
« peu fiable avant défaut » (0,093) de justesse.

## Sorties réelles VS100R (envoi FX, 26/09, sans modèle neuronal)

`real_pairs_vs100r.csv`. Chaque sonde est ajustée sur l'entrée de sa propre prise, puis lue sur
le bouclage de la prise de référence. A = `di_train`, B = `di_test` (sources indépendantes) et
bruit rose. La signature harmonique est mesurée sur les trois sinus glissants, qui ne sont pas
synchronisés.

**Entrée.** Le bouclage de la prise 1 (début de séance) diffère de −42 dB de celui des autres
prises, contre −60 à −70 dB entre prises ultérieures. Sept sections `di_test` contiennent un
glissement (−16 à −24 dB) : elles sont marquées « entrée non commune » pour les comparaisons
directes.

**Lecture préinscrite** (plancher réel = médiane des trois répétitions du centre), sur
`di_test` :

| Métrique | Répétitions | Contrôles négatifs : médiane (max) | Changements réels |
|---|---|---|---|
| Hammerstein 3 | 1,17 ; 1,00 ; 0,23 | 0,13 (1,23) | 0,6 à 3,2 ; gain 0 : 7,7 |
| ESR (amplitude) | 1,20 ; 1,00 ; 0,31 | 0,26 (3,73, entrée non commune) | 0,9 à 5,2 ; gain 0 : 7,2 |
| MR-STFT | 1,21 ; 1,00 ; 0,63 | 0,57 (1,32) | 0,7 à 6,2 ; gain 0 : 15,6 |

- **Contrôles négatifs** (basses, médiums, aigus, réglés avant l'étage de tonalité) : sous 3
  pour toutes les sondes, comme attendu. Le seul dépassement (ESR) vient d'une section à
  glissement.
- **Changements réels** : l'attente « ≥ 10 » n'est tenue par **aucune** métrique. Le plancher
  est dominé par la dérive au cours de la séance. Les deux répétitions avec la prise 1 (14 h 26)
  valent environ 5 fois la répétition 20–39 (2 h d'écart), alors que chaque contrôle est apparié
  à la prise centrale la plus proche dans le temps.

**Lectures post hoc** (non préinscrites) : séparation des changements réels (gain 0 exclu) et
des contrôles négatifs, mesurée par l'AUC avec un p exact unilatéral.

| Segment | Métrique | Toutes les paires (7 / 12) | Entrée commune seulement |
|---|---|---|---|
| `di_test` | Hammerstein 3 | 0,98 (p = 8e-5) | 0,98 (p = 7e-4, 5 / 10) |
| `di_test` | ESR | 0,89 (p = 0,002) | **0,98** (p = 7e-4, 5 / 10) |
| `di_test` | MR-STFT | 0,93 | 0,92 (p = 0,004) |
| bruit rose | ESR | 0,99 | 0,99 |
| sinus glissant −10 dBFS | signature harmonique totale | 0,99 (p = 4e-5) | — |
| sinus glissant −3 dBFS | signature harmonique totale | 0,63 (p = 0,19) | — |

- **À entrée commune, ESR et sondes séparent pareillement.** L'avantage apparent des sondes sur
  toutes les paires vient des sections à glissement. Sur le réel, leur intérêt est de rester
  utilisables malgré l'acquisition, pas de mieux séparer.
- **À −3 dBFS, la signature harmonique ne sépare plus.** Aucun échantillon n'est écrêté à
  l'acquisition (crêtes de −10 à −12 dBFS). Mais la crête de l'envoi ne monte que d'environ 1 dB
  entre −20 et −3 dBFS : la saturation est profonde, et les contrôles négatifs y valent autant
  que les répétitions. Hypothèse non vérifiée : la sonde de degré 5 ne représente plus cette
  saturation, et les réglages y donnent des signatures voisines.
- **Un contour ±2,5 reste dans l'étendue des contrôles négatifs** pour toutes les métriques.

**Conséquence.** Un plancher fait de répétitions éloignées dans le temps surestime le bruit
d'une comparaison rapprochée. Il faudrait des répétitions du centre intercalées entre les
prises, ce qui demande de nouvelles captures.

## Sorties réelles JVM410H OD1 (ToneTwist, sans modèle neuronal)

`real_pairs_jvm.csv`. Échelles d'un seul bouton autour de B5 M5 T5 G5, aux sorties préampli et
haut-parleur, avec une entrée sèche commune. A = 0–150 s et B = 215–300 s du même fichier :
**l'indépendance de A et B n'est pas établie**. Il n'y a pas de répétition, donc pas de
plancher réel.

- **Monotonie observée.** Sur les 40 échelles par sortie (4 boutons × 2 côtés × 5 métriques),
  le score croît strictement avec la distance au centre (ρ de Spearman = +1 partout). Ce n'était
  pas imposé.
- **Couverture.** Hammerstein 5 capte 0,63 (préampli) et 0,62 (haut-parleur) de l'écart, avec
  une part non représentée de 0,75 à 0,76. Sur cet ampli à lampes, les scores de sonde sont des
  signatures partielles. Causes possibles, non départagées : non-linéarité et mémoire réelles
  au-delà de la capacité gelée, A et B non indépendants, niveaux variables de la DI.
- **Borne** : aucune violation.

## Coût de calcul

Une paire, 4 s d'estimation et 4 s de lecture à 48 kHz, machine partagée (charge 15 à 18 sur
32 cœurs) :

| Métrique | Temps |
|---|---|
| ESR | 0,001 s |
| MR-STFT | 0,08 s |
| FIR | 0,11 s |
| IMD | 0,16 s |
| Hammerstein 3 / 5 | 0,39 / 0,68 s |
| Signature harmonique | 0,81 s |
| Volterra–Laguerre 2 / 3 | 1,4 / 6,1 s |
| Wiener–Hammerstein | 32 s pour 1 s d'estimation |

## Commandes de reproduction

```sh
uv run pytest tests/unit/test_probe_metrics.py tests/unit/test_defect_bench.py -q
export CUDA_VISIBLE_DEVICES=""
P=diagnosis/probe_metrics
.venv/bin/python $P/dev_select.py && .venv/bin/python $P/dev_select_v2.py
for f in "noise multisine" "sweep two_tone" plucks di; do
  .venv/bin/python $P/calibrate.py --phase test --families $f
  .venv/bin/python $P/calibrate.py --phase test --probes v2 --families $f
done   # Volterra sur DI : --levels -20
.venv/bin/python $P/calibrate.py --phase test --probes wh --families noise --levels -20 --seconds 1
.venv/bin/python $P/decomposition.py noise && .venv/bin/python $P/decomposition.py plucks
.venv/bin/python $P/real_pairs.py vs100r && .venv/bin/python $P/real_pairs.py jvm
.venv/bin/python $P/cost.py && .venv/bin/python $P/manifest.py
.venv/bin/python $P/summarize.py && .venv/bin/python $P/figures.py && .venv/bin/python $P/report_numbers.py
```
