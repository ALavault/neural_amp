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

- **ESR (ou d_audio)** : écart global. C'est la seule à voir ce qu'aucune sonde ne représente
  (bruit, écho long).
- **Magnitude et phase** (`metrics/spectral.py`, déjà dans le dépôt) : elles séparent le gain
  et les non-linéarités statiques (magnitude) du retard et du passe-tout (phase). Elles
  remplacent la MR-STFT et séparent les changements réels VS100R aussi bien que les sondes.
- **FIR** : repère linéaire très bon marché (0,1 s), qui écarte le bruit de mesure.
- **Hammerstein 3** : H5 lui est redondant ici. Il voit les non-linéarités statiques et garde la
  mémoire linéaire longue.
- **Volterra–Laguerre 3** : il voit les non-linéarités préfiltrées et les interactions
  temporelles que Hammerstein manque. Plus coûteux (~2 min par minute d'audio).
- **Signature harmonique par ordre**, si un sinus glissant intact est disponible : elle sépare
  les défauts pairs et impairs.
- **Dynamique d'enveloppe E2** (ad hoc), si des sauts de niveau sont disponibles : c'est la seule
  métrique spécifique au relâchement (voir plus bas).
- **La part non représentée** de chaque sonde, comme indicateur de couverture, et
  l'avertissement de capacité.
- **Écartés.** Wiener–Hammerstein : peu fiable ici et très coûteux. MR-STFT telle qu'implémentée :
  aveugle à la phase et dominée par le bruit. `envelope_error` : sensible mais non spécifique.

## Wiener–Hammerstein

Étalonnage corrigé après la relecture Codex : la renormalisation de h_in annulait le pas de
Gauss–Newton, et h_out était ajusté sur une autre région que les autres étapes. Le coût ne
remonte plus jamais (test). Conditions : bruit coloré à −20 dBFS, 1 s d'estimation et 1 s de
lecture, 3 départs, graines 200 à 202.

- **Développement.** Capacité 0,186 ; dispersion médiane entre départs, sur les mêmes données,
  de 0,092 du RMS. C'est juste sous le seuil préinscrit de 0,093.
- **Peu fiable dans 35 cellules sur 39** (règle préinscrite : dispersion > 0,5 × score). La
  dispersion entre départs reste d'environ 0,087 du RMS de la sortie, quel que soit le défaut.
  Elle ne passe sous l'écart mesuré que pour les plus gros (redistribution ≥ 0,3, préfiltre
  ≥ 250 Hz). Répétabilité entre graines : médiane 0,034, 90e centile 0,77, la pire du banc.
- **Le score paraît sage, mais c'est un effet du jumelage.** La part captée vaut 0,98 pour le
  gain et 0,88 pour le cubique, parce que les ajustements de référence et d'évaluation partent
  des mêmes initialisations et convergent vers des solutions jumelles. Ce jumelage masque la
  non-unicité : deux minima locaux différents produiraient jusqu'à ~0,09 d'écart fictif.
- **Coût** : 32 s pour 1 s d'estimation. Sonde écartée de l'ensemble minimal dans cette
  configuration.

## Dynamique d'enveloppe : détecteur ad hoc E2

`src/fssr_nam/product/dynamics.py`, préinscrit dans `NOTE.md` avant implémentation.
Résultats : `dynamics_bench.csv` et `dynamics_verdict.json`.

**Pourquoi un détecteur ad hoc.** Aucune métrique existante n'isole la dynamique.
`envelope_error` réagit autant au gain (R ≈ 1,4e4) qu'au relâchement (9,5e3), et sur DI réelle
les sondes ne captent que ~0,6 d'un changement de relâchement.

**Principe.**
1. On excite avec des **sauts de niveau** (bruit coloré, segments de 1,2 s, 22 sauts fort→faible
   et 21 faible→fort).
2. On calcule la trajectoire de gain à court terme après chaque saut, **moyennée sur les sauts**
   de même type.
3. On lui soustrait sa valeur stabilisée, ce qui retire tout effet statique.
4. L'indice est le RMS, en dB, de l'écart entre ces trajectoires pour la référence et la sortie
   évaluée, sur 10 ms–0,8 s.

Les 13 défauts sont appliqués à la **chaîne dynamique**, pour que la spécificité soit réellement
testée.

**Verdict préinscrit : les quatre critères sont remplis aux trois niveaux (−12, −9, −6 dBFS).**

| Critère | Résultat |
|---|---|
| Monotonie | relâchement 60 → 100 → 200 ms : 0,12 → 0,54 → 1,28 dB à −12 dBFS ; 0,20 → 0,88 → 2,06 dB à −6 dBFS |
| Sensibilité | relâchement 60 ms ≫ 3 × le plancher (~6e-5 dB) |
| Spécificité | le pire autre défaut (coloration, 0,05 dB) est 25 à 40 fois plus faible que le relâchement à 200 ms ; gain et polynômes ≈ 0 |
| Contrôle interne | l'indice d'attaque du défaut de relâchement vaut 0,06 à 0,09 de son indice de relâchement, à toutes les sévérités |

**Limites.**
- **Co-répondants déclarés d'avance** : l'écho réagit fortement (0,66–0,81 dB) et le préfiltre
  modérément (0,23–0,27 dB). L'indice n'est donc pas exclusif d'un écho.
- **Excitation dédiée requise.** Le réel est seulement descriptif : les sauts de gain de la DI
  ToneTwist ne sont pas datés, et `bursts_recovery` n'existe que sur une prise micro aux réglages
  connus. Aucune validation sur matériel.
- **Plancher trivial.** Moyennée sur les sauts, la référence bruitée à −80 dB est très basse. Le
  critère de sensibilité ne dit donc rien de la tenue face à un plancher réel.

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
- **À −3 dBFS, la signature harmonique ne sépare plus** (AUC 0,63). Un **glissement
  d'acquisition** à l'intérieur du sinus glissant −3 dBFS de la prise 20 en est un facteur établi.
  Cette prise sert de référence à la plupart des paires (contrôle post hoc `gap_checks.py`,
  `gap_checks_sweep.json`). Entre répétitions, l'écart d'ordre 3 passe de 0,007 à −10 dBFS à 0,21
  à −3 dBFS avec cette prise. L'écart entre réglages baisse aussi de 0,13 à 0,07–0,10 : une
  contribution de la saturation n'est donc pas exclue. Sans la section touchée, AUC 0,88
  (p = 0,13, 4 contre 2 paires seulement). Ajuster sur l'entrée propre ne protège pas la signature
  harmonique : un saut temporel casse la correspondance temps–fréquence du sinus exponentiel.
  D'où la **garde d'intégrité** (`real_pairs.py`).
  - Fonctionnement : alignement global de la section, puis retards par fenêtres de 0,25 s
    comparés au profil modal de la séance. Une section est rejetée dès qu'un pas dépasse 12
    échantillons à 192 kHz, soit un demi-quantum de glissement.
  - Validation : elle détecte des glissements de 24 échantillons insérés à 2,0, 3,5 et 4,6 s,
    sans signaler de décalage global.
  - Zones aveugles : avant 200 Hz, où un glissement agit surtout comme un décalage global pour
    la bande 100 Hz–8 kHz (seule la tranche 100–200 Hz est exposée), et dans les ~0,15
    dernières secondes, au-delà de ~14 kHz, hors bande. Le saut de la prise 38 à −20 dBFS
    tombe dans la première zone.
- **Un contour ±2,5 reste dans l'étendue des contrôles négatifs** pour toutes les métriques.

**Métriques existantes sur le réel** (`real_extra_vs100r.csv`, post hoc, entrée commune
seulement) : séparation des changements réels (gain 0 exclu) et des contrôles négatifs, en AUC.

| Métrique | `di_test` (5 changements / 10 contrôles) | bruit rose (7 / 12) |
|---|---|---|
| magnitude | 0,98 (p = 7e-4) | 0,98 (p = 1e-4) |
| phase | 0,94 (p = 0,002) | 1,00 (p < 1e-4) |
| enveloppe | 0,94 (p = 0,002) | 0,90 (p = 0,001) |

Sur `di_test`, une seule répétition est à entrée commune : le plancher n'y repose que sur elle.

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
- **Métriques existantes** (`real_extra_jvm.csv`, post hoc) : magnitude, phase et enveloppe
  croissent aussi strictement le long des 8 échelles de chaque sortie (ρ = +1).

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

## Usages et coût

**Coût en fonction de la durée.** Temps (s) d'une paire en fonction de T, la durée d'estimation
et de lecture (A = B = T, 48 kHz), mesuré par `cost_scaling.py` sur une machine partagée
(charge 10 à 17). L'exposant k est celui de la loi temps ∝ T^k entre la plus courte et la plus
longue durée.

| Métrique | T = 2 s | T = 8 s | T = 32 s | k | ordre de grandeur pour 1 min d'audio |
|---|---|---|---|---|---|
| ESR | 0,001 | 0,002 | 0,014 | 1,2 | < 0,1 s |
| MR-STFT | 0,05 | 0,17 | 1,19 | 1,1 | ~2 s |
| FIR (256 taps) | 0,06 | 0,24 | 0,95 | 1,0 | ~2 s |
| Hammerstein 3 | 0,20 | 0,57 | 2,6 | 0,9 | ~5 s |
| Hammerstein 5 | 0,34 | 1,0 | 5,4 | 1,0 | ~10 s |
| Volterra–Laguerre 2 | 0,56 | 2,1 | 11,5 | 1,1 | ~20 s |
| Volterra–Laguerre 3 | 6,1 | 33 | 62 | 0,8 | ~2 min |
| Wiener–Hammerstein | 33 (1 s) | 69 (2 s) | 152 (4 s) | 1,1 | ~40 min |
| Signature harmonique | ~0,8 par sinus glissant, quelle que soit la durée du programme | | | | |

Les métriques existantes magnitude, phase et enveloppe sont bon marché (FFT ou boucle simple).
Le coût réel d'un usage dépend aussi de ce qu'il exige en données, qui limite plus que le CPU.

| Usage | Ensemble proposé | Coût pour 1 min d'audio | Données nécessaires |
|---|---|---|---|
| Contrôle rapide, tri de nombreuses sorties pour une même entrée | ESR (amplitude) + magnitude + phase ; FIR si l'on veut écarter le bruit | < 5 s | entrée commune ; seuils fixés sur des répétitions et des défauts acceptables |
| Diagnostic de la nature d'un écart | Hammerstein 3 + Volterra–Laguerre 3, avec part captée et part non représentée ; magnitude/phase ; signature harmonique si un sinus glissant existe ; E2 si des sauts de niveau existent | ~2 min | données d'estimation riches (DI, bruit) ; lecture indépendante ; capacité vérifiée sur la référence ; excitations dédiées (sinus glissant, sauts) |
| Comparaison de captures réelles (prises, séances) | FIR/H3 ajustés sur l'entrée propre de chaque prise ; ESR seulement sur les sections à entrée commune ; garde d'intégrité des sinus glissants | ~5–10 s | bouclage de chaque prise ; **plancher apparié dans le temps** (répétitions intercalées : à capturer) |
| Signature non linéaire contrôlée | signature harmonique par ordre (≤ 7 pour L ≈ 0,7 s) ; IMD | < 1 s par excitation | sinus glissant (idéalement synchronisé, T ≥ 5,9 s pour l'ordre 9) ; deux tons |
| Écart structuré sous le bruit de mesure | FIR/H3 (moyennent sur l'estimation) | ~5 s | capacité de la sonde vérifiée sur la référence (sinon refus) |
| À éviter | WH (peu fiable, ~40 min/min) ; MR-STFT de `spectral.py` (aveugle à la phase, dominée par le bruit) ; rapport de détection seul | — | — |

**Limite décisive, mesurée sur la JVM410H.** Sur un canal à fort gain, aucune classe de sonde
essayée ne ramène le résidu sous ~0,6 du RMS de la sortie en lecture (ESR ≥ 0,35), ni même sous
~0,59 en échantillon pour les sondes linéaires en paramètres. Voir « Angles morts réels ». Dans ce
cas, les sondes ne servent qu'à signaler leur propre insuffisance (avertissement de capacité),
et l'ESR, la magnitude, la phase et l'enveloppe restent les seules lectures globales. Elles sont
toutes monotones le long des échelles JVM.

## Angles morts réels : ce qui les explique

Contrôles post hoc (`gap_checks.py`, `gap_checks_*.json`) :
- **VS100R, −3 dBFS : expliqué.** Un glissement d'acquisition dans le sinus glissant de la prise
  20 en est la cause (voir plus haut). Remède : la garde d'intégrité des sinus glissants.
- **JVM410H : limite de capacité, cause restante non établie** (`gap_checks_jvm*.json`).
  Les résidus sont des rapports RMS (√ESR), pas des parts d'énergie.
  - Hammerstein 5, sur la sortie de *référence* : résidu de 0,71 au gain 5 (ESR ≈ 0,50), contre
    0,06 au gain 0. En échantillon : 0,74 sur 150 s, 0,65 sur 30 s.
  - Aucune autre sonde ne fait mieux qu'environ 0,59 en lecture :
    - 1 024 taps : 0,72 ;
    - base saturante tanh : 0,67 au gain 5, 0,59 au gain 10 ;
    - Volterra–Laguerre 3 : comparable ;
    - Wiener–Hammerstein parallèle à préfiltres de Laguerre fixes : 0,59–0,66 en échantillon,
      0,74–0,96 en lecture ;
    - Wiener–Hammerstein ajusté sur 1 s : 0,39 en échantillon mais 1,08 en lecture
      (surapprentissage).
  - Écartés, par les mesures de `gap_checks_jvm_causes.json` :
    - **bruit** : la sortie pendant les silences d'entrée est à −28 dB du RMS au gain 5 (−26 au
      gain 0, où le résidu de 0,06 ≈ −24 dB s'y explique) ;
    - **dynamique à 0,5 s** : à niveau d'entrée égal, le résidu vaut 0,67 après un passage plus fort
      et 0,66 sans ;
    - **dérive d'horloge** : le retard optimal reste à 0 ± 1 échantillon sur 300 s ;
    - **séparation A/B** : le résultat est le même en échantillon.
  - Non testables avec ces données : le déterminisme de l'enregistrement (aucune répétition) et
    une cascade d'étages non linéaires.

**Métriques existantes face aux angles morts du banc** (`baselines_extra.py`, post hoc) :
- **Magnitude et phase sont complémentaires.** La magnitude (`metrics/spectral.py`) est aveugle
  au passe-tout et au retard (R = 1,6 et 5,5). La phase est exactement nulle pour un gain et
  aveugle au cubique (1,5). Ensemble, elles remplacent la MR-STFT et en expliquent l'angle mort.
- **L'enveloppe (`metrics/transient.py`) n'est pas spécifique** : sensible à tout (gain ×1,4e4,
  relâchement ×9,5e3), elle n'isole pas la dynamique. Aucune métrique existante ne le fait.

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
