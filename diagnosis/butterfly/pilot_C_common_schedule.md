# Pilote C — l'effet de la graine passe-t-il par les décisions ?

Écrit le 2026-09-17, avant tout run C, pendant le pilote A. **Conditionnel** : lancé
seulement si B est soutenue (`pilot_A_fork5_replay.md`) ; sinon, discussion d'abord.
Script et file restent à écrire ; ils suivront ce document sans le modifier.

## Question

Sur Big Muff, SSM-WaveNet ZOH avec garde, l'écart-type entre graines du log de l'ESR
de test vaut 0,37 dans l'analyse emboîtée (3 graines × 2 runs,
`diagnosis/seeds/hypotheses_nested.md`) et 0,53 sur les premiers runs des 3 graines
(effet run, `global_factor.txt` ; corrigé le 2026-09-17, le texte initial disait « sur
les 6 runs »). **C** : cet écart se réduit-il de moitié au moins
quand les décisions prises sur la validation (divisions du learning rate, arrêt) sont
remplacées par un calendrier commun fixé d'avance ?

## Dispositif

- **Graines** : 42 à 49, une run par graine et par bras, 16 runs, mode déterministe,
  lot de 16. Avec 3 graines, l'écart-type n'aurait que 2 degrés de liberté et aucun
  rapport ne serait lisible ; 8 graines en donnent 7.
- **« Décide »** : protocole NablAFx (ReduceLROnPlateau, arrêt anticipé), comme les runs
  du banc.
- **« Fixe »** : learning rate divisé par 2 au début des époques 203, 265, 422, 486,
  565, 608, 661 et 722, arrêt au pas 5 950 (époque 850), sans arrêt anticipé. Ces
  valeurs sont les médianes, rang par rang, des 8 premières divisions et du pas
  d'arrêt des 7 runs SSM-WaveNet existants (6 runs du banc et la chaîne déterministe
  parent + témoin de la fourche 100). Elles ne reprennent le calendrier d'aucune
  graine.
- **Garde active dans les deux bras** : c'est la seule décision sur la validation qui
  reste en « fixe ».
- **Mesures** : ESR de test du dernier checkpoint ; sd = écart-type du log de l'ESR sur
  les 8 graines, par bras ; moyenne du log de l'ESR par bras ; bascules de la garde.

## Prédictions

- **Condition préalable** : sd(décide) ≥ 0,20. Sinon l'écart entre graines n'est pas
  là en mode déterministe et C ne peut pas être testée.
- **C soutenue** : sd(fixe) ≤ sd(décide) / 2.
- **C réfutée** : sd(fixe) ≥ 0,75 × sd(décide).
- **Autres cas** : indécidable.
- **Descriptif** : test F unilatéral du rapport des variances (7 et 7 degrés de
  liberté) ; écart des moyennes entre bras (un calendrier fixe peut changer la qualité
  moyenne, ce n'est pas la question) ; sd(fixe) comparé à s(rejoue) de la fourche 5.
  Si sd(fixe) dépasse nettement s(rejoue), l'effet de la graine qui reste sous calendrier
  fixe naît avant le pas 42 ou dans le découpage des données.
- **Confusion connue** : `random_split` tire le découpage train/validation avec la
  graine, après l'initialisation. En « fixe », la validation ne pilote plus rien, mais
  l'ensemble d'entraînement change de jusqu'à 12 segments sur 130 d'une graine à
  l'autre ; sd(fixe) en contient l'effet, que ce pilote ne sépare pas.
- **Portée** : un modèle, un appareil, 8 graines ; niveau pilote.

## Levée de la condition (2026-09-21)

Ce pilote était conditionné à B, qui **n'a pas tenu** : à la fourche 5, imposer le calendrier du
témoin n'a pas réduit la dispersion, il l'a augmentée — s(rejoue) = 0,365 contre s(décide) = 0,207
(`pilot_A_fork5_replay.md`). La clause prévoyait « sinon, discussion d'abord ». L'utilisateur a
demandé le lancement le 2026-09-21 ; cette section tient lieu de la discussion et dit pourquoi le
pilote garde son sens malgré l'échec de sa condition.

**Ce qui a changé depuis l'écriture.** Le pilote E a montré que la chaîne de données — partition
train/validation et ordre des lots — porte l'essentiel de l'effet de graine : la fixer fait tomber
la dispersion de 0,375 à 0,191, soit le plancher du non-déterminisme seul
(`diagnosis/seeds/pilot_E_split.md`). Et la simulation sur les courbes existantes a montré qu'aucun
seuil fixe ne rend le déclenchement du plateau reproductible
(`diagnosis/seeds/plateau_simulation.md`).

**Ce que C apporte encore, et que E ne donne pas.** E dit que la chaîne de données porte la
variance, pas **par quel canal**. Deux mécanismes restent possibles : le découpage change les
données vues, donc le modèle, directement ; ou bien il change la courbe de validation, donc les
dates de division et d'arrêt, donc le modèle. C sépare exactement ces deux canaux, puisque le bras
« fixe » coupe le second en laissant le premier intact. C'est aussi le seul levier restant du côté
de l'ordonnanceur, la simulation ayant tué le réglage du seuil.

**Prédiction ajoutée, conforme à ce que E laisse attendre.** Si le canal est direct, sd(fixe) sera
proche de sd(décide) et C sera réfutée. Les seuils écrits d'avance sont inchangés : soutenue si
sd(fixe) ≤ sd(décide)/2, réfutée si sd(fixe) ≥ 0,75 × sd(décide), et le domaine intermédiaire est
rapporté tel quel — leçon du pilote D, dont les issues ne partitionnaient pas l'espace.

**Dispositif inchangé**, sauf que les deux options nécessaires n'existaient pas et ont été ajoutées
au script d'entraînement : `--lr-halvings` installe le calendrier fixe et retire le détecteur de
plateau, `--no-early-stopping` laisse le run aller jusqu'à `--max-steps`. Leurs défauts reproduisent
le comportement actuel.

## Verdict (2026-09-22, les seize runs faites)

`ls demo/nablafx_bench/pilotC_*.json | wc -l` → **16**. Calculé par
`diagnosis/butterfly/pilot_C_verdict.py`, écrit le 2026-09-17 et non modifié depuis ; les
seuils ci-dessus ne sont pas touchés, cette section les applique.

| graine | décide | fixe | écart en log |
|---|---|---|---|
| 42 | 0,0358 | 0,0381 | +0,064 |
| 43 | 0,0744 | 0,0638 | −0,154 |
| 44 | 0,0979 | 0,0906 | −0,077 |
| 45 | 0,0832 | 0,0933 | +0,114 |
| 46 | 0,0540 | 0,0697 | +0,255 |
| 47 | 0,0628 | 0,0492 | −0,244 |
| 48 | 0,1580 | 0,0750 | **−0,745** |
| 49 | 0,1781 | 0,0679 | **−0,964** |

**sd(décide) = 0,535 ; sd(fixe) = 0,300 ; rapport = 0,56.** Condition préalable
sd(décide) ≥ 0,20 : remplie, largement.

**Porte du dispositif : franchie, 8 fois sur 8.** Les huit runs « fixe » ont divisé le pas
exactement aux époques 203, 265, 422, 486, 565, 608, 661 et 722, et nulle part ailleurs —
`pilotC_fixe_seed42`, `pilotC_fixe_seed48` et `pilotC_fixe_seed49` comme les cinq autres. Le
détecteur de plateau était bien éteint.

### L'issue : intermédiaire, rapportée telle quelle

0,56 tombe entre les deux seuils : **ni soutenue (≤ 0,50), ni réfutée (≥ 0,75)**. C'est le
troisième cas prévu, écrit d'avance à cause du pilote D dont les issues ne partitionnaient pas
l'espace ; il sert ici exactement à cela.

**Mes deux prédictions étaient fausses, et elles étaient opposées.** Celle du 2026-09-21, ci-dessus,
attendait une réfutation au motif que le pilote E désignait un canal direct. Celle du 2026-09-22,
inscrite dans `diagnosis/seeds/quiet_share_exploration.md` et commitée avant la fin de la seizième
run (8c27a6c), attendait un soutien au motif que la date de division serait le levier. La seconde
était la plus proche ; aucune des deux n'a atteint son seuil.

**Et le libellé tient à une graine.** En retirant une graine à la fois, le rapport va de 0,48 à
0,69 : deux retraits sur huit (les graines 42 et 45) le feraient basculer en « soutenue ». Il
n'atteint **jamais** le seuil de réfutation. La direction est donc robuste, l'étiquette non.

### Les trois descriptifs demandés

- **Test F unilatéral**, sens pré-enregistré : F(7,7) = var(décide)/var(fixe) = 3,19, p = 0,074.
  Non significatif à 5 %, cohérent avec l'intermédiaire.
- **Écart des moyennes** : le bras fixe est *meilleur* de 0,219 en log, soit un ESR moyen à 0,80×.
  Ce n'était pas la question, mais c'est notable : contrairement au pilote F, où un seuil fixe
  coûtait un facteur 2,4 en qualité, un *calendrier* fixe en gagne 20 %.
- **sd(fixe) = 0,300 contre s(rejoue) = 0,365** à la fourche 5. La lecture pré-enregistrée — « si
  sd(fixe) dépasse nettement s(rejoue), l'effet naît avant le pas 42 ou dans le découpage » — ne se
  déclenche pas. Réserve : les deux quantités ne portent pas sur la même population, cinq enfants
  d'une graine contre huit graines.

### Ce que le chiffre agrégé cachait (post hoc, non pré-enregistré)

Le rapport de 0,56 est un mélange, et le tableau par graine le montre : **le calendrier fixe ne
réduit rien pour six graines et sauve les deux autres.**

| sous-ensemble | sd(décide) | sd(fixe) | rapport |
|---|---|---|---|
| les huit | 0,535 | 0,300 | 0,56 |
| sans 48 et 49 | 0,357 | 0,348 | **0,98** |

**Et aucun prédicteur n'identifie la paire sauvée — j'ai d'abord écrit le contraire.** La première
rédaction affirmait que les deux graines sauvées étaient celles dont la division libre tombait le
plus tôt, « 49 à l'époque 141, 48 à 167, contre 234 à 285 pour quatre des six autres ». La phrase
était vraie et malhonnête : elle choisissait sa classe de comparaison pour taire les deux
contre-exemples. Le classement complet est 49 (141), **44 (161), 47 (164)**, 48 (167), puis 234 à
285. Les graines 44 et 47 divisent donc *plus tôt* que 48 et ne sont pas sauvées — leurs gains
valent −0,077 et −0,244. Le classement par part calme échoue de la même façon : 44 y est deuxième.
Relevé par une relecture adverse de Codex, qui a recalculé le classement.

Conséquence, et elle compte : **une division précoce ne suffit pas à produire un mauvais modèle**.
Les graines 44 et 47 divisent tôt et donnent des ESR libres de 0,098 et 0,063, dans le gros de la
distribution. Ce qui rend 48 et 49 mauvaises sous calendrier libre n'est donc pas capté par la date
de division seule, malgré le ρ = −0,81 d'ensemble.

Ce qui subsiste, purement descriptif : coupées à l'époque 200, les quatre graines à division
précoce (44, 47, 48, 49) donnent un rapport de 0,54 et les quatre tardives (42, 43, 45, 46) un
rapport de 0,98. Mais la coupure a été choisie après coup et le 0,54 est porté par 48 et 49 à
l'intérieur de son propre groupe : c'est la même observation réexprimée, pas une confirmation.

**Le test du canal, sans régression vers la moyenne.** Corréler le gain `fixe − décide` avec quoi
que ce soit serait confondu : une graine extrême dans un bras revient vers la moyenne dans l'autre
sans qu'aucun mécanisme n'opère. Chaque bras est donc corrélé séparément à une propriété de la
partition, qui ne dépend d'aucun run :

| prédicteur | → ESR décide | → ESR fixe |
|---|---|---|
| part de trames sous −40 dB | ρ = +0,810 (p 0,015) | ρ = +0,262 (p 0,531) |
| date de la 1ʳᵉ division libre | ρ = −0,810 (p 0,015) | ρ = −0,310 (p 0,456) |
| RMS minimal de validation | ρ = −0,287 (p 0,490) | ρ = +0,060 (p 0,888) |

La corrélation **s'effondre quand l'ordonnanceur est retiré**. Lue avec la précaution qui s'impose
à n = 8 — l'écart entre deux corrélations sur huit points appariés est lui-même très incertain —
elle dit que la partition agit **par** le calendrier plutôt que directement sur la matière
d'entraînement. C'est la question que ce pilote posait, et c'est la première réponse qu'on en tire.

### Ce qui ne peut pas être conclu

- **Le canal n'est pas pur.** sd(fixe) = 0,300 reste bien au-dessus du plancher de non-déterminisme
  de 0,18 : un effet de graine survit au calendrier fixe. La confusion annoncée d'avance tient —
  en « fixe », l'ensemble d'entraînement change encore de jusqu'à 12 segments sur 130 d'une graine
  à l'autre, et sd(fixe) en contient l'effet sans le séparer.
- **La lecture par la queue est post hoc.** Appeler 48 et 49 des valeurs extrêmes est une partition
  décidée après avoir vu les chiffres ; seul le rapport était pré-enregistré. Elle demande d'autres
  graines pour valoir, et huit ne suffisent pas à dire si « le calendrier fixe supprime les
  catastrophes sans resserrer le gros de la distribution » est général.
- **Ce n'est pas une prescription.** Un calendrier fixe tiré des médianes de sept runs existants
  n'est pas un calendrier *choisi* ; le pilote E avait déjà inscrit que fixer n'est pas choisir.
- **Portée** : une architecture, un appareil, un enregistrement, huit graines.

### La prochaine expérience discriminante

Le point faible est la partition post hoc en « queue » et « gros de la distribution », et la
correction ci-dessus interdit de la fonder sur l'identification des graines. Le départager ne
demande pas de nouveau dispositif, seulement **huit graines de plus dans les deux bras**, avec une
règle d'affectation décidée d'avance et hors échantillon : chaque nouvelle graine est rangée dans
le groupe « précoce » ou « tardif » selon la date de division de son propre bras libre, lue avant
que son bras fixe n'existe.

**Prédiction, écrite avant de lancer** : sur les huit nouvelles graines, rapport ≤ 0,6 dans le
groupe précoce et ≥ 0,9 dans le groupe tardif. Si les deux groupes donnent le même rapport, la
lecture par la queue tombe et l'intermédiaire de 0,56 n'était qu'un effet moyen. Si les deux sont
bas, c'est le calendrier lui-même qui resserre, et non la coupure d'une queue.

Réserve à inscrire d'avance : la règle d'affectation utilise une grandeur mesurée sur le bras
libre, donc les groupes ne sont pas équilibrés par construction et pourraient l'être à 6 contre 2.
En dessous de trois graines dans un groupe, aucun rapport n'en sera rapporté.
