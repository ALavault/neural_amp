# Pilote D — fourche 25, la phase précoce sans la garde

**Pré-enregistré le 2026-09-18, avant tout run de la fourche 25.** Demandé par l'utilisateur après le
constat de confusion du bras « rejoue » de la fourche 5 (`pilot_A_fork5_replay.md`, section
« Confusion découverte en cours d'exécution »).

## Question

Le pilote A a montré qu'à la fourche **100**, les décisions pilotées par la validation amplifient une
perturbation d'un pas float32 : s(décide) = 0,156 contre s(rejoue) = 0,009. À la fourche **5**, le
bras « rejoue » ne répond pas à la question, parce que la garde de polarité y reste libre : le parent
bascule aux époques 7, 8 et 9, donc une fourche placée à l'époque 5 laisse chaque enfant redécouvrir
ses propres bascules, et celles-ci sont elles-mêmes des décisions de validation.

La fourche 25 est placée **après la dernière bascule du parent (époque 9)** et **avant sa première
division de pas d'apprentissage (époque 113)**. Les enfants y héritent donc d'un historique de garde
complet et commun, comme à la fourche 100, tout en restant dans la phase précoce.

**Question posée :** l'amplification par les décisions vaut-elle aussi tôt dans l'entraînement, ou la
phase précoce diverge-t-elle d'elle-même, indépendamment de toute décision ?

## Dispositif

Identique au pilote A, seule l'époque de fourche change.

- **Parent** : rejoué avec `--fork-epochs 5 25 100` et l'étiquette `_f25`, pour produire
  `fork_epoch25.ckpt`. L'entraînement étant reproductible au bit près, les points de sauvegarde aux
  époques 5 et 100 doivent être identiques à ceux déjà en place.
- **Enfants** : `k = 0` sans perturbation, `k = 1..4` avec chaque poids déplacé d'un pas float32,
  direction tirée par la graine 1000 + k. Poids, moments d'AdamW, ordonnanceur, arrêt anticipé et
  état de la garde chargés depuis le point de fourche.
- **Bras « décide »** : l'enfant prend ses propres décisions.
- **Bras « rejoue »** : pas d'apprentissage de chaque époque et pas d'arrêt imposés par `décide k0`,
  arrêt anticipé désactivé, garde laissée active.
- 11 runs : 1 parent + 5 + 5. Environ 2 h 30 par enfant.

## Portes, à franchir avant toute lecture

1. **Parent** : `fork_epoch5.ckpt` et `fork_epoch100.ckpt` du parent rejoué identiques au bit près à
   ceux du parent d'origine. Sinon la trajectoire n'est pas la même et la fourche 25 ne se rattache à
   rien : la file s'arrête.
2. **Identité** : `rejoue k0` identique au bit près à `décide k0`, comme aux fourches 5 et 100.
3. **Garde** : l'historique de bascules de chaque enfant doit rester exactement `[7, 8, 9]`, celui du
   parent. Toute liste différente signale une bascule postérieure à la fourche, donc une décision de
   validation restée libre : cet enfant est rapporté à part et la lecture du bras est qualifiée en
   conséquence. C'est le défaut qui a rendu muet le bras « rejoue » de la fourche 5, et il n'est pas
   exclu ici — à la fourche 5, l'enfant témoin a basculé jusqu'à l'époque absolue 72.

## Prédictions, seuils repris du pilote A

- **Soutenu — les décisions amplifient aussi tôt** : s(décide, 25) ≥ 0,10 ; s(rejoue, 25) ≤
  s(décide, 25)/2 ; rapport des moyennes des |Δ| ≥ 2. Dans ce cas la divergence observée à la
  fourche 5 s'explique par la garde restée libre, et le mécanisme du pilote A vaut de l'époque 25 à la
  fin.
- **Réfuté — la phase précoce diverge d'elle-même** : s(rejoue, 25) ≥ 0,10 et rapport < 2. Le
  mécanisme du pilote A serait alors propre à la phase tardive, et la dispersion précoce relèverait
  d'une sensibilité chaotique que le calendrier imposé ne contient pas.
- **Indécidable** : s(décide, 25) < 0,10 et s(rejoue, 25) < 0,10 — la perturbation ne produit rien à
  cette fourche, et le pilote ne départage pas.

## Ce que ce pilote ne pourra pas dire

Cinq runs par bras, une graine, un appareil, une architecture ; les seuils sont de qualité pilote et
un résultat pilote décide de l'expérience complète, pas de la question générale. Il ne dira rien non
plus de l'effet de la **direction** de la perturbation, question ouverte depuis que `k = 1` décroche
aux deux fourches déjà mesurées (`Q-0010` de la mémoire partagée).

## Verdict (2026-09-19, 21 h 22)

`scripts/product_fork_pilot_analysis.py`, bloc `fork25` de `pilot_A_results.json`.

**Les trois portes sont franchies.** Le parent rejoué reproduit ses points de sauvegarde des
époques 5 et 100 au bit près ; `rejoue k0` reproduit `décide k0` au bit près
(`"identical_to_decide_k0": true`) ; et **les dix enfants portent exactement l'historique de
bascules du parent, `[7, 8, 9]`, aucun n'en dévie**. La garde est donc bien éteinte à l'époque 25,
ce qui était toute la raison d'être de cette fourche : le défaut qui a rendu muet le bras « rejoue »
de la fourche 5 est absent ici.

| bras | ESR (k = 0 à 4) | Δ en log | s | pas |
|---|---|---|---|---|
| décide | 0,0386 0,0332 0,0375 0,0432 0,0424 | −0,152 −0,030 +0,112 +0,094 | **0,106** | 5 775 à 8 141, propres |
| rejoue | 0,0386 0,0345 0,0338 0,0371 0,0347 | −0,111 −0,133 −0,041 −0,108 | **0,056** | 6 601 pour tous |

**Aucune des trois issues pré-enregistrées n'est réalisée.** Le pré-enregistrement n'était pas
exhaustif, et la mesure est tombée dans l'intervalle qu'il avait laissé vide :

- *soutenu* exigeait les trois conditions : s(décide) ≥ 0,10 est vérifié (0,106) ; s(rejoue) ≤
  s(décide)/2 = 0,0532 **échoue de 4,8 %** (0,0558) ; le rapport des moyennes des |Δ| ≥ 2 échoue
  largement (0,99) ;
- *réfuté* exigeait s(rejoue) ≥ 0,10, or il vaut 0,056 ;
- *indécidable* exigeait les deux écarts-types sous 0,10, or s(décide) vaut 0,106.

Les seuils n'ont pas été modifiés ; ils sont ceux du pilote A, commités avant tout run.

## Ce que la mesure dit malgré tout

**Le calendrier imposé divise la dispersion par deux**, de 0,106 à 0,056, soit un rapport de 0,52 —
à un cheveu du facteur 2 demandé. La direction est celle du pilote A, l'ampleur ne l'est pas : à la
fourche 100, le même bras faisait tomber la dispersion d'un facteur **17** (0,156 → 0,009).

**Mais il ne rapproche pas les enfants du témoin.** Le rapport des moyennes des |Δ| vaut 0,99 :
sous calendrier imposé, un enfant s'écarte du témoin d'autant qu'en décidant lui-même. Ce que le
calendrier change est la *forme* de l'écart, pas sa taille : les quatre enfants « décide »
s'éparpillent de part et d'autre du témoin (−0,152 à +0,112) tandis que les quatre « rejoue »
se regroupent tous du même côté (−0,041 à −0,133).

**Fait non prévu, à ne pas surinterpréter** : sous calendrier imposé, les quatre enfants perturbés
finissent tous *meilleurs* que le témoin, qui partage pourtant leur calendrier et leur pas d'arrêt.
Quatre signes identiques sur quatre ont une chance sur seize d'être fortuits sous l'hypothèse nulle
d'un signe aléatoire. Avec n = 4, ce n'est pas un résultat ; c'est une observation à vérifier si le
dispositif est repris.

## Lecture d'ensemble des trois fourches

| fourche | s(décide) | s(rejoue) | rapport | garde |
|---|---|---|---|---|
| 5 | 0,207 | 0,365 | 1,76 | libre, historiques différents |
| 25 | 0,106 | 0,056 | 0,52 | éteinte, `[7, 8, 9]` partout |
| 100 | 0,156 | 0,009 | 0,06 | éteinte, `[7, 8, 9]` partout |

Le pouvoir de contention du calendrier imposé croît avec la profondeur de la fourche : il aggrave à
l'époque 5, divise par deux à 25, divise par dix-sept à 100. L'image en deux régimes proposée après
la fourche 5 devient un dégradé, et la fourche 25 se situe dans la zone de transition — ce qui
explique aussi qu'elle tombe entre les mailles d'un pré-enregistrement écrit pour un résultat
tranché.

## Ce que cela ne permet pas de conclure

- **Le pilote D ne tranche pas sa propre question.** Les critères écrits d'avance ne sont pas
  satisfaits, et le résultat ne peut pas être présenté comme soutenant l'hypothèse. Le déplacer
  dans la case « soutenu » demanderait de réécrire un seuil après avoir vu les chiffres.
- Cinq runs par bras, une graine, une architecture, un appareil, des seuils de qualité pilote. La
  différence entre 0,0558 et 0,0532 n'a aucune signification statistique à n = 5 ; c'est la règle
  pré-enregistrée qui lui en donne une, et c'est le prix à payer pour qu'une règle écrite d'avance
  serve à quelque chose.
- Le dégradé entre fourches repose sur trois points, dont un confondu par la garde.

## Prochaine expérience discriminante

Le pré-enregistrement à réécrire, pour une reprise, doit couvrir l'intervalle laissé vide : prévoir
une issue « amplification partielle » lorsque s(rejoue) tombe entre s(décide)/2 et s(décide), et
séparer les deux quantités que ce pilote montre disjointes — la **dispersion** entre enfants, que le
calendrier contient, et la **distance moyenne** au témoin, qu'il ne contient pas. Mesurer les deux
à trois profondeurs de fourche, par exemple 25, 50 et 100, donnerait la courbe de contention plutôt
que trois points isolés.
