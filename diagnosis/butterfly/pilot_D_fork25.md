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
