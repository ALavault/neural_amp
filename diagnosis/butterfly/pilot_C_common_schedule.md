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
