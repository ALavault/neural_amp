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
