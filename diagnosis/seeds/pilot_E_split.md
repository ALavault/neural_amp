# Pilote E — la chaîne de données ou l'initialisation ?

**Pré-enregistré le 2026-09-20, avant tout run.** Piste C de la discussion sur l'origine de la
variance. Demandé par l'utilisateur : « savoir d'où ça vient pour pouvoir le corriger ou au moins
le limiter ».

## Question

La graine tire trois choses à la fois : l'initialisation des poids, l'ordre des lots, et la
partition train/validation de 118/12 segments. Or le RMS minimal de la partition de validation
ordonne l'ESR de test 6 fois sur 6 (`DIAGNOSIS_seeds.md`, corrélation choisie après coup, sans
mécanisme). Si ce que l'on appelle « effet de graine » était d'abord « quels douze segments sont
retirés de l'entraînement », le correctif serait immédiat et sans coût : fixer la partition.

**Question posée :** en rendant la chaîne de données commune à tous les runs, la dispersion entre
graines tombe-t-elle au niveau du seul non-déterminisme ?

## Dispositif

SSM-WaveNet avec les deux changements, comme le banc. Six graines — 42 à 47 — un run chacune, avec
`--split-seed 42` : le générateur est re-tiré juste avant l'entraînement, donc **la partition et
l'ordre des lots sont identiques d'un run à l'autre**, et seule l'initialisation des poids reste
commandée par `--seed`. Environ 2 h par run, 12 h en tout.

Référence de comparaison, déjà mesurée : les six runs du banc (3 graines × 2), effet run
**0,375** en log (`diagnosis/seeds/variance_sources.md`), dont une composante non-déterministe de
**0,18** mesurée à graine égale (`diagnosis/seeds/hypotheses_nested.md`).

## Portes

1. Le défaut de `--split-seed` est `None` et ne change rien : vérifié sur `--help` et par lecture du
   diff, l'option n'agit que si elle est passée.
2. Les six runs doivent partager la même partition. Vérification a posteriori : la perte de
   validation du premier point de contrôle de chaque run porte sur les mêmes douze segments, donc
   les valeurs doivent être comparables entre runs, et surtout la partition recalculée à partir de
   `--split-seed 42` doit reproduire celle du run, au sens de `diagnosis/butterfly/validation_noise.py`.

## Prédictions, et elles partitionnent l'espace

Soit s le écart-type de l'effet run sur les six runs à partition commune.

- **La chaîne de données porte l'essentiel** : s ≤ 0,25. La dispersion tombe de plus de la moitié
  du chemin entre 0,375 et le plancher non-déterministe de 0,18.
- **Elle ne porte pas l'essentiel** : s ≥ 0,32. Fixer la partition ne change presque rien, et la
  variance vient de l'initialisation ou de la dynamique.
- **Intermédiaire** : 0,25 < s < 0,32. Contribution réelle mais partielle ; le pilote ne tranche pas
  et la valeur mesurée est rapportée telle quelle.

## Ce que ce pilote ne pourra pas dire

- Il ne sépare pas la **partition** de l'**ordre des lots** : les deux sont fixés ensemble, parce que
  les deux se tirent du même générateur au même moment. Un résultat positif désignerait « la chaîne
  de données », pas « la partition ». La séparation demanderait de restaurer l'état du générateur
  après le tirage de la partition, donc de toucher au module de données de nablafx.
- Six runs, une architecture, un appareil. Comparer s à une référence mesurée sur six autres runs
  ajoute l'incertitude des deux estimations, qu'un pilote à n = 6 ne maîtrise pas.
- Il ne dit rien de l'effet d'une partition **meilleure** : fixer n'est pas choisir.

## Verdict (2026-09-20, 17 h 46)

| graine | 42 | 43 | 44 | 45 | 46 | 47 |
|---|---|---|---|---|---|---|
| ESR | 0,0811 | 0,0992 | 0,0675 | 0,0653 | 0,0916 | 0,1017 |

**s = 0,191**, pour un seuil de 0,25. **L'issue « la chaîne de données porte l'essentiel » est
réalisée.**

Portes franchies. Les six enregistrements portent `split_seed: 42`. Surtout, la perte de validation
recalculée sur la partition dérivée de cette graine reproduit celle du journal : 0,03489 contre
0,03488 pour la graine 47, 0,03300 contre 0,03353 pour la 42. Une partition différente donnerait un
écart de l'ordre de 30 %, puisque l'écart-type de l'effet segment vaut 1,0 en log.

## Ce que cela permet de conclure

- **La dispersion tombe de 0,375 à 0,191 en fixant la chaîne de données**, c'est-à-dire presque
  exactement au plancher du non-déterminisme GPU seul, **0,18**. Une fois la partition et l'ordre
  des lots communs, l'initialisation des poids ne contribue plus rien de mesurable.
- Autrement dit, ce que la littérature et nos propres notes appellent « effet de graine » est ici,
  pour l'essentiel, **un effet du tirage des données**. Ce n'est pas une propriété de
  l'optimisation, c'est une propriété du protocole d'évaluation.
- **Le correctif est gratuit** : fixer la partition ne coûte pas un pas de calcul. Il ramène la
  comparaison entre architectures à un bruit de 0,18 au lieu de 0,375, soit une variance quatre
  fois moindre.

## Ce que cela ne permet pas de conclure, et un avertissement

- **Fixer n'est pas choisir.** La moyenne géométrique des six runs vaut 0,0831 contre 0,068 pour les
  six runs de référence : cette partition-ci est plus difficile que la moyenne de celles tirées par
  les graines. Figer une partition fige aussi sa difficulté, et un protocole qui en fixerait une
  mauvaise publierait des chiffres pessimistes et incomparables à ceux d'autrui.
- Le pilote ne sépare pas la **partition** de l'**ordre des lots**, fixés ensemble. Il désigne la
  chaîne de données, pas son maillon.
- Six runs contre six, un appareil, une architecture. Comparer deux écarts-types estimés sur six
  points chacun n'a pas la précision que la netteté du résultat suggère : c'est un pilote.
- s = 0,191 reste au-dessus de 0,18 ; rien n'exclut une contribution résiduelle de
  l'initialisation, simplement elle n'est pas séparable du non-déterminisme à ce nombre de runs.

## Prochaine expérience discriminante

Séparer la partition de l'ordre des lots demanderait de restaurer l'état du générateur après le
tirage de la partition, donc de toucher au module de données de nablafx. Plus utile d'abord :
mesurer la **dispersion entre partitions** elle-même — six partitions différentes, graine
d'initialisation commune — qui dirait de combien le choix du jeu de validation déplace le résultat,
et donc ce qu'un protocole gagnerait à en fixer une plutôt qu'à en tirer une.
