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
