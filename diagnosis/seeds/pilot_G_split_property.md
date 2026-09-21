# Pilote G — une propriété de la partition de validation prédit-elle l'ESR final ?

**Pré-enregistré le 2026-09-22, avant toute mesure de corrélation.** Attaque la seule question que
`DIAGNOSIS_seeds.md` section 6 déclare ouverte sans mécanisme : « H2 sans mécanisme ».

## Question

Le pilote E a établi que la chaîne de données porte l'effet de graine : fixer la partition et
l'ordre des lots fait tomber l'écart-type de l'effet run de 0,375 à 0,191, soit le plancher du
non-déterminisme (`pilot_E_split.md`). Mais **on ne sait pas pourquoi une partition est meilleure
qu'une autre**. La seule piste existante est une corrélation sans mécanisme, choisie après coup :
le RMS minimal des douze segments de validation ordonne l'ESR de test 6 fois sur 6
(`DIAGNOSIS_seeds.md`, H2, six runs).

Deux choses ont changé depuis, qui rendent la question testable proprement. La partition est
**reproductible et vérifiée** — la perte de validation recalculée sur la partition dérivée d'une
graine reproduit celle du journal à 0,15 % près (`validation_noise.md`, `pilot_E_split.md`). Et le
pilote C fournit **huit graines** sous conditions identiques, en mode déterministe, là où H2 n'en
avait que trois.

**Question posée :** une propriété mesurable de la partition de validation prédit-elle l'ESR de
test final ?

## Dispositif

- **Mesure avancée.** Les propriétés d'une partition ne dépendent que de la graine, pas du run.
  Elles sont donc calculées et commitées **avant** que les ESR des graines 47, 48 et 49 n'existent,
  ce que l'ordre des commits atteste. Le prédicteur est figé avant l'issue pour trois graines sur
  huit.
- **Partitions** : reproduites en répétant l'ordre de construction du pilote — `seed_everything(s)`,
  puis construction du processeur, puis module de données — pour les graines 42 à 49. Contrôle :
  pour la graine 42, la partition obtenue doit être celle déjà vérifiée dans
  `validation_noise.py`.
- **Issue** : ESR de test du bras « décide » du pilote C, une run par graine, mode déterministe.
- **Propriété primaire, nommée d'avance** : le **RMS minimal** des douze segments de validation,
  celle que H2 désignait. Toute autre est exploratoire et sera rapportée comme telle.
- **Propriétés exploratoires** : RMS moyen, énergie totale, facteur de crête moyen, centroïde
  spectral moyen, proportion de passages sous −40 dB, et le RMS minimal des 118 segments
  d'entraînement — ce qui est retiré de l'entraînement compte peut-être autant que ce qui sert à
  valider.

## Prédictions, et elles partitionnent l'espace

Soit ρ la corrélation de rang de Spearman entre la propriété primaire et le log de l'ESR de test,
sur les huit graines.

- **H2 soutenue** : |ρ| ≥ 0,74, seuil auquel p < 0,05 à n = 8 en bilatéral.
- **H2 réfutée** : |ρ| < 0,50. La corrélation observée sur six runs ne survit pas à huit graines et
  à une partition reproduite, et doit être retirée du diagnostic.
- **Indécidable** : 0,50 ≤ |ρ| < 0,74 — la direction est là, la puissance ne suffit pas à n = 8, et
  le chiffre est rapporté tel quel.

Sur les propriétés exploratoires, aucune conclusion ne sera tirée d'un ρ isolé : avec sept
propriétés testées, la plus forte a une chance sur deux de dépasser 0,7 par hasard seul. Elles ne
servent qu'à désigner une piste pour un dispositif ultérieur.

## Ce que ce pilote ne pourra pas dire

- Huit graines, une architecture, un appareil, un enregistrement. Une corrélation sur huit points ne
  devient un mécanisme qu'en étant confirmée sur un autre matériel.
- Il ne sépare pas la partition de l'**ordre des lots**, que la graine tire aussi : une propriété de
  la partition qui prédirait l'ESR pourrait n'être qu'un marqueur de l'ordre des lots.
- Il ne dit rien de ce qu'une **bonne** partition serait : prédire n'est pas prescrire, et le
  pilote E a déjà montré que figer une partition fige aussi sa difficulté.
- La propriété primaire est définie sur la cible, pas sur l'entrée sèche. Le RMS minimal d'un
  segment de validation dépend donc de l'appareil modélisé autant que du signal de test.
