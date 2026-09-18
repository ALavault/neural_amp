# Par quelle marge un plateau est-il déclaré ?

Lecture seule, journaux existants. Script : `diagnosis/butterfly/plateau_margin.py`. Sortie :
`plateau_margin.json`. Suite de `validation_noise.md`, qui a écarté l'échantillonnage des douze
segments : ils sont fixes à l'intérieur d'un run, donc leur tirage s'annule dans la détection de
plateau. Restait la fluctuation d'une époque à la suivante.

## Hypothèse testée

`ReduceLROnPlateau` divise le pas d'apprentissage après 20 époques sans amélioration de plus de
10⁻⁴ en relatif sur le meilleur atteint. Deux nombres décident donc de tout : de combien le run a
frôlé un nouveau record pendant les vingt époques qui ont déclenché la division, et quelle est
l'amplitude ordinaire de ses fluctuations. Si la première est inférieure à la seconde, l'époque de
division est fixée par le bruit d'optimisation, et deux runs séparés d'un pas float32 n'ont aucune
raison de diviser au même moment.

## Observations, cinq enfants « décide » de la fourche 100

**Le seuil de 10⁻⁴ ne joue aucun rôle.** La fluctuation relative médiane d'une époque à la suivante
vaut 4,4·10⁻³ à 8,8·10⁻³ selon l'enfant, soit **44 à 88 fois le seuil**. Le paramètre
`threshold=1e-4` du protocole est donc inopérant : tout est décidé par des variations deux ordres
de grandeur plus grandes.

**La marge de déclenchement est de l'ordre d'une fluctuation, et parfois trente fois moindre.** Pour
chaque division, l'écart par lequel le meilleur a été manqué pendant la fenêtre de vingt époques,
exprimé en fluctuations médianes du même run :

| enfant | 1ʳᵉ division | 2ᵉ | 3ᵉ |
|---|---|---|---|
| k0 | époque 113, **+4,80** | 213, +1,26 | 292, +1,85 |
| k1 | époque 35, +1,62 | 173, +1,80 | 283, +1,03 |
| k2 | époque 98, +2,08 | 193, +1,15 | 238, +1,35 |
| k3 | époque 175, +1,59 | 248, +4,81 | 308, **+0,19** |
| k4 | époque 114, **+0,03** | 146, +0,87 | 238, +0,20 |

Le cas le plus net est la première division de k4, à l'époque 114 : le meilleur a été manqué de
2,06·10⁻⁴ en relatif, quand la fluctuation ordinaire de ce run vaut 8,18·10⁻³ — **quarante fois
plus**. Une époque de bruit ordinaire aurait établi un nouveau record, remis le compteur à zéro et
repoussé cette division de plusieurs dizaines d'époques.

## Ce que cela permet de conclure

- La chaîne causale du pilote A est mesurée, et elle est courte. Une perturbation d'un pas float32
  change le bruit d'optimisation ; ce bruit décide, à une marge souvent inférieure à sa propre
  amplitude, si un record tombe dans une fenêtre de vingt époques ; cela déplace une division du pas
  d'apprentissage de plusieurs dizaines d'époques ; et toute la trajectoire suivante en dépend. Les
  premières divisions des cinq enfants tombent aux époques 35, 98, 113, 114 et 175.
- Le seuil de 10⁻⁴ donne l'illusion d'un critère fin. Il est inopérant de deux ordres de grandeur
  face aux fluctuations réelles. Un praticien qui règle ce paramètre règle quelque chose qui ne
  s'applique jamais.
- Conséquence transférable : sur une courbe de validation dont le bruit relatif atteint le
  centième, une détection de plateau à patience fixe est un tirage au sort dont la graine est le
  bruit d'optimisation.

## Ce que cela ne permet pas de conclure

- Toutes les divisions ne sont pas arbitraires : certaines marges valent 4,8 fluctuations, donc
  robustes. C'est la fragilité de **certaines** d'entre elles qui suffit à faire diverger les
  trajectoires, pas la fragilité de toutes.
- La « marge » mesurée ici simplifie l'automate du rappel : elle regarde si la fenêtre de vingt
  époques a battu le meilleur antérieur, là où le rappel entretient un compteur qui se remet à zéro
  à chaque record. L'époque relevée peut être décalée d'une unité.
- Cinq runs, une graine, une architecture, un appareil.

## Prochaine expérience discriminante

Le correctif que cette mesure suggère est testable et bon marché : remplacer la détection de
plateau par un calendrier fixe, ce qui est exactement le pilote C déjà pré-enregistré
(`pilot_C_common_schedule.md`). La prédiction devient chiffrée : si l'époque de division est fixée
par le bruit, huit graines sous calendrier commun doivent voir leur dispersion d'ESR tomber, et la
prédiction écrite — sd(fixe) ≤ sd(décide)/2 — porte sur la même quantité que celle mesurée ici.
