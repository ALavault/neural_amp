# Le bruit d'estimation de la validation explique-t-il le découplage ?

Lecture seule, CPU. Script : `diagnosis/butterfly/validation_noise.py`. Sortie :
`validation_noise.json`. Fait suite à l'expérience discriminante annoncée dans
`decided_when.md`.

## La partition de validation, retrouvée et vérifiée

Toutes les décisions du protocole se prennent sur la perte de validation de douze segments de 3 s,
tirés par la graine. Cette partition n'était pas enregistrée. Elle est reproduite ici en répétant
l'ordre de construction du pilote — graine 42, puis processeur, puis module de données — ce qui
donne les indices `[34, 37, 91, 51, 127, 26, 13, 55, 46, 123, 15, 29]`.

**Vérification** : la perte recalculée sur ces douze segments reproduit celle du journal pour les
neuf enfants de la fourche 100, à 0,15 % près (0,02738 contre 0,02742 pour le témoin), l'écart
résiduel étant celui du GPU contre le CPU. La partition est donc la bonne.

## Observations

- À l'intérieur d'un même enfant, la perte varie d'un facteur 3 d'un segment à l'autre : de 0,0170
  à 0,0485 pour le témoin.
- Écart entre enfants, écart-type des neuf moyennes : **0,00080**.
- Erreur type d'une moyenne sur douze segments, médiane sur les enfants : **0,00308**.
- Rapport bruit sur signal : **3,86**.
- Bootstrap sur les douze segments, 10 000 tirages : le **classement complet** des neuf enfants ne
  survit que dans **15,6 %** des tirages ; le **meilleur** enfant garde son nom dans **81,2 %**.

Le rapport de 3,86 est trompeur pris seul, et le bootstrap est la mesure juste. L'erreur type
compare les segments entre eux **à l'intérieur** d'un enfant, alors que la difficulté d'un segment
est commune à tous : le segment 1 est dur pour tout le monde. Le bootstrap, qui retire les douze
segments **conjointement** pour tous les enfants, préserve cette corrélation, et c'est pourquoi le
gagnant tient à 81 % là où le rapport naïf laisserait croire à un tirage au sort.

## Ce que cela permet de conclure

- **Comparer deux runs par leur perte de validation est instable au choix des segments de
  validation.** Un autre tirage de douze segments aurait changé le classement complet quatre fois
  sur cinq, et le meilleur une fois sur cinq. C'est un mécanisme direct du découplage mesuré dans
  `decided_when.md` : la perte de validation et l'ESR de test ne classent pas pareil, en partie
  parce que la perte de validation ne se classe pas elle-même de façon stable.
- Corollaire pratique, indépendant de ce projet : sélectionner un modèle sur une validation de
  douze segments d'un même enregistrement n'a pas la résolution qu'on lui prête.

## Ce que cela ne permet pas de conclure, et correction d'une hypothèse que j'avais avancée

`decided_when.md` proposait le bruit d'estimation sur douze segments comme cause candidate du
comportement divergent des décisions. **Cette hypothèse ne tient pas telle qu'elle était écrite.**
L'ordonnanceur ne compare pas des runs entre eux : il compare un run à lui-même, d'une époque à la
suivante, sur **les mêmes** douze segments. Le tirage des segments est donc constant tout au long
d'un run et s'annule dans la détection de plateau. Il déplace le niveau de la courbe, pas ses
variations.

Ce qui reste à mesurer pour expliquer les décisions divergentes est donc autre chose : la
fluctuation d'une époque à la suivante, à partition fixée, comparée au seuil relatif de 10⁻⁴ que le
`ReduceLROnPlateau` utilise. C'est une propriété du bruit d'optimisation, pas de l'échantillonnage.

Autres réserves : neuf enfants d'une seule graine, une seule partition, un seul appareil. Le
bootstrap sur douze segments est lui-même une estimation grossière à n = 12.

## Prochaine expérience discriminante

Mesurer, dans les journaux déjà écrits, la distribution des variations relatives de la perte de
validation d'une époque à la suivante, et la comparer au seuil de 10⁻⁴ et à la patience de 20
époques. Si les fluctuations dépassent couramment le seuil, la détection de plateau se déclenche sur
du bruit d'optimisation, et l'époque de division devient essentiellement aléatoire — ce qui
expliquerait que les premières divisions tombent aux époques 35, 98, 113, 114 et 175 chez des
enfants qui ne diffèrent que d'un pas float32. Aucun calcul nouveau, les colonnes sont dans
`metrics.csv`.
