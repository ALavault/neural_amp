# Pilote F — remonter le seuil de plateau au-dessus du bruit

**Pré-enregistré le 2026-09-20, avant tout run.** Piste D de la discussion sur l'origine de la
variance, et seul correctif que la mesure de `diagnosis/butterfly/plateau_margin.md` désigne
directement.

## Question

`ReduceLROnPlateau` divise le pas d'apprentissage après 20 époques sans amélioration relative de
plus de 10⁻⁴. Or la fluctuation relative médiane de la perte de validation d'une époque à la
suivante vaut **4,4 à 8,8·10⁻³** selon le run, soit 44 à 88 fois ce seuil. Le seuil ne s'applique
donc jamais : une époque de bruit ordinaire suffit à établir un « record » et à remettre le
compteur à zéro. La marge de déclenchement mesurée est de l'ordre d'une fluctuation, et vaut parfois
0,03 fluctuation.

L'analogie est celle d'un comparateur sans hystérésis sur un signal bruité. Le geste
correspondant est de placer le seuil **au-dessus** du bruit, pour qu'une amélioration ne compte que
si elle dépasse la fluctuation ordinaire.

**Question posée :** un seuil de 2·10⁻², au-dessus du bruit mesuré, réduit-il la dispersion entre
runs — et à quel prix sur la qualité ?

## Dispositif

SSM-WaveNet avec les deux changements. Trois graines, 42 à 44, deux runs chacune, avec
`--plateau-threshold 0.02` : exactement le plan des six runs de référence du banc, un seul
paramètre changé. Environ 2 h par run, 12 h en tout.

Référence déjà mesurée, même plan à 10⁻⁴ : effet run **0,375**, composante non-déterministe
**0,18**, moyenne géométrique de l'ESR **0,068**.

**Les deux quantités comptent.** Un correctif qui réduit la dispersion en dégradant tout le monde
n'est pas un correctif. Le sens attendu de l'effet secondaire est connu d'avance : un seuil plus
haut rend les améliorations plus rares, donc les divisions plus précoces et l'entraînement plus
court ; la qualité moyenne peut en souffrir.

## Portes

1. Le défaut de `--plateau-threshold` est 10⁻⁴, la valeur de PyTorch, et ne change rien : vérifié.
2. Les six runs doivent effectivement diviser plus tôt que la référence. Sinon le paramètre n'a pas
   pris et le pilote est nul : vérification sur `lr_by_epoch` du premier run.

## Prédictions, et elles partitionnent l'espace

Soit s l'écart-type de l'effet run sur les six runs, et g la moyenne géométrique de leur ESR.

- **Correctif utile** : s ≤ 0,25 **et** g ≤ 0,082, c'est-à-dire une dispersion nettement réduite pour
  une qualité qui ne se dégrade pas de plus de 20 %.
- **Correctif au prix de la qualité** : s ≤ 0,25 **et** g > 0,082. La stabilité s'achète, et le taux
  de change est mesuré.
- **Sans effet sur la dispersion** : s > 0,25. Le seuil n'était pas le levier, quelle que soit g.

## Ce que ce pilote ne pourra pas dire

- Un seuil fixe n'est pas la bonne forme du correctif : le bruit de la courbe de validation décroît
  avec la convergence, donc 2·10⁻² sera au-dessus du bruit au début et bien au-dessus à la fin. La
  version adaptée — seuil proportionnel à une estimation courante de la fluctuation — est ce qu'il
  faudrait tester si celle-ci marche. Ce pilote teste la forme la plus simple, pas la meilleure.
- Six runs, une architecture, un appareil, des seuils de qualité pilote.
- Il ne sépare pas l'effet du seuil sur la **date** des divisions de son effet sur leur **nombre**.
