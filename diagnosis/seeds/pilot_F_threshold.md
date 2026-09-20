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

## Verdict (2026-09-20, 23 h 21)

| run | ESR | pas |
|---|---|---|
| graine 42 | 0,1963 | 3 521 |
| graine 42, répétition | 0,1441 | 4 263 |
| graine 43 | 0,0827 | 3 640 |
| graine 43, répétition | 0,1882 | 2 891 |
| graine 44 | 0,1996 | 2 709 |
| graine 44, répétition | 0,1893 | 2 716 |

| | seuil 2·10⁻² | référence, seuil 10⁻⁴ |
|---|---|---|
| écart-type du log | **0,344** | 0,452 |
| moyenne géométrique de l'ESR | **0,160** | 0,068 |
| pas d'entraînement moyens | 3 290 | 5 778 |

**L'issue « sans effet sur la dispersion » est réalisée** : s = 0,344 dépasse le seuil de 0,25.
Le correctif est écarté.

Note sur la référence : l'écart-type de ces six runs vaut 0,452 et non le 0,375 cité dans le
pré-enregistrement. Les deux chiffres portent sur les mêmes runs mais pas sur la même quantité —
0,375 est l'effet run de la décomposition à deux facteurs (`variance_sources.md`), 0,452 l'écart-type
brut des moyennes par run. Le seuil pré-enregistré de 0,25 était fixé en référence au premier ; le
verdict ne change pas, s = 0,344 le dépasse dans les deux lectures.

## Ce que cela permet de conclure

- **Remonter le seuil au-dessus du bruit ne stabilise pas l'entraînement**, et l'abîme : la qualité
  moyenne passe de 0,068 à 0,160, soit 2,4 fois pire, pour une dispersion qui reste du même ordre.
- Le mécanisme opère pourtant exactement comme prévu. Les divisions arrivent plus tôt et plus
  souvent — douze contre neuf chez la graine 42, dès le pas 1113 contre 1295 — le pas
  d'apprentissage s'effondre jusqu'à 2,4·10⁻⁶, et l'arrêt anticipé tombe à 3 290 pas en moyenne
  contre 5 778. Le seuil a donc bien pris ; c'est l'effet escompté sur la dispersion qui n'existe pas.
- **L'analogie du comparateur sans hystérésis est réfutée comme levier.** Elle décrivait bien le
  symptôme mesuré dans `plateau_margin.md` — un seuil cinquante fois sous le bruit, une marge de
  déclenchement inférieure à la fluctuation — mais corriger ce symptôme ne corrige pas la
  dispersion. Diagnostic juste, correctif faux.
- Mis en regard du pilote E, l'enseignement est net : la variance ne vient pas du réglage de
  l'ordonnanceur, elle vient du tirage des données. Un correctif gratuit existe, et ce n'est pas
  celui-ci.

## Ce que cela ne permet pas de conclure

- **Un seul seuil a été testé, et il est trop haut.** 2·10⁻² vaut deux à quatre fois la fluctuation
  mesurée. Rien ne dit qu'une valeur intermédiaire — 5·10⁻³, par exemple, juste au-dessus du bruit
  sans l'écraser — se comporterait comme celle-ci. Le pilote réfute ce point de la courbe, pas la
  courbe.
- Le seuil fixe n'est de toute façon pas la bonne forme : le bruit de la courbe de validation décroît
  avec la convergence. La version adaptée n'a pas été testée.
- Six runs, une architecture, un appareil, des seuils de qualité pilote.
- La dégradation de qualité et l'absence d'effet sur la dispersion ne sont pas séparables ici : un
  entraînement deux fois plus court a sa propre dispersion, qu'on n'a pas mesurée indépendamment.

## Prochaine expérience discriminante

Si l'on veut savoir si un seuil mieux calé aide, la mesure économique n'est pas un nouveau pilote à
six runs : c'est de rejouer les courbes de validation déjà écrites à travers une simulation de
`ReduceLROnPlateau` à seuils variés, et de regarder la dispersion des **époques de division** que
chaque seuil produirait. Si aucun seuil ne resserre ces époques, la forme fixe est morte et seule la
version adaptée mérite du GPU. Aucun entraînement, les journaux suffisent.
