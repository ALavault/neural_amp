# Aucun seuil fixe ne rend la division reproductible

Lecture seule, journaux existants. Script : `diagnosis/seeds/plateau_simulation.py`, sortie
`plateau_simulation.json`. Suite du pilote F, qui a réfuté un seuil sans dire si un autre aurait
marché.

## Ce que la simulation mesure, et ce qu'elle ne mesure pas

Les courbes de validation déjà écrites sont rejouées à travers une simulation fidèle de
`ReduceLROnPlateau` (mode min, seuil relatif), à huit seuils et deux patiences. On relève l'époque
de la **première division** et le **nombre** de divisions.

Un autre seuil changerait la courbe réelle, donc ceci **ne prédit pas** le résultat d'un
entraînement. Ce que cela mesure est la sensibilité du déclencheur lui-même : à courbes données,
dont certaines ne diffèrent que d'un pas float32, avec quelle constance la même règle se
déclenche-t-elle ?

## Observations, patience 20

Écart-type de l'époque de première division, et nombre de divisions :

| seuil | 5 enfants à un pas float32 | 6 graines du banc | 6 runs à chaîne commune |
|---|---|---|---|
| 10⁻⁴ (protocole) | **49,9** (5 à 10 div.) | 20,4 | 48,8 |
| 10⁻³ | 33,7 | 20,4 | 48,8 |
| 5·10⁻³ | 30,4 | 22,4 | 51,2 |
| 10⁻² | 37,0 | 20,8 | 65,9 |
| 2·10⁻² | **14,4** (22 à 23 div.) | 22,5 | 44,5 |
| 5·10⁻² | 34,3 | 22,3 | **14,4** (14 à 17 div.) |

**Aucun seuil ne rend le déclenchement reproductible.** Sur les enfants qui ne diffèrent que d'un
pas float32, l'écart-type de la première division ne descend jamais sous 14 époques et vaut 50 au
réglage du protocole. Sur les graines du banc, il est plat autour de 21 : le seuil n'y change rien
du tout.

**Et les minima sont dégénérés.** Là où la dispersion tombe — 14,4 à 2·10⁻² pour les enfants, 14,4 à
5·10⁻² pour la chaîne commune — le nombre de divisions a explosé à 22 sur 23 et 14 sur 17. Le
déclencheur ne se stabilise pas, il se déclenche en permanence : le calendrier devient « diviser
sans cesse », ce que le pilote F a mesuré comme 2,4 fois pire en qualité. La régularité s'achète en
détruisant la fonction.

**La patience 40 donne l'autre régime dégénéré.** Aux seuils bas, elle produit zéro ou une division
en 500 à 660 époques : avec une courbe bruitée, un nouveau record apparaît presque toujours en
moins de quarante époques, et l'ordonnanceur devient inerte.

## Ce que cela permet de conclure

- **La forme « seuil fixe » est morte comme levier de reproductibilité.** Entre le régime bruité et
  les régimes dégénérés, il n'existe pas de réglage qui déclenche à date reproductible tout en
  laissant l'entraînement se faire. Inutile d'y consacrer du GPU.
- Le pilote F n'avait pas seulement mal choisi sa valeur : aucune valeur n'aurait donné le résultat
  espéré. Le seuil de 2·10⁻² est même celui qui minimise la dispersion chez les enfants — et c'est
  celui qui a dégradé la qualité d'un facteur 2,4.
- Ce qui reste comme levier sur l'ordonnanceur est de **supprimer le déclencheur**, pas de le
  régler : un calendrier fixe, c'est-à-dire le pilote C déjà pré-enregistré
  (`diagnosis/butterfly/pilot_C_common_schedule.md`).

## Ce que cela ne permet pas de conclure

- La simulation tient la courbe fixe alors qu'un autre seuil la changerait. Elle est valide pour
  juger la sensibilité du déclencheur, pas pour prédire une qualité finale.
- Une règle **adaptative** — seuil proportionnel à une estimation courante de la fluctuation — n'est
  pas testée ici et n'est pas réfutée. Elle ne se simule pas sur ces courbes, puisqu'elle changerait
  le comportement dès la première époque.
- Deux patiences, huit seuils, trois groupes de cinq ou six runs, une architecture, un appareil.
