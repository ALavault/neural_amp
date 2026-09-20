# D'où vient la variance d'un run à l'autre ? (1) Ce que disent les données déjà écrites

Lecture seule, CPU. Script : `diagnosis/seeds/variance_sources.py`, valeurs par segment dans
`variance_sources.json`. Deux des cinq pistes ouvertes se tranchent sans nouveau run.

## Piste B — la règle d'arrêt : réfutée, et c'était gratuit

Le protocole teste `last.ckpt`, mais chaque run conserve aussi son meilleur point de contrôle au
sens de la perte de validation, atteint à un pas différent. Si une part de la variance venait de
« l'endroit où le run s'est arrêté », les deux règles de sélection donneraient des dispersions
différentes.

Elles donnent la même chose, à 0,0005 d'ESR près sur les quinze runs examinés : effet run 0,375
contre 0,376 chez les graines, 0,110 contre 0,109 chez les enfants. **La règle de sélection ne
contribue pas à la variance.** L'explication est mécanique : après une dizaine de divisions du pas
d'apprentissage, le modèle ne bouge presque plus, et `best` et `last` sont le même modèle.

Réserve : ceci ne teste pas la version forte de la piste, « les runs s'entraînent des durées
différentes » (4 599 à 9 471 pas). Elle demanderait des points de contrôle intermédiaires, que le
protocole ne conserve pas — deux fichiers par run.

## Piste A — la mesure contre le modèle : la variance est réelle, le classement ne l'est pas

**Correction d'un calcul faux que j'ai d'abord écrit.** J'avais soustrait de la variance inter-runs
la variance d'échantillonnage de la moyenne sur douze segments. C'est invalide : tous les runs sont
évalués sur **les mêmes** douze segments, donc leur difficulté est commune et ne contribue pas aux
différences entre runs. Le symptôme était une « variance modèle » négative chez les enfants.

Décomposition correcte à deux facteurs du log de l'ESR par segment :

| groupe | effet run | effet segment | interaction | classement conservé au bootstrap |
|---|---|---|---|---|
| 6 runs, 3 graines × 2 | **0,375** | 1,016 | 0,241 | **59 %** |
| 9 enfants de la fourche 100 | **0,110** | 0,811 | 0,074 | **17 %** |

- **L'effet segment domine tout** (écart-type 1,0 en log, soit un facteur 2,7 entre segments faciles
  et difficiles) mais il est commun à tous les runs : il ne crée aucune différence entre eux. Il
  fixe en revanche l'incertitude de la valeur absolue rapportée, donc toute comparaison à un chiffre
  publié sur un autre matériel.
- **La dispersion entre runs est réelle** : 0,375 pour les graines, 0,110 pour les enfants. Ce n'est
  pas du bruit de mesure.
- **Mais le classement des runs ne survit pas au choix du matériel.** Un autre tirage de douze
  segments réordonne les six runs de graines dans 41 % des cas, et les neuf enfants dans 83 %.
- **Les runs ne diffèrent pas seulement par un facteur global** : l'interaction vaut 0,241 chez les
  graines, soit 64 % de l'effet run. La formulation « la graine multiplie l'erreur de tous les
  segments par un même facteur » (F-0007) est une approximation utile, pas une égalité.

## Ce que cela permet de conclure

- La variance inter-runs n'est pas un artefact d'évaluation : elle est portée par les modèles.
  Chercher son origine dans l'entraînement est donc justifié.
- En revanche, **toute affirmation d'ordre fondée sur douze segments est fragile**. Cela touche
  directement la formulation autorisée du banc, « un ESR plus bas à chacune des trois graines » :
  l'énoncé reste vrai sur ce jeu de test, et aurait 41 % de chances d'être réordonné sur un autre
  tirage de douze segments de la même durée.
- La règle d'arrêt est hors de cause.

## Ce que cela ne permet pas de conclure

- Six runs et neuf enfants, un appareil, une architecture. Le bootstrap sur douze segments est
  lui-même grossier à n = 12.
- Le bootstrap rééchantillonne les segments existants ; il n'estime pas ce que donnerait un autre
  enregistrement, seulement une autre découpe du même.
- Rien ici ne dit **pourquoi** deux runs produisent des modèles différents : la piste est confirmée
  comme réelle, pas expliquée. Les pistes C (le découpage train/validation), D (le seuil de plateau
  sous le bruit) et E (sous-paramétrage) restent ouvertes.
