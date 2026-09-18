# Quand le sort d'un run se joue-t-il ?

Lecture a posteriori des journaux existants, sans nouveau run.
Script : `diagnosis/butterfly/decided_when.py`. Sorties : `decided_when.txt`, `decided_when.json`.

## Hypothèse testée

Si le sort d'un run se décidait tôt, la perte de validation à une époque précoce ordonnerait
déjà l'ESR de test final. On mesure donc la corrélation de rang de Spearman entre la perte de
validation à l'époque *e* et l'ESR de test final, sur des runs qui partagent le même découpage
train/validation — condition nécessaire, puisque le découpage est tiré par le seed et qu'il
ordonne à lui seul l'ESR final entre seeds (DIAGNOSIS §4, H2).

## Méthode

Trois groupes, tous en lecture seule sur `demo/runs/*/logs/metrics.csv` et `demo/butterfly/*.json` :

- les cinq enfants « décide » de la fourche 100, qui partagent le découpage du seed 42 et ne
  diffèrent que par une perturbation d'un pas float32 puis par leurs propres décisions ;
- les cinq enfants « rejoue », même découpage, schéma d'apprentissage imposé ;
- les trois paires même-seed du banc (run initial contre répétition), où le découpage est commun
  à l'intérieur d'une paire mais pas entre paires : comparaison binaire, 3 paires.

Deux signaux par époque : la perte de validation de l'époque, et le minimum courant, qui est ce
que lisent réellement `ReduceLROnPlateau` et l'arrêt anticipé. Le point apparié à l'ESR final est
la perte de l'époque terminale de chaque enfant, puisque le test porte sur `last.ckpt`.

## Baseline

Aucune baseline externe : la référence est la prédiction triviale « la perte de validation
ordonne l'erreur de test », qui est l'hypothèse implicite de tout arrêt sur validation.

## Observations

**Négatif, et c'est le résultat principal.** Le classement des cinq enfants « décide » par le
minimum courant de validation ne coïncide **jamais** avec leur classement par ESR final, à aucune
époque, pas même à la dernière époque commune (656). ρ erre : −0,70 aux époques 25–50, +0,80 à
150, +0,30 à 300, +0,70 à 656. Le classement de validation lui-même ne se fige qu'à l'époque 652,
c'est-à-dire à la fin. Chez les enfants « rejoue », même absence de coïncidence, avec un ρ terminal
négatif (−0,50) — attendu, leurs ESR tiennent dans 2 %.

**Positif, faible.** À l'époque terminale de chaque enfant, le lien existe mais reste bruité :
ρ = +0,70 (p = 0,19, n = 5) dans le bras « décide » ; +0,70 (p = 0,036, n = 9) sur les neuf enfants
distincts des deux bras réunis. Sur les paires même-seed du banc, le run de plus faible perte de
validation finale est aussi celui de plus faible ESR final dans 3 cas sur 3 — mais l'accord
oscille en cours de route (0/3 à l'époque 10, 3/3 à 100, 1/3 à 200 et 300, 3/3 à la fin).

**Décomposition de l'effet papillon.** Écarts-types en log à l'époque terminale :

| bras | perte de validation | ESR de test | rapport |
|---|---|---|---|
| décide | 0,032 | 0,156 | 4,9 |
| rejoue | 0,002 | 0,009 | 3,8 |

Les décisions dilatent l'écart de **validation** d'un facteur 16 (0,002 → 0,032) ; l'application
validation → test multiplie ensuite par 4 à 5 ce qu'elle reçoit, **identiquement dans les deux
bras**. L'amplification totale 0,009 → 0,156 est le produit des deux, et seul le premier facteur
dépend du bras.

## Ce que cela permet de conclure

- Le sort d'un run ne se lit pas dans sa courbe de validation avant la toute fin : aucune époque
  précoce n'ordonne l'ESR final. Une sélection de run sur validation précoce — pratique courante
  pour économiser du calcul — n'a ici aucun fondement.
- L'effet papillon n'est pas créé par le passage validation → test : ce passage amplifie de la même
  façon (×4–5) dans les deux bras. Ce que font les décisions, c'est dilater l'écart en amont, sur
  la quantité même qu'elles observent.

## Ce que cela ne permet pas de conclure

- n = 5 par bras, une fourche, un seed, un jeu de données. Le ρ groupé (n = 9, p = 0,036) mêle deux
  bras dont les plages d'ESR se chevauchent à peine : il mesure en partie un effet entre groupes,
  pas seulement l'ordonnancement à l'intérieur d'un bras.
- Le facteur 4–5 entre écart de validation et écart de test est estimé sur douze segments de
  validation et douze segments de test d'un seul enregistrement. Rien n'assure sa stabilité
  ailleurs.
- Rien ici ne dit *pourquoi* validation et test se découplent. Les deux ensembles font douze
  segments chacun ; le candidat évident est le bruit d'estimation sur douze segments, non mesuré.

## Prochaine expérience discriminante

Mesurer la perte de validation par segment sur les douze segments de validation, pour chacun des
dix enfants de la fourche 100 : si le découplage vient du bruit d'estimation, l'écart-type entre
segments d'un même enfant doit dominer l'écart entre enfants, et un ré-échantillonnage bootstrap
des douze segments doit rendre le classement des enfants instable. Coût : dix inférences CPU sur
l'ensemble de validation, aucun entraînement.
