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

**Le classement exact des cinq enfants n'est jamais atteint** — mais ce test est décidé par ses
quasi-ex æquo : trois des cinq enfants terminent entre 0,0365 et 0,0382 d'ESR, soit 4,6 % d'écart.
Exiger que leur ordre exact soit reproduit revient à demander à la validation de trancher des
égalités. La lecture honnête se fait paire par paire.

**Résultat principal, par paire.** Époque à partir de laquelle le minimum courant de validation
conserve définitivement le bon ordre, en regard de l'écart final de la paire :

| paire | \|Δ log ESR\| | époque |
|---|---|---|
| k0–k1 | 0,405 | 191 |
| k1–k4 | 0,337 | 115 |
| k1–k2 | 0,303 | 98 |
| k1–k3 | 0,292 | 176 |
| k0–k3 | 0,112 | 652 |
| k0–k2 | 0,102 | 296 |
| k0–k4 | 0,068 | 269 |
| k3–k4 | 0,045 | jamais |
| k2–k4 | 0,034 | 454 |
| k2–k3 | 0,011 | jamais |

Les quatre paires d'écart ≥ 0,29 en log — toutes celles qui opposent k1, l'enfant nettement le
plus mauvais — sont tranchées entre les époques 98 et 191, c'est-à-dire dans le premier quart de
l'entraînement (k0 s'arrête à 922). Les écarts ≤ 0,11 ne sont tranchés que tard (269 à 652) ou
jamais. Chez les enfants « rejoue », dont tous les écarts sont ≤ 0,023, sept paires sur dix ne
sont jamais tranchées — cohérent, il n'y a rien à trancher.

À noter : k1 diverge par une division de pas d'apprentissage dès l'époque 35, mais son infériorité
ne devient définitivement lisible qu'entre 98 et 191. La décision précède la trace mesurable de
60 à 160 époques.

**Positif, faible.** À l'époque terminale de chaque enfant, le lien existe mais reste bruité :
ρ = +0,70 (p = 0,19, n = 5) dans le bras « décide » ; +0,70 (p = 0,036, n = 9) sur les neuf enfants
distincts des deux bras réunis. Sur les paires même-seed du banc, le run de plus faible perte de
validation finale est aussi celui de plus faible ESR final dans 3 cas sur 3 — mais l'accord
oscille en cours de route (0/3 à l'époque 10, 3/3 à 100, 1/3 à 200 et 300, 3/3 à la fin).

**Décomposition de l'effet papillon.** Écarts-types en log à l'époque terminale. Attention : l'ESR
est quadratique en amplitude d'erreur là où la perte L1 + 0,1 MR-STFT est linéaire, donc un facteur
2 entre les deux colonnes est imposé par les définitions, avant tout découplage.

| bras | perte de validation | ESR de test | rapport | au-delà du facteur 2 |
|---|---|---|---|---|
| décide | 0,032 | 0,156 | 4,9 | 2,5 |
| rejoue | 0,002 | 0,009 | 3,8 | 1,9 |

Les décisions dilatent l'écart de **validation** d'un facteur 16 (0,002 → 0,032) ; l'application
validation → test ajoute ensuite un facteur 2 environ au-delà de celui qu'imposent les définitions,
**du même ordre dans les deux bras**. Seul le premier facteur dépend du bras.

## Ce que cela permet de conclure

- Le sort d'un run se lit d'autant plus tôt qu'il est plus tranché. Un run inférieur de 35 % en
  ESR (0,3 en log) devient définitivement identifiable vers l'époque 100 à 190 sur 900, soit le
  premier quart de l'entraînement. Une sélection de run sur validation précoce sépare donc ce type
  d'écart, mais ne sépare pas des runs distants de moins de ~12 % : sur ceux-là elle tranche tard
  ou jamais.
- L'effet papillon n'est pas créé par le passage validation → test : au-delà du facteur 2 imposé
  par les définitions, ce passage ajoute environ ×2 dans les deux bras. Ce que font les décisions,
  c'est dilater l'écart en amont, sur la quantité même qu'elles observent.

## Ce que cela ne permet pas de conclure

- n = 5 par bras, une fourche, un seed, un jeu de données. Le ρ groupé (n = 9, p = 0,036) mêle deux
  bras dont les plages d'ESR se chevauchent à peine : il mesure en partie un effet entre groupes,
  pas seulement l'ordonnancement à l'intérieur d'un bras.
- Les époques du tableau par paire reposent chacune sur une seule paire de runs : elles décrivent
  ces dix comparaisons, elles n'établissent pas un seuil général.
- Le facteur restant entre écart de validation et écart de test est estimé sur douze segments de
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
