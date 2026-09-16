# Pilote A — les décisions prises sur la validation amplifient-elles l'instabilité ?

Brouillon écrit avant tout run du pilote (2026-09-16). Aucune donnée d'enfant n'existe.

## Question

Les études de l'effet papillon (Kwok et al. 2025, Fort et al. 2020, Summers & Dinneen
2021) entraînent avec un calendrier fixe : une perturbation minime injectée après la
phase chaotique initiale ne change presque plus le résultat. Le protocole NablAFx, lui,
prend trois décisions tout-ou-rien sur 12 segments de validation : division du learning
rate par 2 après 20 époques sans amélioration relative d'au moins 1e-4
(ReduceLROnPlateau par défaut), arrêt après 50 époques sans amélioration (min_delta 0),
garde de polarité. **A** : ces décisions réinjectent de la divergence tard dans l'entraînement, si
bien qu'une perturbation minime après la phase chaotique change encore la qualité finale
quand chaque run prend ses propres décisions, et presque plus quand il rejoue celles d'un
témoin.

Faits qui fixent les points de fourche (runs SSM-WaveNet existants, logs CSV) : la garde
bascule au plus tard à l'époque 20 ; la première division du learning rate tombe entre
les époques 171 et 223 ; l'arrêt entre 666 et 1001. À graine identique, le seul
non-déterminisme GPU donne un écart-type de 0,18 sur le log de l'ESR de test
(`hypotheses_nested.md`).

## Dispositif

- **Condition** : SSM-WaveNet, ZOH, exemption de weight decay, garde ; graine 42 ; lot
  de 16 (7 pas par époque, drop_last) ; `--deterministic` (algorithmes CUDA
  déterministes, padding par réflexion reconstruit par découpage).
- **Prérequis** : au lot de 2, vérifié pour SSM-WaveNet et S4-TF-L-16
  (`scripts/product_determinism_check.py`, `determinism.json`) : deux runs
  déterministes de 200 pas identiques bit à bit (poids et moments d'Adam), deux reprises
  d'un même checkpoint identiques entre elles, deux runs non déterministes différents.
  Au lot de 16, le parent et son jumeau (même commande) doivent produire des
  checkpoints de fourche identiques bit à bit, sinon la file s'arrête
  (`scripts/product_fork_pilot_queue.sh`). Test de mécanique au lot de 2 : l'enfant
  « rejoue » k = 0 reproduit bit à bit l'enfant « décide » k = 0.
- **Parent** : checkpoints enregistrés au début des époques 6 et 101, donc après les
  décisions de fin des époques 5 et 100 ; arrêt ensuite.
- **Enfants** : un entraînement neuf dans lequel on charge, avant le premier pas, les
  poids, les moments d'AdamW, l'état de ReduceLROnPlateau et ceux de l'arrêt anticipé
  et de la garde. (La reprise de Lightning restaure aussi ses compteurs de boucle et,
  depuis un début d'époque, saute la validation de l'époque reprise.) Pas et époques
  comptent depuis la fourche ; le plafond de 15 000 pas est diminué du pas de la
  fourche. L'état du générateur aléatoire global n'est pas repris : tous les enfants
  d'une fourche tirent la même suite de permutations des lots.
- **Perturbation k** (k = 1…4) : avant le premier pas de l'enfant, chaque poids passe à
  la valeur float32 voisine, vers le haut ou vers le bas selon un tirage d'un générateur
  séparé de graine 1000 + k (`nextafter`). Tous les paramètres sont réels. Moments
  d'Adam inchangés. k = 0 : aucune perturbation. Un seul poids déplacé d'un pas float32
  ne suffit pas : dans le test de mécanique (lot de 2, fourche à l'époque 2), 100 pas
  plus tard seul ce poids différait, du même pas, les 93 autres tenseurs étant
  identiques bit à bit.
- **Bras « décide »** : protocole publié, états du scheduler, de l'arrêt anticipé et de
  la garde restaurés depuis le checkpoint ; chaque run décide pour lui-même.
- **Bras « rejoue »** : mêmes perturbations k ; le learning rate de chaque époque et le
  pas d'arrêt sont ceux du témoin k = 0 du bras « décide » ; pas d'arrêt anticipé ; la
  garde reste active (déclenchements enregistrés).
- **Runs** (15) : fourche 100, « décide » k = 0…4 ; fourche 100, « rejoue » k = 0…4 ;
  fourche 5, « décide » k = 0…4 (contrôle positif, dans la phase de bascules).
- **Mesures** : ESR de test du dernier checkpoint (protocole) ; Δk = log ESR(k) − log
  ESR(0) dans chaque bras ; s = écart-type du log ESR sur les 5 runs d'un bras ;
  historique des décisions (époques des divisions, époque d'arrêt, bascules) ; ESR entre
  la sortie de l'enfant et celle du témoin sur l'entrée de test.

## Prédictions

- **Validité du rejeu** : le run « rejoue » k = 0 est identique bit à bit au témoin
  « décide » k = 0 (poids du dernier checkpoint). Sinon le bras « rejoue » est invalide.
- **P0, contrôle positif** : fourche 5, « décide », s ≥ 0,10. Sinon une perturbation d'un
  bit ne compte pas dans ce régime et le pilote ne peut pas conclure sur A.
- **A soutenue** : fourche 100, s(décide) ≥ 0,10, s(rejoue) ≤ s(décide) / 2, et la
  moyenne des |Δk| en « décide » vaut au moins deux fois celle en « rejoue » (mêmes k).
- **A réfutée, dynamique seule** : fourche 100, s(rejoue) ≥ 0,10 et rapport des
  moyennes des |Δk| < 2 : l'instabilité persiste sans décisions.
- **A réfutée, stabilité** : fourche 100, s < 0,10 dans les deux bras.
- **Autres cas** : indécidable avec 4 perturbations par bras.
- **Portée** : s est estimé sur 5 runs (4 degrés de liberté) et les seuils 0,10 et ½
  sont de niveau pilote ; un résultat du pilote décide de l'expérience complète, pas
  de A.
- **Mécanisme, descriptif** : en « décide », les enfants dont la première décision
  diffère du témoin plus tôt ont un |Δk| plus grand.
