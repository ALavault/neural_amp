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

## Exécution (ajouté après le lancement, prédictions inchangées)

- File lancée le 2026-09-16 à 16 h 11 au commit bb08337 ; journal
  `demo/runs/fork_pilot_queue.log`, enregistrements `demo/butterfly/`.
- **Parent** : fourches aux pas 42 (époque 5) et 707 (époque 100), 13,8 min pour 707 pas.
  La garde a basculé aux époques 7, 8 et 9 : la fourche 5 précède toutes les bascules.
- **Jumeau** : mêmes bascules, même pas final ; checkpoints des fourches 5 et 100
  identiques bit à bit à ceux du parent (poids et moments d'Adam). Le prérequis du lot de
  16 est satisfait.
- **Coût du mode déterministe** : 51 et 48 pas par minute pour le parent et son jumeau,
  contre 53 à 56 pour les répétitions non déterministes des graines 43 et 44 ; environ
  5 à 10 % plus lent. Mesure grossière : le temps inclut le chargement des données et le
  GPU est partagé.
- **Témoin, fourche 100, « décide » k = 0** : arrêt anticipé à l'époque 922 après la
  fourche (6 461 pas), ESR de test 0,0341, contre 0,0358 et 0,0470 pour les deux runs non
  déterministes du banc à la graine 42. Dix divisions du learning rate, aux époques 113,
  213, 292, 385, 647, 706, 808, 854, 894 et 915.
- **Rejeu k = 0** : dernier checkpoint identique bit à bit au témoin, même ESR de test ;
  pertes de validation identiques sur les 923 époques. Le bras « rejoue » est valide.

## Fourche 100, les deux bras complets (2026-09-17 22 h 55, mesures)

Chiffres de `scripts/product_fork_pilot_analysis.py`, écrits dans
`diagnosis/butterfly/pilot_A_results.json`.

- **« Décide »** : ESR 0,0341 / 0,0512 / 0,0378 / 0,0382 / 0,0365 ; **s = 0,156** ;
  Δ = +0,405, +0,102, +0,112, +0,068 ; arrêts entre 4 599 et 6 461 pas. Les dix divisions
  du learning rate tombent à des époques différentes chez chaque enfant ; la première à
  35, 98, 175 et 114 contre 113 pour le témoin.
- **« Rejoue »** : ESR 0,0341 / 0,0339 / 0,0336 / 0,0339 / 0,0334 ; **s = 0,009** ;
  Δ = −0,007, −0,015, −0,007, −0,023 ; mêmes bascules de garde que le témoin ; écart de
  validation aux divisions imposées au plus 0,063 en log, atteint par k = 3 à la deuxième
  division.
- **Écart de sortie au témoin** : « décide » 8,7·10⁻⁴ à 7,2·10⁻³ ; « rejoue » 4,2·10⁻⁴ à
  7,3·10⁻⁴, soit 2 à 17 fois moins. L'erreur du témoin vaut 0,0245 sur la même mesure.
- **Critères pré-enregistrés de A, à la fourche 100** : s(décide) ≥ 0,10 → 0,156 ;
  s(rejoue) ≤ s(décide)/2 = 0,078 → 0,009 ; rapport des moyennes des |Δ| ≥ 2 → 13,2. Les
  trois sont satisfaits. **Le verdict du pilote reste suspendu au contrôle positif P0 de
  la fourche 5**, comme écrit avant les runs.
- **Mécanisme descriptif** : k = 1 diverge le plus tôt (division à l'époque 35 contre 113)
  et porte le plus grand |Δ| (0,405). Les trois autres divergent aux époques 98, 113 et
  113 et ne se départagent pas (0,102, 0,112, 0,068).
- Coût : 15 runs de 96 à 182 min, le GPU étant partagé. Une tentative a échoué faute de
  mémoire (k = 4, « décide ») et a été relancée par la file.

## Lectures annexes sur les mêmes runs (CPU, lecture seule)

Trois mesures faites sur les enfants de la fourche 100 pendant que la file GPU avance, sans
nouveau run et sans toucher au pré-enregistrement ci-dessus :

- `decided_when.md` — le sort d'un run ne se lit pas dans sa courbe de validation ; les
  décisions dilatent l'écart de validation d'un facteur 16, et l'application validation → test
  multiplie par 4 à 5 dans les deux bras.
- `mode_connectivity.md` — l'enfant le plus divergent franchit une vraie barrière (12 segments
  sur 12), mais la hauteur de barrière suit la distance parcourue (ρ = +0,90) et non le bras :
  la connectivité linéaire ne fournit pas la signature cherchée.
- `listening.md` — page d'écoute aveugle A/B/X et sa lecture, écrite avant toute écoute. Aucune
  écoute n'a encore eu lieu.

## Verdict, le contrôle positif ayant été mesuré (2026-09-18 11 h 00)

`scripts/product_fork_pilot_analysis.py`, sortie dans `pilot_A_results.json`.

**Contrôle positif P0, fourche 5, bras « décide »** : ESR 0,0422 / 0,0684 / 0,0438 / 0,0443 /
0,0422, **s = 0,207** contre un seuil de 0,10. La fourche 5 reproduit donc bien, en pleine phase
chaotique, la dispersion que le dispositif doit être capable de révéler : la faible dispersion du
bras « rejoue » à la fourche 100 n'est pas un artefact du dispositif de fourche.

**Verdict de A : soutenu.** Les trois critères pré-enregistrés sont satisfaits à la fourche 100 et
le contrôle positif tient : s(décide) = 0,156 ≥ 0,10 ; s(rejoue) = 0,009 ≤ 0,078 ; rapport des
moyennes des |Δ| = 13,2 ≥ 2. Une perturbation d'un pas float32 sur chaque poids, appliquée à
l'époque 100, change l'ESR final de 0,156 en log quand le run prend ses propres décisions de
validation, et de 0,009 quand il rejoue celles du témoin.

Observations qui n'appartenaient pas au pré-enregistrement :

- À la fourche 5, la structure est la même qu'à la fourche 100, en plus marqué : un enfant décroche
  seul (+0,482 en log contre +0,405) et les trois autres tiennent dans 0,05. Les écarts de sortie au
  témoin y valent 8,7·10⁻³ à 2,0·10⁻², contre 8,7·10⁻⁴ à 7,2·10⁻³ à la fourche 100.
- C'est k = 1 qui décroche aux deux fourches. La direction de perturbation est tirée avec la graine
  1000 + k, donc k = 1 porte le même motif de directions sur deux parents différents. Avec quatre
  enfants, l'événement a une chance sur quatre d'être fortuit ; si le bras « rejoue » de la fourche 5
  place encore k = 1 en tête, la direction cessera d'être indifférente, ce que le pré-enregistrement
  suppose pourtant.
- Les enfants de la fourche 5 s'arrêtent entre 6 076 et 9 471 pas, contre 4 599 à 6 461 à la
  fourche 100 : perturber tôt allonge aussi l'entraînement.
