# Porte impaire (structure KIPR) contre la porte actuelle : préinscription

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-06, avant tout
entraînement de cette piste.** Les règles ci-dessous ne changent plus après les chiffres.

## Question

La 2ᵉ harmonique en excès des modèles VS100R vient de la porte `tanh(a)·sigmoid(b)`. Le produit de
deux signaux audio domine, et les décalages constants contribuent aussi
(`diagnosis/hammerstein/NOTE.md`, sources dans la porte). Une porte impaire à asymétrie explicite,
dans l'esprit du moteur KIPR, retire-t-elle cet excès sans perdre en ESR ?

**Décision que la mesure change** : proposer ou non la porte impaire comme porte des nouveaux
prototypes. Cette décision reste à prendre séparément : la mesure ne change aucun défaut.

## Modèle (`act_type="odd"`, `src/fssr_nam/models/ssm_wavenet.py`)

- Porte `OddGate` : `tanh(a)·sigmoid(c + s_b + w·e)`, où e est un passe-bas à un pôle de |b|.
  L'état initial est nul, la constante de temps 2 ms au départ. Aucun décalage dans a ni dans b.
  Le décalage FiLM de b s'ajoute après l'enveloppe ; celui de a sert d'asymétrie β.
- L'asymétrie est explicite : sortie positive × e^β, négative × e^−β, avec β appris par canal plus
  le terme FiLM. Pour β = 0, chaque bloc est impair.
- Les 1×1 des blocs, l'entrée et la tête n'ont pas de biais. La tête est en `tanh` : le réseau
  entier est impair quand toutes les asymétries sont nulles. Les biais sont retirés après
  construction, donc les tirages aléatoires sont ceux du modèle à porte actuelle.
- Contrôles : `tests/numerical/test_ssm_odd_gate.py` (imparité, zéro → zéro, asymétrie, causalité,
  parité porte et bloc entier/pas à pas, remise à zéro, mêmes tirages).
- La comparaison porte sur **le modèle impair entier** (porte, absence de biais, tête `tanh`)
  contre le bras C ; elle ne sépare pas ces trois changements.

## Bras

- **VS100R** : `vs100r48odd_film_s42`, `_s43`, `_s44`. Mêmes drapeaux que le bras C
  (`vs100r48fast_film_s*`), plus `--act-type odd --head-act tanh`, avec les pas de division du lr
  du jumeau A, comme C (42 : 7 830 et 10 701 ; 43 : 11 484 et 14 355 ; 44 : 6 438 et 14 181).
  Référence : le bras C, par graine.
- **KIPR** : `kipr_{sym,asym}_s{42,43,44}_odd`, `known_answer.py --tag odd`, protocole de la
  reprise r2 (avance de 196, même entraînement). Référence : `kipr_*_r2`.

Avant les runs VS100R : porte au bit du chemin par défaut (300 pas contre
`vs100r_gc_film_joint_s925_r2`), puis un smoke de 100 pas du modèle impair. Après chaque run
VS100R : validation identique à celle du bras C de même graine. La file attend les files en cours.

## Lectures (script `demo/odd_gate/evaluate.py --arm odd`, commité avec ce texte)

**R1, 2ᵉ harmonique VS100R** (`diagnosis/hammerstein/vs100r_hammerstein.py` relancé avec le bras
ODD). Écart signé de D2 au modèle de l'appareil (dB) : médiane sur les bandes de 200 Hz à 8 kHz
pour chaque prise et run, puis médiane sur les 8 prises de test et les 3 graines, à −20, −10 et
−3 dBFS. On compare Δ = ODD − C.
- « réduit » si Δ ≤ −6 dB aux trois niveaux ;
- « inchangé » si |Δ| ≤ 3 dB aux trois niveaux ;
- sinon « partiel ».

Pour mémoire, C vaut +12,7 dB à −20 dBFS avec cette agrégation.

**R2, ESR VS100R** : lecture P de `demo/train_length/predictions.md` (35 prises hors faible
signal, di_test, moyenne des ln ESR par couple gain–contour, couple non vu exclu, checkpoint `last`),
d_i = s(ODD) − s(C). « Acceptable » si d ≤ +0,10 et chaque d_i ≤ +0,25. Les étiquettes
meilleur / moins bon / équivalent / non tranché sont rapportées.

**Verdict de la piste** :
- « porte impaire retenue » si R2 est acceptable et R1 « réduit » ;
- « porte impaire rejetée (ESR) » si R2 n'est pas acceptable ;
- sinon « porte impaire non retenue (harmoniques paires non réduites) ».

**KIPR** (`known_answer.py measure --tag odd`). Mêmes portes que r2 : instrument, alignement des six
runs, validité niveau par niveau. Même lecture : plancher si E_sym ≥ +15 dB, aucun si ≤ +5 dB.
Comparaison à r2 : « réduit » si E_sym(odd) ≤ E_sym(r2) − 10 dB, seulement si les deux lectures
sont valides (verdict plancher, pas de plancher ou plancher partiel). Si ce bras n'en a pas : aucune
lecture ; si r2 n'en a pas : descriptive. Le verdict KIPR est rapporté à côté de celui de la
piste, sans le modifier.

**Descriptif, sans seuil** :
- |D3| (harmoniques impaires), médiane des deux bras par niveau ;
- prises au gain 2,5 (16 et 36) : écart de ln ESR (di_test, `last`) ODD − C par graine, et écart
  signé de D2 à −10 dBFS par bras (médiane des graines) ;
- nombre de paramètres des deux bras (champ `parameters` des records).

## Ce qui ne pourra pas être conclu

- Rien sur un autre appareil, ni à l'écoute.
- Rien sur la porte seule, puisque la tête et les biais changent avec elle.
- Avec 3 graines, seuls des effets nets ressortent.

## Amendement avant tout entraînement (2026-10-06, après la lecture de r2)

La reprise r2 est sans lecture : deux runs sur six sont de polarité inversée
(`demo/kipr_known_answer/predictions.md`, résultats r2). Pour les runs KIPR de cette piste, la
mesure (`known_answer.py measure`) prend pour la sortie le signe de sa corrélation avec la cible
sur les segments `train_noise`, jamais sur `test_noise`. Ensuite seulement viennent la porte
d'alignement et les lectures, inchangées. L'entraînement ne change pas. r2 n'ayant pas de lecture
valide, la comparaison KIPR à r2 sera descriptive (règle déjà écrite). Les lectures VS100R ne
changent pas : le banc VS100R a sa garde de polarité.

## Résultats KIPR (2026-10-06, six runs, exit 0 ; lecture intermédiaire, VS100R en attente)

Lecture `known_answer.py measure --tag odd`, sortie `known_answer_odd.json`.

**Verdict KIPR préinscrit : non concluant (témoin positif ou H3 de la cible symétrique).** Comparaison à r2 : descriptive (r2 sans lecture valide).

- `sym` : excès de H2/H1 (médiane des graines) +1.2 dB à -24 dBFS, +37.4 dB à -12 dBFS, +46.9 dB à -6 dBFS ; écart de H3/H1 -0.2, +0.3, -2.4 dB.
- `asym` : excès de H2/H1 (médiane des graines) +1.2 dB à -24 dBFS, +5.6 dB à -12 dBFS, +9.5 dB à -6 dBFS ; écart de H3/H1 -0.2, -0.2, -0.7 dB.
- Alignement à −12 dBFS (décalage, ESR, polarité appliquée) : `kipr_sym_s42_odd` 0, 0.22, +1 ; `kipr_sym_s43_odd` 0, 0.22, +1 ; `kipr_sym_s44_odd` 0, 0.24, +1 ; `kipr_asym_s42_odd` 0, 0.2, +1 ; `kipr_asym_s43_odd` 0, 0.21, +1 ; `kipr_asym_s44_odd` 0, 0.18, +1.
- **Descriptif, sans interprétation de verdict** : la porte impaire n'empêche pas l'excès sur la cible
  dite symétrique (+37 et +47 dB). Le modèle est exactement impair à β nul, il a donc appris une
  asymétrie par β. Or cette cible n'est symétrique qu'en forte saturation : son H2/H1 sur les sons
  vaut −12 dB à −60 dBFS, −23 à −48 et −35 à −36 dBFS (`metrics.json` KIPR). Un modèle entraîné sur
  du bruit à quatre niveaux y apprend une asymétrie dépendante du niveau. La prémisse du banc
  (« cible symétrique ») ne vaut que pour les sons de −24 à −6 dBFS.

## Note d'exécution (2026-10-07, avant tout run VS100R de la piste)

Le smoke VS100R `smoke_kiprtracks_odd` est sorti en mémoire GPU épuisée (23,3 Gio sur 23,4, le
processus seul ; échec inscrit dans `demo/RUNS.jsonl`). Cause mesurée sur CPU (tenseurs gardés pour
la rétropropagation, drapeaux de la file dont `--discretization zoh`, lot de 2 et 0,5 s au lieu de la
taille réelle) : les tampons FFT de l'enveloppe gardent 1,71 fois la mémoire du bras C. Correctif :
la porte impaire est recalculée pendant la rétropropagation (`torch.utils.checkpoint`), ce qui
ramène le rapport à 0,99 ; c'est une estimation CPU, le smoke GPU `_r2` la confirmera ou non.
Sortie et 102 gradients de paramètres identiques au bit avec et sans recalcul (float64, CPU, mêmes
drapeaux dont `zoh` ; une première vérification en discrétisation par défaut donnait 110 sur 110).
La parité au bit sur GPU n'est pas vérifiée (relecture Codex). Le modèle, ses paramètres, les
drapeaux, les graines et les lectures ne changent pas ; les runs KIPR déjà faits ne sont pas
touchés. La file des pistes est relancée avec des identifiants de porte et de smoke suffixés `_r2` ;
les identifiants des runs VS100R préinscrits restent ceux de ce texte.
