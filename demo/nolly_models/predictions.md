# Modèles entraînés sur les rendus Nolly : préinscription

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Décision D-0015 (research-memory,
2026-10-11). **Écrit le 2026-10-11, avant l'arrivée de tout rendu Nolly** et avant toute lecture
de la file boîte grise approfondie (D-0014), encore en cours. Ce texte, `evaluate.py`,
`scripts/product_nolly_dataset.py` et `scripts/product_nolly_queue.sh`, commités ensemble, ne
changent plus après les chiffres. Un écart éventuel va dans une section datée « Écarts ».

## Question

Demande de l'utilisateur, le 2026-10-11 : sur les rendus Nolly, en plus de la lecture des
sondes (`diagnosis/probe_metrics/NOLLY_READING.md`), entraîner la boîte grise et le bras C.

**Décisions que la mesure change :**
- savoir si une boîte grise approfondie s'approche d'un réseau quand l'acquisition ne
  brouille pas la comparaison : ni bruit, ni dérive, ni décalage entre entrées. Reaper compense
  la latence déclarée, et la règle ci-dessous compense un retard commun résiduel à
  l'échantillon entier près, à 48 kHz ; un reste fractionnaire commun est possible. C'est le
  cas le plus propre dont on dispose ;
- savoir si la portée linéaire et la profondeur gardent leur rôle sur un ampli dont les cinq
  commandes de tonalité agissent. Sur l'envoi FX de la VS100R, seuls le gain et le contour
  comptaient.

## Données

- **Rendus** : Neural DSP Archetype: Nolly, ampli Rhythm, baffle coupé.
  - Feuille : `demo/capture/v2/SESSION.md`, avec les rôles de `settings.csv`.
  - Entrée commune : le fichier source `reamp_v2_192k.wav`, sha256 `9ebc26a3…`.
- **Prises des modèles** (`fssr_nam.product.nolly.model_takes`) :
  - les 31 prises « train » moins les répétitions 20 et 39 de la prise 01, soit **29**. Ces
    répétitions servent au contrôle de déterminisme. Identiques au bit, elles mettraient les
    mêmes segments dans l'entraînement et dans la validation, et pèseraient triple sur le
    centre ;
  - les **8** combinaisons de test, jamais entraînées ;
  - les trois contrôles baffle sont exclus.
- **Sections**, aux positions du `layout.json` v2, identiques à celles de la VS100R :
  - entraînement et validation : `di_train` + `di_validation`, 150 s, 50 segments de 3 s par
    prise ;
  - test : `di_test`, 30 s, des 8 combinaisons ;
  - réglages vus : `di_test` des 29 prises d'entraînement.
- **48 kHz** : chaque section est décimée par `resample_poly(·, 1, 4)`.
- **Latence, règle figée** :
  - m est la médiane des retards aux marqueurs des prises sans baffle, à 192 kHz, lue dans les
    portes de `nolly_reading.py` ;
  - une seule avance commune de round(m / 4) échantillons à 48 kHz, arrondie au plus proche
    (moitié vers l'extérieur), est appliquée à toutes les prises, entraînement et test. La cible
    est lue d'autant plus tard à 192 kHz, avant décimation ;
  - rien n'est recalé si l'avance vaut 0 ;
  - les retards propres à chaque prise (phase des filtres de tonalité) ne sont jamais
    compensés : ils font partie de l'appareil.
- **Comptes** :
  - 29 × 50 = 1 450 segments ;
  - le partage 0,9 de nablafx donne **1 306 / 144**, vérifié sur le substitut synthétique. Il
    est tiré après `seed_everything(925)`, donc commun à tous les bras.
- **Condition** : la lecture des sondes doit avoir passé ses portes : bouclage, stéréo, chaque
  prise des modèles, ancres et latence. Sinon, le constructeur du jeu refuse, et rien n'est
  entraîné.

## Bras

Graines 42 à 44, 48 kHz, partition `--split-seed 925`. Drapeaux communs, ceux de C sur la
VS100R (`vs100r48fast_film_s*`) avec `--device nolly` :

`--device nolly --discretization zoh --honor-optim --deterministic --split-seed 925
--l1-weight 0.5 --mrstft-weight 0.5 --cond film --cond-head joint --cond-hidden 32
--no-early-stopping --fft-size fast --no-tf32`

| Bras | Run | Configuration | Paramètres (5 boutons) |
|---|---|---|---|
| C | `nolly_C_s*` | SSM-WaveNet, FiLM joint, tête leaky | 28 417 |
| GREY | `nolly_grey_s*` | GreyBoxAmp : 1 étage, FIR 32 | 1 620 |
| L | `nolly_gbL_s*` | 1 étage, FIR 512 | 2 580 |
| F3 | `nolly_gbF3_s*` | 3 étages, FIR 512, biais, enveloppes 50 et 300 ms | 8 166 |

**Taux d'apprentissage**
- **C** décide ses divisions sur plateau : 15 000 pas, sans arrêt anticipé. Sur la VS100R, C
  rejouait les divisions de son jumeau A48 ; il n'en a pas ici.
- **GREY, L et F3** rejouent les pas de division du C de même graine, relevés dans son
  `metrics.csv` (changements de `lr-AdamW/pg1`). C de chaque graine finit donc avant eux.
- Si C ne divise jamais, la file s'arrête et l'utilisateur décide.

**File** (`scripts/product_nolly_queue.sh`, runner épinglé, enchaînée après la file en cours) :
1. les comptes du jeu sont inscrits dans `demo/RUNS.jsonl` ;
2. porte au bit ;
3. essai de 100 pas de chaque bras ;
4. par graine : C, F3, GREY, L.

Chaque run, échecs compris, va dans `demo/RUNS.jsonl`, ainsi que chaque essai court réussi.

**Relance** : une tentative interrompue sans fiche n'est jamais reprise sous le même
identifiant. La file s'arrête sur elle. La relance la liste dans `RELAUNCHED` et la fait
repartir sous `<id>_r2`, identifiant que `evaluate.py` lit quand `<id>` n'a pas de fiche. Les
pas de division et les contrôles suivent l'identifiant réellement lancé.

**Contrôle après chaque run** : 15 000 pas, 1 306 segments d'entraînement, et même partition que
le C de même graine (sauf pour C).

**Portée de la porte au bit** : elle compare le chemin par défaut du banc (300 pas) à
`vs100r_gc_film_joint_s925_r2`. Elle prouve seulement que le chemin VS100R n'a pas bougé. Aucun
run Nolly n'est comparable au bit à un run existant : une tête à 5 entrées tire d'autres valeurs
initiales.

## Lectures (`demo/nolly_models/evaluate.py`, commité avec ce texte)

Dernier checkpoint de chaque run. Rendu complet de `di_test`, région notée de `ev.scored` (sans
la première demi-seconde ni le dernier dixième de seconde).

**P Nolly** :
- s = moyenne des ln ESR des 8 combinaisons de test. Il n'y a pas d'exclusion : chacune est un
  réglage unique jamais vu ;
- d_i = s(bras) − s(référence) par graine ;
- étiquettes `tl.verdict` et acceptabilité (d ≤ +0,10 et chaque d_i ≤ +0,25), comme pour la
  VS100R.

Lectures principales :
1. **F3 contre GREY** : verdict « boîte grise approfondie meilleure que GREY » si l'étiquette
   est « meilleur » ; sinon, l'étiquette telle quelle.
2. **F3 contre C** : « boîte grise approfondie équivalente à C (ESR) » si acceptable ; sinon
   « moins bonne que C (ESR) ». L'étiquette est rapportée à côté.

**Lectures secondaires, descriptives** :
- L contre GREY (portée linéaire) et L contre C ;
- les mêmes comparaisons sur les réglages vus (`di_test` des 29 prises d'entraînement). Elles
  ne portent que sur des valeurs finies ; un bras avec une valeur non finie n'a pas cette
  lecture ;
- **co-lecture à retard compensé**, règle de `coreading_delay.py` inchangée :
  - grille de −2 à +3 échantillons à 48 kHz, au pas de 0,05 ;
  - retard commun par run, celui qui minimise l'ESR médiane des 8 prises de test sur leur
    `di_validation` (jamais entraînée), appliqué tel quel à `di_test` ;
  - une graine au bord de la grille retire l'étiquette ;
  - lue seulement si l'effet de bord du décalage reste sous 1 % de la plus petite ESR recalée
    des 8 prises de test, sur les runs valides hors bord de grille ;
- **mémoire** (`demo/greybox_deep/HISTORY_MEASURE.md`) :
  - Δγ_T = γ_T(modèle) − γ_T(appareil), T = 30, 100 et 300 ms ;
  - médiane sur les graines et les 8 prises de test, par bras ;
  - rapporté à côté d'un affaissement de 3 dB (0,19 à 100 ms) et du plancher S0. S0 a été
    mesuré sur les boucles DI réelles de la VS100R (`di_test`), c'est-à-dire des
    enregistrements du signal source dont le `di_test` v2 est la copie au bit ;
- **γ de l'appareil** le long de l'échelle de gain : prises 30, 16, 01, 06 et 07, gain de 0 à
  10, les autres boutons à 5.

**Aussi rapportés** : paramètres (traitement et tête séparés), avance appliquée, retard compensé
de chaque run, chaque étiquette même si elle contredit les attentes.

**Avant de lire** :
- chaque run a 15 000 pas et son contrôle est `ok` ;
- les valeurs lues sont finies.

Un bras incomplet n'est pas lu ; ses runs présents sont rapportés tels quels.

## Attentes (repères, pas des seuils)

- **F3 contre C** : « moins bon ». Le réseau garde l'avantage sur une cible sans bruit.
- **F3 et L contre GREY** : « meilleur », avec un écart d'au moins le facteur 2 des oracles
  VS100R (d ≤ −0,7). Avec cinq filtres de tonalité actifs, des FIR de 32 coefficients (0,7 ms)
  ne peuvent pas suivre le grave.
- **Réglages vus contre combinaisons de test** : pas d'attente chiffrée.
- **Mémoire** : pas d'attente. On ignore si Nolly modélise un affaissement.

## Ce qui ne pourra pas être conclu

- **Le matériel** : rien sur un ampli réel ni sur une chaîne d'acquisition. Nolly est lui-même
  un modèle logiciel ; rien non plus sur l'ampli dont il s'inspire.
- **La structure interne du plugin** : une boîte grise qui réussit ne dit pas que Nolly est
  construit ainsi.
- **La généralité** : un ampli, un canal, baffle coupé, une séance, 48 kHz seulement ; trois
  graines, une partition.
- **Le choix des bras** : leur conception (FIR 512, biais, enveloppes) vient des oracles VS100R,
  pas de Nolly.
- **L'écoute** : rien.

## Budget

Mesuré sur la VS100R, à 15 000 pas et à nombre de segments voisin (1 396) :

| Bras | Durée par run |
|---|---|
| C | 4 h à 5 h 30 |
| F3 | 2 h 10 |
| GREY | 51 min |
| L | 50 min |

Total : 12 runs, environ **26 h de GPU**, plus 1 h pour la porte et les essais.

## Ordre au moment du lancement

1. Lecture des sondes, `nolly_reading.py`, et ses portes.
2. `scripts/product_nolly_dataset.py --reading diagnosis/probe_metrics/nolly_reading.json`. Il
   construit `datasets/raw/internal/capture_v2/nolly_48k` et son manifeste (comptes, médiane,
   avance), ou refuse si une porte a échoué.
3. Runner `neural_amp_runner_nolly` épinglé sur le commit de ce texte, ou sur un commit qui ne
   touche ni ce texte ni ses scripts.
4. Lancement par `nohup setsid`, enchaîné après la file boîte grise par `kill -0`.
