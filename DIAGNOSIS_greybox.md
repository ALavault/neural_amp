# Diagnostic : écart d'ESR de la boîte grise façon KIPR sur la VS100R

Voie produit (D-PRODUCT-001), aucune revendication scientifique. Régime strict. Commandé par
D-0014 (research-memory, 2026-10-10) : avant d'approfondir la boîte grise (plus d'étages, plus
de mémoire), établir d'où vient son écart. Sections 1 à 3 écrites **avant toute nouvelle
mesure** ; scripts et tables dans `diagnosis/greybox/`.

## 1. Écart

- **Métrique** : lecture P de `demo/greybox/predictions.md`, ln ESR par prise sur `di_test` à
  48 kHz (prises non faibles, OD2, PD off), moyenne par couple gain-contour, couple non vu exclu.
- **Valeurs** (`demo/greybox/evaluation.json`) : s(C) = −3,952 / −3,967 / −3,929 et
  s(GREY) = −1,552 / −0,832 / −1,717 pour les graines 42, 43, 44 ; d = +2,58 (ESR × 13,2).
- **Dispersion** : d_i de +2,21 à +3,14 ; l'écart entre graines du bras C est de 0,04. L'écart
  dépasse de loin la dispersion : il est réel. Trois runs par bras.

## 2. Comparabilité

- **Jeu de test, données, prétraitement** : identiques (même script `evaluate.py`, mêmes prises,
  même décimation `resample_poly(1, 4)`, même fenêtre notée).
- **Protocole** : mêmes drapeaux que C (15 000 pas, lr 0,01, divisions aux pas de C, perte
  L1 + MR-STFT, `--honor-optim`), contrôle `ok` dans `demo/RUNS.jsonl` (1 396 segments).
- **Budget réel** : 15 000 pas sur 15 000 pour les deux bras.
- **Taille du jeu d'évaluation** : 12 couples, 29,4 s notées par prise ; l'écart (× 13) n'est
  pas à la limite de résolution.
- **Instrument, à vérifier** : (a) la chaîne recalculée doit redonner les s ci-dessus ;
  (b) défaut connu, commun aux bras : les modèles neuronaux devancent l'appareil d'environ
  0,8 échantillon et l'ESR brute le pénalise (co-lecture à retard compensé du bras C,
  `demo/train_length/eval_train_length_aligned.json` : s(C) passe de −3,95 à −5,52 / −5,63 /
  −5,44). GREY a des FIR causaux et peut avoir appris ce retard. La même co-lecture (règle de
  `coreading_delay.py`, inchangée) sera appliquée à GREY ; si GREY ne gagne rien, l'écart
  compensé est plus grand que l'écart brut, pas plus petit.

## 3. Hypothèses

**Lectures déjà faites, et ce qu'elles suggèrent (marquées comme telles)** :
- 2ᵉ harmonique (`demo/greybox/evaluation.json`) : écart de D2 à l'appareil +20,8 dB pour GREY
  contre +12,7 pour C à −20 dBFS, mais +3,0 / +2,8 à −10 dBFS. *Suggère* une asymétrie mal
  dosée à bas niveau (H2).
- Validation (`demo/runs/nablafx_vs100r48grey_s4*/logs/metrics.csv`) : perte plate vers 0,48 du
  pas 1 300 au pas 11 000 environ, puis chutes tardives (graine 44 : 0,48 → 0,42 vers le pas
  11 000 ; graine 42 : 0,47 → 0,44 sur les 200 derniers pas) ; le meilleur point de contrôle
  est le dernier. C plafonne vers 0,19–0,21. *Suggère* une optimisation inachevée (H4).

**Hypothèses, écrites avant toute nouvelle mesure.** Décomposition commune à H1–H3 : pour
chaque run et chaque prise, erreur e = y_appareil − y_modèle sur `di_test` ; des sondes
linéaires en paramètres, **emboîtées**, sont ajustées de x vers e sur la première moitié de la
fenêtre notée (A) et lues sur la seconde (B). Part captée = 1 − ‖e_B − ê_B‖² / ‖e_B‖². Même
décomposition de l'erreur de C, comme témoin.

- **H1, mémoire linéaire insuffisante (capacité linéaire).** Mécanisme : la partie linéaire de
  GREY (deux FIR de 32 coefficients, soit 0,67 ms, et un filtre du premier ordre avant et après
  chaque coupe-pic) ne reproduit pas la réponse linéaire de l'appareil (passe-haut vers 120 Hz
  dont la réponse dépasse 5 ms, retard de groupe dispersif, E-0037). Mesure : part de ‖e‖²
  captée par un FIR de 512 coefficients (10,7 ms). Prédiction : **soutenue si la part linéaire
  médiane ≥ 0,5, réfutée si ≤ 0,2.**
- **H2, forme statique (capacité non linéaire statique).** Mécanisme : un coupe-pic par branche
  et une asymétrie par gains e^±β, homogène de degré 1 : sous l'écrêtage, H2/H1 ne dépend pas du
  niveau, alors qu'une asymétrie douce donne H2/H1 croissant avec lui. Mesures : (a) part
  supplémentaire captée par un Hammerstein de degré 5 aux mêmes 512 coefficients (emboîte le
  FIR) ; (b) pente de H2/H1 entre −20 et −10 dBFS, GREY et appareil, lue dans
  `diagnosis/hammerstein/vs100r_hammerstein.json`. Prédiction : **soutenue si la part statique
  médiane ≥ 0,3 et si la pente de GREY est ≤ +2 dB pour 10 dB quand celle de l'appareil est
  ≥ +6 dB ; réfutée si la part statique ≤ 0,1.**
- **H3, mémoire non linéaire longue.** Mécanisme : l'appareil dépend du niveau passé sur 5 à
  300 ms (affaissement, déplacement de polarisation) ; GREY n'a qu'une enveloppe par branche,
  qui ne module que le gain d'attaque. Mesure : part supplémentaire captée en ajoutant au
  Hammerstein des branches modulées par des enveloppes de |x| à 5, 50 et 300 ms (env, x·env,
  x²·env ; emboîte H2). Prédiction : **soutenue si cette part médiane ≥ 0,15, réfutée si
  ≤ 0,05.**
- **H4, optimisation inachevée.** Mécanisme : paysage à plateaux ; à 15 000 pas, les
  coefficients ne sont pas à l'optimum que la structure permet. Mesure, **oracle** (ajusté sur la
  prise de test, étiqueté comme tel) : pour une prise, coefficients de GREY à ce réglage, puis
  ré-ajustement des seuls coefficients de structure (sans la tête) sur la moitié A, lecture sur
  B ; et le même ajustement depuis l'initialisation par défaut. Prédiction : **soutenue si
  l'oracle divise l'ESR de B par ≥ 3, réfutée si par ≤ 1,5** (la structure est alors à sa
  capacité, et H1 à H3 disent où).
- **H5, conditionnement par les boutons.** Mécanisme : la tête (2 → 32 → 40) n'ajuste pas les
  coefficients au réglage ; l'excès viendrait de certains réglages. Mesure : excès de ln ESR
  GREY − C par prise, face au gain et au contour. Prédiction : **soutenue si l'excès varie de
  ≥ 1,5 en ln entre prises et suit le gain (|ρ de Spearman| ≥ 0,6), réfutée si sa plage est
  ≤ 0,5.**

### Écart à la mesure prévue de H1–H3 (2026-10-10, avant toute mesure de remplacement)

**La décomposition par sondes échoue à son propre contrôle d'instrument.** Contrôle fait sur la
sortie de l'appareil elle-même, prise 1, avant toute erreur de modèle (`decompose.py`,
fonction `decompose` ; essai interactif) :
- Les deux moitiés de la fenêtre notée n'ont pas le même niveau : RMS de l'entrée 0,073 sur A,
  0,042 sur B (−5 dB) ; crête 0,46 contre 0,30.
- L'appareil sature fort (OD2). Son gain linéaire apparent dépend donc du niveau :
  - un FIR de 512 coefficients capte 0,61 de la sortie sur A et 0,17 sur B ;
  - Hammerstein 5, qui contient ce FIR, capte 0,77 sur A et **−0,23** sur B.
- Une crête de régularisation 100 fois plus forte ne corrige pas : Hammerstein 5 vaut encore
  −0,08 sur B.
- Les polynômes extrapolent mal hors de leurs amplitudes, et une saturation dure n'est pas
  polynomiale. C'est l'angle mort déjà connu des sondes sur un appareil à fort gain
  (`diagnosis/probe_metrics/REPORT.md`, Q-0014).

Les parts captées par ces sondes ne mesurent donc pas ce que H1–H3 demandent. Cette
mesure est abandonnée, sans aucun chiffre lu sur une erreur de modèle.

**Mesure de remplacement : oracles emboîtés par prise.** Elle fait varier la variable du
mécanisme dans la structure de GREY elle-même.

Cadre commun :
- **Prises** : les 8 prises de test (`split == "test"`).
- **Découpage** : blocs de 1 s alternés, A les impairs et B les pairs, de sorte que A et B
  aient la même distribution de niveaux.
- **Ajustement** : coefficients ajustés sur A (Adam, ESR), lus sur B, tous étiquetés
  **oracle**.
- **Point de départ (V0)** : structure de GREY sans tête, chargée avec les coefficients de GREY
  graine 42 au réglage de la prise, puis ré-ajustée sur A.
- **Variantes** : chacune part de la solution V0 et n'ajoute qu'un composant, initialisé à la
  neutralité. Elle contient donc V0.
  - **V1, portée linéaire (H1)** : FIR d'entrée et de sortie portés de 32 à 512 coefficients,
    les nouveaux à zéro.
  - **V2, forme statique (H2)** : un biais par branche avant le coupe-pic, dont on retranche la
    sortie statique (zéro en entrée donne zéro en sortie), initialisé à 0.
  - **V3, mémoire longue (H3)** : deux enveloppes lentes de |u| (50 et 300 ms) qui modulent le
    gain d'attaque et le biais, avec des poids initialisés à 0. V3 inclut V2 ; on la compare
    donc à V2.
  - **V4, profondeur (descriptif)** : un second étage complet en série, initialisé à
    l'identité par un chemin direct de poids 1 et des branches de poids 0.

Mesure : r = ESR_B(référence) / ESR_B(variante), en médiane sur les 8 prises. La référence
est V0 pour V1, V2 et V4, et V2 pour V3.

Prédictions, écrites avant ces mesures :
- **H1** : soutenue si r(V1) ≥ 2, réfutée si r(V1) ≤ 1,2.
- **H2** : soutenue si r(V2) ≥ 2 et si la pente de H2/H1 suit le critère (b) inchangé ;
  réfutée si r(V2) ≤ 1,2.
- **H3** : soutenue si r(V3) ≥ 1,5, réfutée si r(V3) ≤ 1,1.
- **H4**, mesure inchangée et faite sur ce même découpage : r = ESR_B(GREY conditionné) /
  ESR_B(V0). Soutenue si r ≥ 3, réfutée si r ≤ 1,5. On calcule V0 depuis les graines 42, 43
  et 44, et aussi depuis l'initialisation par défaut.

Témoin : l'ESR_B de C (graine 42) sur les mêmes blocs, qui montre ce qu'un modèle atteint.
Mesures inchangées : H2 (b), H5 et les deux contrôles d'instrument.

### Écarts issus de la relecture Codex (2026-10-10, en cours de mesure)

**Déjà vu à ce moment-là :**
- les oracles de 3 prises de test sur 8 (V0 à V4) ;
- 1 prise de la section de validation (V0, V1, V4) ;
- les contrôles d'instrument et la co-lecture compensée.

**Écarts :**
- **Témoins d'optimisation.** Chaque variante repart de son parent ajusté et reçoit 1 500 pas
  de plus sur tous ses coefficients. Son gain peut donc venir de l'optimisation supplémentaire,
  pas du composant ajouté, surtout si le parent n'a pas convergé (H4).
  - Deux témoins sont ajoutés, sans rien ajouter au modèle :
    - V0c : V0 ré-ajusté 1 500 pas de plus ;
    - V2c : V2 ré-ajusté 1 500 pas de plus.
  - **Les rapports de H1 à H3 se lisent désormais contre ces témoins**, au même nombre total de
    pas : r(V1) = ESR_B(V0c) / ESR_B(V1), de même pour V2 et V4, et r(V3) = ESR_B(V2c) /
    ESR_B(V3).
  - Les seuils ne changent pas. Les rapports contre V0 et V2 restent rapportés, à titre
    descriptif.
  - Les témoins tournent dans une passe séparée, avec la même graine 0 ; on vérifie que V0
    y est identique.
- **Contrôle des sondes** : les branches xᵏ étaient normalisées sur A et B ensemble, ce qui
  laissait passer des caractéristiques de l'entrée B dans l'ajustement sur A. Elles sont
  maintenant normalisées sur A seulement, et le contrôle est refait. La conclusion
  d'instrument est rapportée avec les nouveaux chiffres.

## 4. Mesures et verdict

Sorties :
- `diagnosis/greybox/decompose.json` : contrôles d'instrument, H2 (b), H5 ;
- `oracle.json`, `oracle_controls.json`, `oracle_di_validation.json` et
  `oracle_di_validation_controls.json` : les oracles ;
- `summary.json` : synthèse par `summarize.py`.

Les oracles sont ajustés sur la prise elle-même et étiquetés comme tels. Valeurs : médiane
[min, max] sur les 8 prises.

**Instrument :**
- **(a) Tenu.** Rendus refaits sur GPU, la chaîne redonne les s de la lecture P à moins de
  1e-5 près, pour les six runs.
- **(b) Défaut confirmé, il aggrave l'écart.** Retard commun estimé sur `di_validation` :
  −0,55, −0,40 et −0,50 échantillon pour GREY.
  - GREY gagne au plus 0,025 en ln (s compensé : −1,577, −0,838 et −1,732).
  - C gagne environ 1,6 en ln (−5,517, −5,633 et −5,436).
  - L'écart compensé est donc d = +4,15, soit une ESR × 63, contre × 13,2 brut.
- **Contrôle des sondes**, refait avec la normalisation sur A seul : Hammerstein 5 capte
  0,767 sur A et −0,235 sur B (crête relative 1e-4), puis −0,067 sur B (crête 1e-2). La
  conclusion ne change pas : la décomposition par sondes reste abandonnée.

**Témoins d'optimisation.** Prolonger V0 de 1 500 pas sans rien ajouter (V0c) divise
l'ESR_B par 1,044 sur test et par 1,044 sur validation.

**Déterminisme.** V0 n'est pas identique au bit d'une passe à l'autre, à cause des
arrondis GPU : l'écart relatif atteint 0,18 % sur test et 2,2 % sur validation, faible
devant les rapports lus.

| Hypothèse | Mesure | Test | Validation | Verdict |
|---|---|---|---|---|
| H1, portée linéaire | r(V1) contre V0c | **2,94** [2,05 ; 3,70] | **2,36** [2,01 ; 2,65] | **soutenue** (≥ 2) |
| H2, forme statique | r(V2) contre V0c | 1,025 [1,00 ; 1,06] | — | **réfutée** (≤ 1,2) |
| H2 (b), pente de H2/H1 de −20 à −10 dBFS | GREY / appareil (dB pour 10 dB) | +2,6 / +17,6 (C : +5,6) | — | critère non tenu (borne +2) ; ne change pas le verdict |
| H3, mémoire longue | r(V3) contre V2c | 1,017 [0,995 ; 1,034] | — | **réfutée** (≤ 1,1) |
| H4, optimisation | ESR_B(GREY) / ESR_B(V0), graines 42 / 43 / 44 | 1,67 / 2,26 / 1,49 | 1,71 (graine 42) | **indécidable** (entre 1,5 et 3) |
| H5, conditionnement | plage de l'excès ; ρ avec gain et contour | 2,48 ; −0,05 et −0,19 | — | **indécidable** : excès très inégal, mais sans lien avec les boutons |
| profondeur (descriptif) | r(V4) contre V0c | 1,21 [1,03 ; 2,16] | 1,19 [1,02 ; 1,63] | apport modeste en médiane, net sur 2 prises |

**Descriptif :**
- **Initialisation par défaut.** Partir des constantes par défaut au lieu des coefficients
  de GREY donne, en 1 500 pas, une ESR_B 3,7 fois plus grande. La structure s'optimise mal
  à froid.
- **Comparaison à C.** L'oracle V1 reste en médiane au double de C sur les mêmes blocs
  (0,038 contre 0,019 sur test, 0,050 contre 0,027 sur validation). Il ne passe sous C que
  sur 2 prises sur 8, dont la prise 16, où C est faible.
- **Profondeur selon les prises.** Le second étage aide surtout sur les prises 30
  (gain 10) et 33 (gain 7,5, contour 2,5) : ×2,2 sur test, ×1,5 à 1,6 sur validation.
  Il n'aide presque pas sur la prise 2, qui a pourtant le même réglage que la prise 33
  (×1,03).

## 5. Contribution retenue

**H1, portée linéaire.**
- **Prédicat** : l'écart se concentre dans ce que deux FIR de 32 coefficients (0,67 ms) et
  des filtres du premier ordre ne peuvent pas rendre. Ce sont la réponse linéaire longue de
  l'envoi FX (passe-haut vers 120 Hz, retard de groupe dispersif, E-0037).
- **Mesure** : porter les FIR à 512 coefficients (10,7 ms) divise l'ESR des oracles par 2,9
  (test) et 2,4 (validation), contre un témoin qui reçoit le même nombre de pas.
- **Ce que l'oracle ne mesure pas** : ni le biais, ni les enveloppes lentes, ni la forme
  statique n'apportent plus de 6 %.
- **Prédiction pour le correctif** : le bras L de `demo/greybox_deep/predictions.md` (GREY
  avec des FIR de 512 coefficients, entraîné comme GREY) doit atteindre d ≤ −0,7 contre
  GREY, soit une ESR au moins divisée par 2.

## 6. Non expliqué

- **Le reste de l'écart à C.** Même ré-ajusté prise par prise avec des FIR longs, l'oracle
  reste en médiane au double de C. Un modèle conditionné, entraîné sur toutes les prises,
  fera moins bien qu'un oracle.
  - Une part vient de l'entraînement (H4 indécidable : de 1,5 à 2,3 selon la graine).
  - Une autre vient de l'initialisation : à froid, la structure s'optimise mal.
  - Le reste n'est pas localisé.
- **La croissance de H2 avec le niveau** (+17,6 dB pour 10 dB sur l'appareil) n'est
  reproduite ni par GREY (+2,6) ni par C (+5,6). Elle ne pèse presque pas sur l'ESR (H2
  réfutée par l'oracle), mais reste un écart audible possible.
- **À tester :**
  - **La profondeur aide sur certaines prises seulement** (30 et 33), sans lien simple avec
    les boutons : les prises 2 et 33 ont le même réglage. Mesure : le bras F3 contre F1 de la
    piste, prise par prise.
  - **Si l'excès de GREY suit le passage de la séance** (les prises n'ont pas toutes été
    enregistrées dans le même passage Reaper) plutôt que le réglage. Mesure : l'excès par
    prise, groupé par passage (`passes.json`).
- **Non fait** : la décomposition par sondes de H1 à H3, abandonnée parce qu'elle échoue à
  son contrôle d'instrument sur cette saturation.
