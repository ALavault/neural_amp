# Longueur d'entraînement par prise : 30 s ou 60 s au lieu de 150 s

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-05, avant tout
run.** Les règles ci-dessous ne changent plus après les chiffres.

## Question

Chaque prise VS100R contient 150 s de guitare pour l'entraînement (di_train 120 s et
di_validation 30 s, découpés en 50 segments de 3 s ; 10 % des segments, tirés au sort,
servent à la validation). Un modèle entraîné sur 30 s ou 60 s par prise est-il aussi bon ?

**Décision que la mesure change** : la longueur de la partie d'entraînement du fichier de la
prochaine campagne de captures. Une partie plus courte permet d'enregistrer plus de couples
gain–contour dans le même temps. Motivation, non établie : chez Miklánek et al. (DAFx-23,
`reports/neural_amp-research/2026-09-22/revue_va_neuronal.md`), un modèle boîte noire entraîné sur
4 min d'un seul réglage s'effondre sur les réglages non vus, et il tient avec plus de réglages ; mais
la durée totale et le nombre de réglages y changent ensemble, et le résultat dépend de
l'architecture. Rien n'y isole la durée par réglage, que ce test mesure.

## Bras

Tous : VS100R à 48 kHz, SSM-WaveNet ZOH, FiLM joint (32), tête `leaky`, `--honor-optim`,
`--deterministic`, `--split-seed 925`, L1 0,5 et MR-STFT 0,5, lot de 16, 15 000 pas,
`--fft-size fast --no-tf32 --no-early-stopping` (réglage des nouveaux prototypes depuis
`demo/fft_size/predictions_fast_notf32.md`), et `--lr-halving-steps` aux pas déjà rejoués par
le bras C de même graine (42 : 7 830 et 10 701 ; 43 : 11 484 et 14 355 ; 44 : 6 438 et 14 181).

- **Référence C** (existante) : `vs100r48fast_film_s42`, `_s43`, `_s44`, tous les segments
  d'entraînement (1 396 segments, 69,8 min).
- **L10** (nouveau) : `vs100r48len10_film_s42`, `_s43`, `_s44`, mêmes drapeaux plus
  `--train-segments-per-take 10` : segments aux positions 0, 5, 10, …, 45 de chaque prise
  (30 s nominales), 280 segments (14,0 min).
- **L20** (nouveau) : `vs100r48len20_film_s42`, `_s43`, `_s44`, avec
  `--train-segments-per-take 20` : positions 0, 2, 5, 7, …, 47 (60 s nominales), 564 segments
  (28,2 min).

**Ce que la troncature simule.** Toutes les prises jouent le même fichier DI : un fichier de
campagne plus court mettrait le même extrait à tous les réglages. Les positions gardées sont
donc les mêmes dans les 31 prises. Elles sont réparties sur tout le matériau (di_train et
di_validation) : cela correspond à un fichier court construit en enchaînant ces extraits, plus
varié qu'un seul passage continu de 30 s. Comme pour la référence, environ 10 % des positions
gardées tombent dans la validation et ne servent pas à l'entraînement.

**Appariement.** Le filtre s'applique après le tirage de la validation et ne consomme aucun
tirage aléatoire. Vérifié sur CPU avant les runs : la validation de L10 et de L20 est
identique à celle de C (mêmes 154 segments). À graine égale, l'initialisation est la même ;
l'ordre des lots diffère, puisque l'ensemble d'entraînement diffère.

**Budget fixe.** Les trois bras font 15 000 pas. L10 voit chaque segment environ 5 fois plus
souvent que C : le surapprentissage est attendu. D'où deux lectures (voir plus bas), écrites
avant les chiffres.

**Les pertes de validation ne se comparent pas entre bras.** Dans C, l'audio de chaque segment
de validation est entraîné à la même position dans d'autres prises ; dans L10 et L20, la
plupart des positions de validation sont de l'audio jamais entraîné. La validation sert
seulement à choisir le meilleur checkpoint à l'intérieur d'un bras.

## Contrôles avant et après les runs

- Avant : porte au bit du chemin par défaut sur 300 pas contre
  `vs100r_gc_film_joint_s925_r2`, comme pour la file FFT (le banc a changé :
  `--train-segments-per-take`, inactif par défaut). La file s'arrête si elle échoue.
- Avant : smoke de 300 pas de la référence et de L10, pour mesurer le surcoût de la
  validation plus fréquente (L10 : environ 17 pas par époque contre 87).
- Après chaque run : `val_indices` identique à celui du bras C de même graine, et
  `train_segments` égal à 280 ou 564. Sinon, le run est exclu et la raison notée.

## Évaluation (script `demo/train_length/eval_train_length.py`, commité avec ce texte)

Toutes les prises VS100R hors faible signal, test et test_seen (35 prises). Section di_test :
aucun bras ne l'a entraînée. Région notée et décimation de `product_vs100r_eval.py`. Inférence
avec la longueur de FFT par défaut.

**P, primaire.** Pour chaque run et le checkpoint `last` : moyenne des ln ESR des prises de
chaque couple gain–contour, puis moyenne sur les 12 couples. Le couple non vu (2,5 ; 7,5),
c'est-à-dire la prise 16, est exclu. Pour chaque graine, d_i = s(bras) − s(C) ; d = moyenne
des trois.
- « meilleur » si d ≤ −0,1 et les trois d_i < 0 ;
- « moins bon » si d ≥ +0,1 et les trois d_i > 0 ;
- « équivalent » si |d| ≤ 0,05 et les trois |d_i| ≤ 0,15 ;
- sinon, non tranché.

**Pourquoi P et non la métrique M1 du test FFT** (8 prises de test). On sait avant les runs que
M1 est dominée par la seule prise 16 (`demo/fft_size/predictions_fast_notf32.md`, résultats :
écarts de ln ESR de +0,76, −1,70 et +0,61 sur cette prise, contre des moyennes de −0,035, +0,048 et
+0,007 sur les sept autres, où les écarts par prise vont de −0,16 à +0,14).
L'agrégation par couple évite aussi que le couple (5 ; 5), qui a 15 prises, pèse pour plus de
40 % du total.

**Lectures secondaires.**
- La même lecture P avec le checkpoint `best` : celui que le banc retient (sauvegarde tous les
  100 pas, classée sur la dernière perte de validation connue au moment de la sauvegarde). Ce
  n'est pas exactement le minimum de validation : un minimum atteint entre deux sauvegardes peut
  être manqué, dans tous les bras.
- Rapportés sans seuil : M1 sur les 8 prises de test (continuité avec le test FFT) et la
  prise 16 seule (seul couple non vu), pour `last` et `best`.

**Seuil d'acceptation (indépendant du verdict).** Une lecture P est « acceptable » si d ≤ +0,10
et si chaque d_i ≤ +0,25 : ESR moyenne au plus 10,5 % plus haute, et aucune graine plus de 28 %
plus haute. « Non tranché » ne suffit pas.

**Lecture fixée d'avance, descriptive.** Elle dit ce qui est observé sous ce protocole, sans
attribuer de cause. Une perte en `last` rattrapée en `best` peut venir du surapprentissage, de
l'optimisation ou de la sélection ; une perte dans les deux peut venir du manque d'audio comme de
ces mêmes causes. Ce test ne les sépare pas.

## Règle de décision pour la campagne

La partie d'entraînement du fichier de campagne prend la plus courte longueur parmi 30 s (L10) et
60 s (L20) dont la lecture P est « acceptable » en `last`, ou, à défaut, en `best` (il faut alors
garder la sélection de checkpoint sur la validation pour la campagne). Si aucune ne l'est, elle
reste à 150 s. On tolère ainsi une ESR moyenne jusqu'à 10,5 % plus haute, en échange de plus de
couples par séance. Dans tous les cas, la partie de test (30 s de DI, jamais
entraînée) et les signaux de mesure (silence, bruit rose, trois sinus glissants) restent
identiques aux 39 prises existantes.

## Ce qui ne pourra pas être conclu

- Rien sur un autre appareil, ni sur un autre modèle.
- Rien sur un extrait continu (un seul passage de 30 s) : seuls des extraits répartis sont
  testés.
- Rien sur des signaux synthétiques à la place de la guitare.
- Peu de chose sur les réglages non vus : un seul couple (prise 16), rapporté sans seuil.
- Avec 3 paires, seuls des effets nets ressortent.

## Co-lecture à retard compensé (règle ajoutée le 2026-10-09, avant la lecture P)

**Écrite avant que l'évaluation ne tourne.** `eval_train_length.json` n'existe pas, et aucun
chiffre de la lecture P n'a été calculé pour L10 ou L20. En revanche, les ESR de test du banc
(`test_last` dans `demo/nablafx_bench/vs100r48len*.json`) existent déjà sur disque pour les six
runs L10 et L20 : elles portent sur une autre agrégation. La session qui écrit cette section n'en
a pas affiché les valeurs ; on ne peut pas exclure qu'une session précédente les ait vues.

**Ordre d'exécution.** La lecture P tourne d'abord, avec `eval_train_length.py` inchangé depuis
36d08bb. La co-lecture tourne ensuite, dans un script séparé qui ne modifie ni ce script ni son
JSON. Elle est rapportée comme « règle écrite avant P, exécutée après P ».

**Pourquoi.** Sur les modèles VS100R à 48 kHz, un retard *oracle* ajusté par prise sur la section
de test vaut 0,80 à 0,85 échantillon en médiane, et ce réajustement divise l'ESR par 2 à 9,4 sur
la plupart des prises (`DIAGNOSIS_vs100r_192k.md`). L'origine de ce décalage n'est pas établie : un
oracle peut absorber aussi une erreur de phase du modèle. Si c'est un retard pur (environ 18 µs),
il est inaudible. Une différence d'ESR brute entre bras peut donc venir en bonne partie de la façon
dont chaque bras approche cette phase sous-échantillon. La lecture P préinscrite n'en tient pas
compte. Cette co-lecture le fait sans oracle sur le test, sans remplacer P.

**Règle.** Pour chaque run des trois bras, checkpoint `last` :
- un seul retard commun τ, sur la grille de −2 à +3 échantillons au pas de 0,05, celui qui
  minimise l'ESR médiane sur les 35 prises hors faible signal, **section `di_validation`** (30 s,
  hors section notée ; c'est du matériau d'entraînement pour la plupart des prises) ;
- ce retard est appliqué tel quel à la sortie du modèle sur `di_test`, sans nouvel ajustement :
  pas d'oracle par prise de test, qui absorberait aussi des erreurs de phase ou de timbre ;
- retard fractionnaire par rampe de phase FFT (`delayed()` de
  `demo/capture/valvestate/timing_shift.py`), après un remplissage de 0,1 s de zéros de chaque
  côté contre le repliement circulaire. Le noyau d'un retard fractionnaire reste à support
  infini : un effet de bord résiduel subsiste dans la région notée (`scored()`, inchangée). Il
  est mesuré (contrôles ci-dessous), pas supposé nul ;
- mêmes agrégats que P (moyenne des ln ESR par couple gain–contour, couple non vu exclu ; d_i
  par graine ; d), mêmes étiquettes et même seuil d'acceptation, calculés à titre descriptif. Les
  retards τ de chaque run sont rapportés : un retard qui change selon le bras est en soi une
  information.

**Lecture fixée d'avance.**
- Si P et la co-lecture donnent la même étiquette et la même acceptabilité : le résultat est
  rapporté comme robuste au calage.
- Si elles divergent : les deux sont rapportées, et l'écart est attribué au calage sous ce
  protocole, sans autre cause. **La règle de décision pour la campagne reste celle de P.** Une
  divergence sur l'acceptabilité est soumise à l'utilisateur, qui tranche ; elle ne modifie pas
  la règle d'elle-même.

**Contrôles de l'instrument.**
- **Effet de bord.** Sur la première prise de la liste, la sortie FX de `di_test` est retardée de
  0,8 échantillon de deux façons : sur la section seule (comme la co-lecture), et sur la section
  prolongée d'une seconde de chaque côté puis recoupée. L'énergie de leur différence dans la
  région notée, rapportée à celle du signal, doit rester sous 1 % de la plus petite ESR lue par
  la co-lecture. Sinon, la co-lecture est déclarée non fiable et n'est pas lue.
- **Bord de grille.** Un run dont τ tombe sur −2 ou +3 est exclu de la co-lecture, avec sa
  raison ; les paires de graines qui en dépendent ne sont pas lues.
- **Retard du bras C.** Les τ des trois runs C sont rapportés sans seuil ; d'après
  `DIAGNOSIS_vs100r_192k.md`, on attend une valeur absolue proche de 0,8.

## Résultats (2026-10-09, après les neuf runs)

Évaluation lancée le 2026-10-09 sur le feu vert de l'utilisateur. Les six contrôles après run
sont passés : `val_indices` est identique au bras C de même graine, avec 280 ou 564 segments
d'entraînement (`demo/RUNS.jsonl`). `eval_train_length.py` n'a pas changé depuis 36d08bb ; il a
tourné sur CPU et sa sortie est `eval_train_length.json`. Contrôle de cohérence : les M1 de C en
`last` (−3,515 ; −3,677 ; −3,477) sont identiques aux s(C) du test FFT (`eval_fast.json`).

### Lecture P (primaire, checkpoint `last`)

| graine | s(C) | s(L10) | s(L20) | d_i L10 | d_i L20 |
|---|---|---|---|---|---|
| 42 | −3,952 | −4,043 | −3,849 | −0,091 | +0,104 |
| 43 | −3,967 | −4,005 | −3,978 | −0,038 | −0,011 |
| 44 | −3,929 | −4,154 | −3,905 | −0,225 | +0,024 |

- **L10 : « meilleur »** (d = −0,118, les trois d_i < 0) ; acceptable.
- **L20 : « équivalent »** (d = +0,039, tous les |d_i| ≤ 0,15) ; acceptable.
- **`best`** :
  - L10 non tranché (d = −0,029 ; +0,080, +0,082, −0,249) ;
  - L20 non tranché (d = +0,076 ; +0,009, +0,201, +0,017) ;
  - les deux sont acceptables.
- **M1, descriptif, `last`** : L10 d = −0,078 (−0,040 ; +0,082 ; −0,275) ; L20 d = 0,000
  (−0,066 ; +0,153 ; −0,086).
- **Prise 16, descriptif, `last`** : L10 −0,052 ; +0,544 ; −0,954. L20 −0,996 ; +1,089 ;
  −0,836. Les |d_i| atteignent environ 1 et changent de signe selon la graine, comme au test
  FFT.
- **M1 et prise 16, descriptif, `best`** :
  - M1 : L10 d = −0,010 (+0,096 ; +0,166 ; −0,291) ; L20 d = +0,015 (−0,125 ; +0,290 ;
    −0,122).
  - Prise 16 : L10 +0,272 ; +0,485 ; −0,944. L20 −0,879 ; +0,897 ; −1,030.
- **Règle de décision appliquée à P** : L10 est acceptable en `last`, donc la partie
  d'entraînement du fichier de campagne prendrait 30 s.

### Co-lecture à retard compensé

La règle a été figée en 6f62341, avant P ; la co-lecture a été exécutée après P. Le script est
`coreading_delay.py` (63b5e90, relu par Codex) et la sortie `eval_train_length_aligned.json`.

**Écart d'instrument : l'inférence tourne sur GPU** (fp32, sans TF32), alors que P tourne sur
CPU. Contrôle de parité (`parity_gpu_cpu.py`, sortie `parity_gpu_cpu.json`) : sans retard, sur
les neuf runs et les 35 prises, la ln ESR du chemin GPU s'écarte de celle de P d'au plus 1,9e-5.
Un d_i peut donc bouger d'au plus 3,9e-5.

**Contrôles de l'instrument** : tous passés.
- Effet de bord : 8,6e-13, pour une borne de 1,0e-5.
- Aucun τ sur un bord de grille.
- τ de C : +1,00 ; +0,95 ; +1,05. On attendait environ 0,8 en valeur absolue. Mais
  `DIAGNOSIS_vs100r_192k.md` utilisait un oracle par prise de test, et non un retard par run
  estimé sur `di_validation`.

| graine | τ C | τ L10 | τ L20 | s(C) | s(L10) | s(L20) | d_i L10 | d_i L20 |
|---|---|---|---|---|---|---|---|---|
| 42 | +1,00 | +0,85 | +1,00 | −5,517 | −5,158 | −5,263 | +0,359 | +0,254 |
| 43 | +0,95 | +0,85 | +1,00 | −5,633 | −5,011 | −5,855 | +0,622 | −0,222 |
| 44 | +1,05 | +0,80 | +1,05 | −5,436 | −5,371 | −5,688 | +0,064 | −0,252 |

- **L10 : « moins bon »** (d = +0,348, les trois d_i > 0) ; **non acceptable**.
- **L20 : « non tranché »** (d = −0,074, signes mélangés) ; **non acceptable**. d₄₂ = +0,254
  dépasse la borne par graine de 0,25 de 0,004, soit environ cent fois l'effet possible du
  device (3,9e-5).
- **Le calage abaisse s en moyenne de** 1,58 (C), 1,11 (L10) et 1,69 (L20) : l'ESR est divisée
  par environ 4,9, 3,0 et 5,4.
- **Le retard dépend du bras** : 0,80 à 0,85 pour L10, contre 0,95 à 1,05 pour C et L20.
- **M1, descriptif** : L10 d = +0,187 (+0,222 ; +0,447 ; −0,108) ; L20 d = −0,077 (−0,046 ;
  +0,075 ; −0,261).

**Confrontation, selon la lecture fixée d'avance.**
- Les deux lectures divergent pour les deux bras, sur l'étiquette comme sur l'acceptabilité.
- L'écart est attribué au calage sous ce protocole, sans autre cause.
- La règle de décision reste celle de P (30 s). La divergence sur l'acceptabilité est soumise à
  l'utilisateur, qui tranche.

### Contrôle post hoc, hors règle : τ estimé sur `di_test`

**Motif.** La co-lecture estime τ sur `di_validation`. Pour C, cette section est presque
entièrement entraînée ; pour L10 et L20, elle est en majorité non vue. Ici, τ est un retard
commun par run, jamais par prise, estimé sur `di_test`, qu'aucun bras n'a vu. Script :
`posthoc_test_delay.py` ; sortie : `eval_train_length_posthoc_test_tau.json`. Ce contrôle ne
change aucune étiquette.

**Retards obtenus** :
- C : +1,00 ; +1,05 ; +1,05 ;
- L10 : +0,85 ; +0,80 ; +0,75 ;
- L20 : +0,95 ; +0,90 ; +1,10.

**Lectures** :
- **L10** : d = +0,358 (+0,359 ; +0,617 ; +0,097), les trois d_i > 0.
- **L20** : d = −0,031 (+0,275 ; −0,137 ; −0,229).

**Conséquence.** L'inversion de L10 persiste avec les deux sections d'estimation examinées. Le
contrôle n'est pas une évaluation indépendante : `di_test` sert à ajuster τ puis à noter. Après
calage, l'ESR de L10 sur `di_test` est plus haute que celle de C, d'environ exp(0,36) ≈ 1,4 fois
en moyenne géométrique. Son avantage en ESR brute accompagne un τ plus petit : 0,75 à 0,85 pour
L10, contre 1,00 à 1,05 pour C dans ce contrôle. Le calage peut compenser un retard pur comme une
erreur de phase du modèle. Ce test ne change aucune étiquette préinscrite et n'établit pas de
cause.

### Ce qui reste à trancher (utilisateur)

**La règle préinscrite suit P** : 30 s (L10), dans le périmètre testé.

**Si l'utilisateur retient l'ESR après calage**, L10 et L20 ne sont pas acceptables dans la
co-lecture. Celle-ci ne porte que sur `last` : le repli sur `best` de la règle initiale n'y a pas
d'équivalent. Garder 150 s serait alors une décision de l'utilisateur, hors règle.

**L'effet d'un réentraînement à cible avancée sur cette comparaison reste inconnu** : il n'a pas
été mesuré.

## Décision (utilisateur, 2026-10-10)

> « 150 s, on garde la longueur actuelle. Pour le moment. Dans l'idéal, contenir les 30s dans les
> 150 (une pierre, deux coups) »

**La partie d'entraînement du fichier de la prochaine capture reste à 150 s**, à titre provisoire.
C'est une décision de l'utilisateur, hors règle : la règle préinscrite donnait 30 s (L10, d'après
P).

**Les 30 s restent contenues dans les 150 s, par construction.** `--train-segments-per-take 10`
garde les positions `k × 50 // 10` du découpage en 50 segments de 3 s
(`keep_segments_per_take`, `scripts/product_nablafx_bench.py`) : les positions 0, 5, …, 45, dont
deux (40 et 45) tombent dans `di_validation`. Une capture faite avec le même fichier sert donc aux
deux longueurs, à une condition : `di_train` et `di_validation` (150 s, 50 × 3 s) doivent rester
identiques au bit près à ceux de `reamp_valvestate_192k.wav`, dans le même ordre. Les nouveaux
signaux de mesure se placent hors de cette partie, et `layout.json` est régénéré. Le contrôle porte
sur une empreinte des seuls échantillons de cette partie, l'empreinte du fichier entier changeant
avec les ajouts.

Le sous-ensemble ainsi contenu reste celui de ce test : des extraits répartis, joués dans leur
contexte de 150 s. Un bloc continu de 30 s serait un autre plan, non testé ici.
