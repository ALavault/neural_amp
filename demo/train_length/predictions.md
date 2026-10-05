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
