# FFT de 288 000 points sans TF32 : entraînement complet contre le réglage actuel

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-04, avant tout
run.** Les règles ci-dessous ne changent plus après les chiffres. Mesures préalables :
`demo/fft_size/NOTE.md`.

## Question

Sur un lot, « FFT de 288 000 points sans TF32 » est 16 % plus rapide, prend 3,8 Gio de moins et
donne un gradient bien plus proche du float64 (0,47 % contre 21 %) que le réglage actuel. Un
entraînement complet avec ce réglage donne-t-il un modèle au moins aussi bon ?

**Décision que la mesure change** : adopter `--fft-size fast --no-tf32` par défaut pour les
nouveaux prototypes, ou non.

## Bras

Tous : VS100R à 48 kHz (`--device vs100r`), SSM-WaveNet ZOH, FiLM joint (32), tête `leaky`,
`--honor-optim`, `--deterministic`, `--split-seed 925`, L1 0,5 et MR-STFT 0,5, lot de 16,
15 000 pas.

- **Témoin A** (existant, réglage actuel : FFT pow2, TF32) : `vs100r_gc_film_joint_s925_r2`
  (graine 42), `vs100r48_film_s43`, `vs100r48_film_s44`.
- **Bras C** (nouveau) : `vs100r48fast_film_s42`, `_s43`, `_s44`, mêmes drapeaux plus
  `--fft-size fast --no-tf32 --no-early-stopping`, et `--lr-halving-steps` aux pas de division
  enregistrés par le témoin de même graine (42 : 7 830 et 10 701 ; 43 : 11 484 et 14 355 ;
  44 : 6 438 et 14 181).

**Appariement.** Ni la longueur de FFT ni le TF32 ne consomment de tirage aléatoire : à graine
égale, C part de la même initialisation et voit les mêmes lots dans le même ordre que A. Avec le
calendrier rejoué, seule la précision numérique diffère. Écarts déclarés : le témoin de graine 42
avait l'arrêt anticipé actif, mais il ne s'est pas déclenché ; le témoin de graine 44 a eu un
retournement de polarité à l'époque 0, que C peut ne pas reproduire.

## Contrôle avant les runs (la file s'arrête s'il échoue)

Porte au bit du chemin par défaut (FFT pow2, TF32) sur 300 pas contre
`vs100r_gc_film_joint_s925_r2`, au commit de la file : le code a changé depuis la dernière
porte (option `fft_size`).

## Évaluation (script `demo/fft_size/eval_fast.py`, pas encore écrit)

Les 8 prises de test VS100R (`split == "test"`, hors faible signal), région notée et décimation de
`product_vs100r_eval.py`, comme `demo/capture/valvestate/eval_192k.py`. L'inférence se fait avec
la longueur de FFT par défaut pour tous les modèles.

**M1, ESR appariée.** Pour chaque run, s = moyenne sur les 8 prises de ln ESR. Pour chaque graine,
d_i = s(C) − s(A) ; d = moyenne des trois d_i.
- « C meilleur » si d ≤ −0,1 et les trois d_i < 0.
- « C moins bon » si d ≥ +0,1 et les trois d_i > 0.
- « Équivalent » si |d| ≤ 0,05 et les trois |d_i| ≤ 0,15.
- Sinon, non tranché.

**M2, aigu.** Écart moyen |e| au tiers d'octave de 16 kHz (niveaux par bande contre le réel,
Welch 8192), moyenne sur les graines de C moins celle de A. Lecture descriptive : rapportée
sans seuil.

**Temps et mémoire (descriptif).** `minutes_this_session` et `peak_gpu_reserved_gib` des rapports
du banc, par paire. Le GPU est partagé, donc la durée de bout en bout est bruitée ; le gain de
vitesse attendu reste celui du micro-benchmark (−16 %).

**Règle d'adoption.** `--fft-size fast --no-tf32` devient le réglage par défaut des nouveaux
prototypes si M1 n'est pas « C moins bon » et si le pic mémoire de C est inférieur d'au moins
2 Gio à celui de A. Sinon, on garde le réglage actuel. Dans les deux cas, les runs existants ne
changent pas.

## Ce qui ne pourra pas être conclu

- Rien sur un autre appareil que le VS100R, ni à une autre fréquence.
- Rien de séparé sur la FFT et le TF32 : le bras C change les deux à la fois. Il faudrait un bras
  par facteur pour les séparer.
- Avec 3 paires, seuls des effets nets ressortent ; l'appariement réduit le bruit sans
  l'annuler (la précision numérique fait diverger les trajectoires au fil de l'entraînement).
