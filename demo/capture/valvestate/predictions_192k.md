# VS100R : entraîner à 192 kHz puis décimer, contre entraîner à 48 kHz

Voie produit (D-PRODUCT-001), aucune revendication scientifique. **Écrit le 2026-10-02, avant tout
run et avant le code d'entraînement.** Les règles ci-dessous ne changent plus après les chiffres.

## Origine et question

Le transfert d'un modèle à 192 kHz par rééchelonnement de Δ, sans réentraînement, ne rapproche pas
du réel (`demo/sr_transfer/predictions_brightness.md`, lecture par bande). L'utilisateur propose
le test qui a un sens : la capture VS100R existe à 192 kHz. On entraîne un modèle directement à
192 kHz, on décime sa sortie à 48 kHz, et on le compare à l'écoute et en mesure au modèle entraîné
à 48 kHz. Les deux sont jugés contre la même référence réelle : la capture à 192 kHz décimée par
`resample_poly(·, 1, 4)`, la décimation qui a produit le jeu à 48 kHz.

Question : le modèle entraîné à 192 kHz puis décimé est-il plus proche du réel que le modèle
entraîné à 48 kHz ?

## Bras

Les deux bras partagent : SSM-WaveNet ZOH, FiLM joint (32), tête `leaky`, `--honor-optim`,
`--deterministic`, `--split-seed 925`, L1 0,5 et MR-STFT 0,5, lot de 16, 15 000 pas.

- **A48** : appareil `vs100r` (jeu `explore_48k`, segments de 144 000 échantillons, soit 3 s).
  Graine 42 : `vs100r_gc_film_joint_s925_r2`, déjà entraîné. Graines 43 et 44 : nouveaux runs
  `vs100r48_film_s43` et `_s44`, mêmes drapeaux, avec `--no-early-stopping`. Le run de graine 42
  avait l'arrêt anticipé actif, mais il ne s'est pas déclenché : il est allé aux 15 000 pas.
- **B192** : appareil `vs100r_192k` (jeu `explore_192k`, mêmes prises et mêmes coupes, non
  décimées). Runs `vs100r192_film_s42`, `_s43`, `_s44`.

## Choix fixés pour B192, et leurs confondants

- **Segments de 144 000 échantillons**, soit 0,75 s de son, pour le même coût et la même mémoire
  par pas. 3 s à 192 kHz ne tiennent pas sur la carte. À pas égal, B192 voit 4 fois moins de durée
  de son.
- **Calendrier du taux d'apprentissage rejoué par pas.** L'ordonnanceur à plateau compte des
  époques de validation. À 192 kHz, une époque contient environ 4 fois plus de pas, et la perte
  de validation n'est pas la même. Chaque run B192 divise donc le taux aux pas exacts où son
  jumeau A48 l'a divisé (graine 42 : pas 7 830 et 10 701, lus dans
  `demo/runs/nablafx_vs100r_gc_film_joint_s925_r2/logs/metrics.csv`). Il n'a pas d'arrêt
  anticipé. Le plateau ne décide rien dans B192.
- **MR-STFT à la même résolution en secondes et en Hz** : tailles de FFT, pas et fenêtres
  multipliés par 4 en échantillons (FFT 4096, 8192, 2048 ; pas 480, 960, 200 ; fenêtres 2400,
  4800, 960). Confondant déclaré : le terme log-magnitude moyenne sur les cases, et environ 75 %
  des cases sont au-dessus de 24 kHz. Ce domaine contient surtout des harmoniques de l'ampli et le
  bruit de l'appareil, l'entrée bouclée étant limitée vers 24 kHz. C'est ce que veut dire
  « entraîner à 192 kHz » ; une perte calculée après décimation serait une autre expérience.
- **Initialisation de Δ** : log Δ décalé de log(48 000 / 192 000) après construction du modèle,
  pour que les constantes de temps initiales soient les mêmes en secondes. Le décalage ne doit
  consommer aucun tirage aléatoire (contrôle ci-dessous).
- **Bras non appariés** : l'ordre des données diffère, puisqu'il y a environ 4 fois plus de
  segments. La réduction de variance du protocole apparié ne s'applique pas.

## Contrôles avant le premier run (la file s'arrête si l'un échoue)

1. **Porte au bit du chemin A48** : 300 pas de `vs100r_gc_film_joint_s925_r2` rejoués avec le code
   modifié, identiques au bit à leur référence (`diagnosis/seeds/repro_gate.py`).
2. **Données à 192 kHz** : chaque fichier de `explore_192k` est à 192 000 Hz, et le jeu de
   données le charge à 192 000 Hz sans rééchantillonnage (`sample_rate` lu dans la spécification
   de l'appareil). Un segment chargé a de l'énergie au-dessus de 24 kHz.
3. **Initialisation** : à graine égale, tous les paramètres sauf `log_dt` sont identiques au bit
   entre un modèle 48 kHz et un modèle 192 kHz, et `log_dt` diffère exactement de log(1/4).
4. **Pré-vol à 192 kHz** : quelques dizaines de pas sans enregistrement, pour mesurer la mémoire
   maximale et le temps par pas. `GPU_FREE_MIB` est fixé d'après cette mesure.

## Évaluation, à 48 kHz (script `demo/capture/valvestate/eval_192k.py`, pas encore écrit)

Les 8 prises de test (`split == "test"`, hors faible signal), région notée de
`product_vs100r_eval.py`. A48 reçoit l'entrée bouclée décimée. B192 reçoit l'entrée bouclée à
192 kHz, et sa sortie est décimée par `resample_poly(·, 1, 4)`. L'ESR de test enregistrée par le
banc pour B192 est calculée à 192 kHz : elle n'est pas comparable et n'est pas lue.

**M1, ESR.** Pour chaque run, s = moyenne sur les 8 prises de ln ESR. d = moyenne des s de B192
moins moyenne des s de A48 (3 graines par bras). σ = écart-type commun des s à l'intérieur des
bras.
- « B192 plus proche » si d ≤ −ln 1,5 et d < −2 σ √(2/3).
- « A48 plus proche » si d ≥ +ln 1,5 et d > +2 σ √(2/3).
- Sinon, non tranché.

**M2, niveaux par bande.** Tiers d'octave de 63 Hz à 16 kHz (Welch 8192, Hann) sur les régions
notées mises bout à bout. Pour chaque run : e_b = L_run(b) − L_réel(b). On lit le tiers d'octave
de 16 kHz explicitement, ainsi que la moyenne de |e_b| sur 1 à 16 kHz. Le tableau complet par bande
est rapporté.
- « B192 plus proche dans l'aigu » si la moyenne sur graines de |e_16k| est inférieure d'au moins
  0,5 dB pour B192, et pour au moins 2 graines sur 3 prises une à une. Règle symétrique pour A48.
  Sinon, non tranché.

**Écoute** : une section de la page d'écoute (`demo/listening/results.html`) avec le réel à
48 kHz, A48 et B192 décimé, en mode aveugle possible. Sans lecture chiffrée.

**Descriptif, sans lecture** : la part d'énergie hors grille harmonique sur sinus, avec les
corrections relevées par Codex le 2026-10-02 (contrôle de l'établissement du réseau, spectres
conservés).

## Engagement sur l'extension

Les graines 43 et 44 des deux bras sont lancées quoi qu'on entende à la graine 42. Ordre de la
file : porte A48, pré-vol, `vs100r192_film_s42`, `vs100r48_film_s43`, `vs100r192_film_s43`,
`vs100r48_film_s44`, `vs100r192_film_s44`. Chaque B192 attend son jumeau A48, dont il rejoue le
calendrier. Écouter la graine 42 seule ne permet aucun classement.

## Ce qui ne pourra pas être conclu

- Rien sur un modèle entraîné à 192 kHz avec une perte limitée à 24 kHz : c'est une autre
  expérience.
- Rien hors VS100R, OD2, Power Dimension off.
- Avec 3 graines par bras, seuls de grands écarts peuvent ressortir (σ attendu autour de 0,4 sur
  ln ESR).

## Précisions déclarées avant les runs (2026-10-02, après les contrôles 2 et 3 et la relecture Codex)

- Contrôle 2 tenu : 140 fichiers à 192 000 Hz, chargés à 192 000 Hz sans rééchantillonnage ;
  5 581 segments d'entraînement contre 1 396 à 48 kHz (348 pas par époque contre 87) ; part
  d'énergie au-dessus de 24 kHz de 3·10⁻⁵ pour la cible et 1·10⁻⁵ pour l'entrée sur un segment.
- Contrôle 3 tenu : à la graine 42, tous les paramètres sauf `log_dt` sont identiques au bit, et
  le tirage suivant aussi ; `log_dt` diffère de log(1/4) à 4,8·10⁻⁷ près, l'arrondi float32
  (« exactement » plus haut s'entend à cet arrondi près).
- La MR-STFT par réglage que le banc enregistre au test (`metrics_by_setting`) garde les tailles
  de 48 kHz pour B192. Comme l'ESR de test du banc à 192 kHz, elle n'est pas lue.
- Les segments de test du banc font 240 000 échantillons à 192 kHz (1,25 s). Sans effet sur
  l'évaluation pré-inscrite, faite à 48 kHz par `eval_192k.py`.
