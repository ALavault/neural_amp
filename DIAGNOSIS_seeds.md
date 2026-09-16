# DIAGNOSIS_seeds — dispersion inter-graines sur ToneTwist Big Muff

Scripts et tables : `diagnosis/seeds/` (`hypotheses.md`, `measure.py`,
`validation_loss.py`, `measurements.json`, `validation_loss.json`, `spread.txt`).
Mesures en lecture seule sur les douze runs de `demo/nablafx_bench/`.

## 1. Écart

Métrique : ESR de test du protocole NablAFx, moyenne des ESR de douze segments de
5 s, dernier checkpoint. Trois graines (42, 43, 44) par condition, douze runs.

| Condition | ESR par graine | Moyenne | Écart-type | CV |
|---|---|---|---|---|
| SSM-WaveNet 11,3k, ZOH + exemption + garde | 0,0358 / 0,1277 / 0,0832 | 0,0822 | 0,0460 | 56 % |
| S4-TF-L-16 70,2k, mêmes changements | 0,0888 / 0,1354 / 0,1568 | 0,1270 | 0,0348 | 27 % |
| S4-TF-L-16 70,2k, entraînement publié | 0,1373 / 0,1569 / 0,1171 | 0,1371 | 0,0199 | 15 % |
| SSM-WaveNet 12,4k, `b` appris, entraînement publié | 0,2363 / 0,0738 / 0,1656 | 0,1586 | 0,0815 | 51 % |

Référence : l'étude publiée ne rapporte qu'un run par modèle, donc aucune
dispersion de référence. L'écart que le papier revendique (0,0822 contre 0,1270,
soit 0,045) est du même ordre que l'écart-type intra-condition. Il n'est pas
« non distinguable du bruit » — la comparaison est appariée et SSM-WaveNet est
plus bas sur les trois graines — mais il n'est pas résoluble sur un run unique.
L'objet du diagnostic est la dispersion elle-même, et chaque condition a bien
trois graines.

## 2. Comparabilité

- **Jeu de test** : les douze mêmes segments de 5 s pour les douze runs
  (`test_segments.json`, champ `segment_rms` commun).
- **Données, métrique, protocole** : même archive Zenodo (sha256 dans
  `paper/icassp2027/README.md`), même rééchantillonnage, `auraloss` ESR par
  segment puis moyenne, dernier checkpoint, même code
  (`scripts/product_nablafx_bench.py`) avec le commit enregistré par run.
- **Budget réel** : plafond de 15k pas jamais atteint ; arrêt entre 4 137 et
  8 764 pas ; 11 329 / 12 353 / 70 193 paramètres.
- **Taille du jeu d'évaluation** : la moyenne sur douze segments a une erreur
  type de 19 à 24 % de sa valeur (`spread.txt`). Les runs partageant les mêmes
  segments, cette erreur n'explique pas les différences entre runs, mais elle
  interdit de comparer un run unique à une valeur publiée à mieux que ±20 %.

Aucun défaut de comparabilité : le diagnostic porte sur l'entraînement.

## 3. Hypothèses

Écrites avant toute mesure (`diagnosis/seeds/hypotheses.md`, commit `4f7809e`).

- **H1, point d'arrêt (optimisation)** : l'arrêt anticipé et le palier de
  learning rate s'appuient sur douze segments ; les runs s'arrêtent à des
  maturités très différentes. *Prédiction* : ρ(pas, ESR) ≤ −0,8 par condition.
- **H2, partition train/val (données)** : la graine choisit les douze segments
  retirés de l'entraînement, donc le signal d'arrêt et la matière retirée.
  *Prédiction* : l'ESR de la moitié calme suit une statistique de niveau de la
  partition.
- **H3, tirage des pôles (capacité)** : 512 pôles contre 4 096 pour S4 ; un
  tirage pauvre en modes lents n'est pas rattrapable. *Prédiction* : la meilleure
  graine a nettement plus de pôles à τ > 10 ms que la pire.
- **H4, définition de la métrique (mesure)** : la moyenne par segment est
  dominée par quelques segments. *Prédiction* : un ESR pondéré par l'énergie
  réduit le CV d'au moins 30 % en relatif.

## 4. Mesures et verdict

- **H1** — corrélation de rang pas/ESR : −1,00 dans les deux conditions « avec
  changements », **+1,00** dans les deux conditions « entraînement publié »,
  +0,09 sur les douze runs. Le signe s'inverse d'une condition à l'autre.
  **Indécidable** : trois points par condition ne séparent pas ±1 du hasard.
- **H2** — reconstruction des partitions : le RMS minimal du sous-ensemble de
  validation ordonne exactement l'ESR de test dans les deux conditions gardées,
  six runs sur six (SSM : 0,0308 → 0,036 ; 0,0201 → 0,083 ; 0,0002 → 0,128 ;
  S4 : 0,0284 → 0,089 ; 0,0201 → 0,135 ; 0,0127 → 0,157). Mais le mécanisme
  proposé est faux : le segment quasi silencieux de la graine 43, d'ESR 7,04,
  ne pèse que 6,3 % de la perte de validation. **Corrélation soutenue, mécanisme
  réfuté** ; statistique retenue après coup parmi cinq.
- **H3** — constantes de temps apprises : les runs avec exemption de weight decay
  ont 99 à 118 pôles au-delà de 10 ms contre 4 à 21 pour l'ablation, ce qui
  confirme l'effet du weight decay *entre* conditions ; à l'intérieur d'une
  condition, le nombre de pôles longs n'ordonne pas l'ESR (graine 44 : 118 pôles
  longs et ESR moyen ; graine 42 : 99 et meilleur ESR). **Réfutée** pour la
  dispersion intra-condition.
- **H4** — ESR pondéré par l'énergie : CV 50 % contre 56 % (SSM), 29 % contre
  27 % (S4), soit 11 % en relatif au lieu des 30 % prédits. **Réfutée.**

## 5. Contribution retenue

**Prédicat.** Dans les quatre conditions, 77 à 81 % de l'écart entre la meilleure
et la pire graine se situe sur les six segments de test les plus calmes (RMS
0,029–0,044) ; sur les six plus forts l'écart est trois à quatre fois plus petit.
Sur ces mêmes segments calmes, l'ESR vaut 2,6 à 4,8 fois celui des segments forts
pour les douze runs.

**Mécanisme.** Le critère qui pilote l'entraînement et la sélection — perte
L1 + 0,1 MR-STFT calculée sur le lot de validation — est une erreur *absolue*,
dominée par le matériel fort, alors que la métrique rapportée est une erreur
*relative* moyennée par segment, dominée par le matériel calme. La mesure le
montre directement : un segment de validation dont l'ESR vaut 7,04 ne pèse que
6,3 % de la perte de validation. Le régime de bas niveau n'est donc contraint ni
par l'arrêt anticipé, ni par le palier de learning rate, ni par le choix du
checkpoint : il est laissé à la graine. C'est cohérent avec la corrélation H2,
dont le mécanisme reste à établir.

**Prédiction chiffrée.** Un correctif qui aligne la sélection sur la métrique —
validation stratifiée par niveau, chaque tercile représenté, et sélection sur
l'ESR moyen par segment calculé au-dessus d'un plancher de niveau — doit ramener
le CV inter-graines de SSM-WaveNet (graines 42, 43, 44) de 56 % à ≤ 25 %, sans
dégrader la moyenne au-delà de 0,082, et doit ramener le rapport ESR
calme/fort sous 3,0 (aujourd'hui 2,6 / 3,7 / 3,9 sur ces trois runs).

## 6. Non expliqué, non vérifié

- La corrélation H2 reste sans mécanisme : six runs, statistique choisie après
  coup. Trois graines de plus par condition trancheraient.
- Part non attribuée : même en admettant le mécanisme, rien ne dit quelle
  fraction de la dispersion disparaîtrait. La seule borne mesurée est la
  localisation (77–81 % sur la moitié calme).
- **Variance à graine constante : mesurée après coup** (run
  `ssmzoh_guard_seed42_repeat`). L'entraînement n'est pas déterministe sur GPU
  (cuDNN benchmark, `use_deterministic_algorithms` à False, comme dans NablAFx) :
  la répétition de la graine 42 donne 0,0470 contre 0,0358, soit 0,0112, quand
  l'écart entre graines vaut 0,0919. La non-reproductibilité n'explique donc pas
  la dispersion, mais elle rend toute valeur individuelle incertaine à ±0,011.
  Une seule répétition : la distribution reste inconnue.

- Non séparés : initialisation, partition et ordre des lots changent ensemble
  avec la graine.

Correctif : voir skill controlled-fix.
